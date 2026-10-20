// Frame operations beyond the basics: groupby, merge, rolling windows, time parsing and resampling, reshaping,
// summaries, and array files (.npy / .safetensors).
// pycli ops covered: frame_groupby, frame_merge, frame_rolling, frame_resample, frame_select (sort/where), frame_inspect,
// frame_export, array_import/array_inspect (npy, safetensors), frame_to_array(_to)/array_to_frame.
#include "frame.hpp"
#include "out.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <set>

namespace lmnum {
namespace {

const std::string Ft = "Frame", Vt = "Vec", St = "StrVec", Mt = "Mat", Rt = "Res";
Frame& ff(void* s) { return *static_cast<Frame*>(s); }

// ---- aggregation ----------------------------------------------------------------------------------------------------
const char* const AGGS = "sum, mean, count, size, min, max, std, var, median, first, last, nunique, prod";
double aggregate(const std::string& agg, const std::vector<double>& all) {
  std::vector<double> v;
  for (double x : all) if (!std::isnan(x)) v.push_back(x);
  if (agg == "size") return double(all.size());
  if (agg == "count") return double(v.size());
  if (agg == "sum") { double s = 0, c = 0; for (double x : v) { double t = s + x; c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s; s = t; } return s + c; }
  if (agg == "prod") { double p = 1; for (double x : v) p *= x; return p; }
  if (v.empty()) return NAN;
  if (agg == "mean") { double s = 0; for (double x : v) s += x; return s / double(v.size()); }
  if (agg == "min") return *std::min_element(v.begin(), v.end());
  if (agg == "max") return *std::max_element(v.begin(), v.end());
  if (agg == "first") return v.front();
  if (agg == "last") return v.back();
  if (agg == "nunique") { std::set<double> u(v.begin(), v.end()); return double(u.size()); }
  if (agg == "median") {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
  if (agg == "std" || agg == "var") {
    if (v.size() < 2) return NAN;
    double m = 0; for (double x : v) m += x; m /= double(v.size());
    double s = 0; for (double x : v) s += (x - m) * (x - m);
    double var = s / double(v.size() - 1);
    return agg == "var" ? var : std::sqrt(var);
  }
  fail("unknown aggregation '" + agg + "' (known: " + AGGS + ")");
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) { if (c == sep) { out.push_back(cur); cur.clear(); } else cur += c; }
  out.push_back(cur);
  return out;
}
// "sales:sum|mean|count, net:sum" -> [(sales,sum), (sales,mean), ...]
std::vector<std::pair<std::string, std::string>> parse_spec(const std::string& spec, const char* op) {
  std::vector<std::pair<std::string, std::string>> out;
  for (auto& part : split(spec, ',')) {
    std::string p = trim(part);
    if (p.empty()) continue;
    size_t colon = p.rfind(':');
    if (colon == std::string::npos) fail(std::string(op) + ": spec entry '" + p + "' must be column:agg (aggs joined by |, e.g. \"sales:sum|mean\")");
    std::string col = trim(p.substr(0, colon));
    for (auto& a : split(p.substr(colon + 1), '|')) {
      std::string agg = trim(a);
      aggregate(agg, {});  // validates the name
      out.push_back({col, agg});
    }
  }
  if (out.empty()) fail(std::string(op) + ": empty aggregation spec");
  return out;
}

// A row key over several columns; missing values sort last.
struct KeyCell { bool missing; bool numeric; double num; std::string text; };
bool key_less(const std::vector<KeyCell>& a, const std::vector<KeyCell>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].missing != b[i].missing) return !a[i].missing;
    if (a[i].missing) continue;
    if (a[i].numeric) { if (a[i].num != b[i].num) return a[i].num < b[i].num; }
    else if (a[i].text != b[i].text) return a[i].text < b[i].text;
  }
  return false;
}
std::vector<KeyCell> row_key(const std::vector<const Column*>& cols, size_t r) {
  std::vector<KeyCell> k;
  for (auto* c : cols) k.push_back({c->is_missing(r), c->numeric, c->numeric ? c->num[r] : 0.0, c->numeric ? "" : c->text[r]});
  return k;
}
struct KeyLess { bool operator()(const std::vector<KeyCell>& a, const std::vector<KeyCell>& b) const { return key_less(a, b); } };

Frame* groupby(const Frame& f, const std::vector<std::string>& keys, const std::string& spec, bool dropna) {
  if (keys.empty()) fail("groupby: no key columns");
  std::vector<const Column*> kc;
  for (auto& k : keys) kc.push_back(&f.get(k));
  auto aggs = parse_spec(spec, "groupby");
  for (auto& a : aggs) if (a.second != "size" && a.second != "count" && a.second != "nunique") f.num_col(a.first);
  std::map<std::vector<KeyCell>, std::vector<size_t>, KeyLess> groups;
  for (size_t r = 0; r < f.nrows(); ++r) {
    auto k = row_key(kc, r);
    bool miss = false;
    for (auto& c : k) miss |= c.missing;
    if (miss && dropna) continue;
    groups[k].push_back(r);
  }
  auto* out = make<Frame>();
  for (size_t j = 0; j < kc.size(); ++j) {
    Column c; c.name = kc[j]->name; c.numeric = kc[j]->numeric;
    bool any_missing = false;
    for (auto& g : groups) any_missing |= g.first[j].missing;
    for (auto& g : groups) {
      if (c.numeric) c.num.push_back(g.first[j].missing ? NAN : g.first[j].num);
      else { c.text.push_back(g.first[j].text); if (any_missing) c.missing.push_back(g.first[j].missing); }
    }
    out->cols.push_back(std::move(c));
  }
  for (auto& a : aggs) {
    Column c; c.name = a.first + "__" + a.second;
    const Column& src = f.get(a.first);
    for (auto& g : groups) {
      std::vector<double> vals;
      for (size_t r : g.second) vals.push_back(src.numeric ? src.num[r] : (src.is_missing(r) ? NAN : 0.0));
      if (!src.numeric && a.second == "nunique") {
        std::set<std::string> u; for (size_t r : g.second) if (!src.is_missing(r)) u.insert(src.text[r]);
        c.num.push_back(double(u.size()));
      } else c.num.push_back(aggregate(a.second, vals));
    }
    out->cols.push_back(std::move(c));
  }
  return out;
}

Frame* merge(const Frame& l, const Frame& r, const std::vector<std::string>& on, const std::string& how, const Opts& o) {
  std::string validate = o.str("validate", "");
  bool indicator = o.flag("indicator", false);
  const std::vector<std::string>* suffixes = o.strs("suffixes");
  o.finish("merge");
  if (how != "inner" && how != "left" && how != "right" && how != "outer") fail("merge: how must be inner, left, right or outer; got '" + how + "'");
  if (on.empty()) fail("merge: no key columns");
  std::string sx = suffixes && suffixes->size() == 2 ? (*suffixes)[0] : "_x", sy = suffixes && suffixes->size() == 2 ? (*suffixes)[1] : "_y";
  std::vector<const Column*> lk, rk;
  for (auto& k : on) {
    lk.push_back(&l.get(k)); rk.push_back(&r.get(k));
    if (lk.back()->numeric != rk.back()->numeric) fail("merge: key '" + k + "' is numeric on one side and text on the other");
  }
  std::map<std::vector<KeyCell>, std::vector<size_t>, KeyLess> rindex;
  for (size_t i = 0; i < r.nrows(); ++i) rindex[row_key(rk, i)].push_back(i);
  std::map<std::vector<KeyCell>, size_t, KeyLess> lcount;
  for (size_t i = 0; i < l.nrows(); ++i) lcount[row_key(lk, i)]++;
  if (!validate.empty()) {
    bool lu = true, ru = true;
    for (auto& kv : lcount) lu &= kv.second == 1;
    for (auto& kv : rindex) ru &= kv.second.size() == 1;
    if (validate == "one_to_one" || validate == "1:1") { if (!lu || !ru) fail("merge: validate one_to_one failed: keys are duplicated"); }
    else if (validate == "one_to_many" || validate == "1:m") { if (!lu) fail("merge: validate one_to_many failed: left keys are duplicated"); }
    else if (validate == "many_to_one" || validate == "m:1") { if (!ru) fail("merge: validate many_to_one failed: right keys are duplicated"); }
    else if (validate != "many_to_many" && validate != "m:m") fail("merge: validate must be one_to_one, one_to_many, many_to_one or many_to_many");
  }
  // (left row or npos, right row or npos)
  std::vector<std::pair<size_t, size_t>> pairs;
  const size_t NONE = size_t(-1);
  std::vector<bool> rused(r.nrows(), false);
  if (how == "right") {
    std::map<std::vector<KeyCell>, std::vector<size_t>, KeyLess> lindex;
    for (size_t i = 0; i < l.nrows(); ++i) lindex[row_key(lk, i)].push_back(i);
    for (size_t j = 0; j < r.nrows(); ++j) {
      auto it = lindex.find(row_key(rk, j));
      if (it == lindex.end()) pairs.push_back({NONE, j});
      else for (size_t i : it->second) pairs.push_back({i, j});
    }
  } else {
    for (size_t i = 0; i < l.nrows(); ++i) {
      auto it = rindex.find(row_key(lk, i));
      if (it == rindex.end()) { if (how != "inner") pairs.push_back({i, NONE}); }
      else for (size_t j : it->second) { pairs.push_back({i, j}); rused[j] = true; }
    }
    if (how == "outer") {
      for (size_t j = 0; j < r.nrows(); ++j) if (!rused[j]) pairs.push_back({NONE, j});
      // pandas sorts an outer join by the keys (stable within a key)
      std::stable_sort(pairs.begin(), pairs.end(), [&](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
        auto ka = a.first != NONE ? row_key(lk, a.first) : row_key(rk, a.second);
        auto kb = b.first != NONE ? row_key(lk, b.first) : row_key(rk, b.second);
        return key_less(ka, kb);
      });
    }
  }
  auto* out = make<Frame>();
  auto emit = [&](const Column& src, const std::string& name, bool from_left, const Column* other_key) {
    Column c; c.name = name; c.numeric = src.numeric;
    bool any_missing = false;
    for (auto& p : pairs) {
      size_t row = from_left ? p.first : p.second;
      const Column* s = &src;
      if (row == NONE && other_key) { s = other_key; row = from_left ? p.second : p.first; }
      bool miss = row == NONE || s->is_missing(row);
      if (c.numeric) c.num.push_back(miss ? NAN : s->num[row]);
      else { c.text.push_back(miss ? "" : s->text[row]); c.missing.push_back(miss); any_missing |= miss; }
    }
    if (!c.numeric && !any_missing) c.missing.clear();
    out->cols.push_back(std::move(c));
  };
  for (auto& c : l.cols) {
    auto k = std::find(on.begin(), on.end(), c.name);
    if (k != on.end()) { emit(c, c.name, true, rk[size_t(k - on.begin())]); continue; }
    emit(c, r.find(c.name) ? c.name + sx : c.name, true, nullptr);
  }
  for (auto& c : r.cols) {
    if (std::find(on.begin(), on.end(), c.name) != on.end()) continue;
    emit(c, l.find(c.name) ? c.name + sy : c.name, false, nullptr);
  }
  if (indicator) {
    Column c; c.name = "_merge"; c.numeric = false;
    for (auto& p : pairs) c.text.push_back(p.first == NONE ? "right_only" : p.second == NONE ? "left_only" : "both");
    out->cols.push_back(std::move(c));
  }
  return out;
}

std::vector<double> rolling(const std::vector<double>& v, int64_t window, const std::string& agg, const Opts& o) {
  int64_t min_periods = o.integer("min_periods", window);
  bool center = o.flag("center", false);
  o.finish("rolling");
  if (window < 1) fail("rolling: window must be >= 1");
  if (min_periods < 0 || min_periods > window) fail("rolling: min_periods must be in [0, window]");
  aggregate(agg, {});
  std::vector<double> out(v.size(), NAN);
  int64_t offset = center ? window / 2 : 0;
  for (int64_t i = 0; i < int64_t(v.size()); ++i) {
    int64_t end = i + offset;  // inclusive
    int64_t start = end - window + 1;
    std::vector<double> w;
    int64_t valid = 0;
    for (int64_t k = std::max<int64_t>(start, 0); k <= std::min<int64_t>(end, int64_t(v.size()) - 1); ++k) { w.push_back(v[size_t(k)]); valid += !std::isnan(v[size_t(k)]); }
    if (valid >= std::max<int64_t>(min_periods, 1)) out[size_t(i)] = aggregate(agg, w);
  }
  return out;
}

// ---- time -----------------------------------------------------------------------------------------------------------
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {  // Howard Hinnant's algorithm (public domain)
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + int64_t(doe) - 719468;
}
void civil_from_days(int64_t z, int64_t& y, unsigned& m, unsigned& d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = unsigned(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = int64_t(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp < 10 ? mp + 3 : mp - 9;
  y += m <= 2;
}
// ISO 8601: YYYY-MM-DD[(T| )HH:MM[:SS[.frac]]][Z|(+|-)HH[:]MM] -> seconds since 1970-01-01 UTC. `tz` reports an offset.
double parse_time(const std::string& s, bool& tz) {
  int y, mo, d, h = 0, mi = 0;
  double sec = 0;
  const char* p = s.c_str();
  char* e;
  auto num = [&](int digits, int& out) {
    out = 0;
    for (int i = 0; i < digits; ++i) { if (!std::isdigit((unsigned char)*p)) fail("cannot parse time '" + s + "' (ISO 8601 expected)"); out = out * 10 + (*p++ - '0'); }
  };
  num(4, y); if (*p++ != '-') fail("cannot parse time '" + s + "'");
  num(2, mo); if (*p++ != '-') fail("cannot parse time '" + s + "'");
  num(2, d);
  if (*p == 'T' || *p == ' ') {
    ++p; num(2, h);
    if (*p++ != ':') fail("cannot parse time '" + s + "'");
    num(2, mi);
    if (*p == ':') { ++p; sec = std::strtod(p, &e); p = e; }
  }
  double off = 0;
  tz = false;
  if (*p == 'Z') { tz = true; ++p; }
  else if (*p == '+' || *p == '-') {
    int sign = *p == '-' ? -1 : 1; ++p; int oh, om = 0; num(2, oh); if (*p == ':') ++p; if (*p) num(2, om);
    off = sign * (oh * 3600.0 + om * 60.0); tz = true;
  }
  if (*p) fail("cannot parse time '" + s + "' (trailing text)");
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || sec >= 61) fail("time out of range: '" + s + "'");
  return double(days_from_civil(y, unsigned(mo), unsigned(d))) * 86400.0 + h * 3600.0 + mi * 60.0 + sec - off;
}
std::string format_time(double t, bool tz) {
  int64_t days = int64_t(std::floor(t / 86400.0));
  double rem = t - double(days) * 86400.0;
  int64_t y; unsigned m, d;
  civil_from_days(days, y, m, d);
  int h = int(rem / 3600), mi = int((rem - h * 3600) / 60);
  double s = rem - h * 3600 - mi * 60;
  char buf[64];
  if (s == std::floor(s)) std::snprintf(buf, sizeof buf, "%04lld-%02u-%02u %02d:%02d:%02d%s", (long long)y, m, d, h, mi, int(s), tz ? "+00:00" : "");
  else std::snprintf(buf, sizeof buf, "%04lld-%02u-%02u %02d:%02d:%09.6f%s", (long long)y, m, d, h, mi, s, tz ? "+00:00" : "");
  return buf;
}
std::vector<double> column_times(const Column& c, bool& tz) {
  std::vector<double> t;
  tz = false;
  if (c.numeric) { for (double x : c.num) t.push_back(x); return t; }
  for (size_t i = 0; i < c.size(); ++i) {
    if (c.is_missing(i) || c.text[i].empty()) { t.push_back(NAN); continue; }
    bool z; t.push_back(parse_time(c.text[i], z)); tz |= z;
  }
  return t;
}

// Resampling rule: "<n><unit>" with unit s|min|h|D (fixed bins from midnight of the first day, labelled left) or
// W[-DAY] (weeks ending on DAY, default SUN, labelled right), MS/ME (month start/end), QS/QE, YS/YE.
struct Rule { enum Kind { FIXED, WEEK, MONTH_START, MONTH_END, QUARTER_START, QUARTER_END, YEAR_START, YEAR_END } kind; double seconds = 0; int64_t n = 1; int weekday = 6; };
Rule parse_rule(const std::string& r) {
  size_t i = 0;
  while (i < r.size() && std::isdigit((unsigned char)r[i])) ++i;
  Rule rule{Rule::FIXED};
  rule.n = i ? std::stoll(r.substr(0, i)) : 1;
  if (rule.n < 1) fail("resample: the multiple must be >= 1");
  std::string u = r.substr(i);
  if (u == "s" || u == "S") rule.seconds = 1;
  else if (u == "min" || u == "T") rule.seconds = 60;
  else if (u == "h" || u == "H") rule.seconds = 3600;
  else if (u == "D") rule.seconds = 86400;
  else if (u.rfind("W", 0) == 0) {
    rule.kind = Rule::WEEK;
    static const char* days[] = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"};
    if (u.size() > 1) {
      if (u[1] != '-') fail("resample: weekly rule must be W or W-DAY");
      std::string dname = u.substr(2); bool found = false;
      for (int k = 0; k < 7; ++k) if (dname == days[k]) { rule.weekday = k; found = true; }
      if (!found) fail("resample: unknown weekday '" + dname + "'");
    }
    if (rule.n != 1) fail("resample: only 1W weekly bins are supported");
  } else if (u == "MS") rule.kind = Rule::MONTH_START;
  else if (u == "ME" || u == "M") rule.kind = Rule::MONTH_END;
  else if (u == "QS") rule.kind = Rule::QUARTER_START;
  else if (u == "QE" || u == "Q") rule.kind = Rule::QUARTER_END;
  else if (u == "YS" || u == "AS") rule.kind = Rule::YEAR_START;
  else if (u == "YE" || u == "A" || u == "Y") rule.kind = Rule::YEAR_END;
  else fail("resample: unknown rule '" + r + "' (s, min, h, D with an optional multiple; W, W-DAY; MS, ME; QS, QE; YS, YE)");
  if (rule.kind != Rule::FIXED && rule.kind != Rule::WEEK && rule.n != 1) fail("resample: calendar rules take no multiple");
  return rule;
}
// Bin label (seconds) for a timestamp.
double bin_label(const Rule& r, double t, double origin) {
  if (r.kind == Rule::FIXED) { double w = r.seconds * double(r.n); return origin + std::floor((t - origin) / w) * w; }
  int64_t day = int64_t(std::floor(t / 86400.0));
  if (r.kind == Rule::WEEK) {
    int wd = int(((day % 7) + 7 + 3) % 7);  // 1970-01-01 was a Thursday (Mon=0)
    int64_t ahead = (r.weekday - wd + 7) % 7;
    return double(day + ahead) * 86400.0;
  }
  int64_t y; unsigned m, d;
  civil_from_days(day, y, m, d);
  auto month_end = [](int64_t yy, unsigned mm) { unsigned nm = mm == 12 ? 1 : mm + 1; int64_t ny = mm == 12 ? yy + 1 : yy; return days_from_civil(ny, nm, 1) - 1; };
  switch (r.kind) {
    case Rule::MONTH_START: return double(days_from_civil(y, m, 1)) * 86400.0;
    case Rule::MONTH_END: return double(month_end(y, m)) * 86400.0;
    case Rule::QUARTER_START: return double(days_from_civil(y, ((m - 1) / 3) * 3 + 1, 1)) * 86400.0;
    case Rule::QUARTER_END: return double(month_end(y, ((m - 1) / 3) * 3 + 3)) * 86400.0;
    case Rule::YEAR_START: return double(days_from_civil(y, 1, 1)) * 86400.0;
    case Rule::YEAR_END: return double(days_from_civil(y, 12, 31)) * 86400.0;
    default: return NAN;
  }
}
double next_label(const Rule& r, double label) {
  if (r.kind == Rule::FIXED) return label + r.seconds * double(r.n);
  if (r.kind == Rule::WEEK) return label + 7 * 86400.0;
  int64_t day = int64_t(std::floor(label / 86400.0));
  int64_t y; unsigned m, d;
  civil_from_days(day, y, m, d);
  int step = (r.kind == Rule::QUARTER_START || r.kind == Rule::QUARTER_END) ? 3 : (r.kind == Rule::YEAR_START || r.kind == Rule::YEAR_END) ? 12 : 1;
  unsigned nm = m + unsigned(step);
  int64_t ny = y;
  while (nm > 12) { nm -= 12; ++ny; }
  if (r.kind == Rule::MONTH_START || r.kind == Rule::QUARTER_START || r.kind == Rule::YEAR_START) return double(days_from_civil(ny, nm, 1)) * 86400.0;
  unsigned nm2 = nm == 12 ? 1 : nm + 1; int64_t ny2 = nm == 12 ? ny + 1 : ny;
  return double(days_from_civil(ny2, nm2, 1) - 1) * 86400.0;
}

Frame* resample(const Frame& f, const std::string& time_col, const std::string& rule_text, const std::string& spec) {
  bool tz;
  std::vector<double> t = column_times(f.get(time_col), tz);
  Rule rule = parse_rule(rule_text);
  auto aggs = parse_spec(spec, "resample");
  for (auto& a : aggs) if (a.second != "size") f.num_col(a.first);
  double first = INFINITY, last = -INFINITY;
  for (double x : t) if (!std::isnan(x)) { first = std::min(first, x); last = std::max(last, x); }
  auto* out = make<Frame>();
  if (!std::isfinite(first)) {
    Column c; c.name = time_col; c.numeric = false; out->cols.push_back(c);
    for (auto& a : aggs) { Column k; k.name = a.first + "__" + a.second; out->cols.push_back(k); }
    return out;
  }
  double origin = std::floor(first / 86400.0) * 86400.0;
  std::vector<double> labels;
  for (double l = bin_label(rule, first, origin); l <= bin_label(rule, last, origin); l = next_label(rule, l)) labels.push_back(l);
  std::map<double, std::vector<size_t>> rows;
  for (size_t i = 0; i < t.size(); ++i) if (!std::isnan(t[i])) rows[bin_label(rule, t[i], origin)].push_back(i);
  Column tc; tc.name = time_col; tc.numeric = false;
  for (double l : labels) tc.text.push_back(format_time(l, tz));
  out->cols.push_back(std::move(tc));
  for (auto& a : aggs) {
    Column c; c.name = a.first + "__" + a.second;
    const Column& src = f.get(a.first);
    for (double l : labels) {
      std::vector<double> vals;
      auto it = rows.find(l);
      if (it != rows.end()) for (size_t r : it->second) vals.push_back(src.numeric ? src.num[r] : 0.0);
      c.num.push_back(aggregate(a.second, vals));
    }
    out->cols.push_back(std::move(c));
  }
  return out;
}

// ---- array files ----------------------------------------------------------------------------------------------------
// .npy v1/v2/v3, little-endian C order, dtypes <f8 <f4 <i8 <i4 |u1 |b1 (pycli's whitelist), rank 1 or 2.
Eigen::MatrixXd read_npy(const std::string& path, bool& rank1) {
  std::string b = read_file_text(path);
  if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0) fail("read_npy: '" + path + "' is not a .npy file");
  unsigned major = (unsigned char)b[6];
  size_t hlen, hstart;
  if (major == 1) { hlen = (unsigned char)b[8] | ((unsigned char)b[9] << 8); hstart = 10; }
  else if (major == 2 || major == 3) { if (b.size() < 12) fail("read_npy: truncated header"); hlen = 0; for (int i = 0; i < 4; ++i) hlen |= size_t((unsigned char)b[8 + i]) << (8 * i); hstart = 12; }
  else fail("read_npy: unsupported .npy version " + std::to_string(major));
  if (hstart + hlen > b.size() || hlen > 65536) fail("read_npy: bad header length");
  std::string h = b.substr(hstart, hlen);
  auto field = [&](const std::string& key) {
    size_t k = h.find("'" + key + "'");
    if (k == std::string::npos) fail("read_npy: header has no '" + key + "'");
    size_t c = h.find(':', k);
    return trim(h.substr(c + 1));
  };
  std::string descr = field("descr"), fortran = field("fortran_order"), shape = field("shape");
  if (fortran.rfind("False", 0) != 0) fail("read_npy: Fortran order is not supported");
  std::string dt = descr.substr(1, descr.find('\'', 1) - 1);
  size_t width;
  if (dt == "<f8" || dt == "<i8") width = 8;
  else if (dt == "<f4" || dt == "<i4") width = 4;
  else if (dt == "|u1" || dt == "|b1") width = 1;
  else fail("read_npy: dtype '" + dt + "' is not supported (<f8 <f4 <i8 <i4 |u1 |b1)");
  std::vector<int64_t> dims;
  size_t open = shape.find('('), close = shape.find(')');
  for (auto& p : split(shape.substr(open + 1, close - open - 1), ',')) { std::string t = trim(p); if (!t.empty()) dims.push_back(std::stoll(t)); }
  if (dims.empty() || dims.size() > 2) fail("read_npy: only rank 1 and 2 arrays are supported");
  rank1 = dims.size() == 1;
  int64_t rows = dims[0], cols = rank1 ? 1 : dims[1];
  size_t count = size_t(rows * cols), off = hstart + hlen;
  if (off + count * width != b.size()) fail("read_npy: payload size does not match the shape");
  Eigen::MatrixXd m(rows, cols);
  for (size_t i = 0; i < count; ++i) {
    const char* p = b.data() + off + i * width;
    double v;
    if (dt == "<f8") std::memcpy(&v, p, 8);
    else if (dt == "<f4") { float f; std::memcpy(&f, p, 4); v = f; }
    else if (dt == "<i8") { int64_t x; std::memcpy(&x, p, 8); if (x > (int64_t(1) << 53) || x < -(int64_t(1) << 53)) fail("read_npy: integer beyond 2^53"); v = double(x); }
    else if (dt == "<i4") { int32_t x; std::memcpy(&x, p, 4); v = x; }
    else v = (unsigned char)*p;
    m(Eigen::Index(i / size_t(cols)), Eigen::Index(i % size_t(cols))) = v;
  }
  return m;
}
std::string npy_bytes(const Eigen::MatrixXd& m, bool rank1) {
  std::string shape = rank1 ? "(" + std::to_string(m.rows()) + ",)" : "(" + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")";
  std::string h = "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::string out = "\x93NUMPY";
  out += char(1); out += char(0);
  out += char(h.size() & 0xff); out += char(h.size() >> 8);
  out += h;
  for (Eigen::Index r = 0; r < m.rows(); ++r)
    for (Eigen::Index c = 0; c < m.cols(); ++c) { double v = m(r, c); out.append(reinterpret_cast<const char*>(&v), 8); }
  return out;
}
// safetensors: 8-byte little-endian header length, JSON header {"name": {"dtype","shape","data_offsets"}}, payload.
std::string st_bytes(const std::string& key, const Eigen::MatrixXd& m, bool rank1) {
  size_t n = size_t(m.size()) * 8;
  std::string shape = rank1 ? "[" + std::to_string(m.rows()) + "]" : "[" + std::to_string(m.rows()) + "," + std::to_string(m.cols()) + "]";
  std::string h = "{" + jstr(key) + ":{\"dtype\":\"F64\",\"shape\":" + shape + ",\"data_offsets\":[0," + std::to_string(n) + "]}}";
  while (h.size() % 8) h += ' ';
  std::string out(8, '\0');
  uint64_t hl = h.size();
  for (int i = 0; i < 8; ++i) out[size_t(i)] = char((hl >> (8 * i)) & 0xff);
  out += h;
  for (Eigen::Index r = 0; r < m.rows(); ++r)
    for (Eigen::Index c = 0; c < m.cols(); ++c) { double v = m(r, c); out.append(reinterpret_cast<const char*>(&v), 8); }
  return out;
}
Eigen::MatrixXd read_st(const std::string& path, const std::string& key, bool& rank1) {
  std::string b = read_file_text(path);
  if (b.size() < 8) fail("read_safetensors: '" + path + "' is truncated");
  uint64_t hl = 0;
  for (int i = 0; i < 8; ++i) hl |= uint64_t((unsigned char)b[size_t(i)]) << (8 * i);
  if (hl > (1u << 20) || 8 + hl > b.size()) fail("read_safetensors: bad header length");
  std::string h = b.substr(8, hl);
  size_t k = h.find(jstr(key) + ":");
  if (k == std::string::npos) fail("read_safetensors: no tensor '" + key + "' in '" + path + "'");
  size_t obj_end = h.find('}', k);
  std::string spec = h.substr(k, obj_end - k);
  auto grab = [&](const std::string& name) { size_t p = spec.find("\"" + name + "\""); if (p == std::string::npos) fail("read_safetensors: tensor '" + key + "' lacks " + name); return spec.substr(spec.find(':', p) + 1); };
  std::string dtype = grab("dtype");
  dtype = dtype.substr(dtype.find('"') + 1); dtype = dtype.substr(0, dtype.find('"'));
  std::string shape = grab("shape"); shape = shape.substr(shape.find('[') + 1); shape = shape.substr(0, shape.find(']'));
  std::string offs = grab("data_offsets"); offs = offs.substr(offs.find('[') + 1); offs = offs.substr(0, offs.find(']'));
  std::vector<int64_t> dims;
  for (auto& p : split(shape, ',')) { std::string t = trim(p); if (!t.empty()) dims.push_back(std::stoll(t)); }
  auto o = split(offs, ',');
  if (o.size() != 2) fail("read_safetensors: bad data_offsets");
  size_t begin = std::stoull(trim(o[0])), end = std::stoull(trim(o[1]));
  size_t width = dtype == "F64" || dtype == "I64" ? 8 : dtype == "F32" || dtype == "I32" ? 4 : dtype == "U8" || dtype == "BOOL" ? 1 : 0;
  if (!width) fail("read_safetensors: dtype " + dtype + " is not supported (F64 F32 I64 I32 U8 BOOL)");
  if (dims.empty() || dims.size() > 2) fail("read_safetensors: only rank 1 and 2 tensors are supported");
  rank1 = dims.size() == 1;
  int64_t rows = dims[0], cols = rank1 ? 1 : dims[1];
  if (end < begin || 8 + hl + end > b.size() || end - begin != size_t(rows * cols) * width) fail("read_safetensors: data offsets do not match the shape");
  Eigen::MatrixXd m(rows, cols);
  const char* base = b.data() + 8 + hl + begin;
  for (int64_t i = 0; i < rows * cols; ++i) {
    const char* p = base + size_t(i) * width;
    double v;
    if (dtype == "F64") std::memcpy(&v, p, 8);
    else if (dtype == "F32") { float f; std::memcpy(&f, p, 4); v = f; }
    else if (dtype == "I64") { int64_t x; std::memcpy(&x, p, 8); v = double(x); }
    else if (dtype == "I32") { int32_t x; std::memcpy(&x, p, 4); v = x; }
    else v = (unsigned char)*p;
    m(i / cols, i % cols) = v;
  }
  return m;
}

}  // namespace

void register_frame_ops(Api& api) {
  api.section("Table operations", "Grouping, joins, rolling windows, time resampling, reshaping and summaries (pycli `frame_groupby`, "
                                  "`frame_merge`, `frame_rolling`, `frame_resample`, `frame_inspect`). An aggregation spec is "
                                  "`\"col:agg|agg, col:agg\"`; result columns are named `col__agg` like pycli. Aggregations: sum, mean, "
                                  "count (non-missing), size (rows), min, max, std and var (ddof 1), median, first, last, nunique, prod.");
  api.method(Ft, "groupby", {St, "string"}, Ft, "group by key columns and aggregate: `f.groupby(strs(\"region\"), \"sales:sum|mean\")`; groups sorted, a missing key is its own group (pandas dropna=False)",
             [](void* s, Args& a) { return O(groupby(ff(s), obj<StrVec>(a[0]).v, a[1].s, false)); });
  api.method(Ft, "groupby", {St, "string", "bool"}, Ft, "groupby; the bool drops rows whose key is missing (pandas dropna=True)",
             [](void* s, Args& a) { return O(groupby(ff(s), obj<StrVec>(a[0]).v, a[1].s, a[2].b)); });
  api.method(Ft, "merge", {Ft, St, "string"}, Ft, "database join on key columns, how = inner, left, right or outer (pandas.merge: outer is sorted by key; clashing names get _x/_y)",
             [](void* s, Args& a) { return O(merge(ff(s), obj<Frame>(a[0]), obj<StrVec>(a[1]).v, a[2].s, no_opts())); });
  api.method(Ft, "merge", {Ft, St, "string", "Opts"}, Ft, "merge with options: `validate` (one_to_one, one_to_many, many_to_one, many_to_many), `indicator` (bool: adds `_merge` = left_only/right_only/both), `suffixes` (StrVec of two)",
             [](void* s, Args& a) { return O(merge(ff(s), obj<Frame>(a[0]), obj<StrVec>(a[1]).v, a[2].s, obj<Opts>(a[3]))); });
  api.method(Ft, "concat", {Ft}, Ft, "rows of this frame then the other (same column names and kinds)", [](void* s, Args& a) {
    const Frame& x = ff(s); const Frame& y = obj<Frame>(a[0]);
    if (x.cols.size() != y.cols.size()) fail("concat: the frames have different columns (" + x.names_text() + " / " + y.names_text() + ")");
    auto* r = make<Frame>(x);
    for (auto& c : r->cols) {
      const Column& o = y.get(c.name);
      if (o.numeric != c.numeric) fail("concat: column '" + c.name + "' is numeric in one frame and text in the other");
      if (c.numeric) c.num.insert(c.num.end(), o.num.begin(), o.num.end());
      else {
        size_t before = c.text.size();
        c.text.insert(c.text.end(), o.text.begin(), o.text.end());
        if (!c.missing.empty() || !o.missing.empty()) {
          c.missing.resize(before, 0);
          for (size_t i = 0; i < o.text.size(); ++i) c.missing.push_back(o.is_missing(i));
        }
      }
    }
    return O(r); });
  api.method(Ft, "rolling", {"string", "int", "string"}, Vt, "trailing window aggregate of a numeric column (pandas rolling: the first window-1 values are NaN)",
             [](void* s, Args& a) { return vecv(rolling(ff(s).num_col(a[0].s).num, a[1].i, a[2].s, no_opts())); });
  api.method(Ft, "rolling", {"string", "int", "string", "Opts"}, Vt, "rolling with `min_periods` (int) and `center` (bool)",
             [](void* s, Args& a) { return vecv(rolling(ff(s).num_col(a[0].s).num, a[1].i, a[2].s, obj<Opts>(a[3]))); });
  api.method(Vt, "rolling", {"int", "string"}, Vt, "trailing window aggregate (same rules as Frame.rolling)",
             [](void* s, Args& a) { return vecv(rolling(static_cast<Vec*>(s)->v, a[0].i, a[1].s, no_opts())); });
  api.method(Ft, "resample", {"string", "string", "string"}, Ft, "time bins over an ISO-8601 text (or epoch-seconds) column: rule `15min`, `1h`, `1D` (bins from midnight of the first day, labelled left), `W`/`W-DAY` (weeks ending on DAY, labelled right), `MS`/`ME`, `QS`/`QE`, `YS`/`YE`; empty bins are kept (pandas resample)",
             [](void* s, Args& a) { return O(resample(ff(s), a[0].s, a[1].s, a[2].s)); });
  api.method(Ft, "times", {"string"}, Vt, "an ISO-8601 text column as seconds since 1970-01-01 UTC (offsets applied)", [](void* s, Args& a) {
    bool tz; return vecv(column_times(ff(s).get(a[0].s), tz)); });
  api.fn("format_time", {"float"}, "string", "seconds since 1970 UTC as `YYYY-MM-DD HH:MM:SS`", [](Args& a) { return S(format_time(a[0].f, false)); });
  api.fn("parse_time", {"string"}, "float", "ISO-8601 text as seconds since 1970 UTC", [](Args& a) { bool tz; return F(parse_time(a[0].s, tz)); });
  api.method(Ft, "unique", {"string"}, St, "distinct values of a column as text, in first-seen order (missing omitted)", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); std::vector<std::string> u; std::set<std::string> seen;
    for (size_t i = 0; i < c.size(); ++i) { if (c.is_missing(i)) continue; std::string t = c.cell_text(i); if (seen.insert(t).second) u.push_back(t); }
    return strsv(std::move(u)); });
  api.method(Ft, "value_counts", {"string"}, Ft, "frame {value (text), count} sorted by count descending, then value", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); std::map<std::string, double> counts;
    for (size_t i = 0; i < c.size(); ++i) if (!c.is_missing(i)) counts[c.cell_text(i)] += 1;
    std::vector<std::pair<std::string, double>> v(counts.begin(), counts.end());
    std::stable_sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second > y.second; });
    auto* r = make<Frame>(); Column vc; vc.name = "value"; vc.numeric = false; Column cc; cc.name = "count";
    for (auto& p : v) { vc.text.push_back(p.first); cc.num.push_back(p.second); }
    r->cols.push_back(vc); r->cols.push_back(cc); return O(r); });
  api.method(Ft, "describe", {}, Ft, "summary of the numeric columns: a `stat` column (count, mean, std, min, 25%, 50%, 75%, max) and one column each (pandas describe)", [](void* s, Args&) {
    auto* r = make<Frame>(); Column st; st.name = "stat"; st.numeric = false;
    st.text = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}; r->cols.push_back(st);
    for (auto& c : ff(s).cols) {
      if (!c.numeric) continue;
      std::vector<double> v; for (double x : c.num) if (!std::isnan(x)) v.push_back(x);
      std::sort(v.begin(), v.end());
      auto q = [&](double p) -> double { if (v.empty()) return NAN; double h = (double(v.size()) - 1) * p; size_t lo = size_t(std::floor(h)); size_t hi = std::min(lo + 1, v.size() - 1); return v[lo] + (h - double(lo)) * (v[hi] - v[lo]); };
      Column k; k.name = c.name;
      k.num = {double(v.size()), aggregate("mean", v), aggregate("std", v), v.empty() ? NAN : v.front(), q(0.25), q(0.5), q(0.75), v.empty() ? NAN : v.back()};
      r->cols.push_back(k);
    }
    return O(r); });
  api.method(Ft, "dropna", {}, Ft, "rows with no missing value in any column", [](void* s, Args&) {
    std::vector<size_t> rows; for (size_t i = 0; i < ff(s).nrows(); ++i) { bool ok = true; for (auto& c : ff(s).cols) ok &= !c.is_missing(i); if (ok) rows.push_back(i); }
    return O(frame_take(ff(s), rows)); });
  api.method(Ft, "dropna", {St}, Ft, "rows with no missing value in these columns", [](void* s, Args& a) {
    std::vector<const Column*> cs; for (auto& n : obj<StrVec>(a[0]).v) cs.push_back(&ff(s).get(n));
    std::vector<size_t> rows; for (size_t i = 0; i < ff(s).nrows(); ++i) { bool ok = true; for (auto* c : cs) ok &= !c->is_missing(i); if (ok) rows.push_back(i); }
    return O(frame_take(ff(s), rows)); });
  api.method(Ft, "fillna", {"string", "float"}, Ft, "replace NaN in a numeric column", [](void* s, Args& a) {
    auto* r = make<Frame>(ff(s)); r->num_col(a[0].s);
    for (auto& c : r->cols) if (c.name == a[0].s) for (double& x : c.num) if (std::isnan(x)) x = a[1].f;
    return O(r); });
  api.method(Ft, "pivot", {"string", "string", "string", "string"}, Ft, "pivot table (index column, columns column, values column, agg): one row per index value, one column per distinct `columns` value (sorted)", [](void* s, Args& a) {
    const Frame& f = ff(s); const Column& ic = f.get(a[0].s); const Column& cc = f.get(a[1].s); const Column& vc = f.num_col(a[2].s);
    aggregate(a[3].s, {});
    std::map<std::vector<KeyCell>, std::map<std::string, std::vector<double>>, KeyLess> cells; std::set<std::string> colnames;
    for (size_t i = 0; i < f.nrows(); ++i) { if (cc.is_missing(i)) continue; std::string cn = cc.cell_text(i); colnames.insert(cn); cells[row_key({&ic}, i)][cn].push_back(vc.num[i]); }
    auto* r = make<Frame>(); Column idx; idx.name = ic.name; idx.numeric = ic.numeric;
    for (auto& kv : cells) { if (ic.numeric) idx.num.push_back(kv.first[0].missing ? NAN : kv.first[0].num); else { idx.text.push_back(kv.first[0].text); } }
    r->cols.push_back(idx);
    for (auto& cn : colnames) {
      Column k; k.name = cn;
      for (auto& kv : cells) { auto it = kv.second.find(cn); k.num.push_back(it == kv.second.end() ? NAN : aggregate(a[3].s, it->second)); }
      r->cols.push_back(k);
    }
    return O(r); });
  api.method(Ft, "shift", {"string", "int"}, Vt, "a numeric column shifted by n rows (positive: later rows see earlier values; vacated cells NaN)", [](void* s, Args& a) {
    const auto& v = ff(s).num_col(a[0].s).num; std::vector<double> r(v.size(), NAN); int64_t n = a[1].i;
    for (int64_t i = 0; i < int64_t(v.size()); ++i) { int64_t j = i - n; if (j >= 0 && j < int64_t(v.size())) r[size_t(i)] = v[size_t(j)]; }
    return vecv(std::move(r)); });

  api.section("Array files", "NumPy `.npy` and `.safetensors` arrays (pycli's artifact formats). Reads accept the dtypes pycli accepts "
                             "(<f8 <f4 <i8 <i4 |u1 |b1; F64 F32 I64 I32 U8 BOOL), rank 1 or 2; writes are float64 and atomic.");
  api.fn("read_npy", {"string"}, Mt, "a .npy file as a matrix (a rank-1 array becomes one column)", [](Args& a) { bool r1; return matv(read_npy(a[0].s, r1)); });
  api.fn("read_npy_vec", {"string"}, Vt, "a rank-1 (or single-column) .npy file as a vector", [](Args& a) {
    bool r1; Eigen::MatrixXd m = read_npy(a[0].s, r1); if (m.cols() != 1) fail("read_npy_vec: the array has " + std::to_string(m.cols()) + " columns");
    return vecv(Eigen::VectorXd(m.col(0))); });
  api.fn("write_npy", {"string", Vt}, "void", "write a vector as a rank-1 float64 .npy", [](Args& a) { write_file_atomic(a[0].s, npy_bytes(to_eigen(obj<Vec>(a[1]).v), true)); return Value::none(); });
  api.fn("write_npy", {"string", Mt}, "void", "write a matrix as a rank-2 float64 .npy", [](Args& a) { write_file_atomic(a[0].s, npy_bytes(obj<Mat>(a[1]).m, false)); return Value::none(); });
  api.fn("read_safetensors", {"string", "string"}, Mt, "one tensor (file, key) as a matrix (rank 1 becomes one column)", [](Args& a) { bool r1; return matv(read_st(a[0].s, a[1].s, r1)); });
  api.fn("read_safetensors_vec", {"string", "string"}, Vt, "one rank-1 tensor (file, key) as a vector", [](Args& a) {
    bool r1; Eigen::MatrixXd m = read_st(a[0].s, a[1].s, r1); if (m.cols() != 1) fail("read_safetensors_vec: the tensor has " + std::to_string(m.cols()) + " columns");
    return vecv(Eigen::VectorXd(m.col(0))); });
  api.fn("write_safetensors", {"string", "string", Vt}, "void", "write a vector as one F64 tensor (file, key, data)", [](Args& a) {
    write_file_atomic(a[0].s, st_bytes(a[1].s, to_eigen(obj<Vec>(a[2]).v), true)); return Value::none(); });
  api.fn("write_safetensors", {"string", "string", Mt}, "void", "write a matrix as one F64 tensor (file, key, data)", [](Args& a) {
    write_file_atomic(a[0].s, st_bytes(a[1].s, obj<Mat>(a[2]).m, false)); return Value::none(); });
  api.fn("remove_file", {"string"}, "bool", "delete a file; false when it did not exist", [](Args& a) { return B(std::remove(a[0].s.c_str()) == 0); });
  api.fn("write_text", {"string", "string"}, "void", "write a text file (atomic replace)", [](Args& a) { write_file_atomic(a[0].s, a[1].s); return Value::none(); });
  api.fn("read_text", {"string"}, "string", "read a whole text file", [](Args& a) { return S(read_file_text(a[0].s)); });
  api.fn("mat_frame", {Mt, St}, Ft, "a matrix as a frame with these column names (pycli array_to_frame)", [](Args& a) {
    const Eigen::MatrixXd& m = obj<Mat>(a[0]).m; const auto& names = obj<StrVec>(a[1]).v;
    if (size_t(m.cols()) != names.size()) fail("mat_frame: " + std::to_string(m.cols()) + " columns but " + std::to_string(names.size()) + " names");
    auto* f = make<Frame>(); for (size_t j = 0; j < names.size(); ++j) { Column c; c.name = names[j]; Eigen::VectorXd col = m.col(Eigen::Index(j)); c.num = from_eigen(col); f->add(std::move(c)); }
    return O(f); });
}

}  // namespace lmnum
