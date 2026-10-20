// Frame basics: CSV in/out, columns, expression filter/mutate, select/sort/head, to_mat.
// Grouping, joins, rolling windows and time resampling live in frame_ops.cpp.
#include "frame.hpp"
#include "expr.hpp"
#include "out.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unistd.h>

namespace lmnum {

std::string Column::cell_text(size_t i) const {
  if (!numeric) return is_missing(i) ? "" : text[i];
  double v = num[i];
  if (std::isnan(v)) return "";
  if (v == std::floor(v) && std::fabs(v) < 1e15) { char b[32]; std::snprintf(b, sizeof b, "%.0f", v); return b; }
  return jnum(v);
}

const Column* Frame::find(const std::string& name) const {
  for (auto& c : cols)
    if (c.name == name) return &c;
  return nullptr;
}
std::string Frame::names_text() const {
  std::string s;
  for (auto& c : cols) s += (s.empty() ? "" : ", ") + c.name;
  return s;
}
const Column& Frame::get(const std::string& name) const {
  const Column* c = find(name);
  if (!c) fail("frame has no column '" + name + "' (columns: " + names_text() + ")");
  return *c;
}
const Column& Frame::num_col(const std::string& name) const {
  const Column& c = get(name);
  if (!c.numeric) fail("column '" + name + "' is text, not numeric");
  return c;
}
void Frame::add(Column c) {
  if (!cols.empty() && c.size() != nrows() && !(cols.size() == 1 && cols[0].name == c.name))
    fail("column '" + c.name + "' has " + std::to_string(c.size()) + " rows; the frame has " + std::to_string(nrows()));
  for (auto& x : cols)
    if (x.name == c.name) { x = std::move(c); return; }
  cols.push_back(std::move(c));
}

std::string read_file_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot read '" + path + "'");
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

void write_file_atomic(const std::string& path, const std::string& bytes) {
  std::string tmp = path + ".tmp-" + std::to_string(::getpid());
  {
    std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
    if (!o) fail("cannot write '" + path + "' (directory missing or not writable)");
    o.write(bytes.data(), std::streamsize(bytes.size()));
    if (!o) { std::remove(tmp.c_str()); fail("write to '" + path + "' failed"); }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); fail("cannot move the new file into '" + path + "'"); }
}

namespace {

std::vector<std::vector<std::string>> parse_rows(const std::string& t, char sep) {
  std::vector<std::vector<std::string>> rows;
  std::vector<std::string> row;
  std::string cell;
  bool quoted = false, any = false;
  size_t i = 0;
  if (t.size() >= 3 && (unsigned char)t[0] == 0xEF && (unsigned char)t[1] == 0xBB && (unsigned char)t[2] == 0xBF) i = 3;
  for (; i < t.size(); ++i) {
    char c = t[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < t.size() && t[i + 1] == '"') { cell += '"'; ++i; }
        else quoted = false;
      } else cell += c;
      continue;
    }
    if (c == '"') { quoted = true; any = true; }
    else if (c == sep) { row.push_back(cell); cell.clear(); any = true; }
    else if (c == '\n' || c == '\r') {
      if (c == '\r' && i + 1 < t.size() && t[i + 1] == '\n') ++i;
      if (any || !cell.empty()) { row.push_back(cell); rows.push_back(row); }
      row.clear(); cell.clear(); any = false;
    } else { cell += c; any = true; }
  }
  if (quoted) fail("CSV: unterminated quoted field");
  if (any || !cell.empty()) { row.push_back(cell); rows.push_back(row); }
  return rows;
}

bool parse_number(const std::string& s, double& out) {
  if (s.empty() || s == "NA" || s == "nan" || s == "NaN" || s == "null") { out = NAN; return true; }
  const char* b = s.c_str();
  char* e = nullptr;
  out = std::strtod(b, &e);
  if (e == b) return false;
  while (*e == ' ') ++e;
  return *e == 0;
}

Frame* build(const std::vector<std::vector<std::string>>& rows, const Opts& o) {
  bool header = o.flag("header", true);
  const std::vector<std::string>* force_text = o.strs("text");
  const std::vector<std::string>* names_opt = o.strs("names");
  auto* f = make<Frame>();
  if (rows.empty()) return f;
  std::vector<std::string> names;
  size_t first = 0;
  if (names_opt) names = *names_opt;
  else if (header) { names = rows[0]; first = 1; }
  else for (size_t j = 0; j < rows[0].size(); ++j) names.push_back("c" + std::to_string(j));
  if (header && names_opt) first = 1;
  size_t width = names.size();
  for (size_t r = first; r < rows.size(); ++r)
    if (rows[r].size() != width) fail("CSV row " + std::to_string(r + 1) + " has " + std::to_string(rows[r].size()) + " fields; the header has " + std::to_string(width));
  for (size_t j = 0; j < width; ++j) {
    Column c;
    c.name = names[j];
    bool text = force_text && std::find(force_text->begin(), force_text->end(), c.name) != force_text->end();
    std::vector<double> nums;
    if (!text) {
      for (size_t r = first; r < rows.size(); ++r) {
        double v;
        if (!parse_number(rows[r][j], v)) { text = true; break; }
        nums.push_back(v);
      }
    }
    if (text) {
      c.numeric = false;
      for (size_t r = first; r < rows.size(); ++r) c.text.push_back(rows[r][j]);
    } else {
      c.num = std::move(nums);
    }
    for (auto& x : f->cols) if (x.name == c.name) fail("CSV: duplicate column name '" + c.name + "'");
    f->cols.push_back(std::move(c));
  }
  return f;
}

std::string csv_quote(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
  std::string o = "\"";
  for (char c : s) { if (c == '"') o += '"'; o += c; }
  return o + "\"";
}

Frame& ff(void* self) { return *static_cast<Frame*>(self); }

// Compile an expression over the numeric columns; returns the program and the slot -> column mapping.
struct RowExpr {
  ExprProgram prog;
  std::vector<const Column*> slots;
};
RowExpr row_expr(const Frame& f, const std::string& text) {
  RowExpr r;
  std::vector<std::string> names;
  for (auto& n : expr_names(text)) {
    const Column* c = f.find(n);
    if (c && c->numeric) { names.push_back(n); r.slots.push_back(c); }
  }
  names.push_back("row");  // the 0-based row index
  ExprCompiler comp(names);
  r.prog = comp.compile(text);
  return r;
}
std::vector<double> eval_rows(const Frame& f, const RowExpr& e) {
  size_t n = f.nrows();
  std::vector<double> out(n), vars(e.slots.size() + 1);
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < e.slots.size(); ++k) vars[k] = e.slots[k]->num[i];
    vars[e.slots.size()] = double(i);
    out[i] = expr_eval(e.prog, vars.data());
  }
  return out;
}

}  // namespace

Frame* read_csv_file(const std::string& path, const Opts& o) {
  std::string sep = o.str("sep", ",");
  if (sep.size() != 1) fail("read_csv: sep must be one character");
  Frame* f = build(parse_rows(read_file_text(path), sep[0]), o);
  o.finish("read_csv");
  return f;
}
Frame* parse_csv_text(const std::string& text, const Opts& o) {
  std::string sep = o.str("sep", ",");
  if (sep.size() != 1) fail("csv: sep must be one character");
  Frame* f = build(parse_rows(text, sep[0]), o);
  o.finish("csv");
  return f;
}
std::string frame_to_csv(const Frame& f) {
  std::string o;
  for (size_t j = 0; j < f.cols.size(); ++j) o += (j ? "," : "") + csv_quote(f.cols[j].name);
  o += "\n";
  for (size_t i = 0; i < f.nrows(); ++i) {
    for (size_t j = 0; j < f.cols.size(); ++j) o += (j ? "," : "") + csv_quote(f.cols[j].is_missing(i) ? "" : f.cols[j].numeric ? jnum(f.cols[j].num[i]) : f.cols[j].text[i]);
    o += "\n";
  }
  return o;
}
std::string frame_to_json(const Frame& f) {
  std::string o = "{\"columns\": [";
  for (size_t j = 0; j < f.cols.size(); ++j) o += (j ? ", " : "") + jstr(f.cols[j].name);
  o += "], \"rows\": " + std::to_string(f.nrows()) + ", \"data\": {";
  for (size_t j = 0; j < f.cols.size(); ++j) {
    const Column& c = f.cols[j];
    std::string arr;
    if (c.numeric) arr = jvec(c.num);
    else {
      arr = "[";
      for (size_t i = 0; i < c.text.size(); ++i) arr += (i ? "," : "") + (c.is_missing(i) ? std::string("null") : jstr(c.text[i]));
      arr += "]";
    }
    o += (j ? ", " : "") + jstr(c.name) + ": " + arr;
  }
  return o + "}}";
}
Frame* frame_take(const Frame& f, const std::vector<size_t>& rows) {
  auto* r = make<Frame>();
  for (auto& c : f.cols) {
    Column n;
    n.name = c.name;
    n.numeric = c.numeric;
    for (size_t i : rows) {
      if (c.numeric) n.num.push_back(c.num[i]);
      else n.text.push_back(c.text[i]);
    }
    if (!c.numeric && !c.missing.empty())
      for (size_t i : rows) n.missing.push_back(c.missing[i]);
    r->cols.push_back(std::move(n));
  }
  return r;
}

void register_frame(Api& api) {
  const std::string Ft = "Frame", Vt = "Vec", St = "StrVec", Mt = "Mat";
  api.section("Tables", "`Frame` is a column table (pandas DataFrame subset): numeric columns are float64 with NaN for missing values; "
                        "other columns are text. Expressions in `filter`/`mutate` see the numeric columns by name plus `row` (0-based index). "
                        "Frames are immutable: every method returns a new frame.");
  api.type(Ft, "Column table.");
  api.fn("read_csv", {"string"}, Ft, "read a CSV file with a header row (a column is numeric when every cell parses; empty/NA/nan are NaN)",
         [](Args& a) { return O(read_csv_file(a[0].s, no_opts())); });
  api.fn("read_csv", {"string", "Opts"}, Ft, "read_csv with options: `sep` (one char), `header` (bool), `names` (StrVec), `text` (StrVec: columns kept as text)",
         [](Args& a) { return O(read_csv_file(a[0].s, obj<Opts>(a[1]))); });
  api.fn("csv", {"string"}, Ft, "parse CSV text (same rules as read_csv)", [](Args& a) { return O(parse_csv_text(a[0].s, no_opts())); });
  api.fn("frame", {}, Ft, "empty frame (add columns with `with`)", [](Args&) { return O(make<Frame>()); });
  api.method(Ft, "nrows", {}, "int", "row count", [](void* s, Args&) { return I(int64_t(ff(s).nrows())); });
  api.method(Ft, "ncols", {}, "int", "column count", [](void* s, Args&) { return I(int64_t(ff(s).cols.size())); });
  api.method(Ft, "names", {}, St, "column names in order", [](void* s, Args&) {
    std::vector<std::string> n; for (auto& c : ff(s).cols) n.push_back(c.name); return strsv(std::move(n)); });
  api.method(Ft, "has", {"string"}, "bool", "whether a column exists", [](void* s, Args& a) { return B(ff(s).find(a[0].s) != nullptr); });
  api.method(Ft, "is_numeric", {"string"}, "bool", "whether a column is numeric", [](void* s, Args& a) { return B(ff(s).get(a[0].s).numeric); });
  api.method(Ft, "col", {"string"}, Vt, "numeric column as a vector", [](void* s, Args& a) { return vecv(ff(s).num_col(a[0].s).num); });
  api.method(Ft, "text", {"string"}, St, "any column as text (numbers formatted, NaN as empty)", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); std::vector<std::string> t; for (size_t i = 0; i < c.size(); ++i) t.push_back(c.cell_text(i)); return strsv(std::move(t)); });
  api.method(Ft, "num", {"string", "int"}, "float", "one numeric cell (column, row)", [](void* s, Args& a) {
    const Column& c = ff(s).num_col(a[0].s); return F(c.num[index_arg(a[1].i, c.size(), "Frame.num")]); });
  api.method(Ft, "cell", {"string", "int"}, "string", "one cell as text (column, row)", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); return S(c.cell_text(index_arg(a[1].i, c.size(), "Frame.cell"))); });
  api.method(Ft, "is_missing", {"string", "int"}, "bool", "whether a cell is missing (NaN, or an absent text value after a join)", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); return B(c.is_missing(index_arg(a[1].i, c.size(), "Frame.is_missing"))); });
  api.method(Ft, "with", {"string", Vt}, Ft, "new frame with a numeric column added or replaced", [](void* s, Args& a) {
    auto* r = make<Frame>(ff(s)); Column c; c.name = a[0].s; c.num = obj<Vec>(a[1]).v; r->add(std::move(c)); return O(r); });
  api.method(Ft, "with", {"string", St}, Ft, "new frame with a text column added or replaced", [](void* s, Args& a) {
    auto* r = make<Frame>(ff(s)); Column c; c.name = a[0].s; c.numeric = false; c.text = obj<StrVec>(a[1]).v; r->add(std::move(c)); return O(r); });
  api.method(Ft, "mutate", {"string", "string"}, Ft, "new frame with column NAME = expression over numeric columns: `f.mutate(\"bmi\", \"w / (h/100)^2\")`", [](void* s, Args& a) {
    RowExpr e = row_expr(ff(s), a[1].s); auto* r = make<Frame>(ff(s)); Column c; c.name = a[0].s; c.num = eval_rows(ff(s), e); r->add(std::move(c)); return O(r); });
  api.method(Ft, "filter", {"string"}, Ft, "rows where the expression is nonzero: `f.filter(\"x > 0 & y < 3\")`", [](void* s, Args& a) {
    RowExpr e = row_expr(ff(s), a[0].s); auto v = eval_rows(ff(s), e); std::vector<size_t> rows;
    for (size_t i = 0; i < v.size(); ++i) if (v[i] != 0 && !std::isnan(v[i])) rows.push_back(i); return O(frame_take(ff(s), rows)); });
  api.method(Ft, "where", {"string", "string"}, Ft, "rows whose column (as text) equals a value", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); std::vector<size_t> rows;
    for (size_t i = 0; i < c.size(); ++i) if (c.cell_text(i) == a[1].s) rows.push_back(i); return O(frame_take(ff(s), rows)); });
  api.method(Ft, "select", {St}, Ft, "these columns, in this order", [](void* s, Args& a) {
    auto* r = make<Frame>(); for (auto& n : obj<StrVec>(a[0]).v) r->cols.push_back(ff(s).get(n)); return O(r); });
  api.method(Ft, "drop", {St}, Ft, "all columns except these", [](void* s, Args& a) {
    const auto& d = obj<StrVec>(a[0]).v; for (auto& n : d) ff(s).get(n);
    auto* r = make<Frame>(); for (auto& c : ff(s).cols) if (std::find(d.begin(), d.end(), c.name) == d.end()) r->cols.push_back(c); return O(r); });
  api.method(Ft, "rename", {"string", "string"}, Ft, "rename one column", [](void* s, Args& a) {
    ff(s).get(a[0].s); if (a[0].s != a[1].s && ff(s).find(a[1].s)) fail("rename: column '" + a[1].s + "' already exists");
    auto* r = make<Frame>(ff(s)); for (auto& c : r->cols) if (c.name == a[0].s) c.name = a[1].s; return O(r); });
  api.method(Ft, "head", {"int"}, Ft, "first n rows", [](void* s, Args& a) {
    size_t n = std::min<size_t>(size_t(std::max<int64_t>(a[0].i, 0)), ff(s).nrows()); std::vector<size_t> rows(n); std::iota(rows.begin(), rows.end(), 0); return O(frame_take(ff(s), rows)); });
  api.method(Ft, "tail", {"int"}, Ft, "last n rows", [](void* s, Args& a) {
    size_t total = ff(s).nrows(); size_t n = std::min<size_t>(size_t(std::max<int64_t>(a[0].i, 0)), total); std::vector<size_t> rows(n); std::iota(rows.begin(), rows.end(), total - n); return O(frame_take(ff(s), rows)); });
  api.method(Ft, "take", {Vt}, Ft, "rows at these indexes", [](void* s, Args& a) {
    std::vector<size_t> rows; for (double d : obj<Vec>(a[0]).v) rows.push_back(index_arg(int64_t(d), ff(s).nrows(), "Frame.take")); return O(frame_take(ff(s), rows)); });
  api.method(Ft, "sort", {"string", "bool"}, Ft, "stable sort by one column (ascending when true; NaN last)", [](void* s, Args& a) {
    const Column& c = ff(s).get(a[0].s); bool asc = a[1].b; std::vector<size_t> rows(c.size()); std::iota(rows.begin(), rows.end(), 0);
    std::stable_sort(rows.begin(), rows.end(), [&](size_t i, size_t j) {
      if (c.numeric) { double x = c.num[i], y = c.num[j]; if (std::isnan(x) || std::isnan(y)) return !std::isnan(x) && std::isnan(y); return asc ? x < y : x > y; }
      return asc ? c.text[i] < c.text[j] : c.text[i] > c.text[j]; });
    return O(frame_take(ff(s), rows)); });
  api.method(Ft, "to_mat", {St}, Mt, "numeric columns as a matrix (rows x len(names))", [](void* s, Args& a) {
    const auto& names = obj<StrVec>(a[0]).v; Eigen::MatrixXd m(Eigen::Index(ff(s).nrows()), Eigen::Index(names.size()));
    for (size_t j = 0; j < names.size(); ++j) { const Column& c = ff(s).num_col(names[j]); for (size_t i = 0; i < c.size(); ++i) m(Eigen::Index(i), Eigen::Index(j)) = c.num[i]; }
    return matv(std::move(m)); });
  api.method(Ft, "write_csv", {"string"}, "void", "write the frame as CSV (atomic replace)", [](void* s, Args& a) { write_file_atomic(a[0].s, frame_to_csv(ff(s))); return Value::none(); });
  api.method(Ft, "csv", {}, "string", "the frame as CSV text", [](void* s, Args&) { return S(frame_to_csv(ff(s))); });
  api.method(Ft, "json", {}, "string", "{columns, rows, data} JSON text", [](void* s, Args&) { return S(frame_to_json(ff(s))); });
  register_frame_ops(api);
}

}  // namespace lmnum
