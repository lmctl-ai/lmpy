// Statistical models (statsmodels): design matrices from term strings, OLS/WLS with robust covariances, GLM by IRLS.
// pycli ops covered: sm_ols(_to), sm_glm(_to). The time-series tests (sm_tsa_acf, sm_tsa_tools) are in models_tsa.cpp
// and SARIMAX (sm_arima) in models_arima.cpp.
//
// The estimators follow statsmodels 0.14 (BSD-3-Clause) step by step so that the numbers agree to rounding: OLS through
// the SVD pseudo-inverse (RegressionModel.fit(method="pinv")), statsmodels' constant detection, HC0-HC3 and one-way
// cluster sandwiches (statsmodels.stats.sandwich_covariance), and GLM._fit_irls with its starting values, its absolute
// deviance convergence rule and the final WLS refit that yields the covariance. Formulas are re-implemented, not copied.
#include "models_common.hpp"
#include "expr.hpp"
#include "frame.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>

namespace lmpy {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace models;
const double EPS = std::numeric_limits<double>::epsilon();

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Split on a separator at parenthesis depth 0 (quotes respected).
std::vector<std::string> split_top(const std::string& s, char sep) {
  std::vector<std::string> out;
  int depth = 0;
  char quote = 0;
  std::string cur;
  for (char c : s) {
    if (quote) { cur += c; if (c == quote) quote = 0; continue; }
    if (c == '\'' || c == '"') { quote = c; cur += c; continue; }
    if (c == '(' || c == '[') ++depth;
    if (c == ')' || c == ']') --depth;
    if (c == sep && depth == 0) { out.push_back(trim(cur)); cur.clear(); continue; }
    cur += c;
  }
  out.push_back(trim(cur));
  return out;
}

bool wrapped(const std::string& s, const std::string& head) {
  if (s.size() < head.size() + 2 || s.compare(0, head.size() + 1, head + "(") != 0 || s.back() != ')') return false;
  int depth = 0;  // the opening parenthesis must close at the very end
  for (size_t i = head.size(); i < s.size(); ++i) {
    if (s[i] == '(') ++depth;
    if (s[i] == ')' && --depth == 0 && i != s.size() - 1) return false;
  }
  return true;
}
std::string inner(const std::string& s, const std::string& head) { return trim(s.substr(head.size() + 1, s.size() - head.size() - 2)); }
std::string unquote(std::string s) {
  s = trim(s);
  if (s.size() >= 2 && (s[0] == '\'' || s[0] == '"') && s.back() == s[0]) return s.substr(1, s.size() - 2);
  return s;
}

// ---- Design matrices ---------------------------------------------------------------------------------------------
struct Factor {
  enum Kind { NUM, CAT, EXPR } kind = NUM;
  std::string col;   // NUM, CAT
  std::string text;  // EXPR (inside I(...))
};
struct Term {
  bool is_const = false;
  int poly = 0;  // poly(x, k): raw powers 1..k
  std::vector<Factor> factors;
};
struct CatInfo {
  std::vector<std::string> levels;  // observed levels after row selection, sorted
  std::vector<std::string> cell;    // per kept row, the level text
  std::string ref;
  bool ref_given = false;
};
struct Design {
  MatrixXd X;
  std::vector<std::string> names;
  std::vector<size_t> rows;  // kept rows of the frame
  size_t n_dropped = 0;
  std::shared_ptr<Res> coding = std::make_shared<Res>();
  std::vector<std::string> cat_order;
};

std::string level_text(const Column& c, size_t i) { return c.cell_text(i); }

// Parse the term list. `intercept` false removes the automatic constant (as does a "0" or "-1" term).
std::vector<Term> parse_terms(const Frame& f, const std::vector<std::string>& terms, bool& intercept, std::map<std::string, CatInfo>& cats, const char* op) {
  std::vector<Term> out;
  bool const_seen = false;
  auto cat_ref = [&](const std::string& col, const std::string* ref) {
    CatInfo& ci = cats[col];
    if (ref) {
      if (ci.ref_given && ci.ref != *ref) fail(std::string(op) + ": categorical " + col + " has two different reference levels ('" + ci.ref + "' and '" + *ref + "')");
      ci.ref = *ref;
      ci.ref_given = true;
    }
  };
  for (const auto& raw : terms) {
    std::string t = trim(raw);
    if (t.empty()) fail(std::string(op) + ": empty term");
    if (t == "0" || t == "-1") { intercept = false; continue; }
    if (t == "1" || t == "const") {
      if (const_seen) fail(std::string(op) + ": the intercept term appears twice");
      const_seen = true;
      Term c; c.is_const = true; out.push_back(c); continue;
    }
    Term term;
    if (wrapped(t, "poly")) {
      auto args = split_top(inner(t, "poly"), ',');
      if (args.size() != 2) fail(std::string(op) + ": poly needs (column, degree): '" + t + "'");
      int deg = std::atoi(args[1].c_str());
      if (deg < 1 || deg > 6) fail(std::string(op) + ": poly degree must be 1..6: '" + t + "'");
      if (!f.num_col(args[0]).numeric) fail(std::string(op) + ": poly needs a numeric column");
      Factor fa; fa.kind = Factor::NUM; fa.col = args[0];
      term.poly = deg; term.factors.push_back(fa);
      out.push_back(term);
      continue;
    }
    for (const auto& part : split_top(t, ':')) {
      if (part.empty()) fail(std::string(op) + ": empty factor in term '" + t + "'");
      Factor fa;
      if (wrapped(part, "C")) {
        auto args = split_top(inner(part, "C"), ',');
        if (args.empty() || args.size() > 2 || args[0].empty()) fail(std::string(op) + ": C() takes a column and an optional reference level: '" + part + "'");
        fa.kind = Factor::CAT; fa.col = args[0];
        f.get(fa.col);
        if (args.size() == 2) {
          std::string r = args[1];
          if (wrapped(r, "Treatment")) r = inner(r, "Treatment");
          if (r.compare(0, 10, "reference=") == 0) r = r.substr(10);
          r = unquote(r);
          cat_ref(fa.col, &r);
        } else cat_ref(fa.col, nullptr);
      } else if (wrapped(part, "I")) {
        fa.kind = Factor::EXPR; fa.text = inner(part, "I");
        if (fa.text.empty()) fail(std::string(op) + ": empty I() expression");
      } else {
        const Column& c = f.get(part);
        fa.col = part;
        if (c.numeric) fa.kind = Factor::NUM;
        else { fa.kind = Factor::CAT; cat_ref(part, nullptr); }  // text columns are categorical, as in patsy
      }
      term.factors.push_back(fa);
    }
    out.push_back(term);
  }
  if (!const_seen && intercept) { Term c; c.is_const = true; out.insert(out.begin(), c); }
  if (const_seen && !intercept) out.erase(std::remove_if(out.begin(), out.end(), [](const Term& x) { return x.is_const; }), out.end());
  return out;
}

// Build X for the rows without missing values in any used column (or fail, with na_action "error").
Design build_design(const Frame& f, const std::vector<std::string>& term_text, const std::vector<std::string>& extra_cols, const Opts& o,
                    bool& has_const_term, const char* op) {
  bool intercept = o.flag("intercept", true);
  std::string na = o.str("na_action", "error");
  if (na != "error" && na != "drop") fail(std::string(op) + ": na_action must be error or drop; got '" + na + "'");
  ExprBindings bind = expr_bindings(o);
  if (!bind.vector_names.empty()) fail(std::string(op) + ": `$` bindings in model terms must be scalars");
  std::map<std::string, CatInfo> cats;
  std::vector<Term> terms = parse_terms(f, term_text, intercept, cats, op);
  has_const_term = false;
  for (auto& t : terms) has_const_term |= t.is_const;
  size_t n = f.nrows();
  if (n == 0) fail(std::string(op) + ": the frame has no rows");

  // Expressions: evaluate on every row (NaN marks a missing value).
  std::map<std::string, std::vector<double>> expr_vals;
  for (auto& t : terms)
    for (auto& fa : t.factors) {
      if (fa.kind != Factor::EXPR || expr_vals.count(fa.text)) continue;
      std::vector<std::string> names;
      std::vector<const Column*> slots;
      for (auto& nm : expr_names(fa.text)) {
        const Column* c = f.find(nm);
        if (c && c->numeric) { names.push_back(nm); slots.push_back(c); }
      }
      size_t ncol = names.size();
      for (auto& s : bind.scalar_names) names.push_back(s);
      ExprProgram prog = ExprCompiler(names).compile(fa.text);
      std::vector<double> vars(names.size()), vals(n);
      for (size_t k = 0; k < bind.scalar_values.size(); ++k) vars[ncol + k] = bind.scalar_values[k];
      for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < ncol; ++k) vars[k] = slots[k]->num[i];
        vals[i] = expr_eval(prog, vars.data());
      }
      expr_vals[fa.text] = std::move(vals);
    }

  // Missing rows.
  std::vector<char> miss(n, 0);
  auto mark_col = [&](const std::string& name) {
    const Column& c = f.get(name);
    for (size_t i = 0; i < n; ++i) {
      if (c.numeric) {
        if (std::isnan(c.num[i])) miss[i] = 1;
        else if (std::isinf(c.num[i])) fail(std::string(op) + ": column " + name + " contains infinite values");
      } else if (c.text[i].empty()) miss[i] = 1;
    }
  };
  for (auto& name : extra_cols) mark_col(name);
  for (auto& t : terms)
    for (auto& fa : t.factors) {
      if (fa.kind == Factor::EXPR) {
        for (size_t i = 0; i < n; ++i) {
          double v = expr_vals[fa.text][i];
          if (std::isnan(v)) miss[i] = 1;
          else if (std::isinf(v)) fail(std::string(op) + ": I(" + fa.text + ") is infinite in row " + std::to_string(i));
        }
      } else mark_col(fa.col);
    }
  Design d;
  for (size_t i = 0; i < n; ++i) if (!miss[i]) d.rows.push_back(i);
  d.n_dropped = n - d.rows.size();
  if (d.n_dropped && na == "error")
    fail(std::string(op) + ": " + std::to_string(d.n_dropped) + " row(s) have missing values in the used columns; set na_action \"drop\" to drop them (n_dropped is reported)");
  if (d.rows.empty()) fail(std::string(op) + ": no rows left after dropping missing values");
  size_t m = d.rows.size();

  // Categorical levels on the kept rows: text sorted by code point, numbers numerically.
  for (auto& kv : cats) {
    const Column& c = f.get(kv.first);
    CatInfo& ci = kv.second;
    ci.cell.resize(m);
    for (size_t r = 0; r < m; ++r) ci.cell[r] = level_text(c, d.rows[r]);
    if (c.numeric) {
      std::set<double> lv;
      for (size_t r = 0; r < m; ++r) lv.insert(c.num[d.rows[r]]);
      for (double v : lv) { Column tmp; tmp.num = {v}; ci.levels.push_back(tmp.cell_text(0)); }
    } else {
      std::set<std::string> lv(ci.cell.begin(), ci.cell.end());
      ci.levels.assign(lv.begin(), lv.end());
    }
    if (!ci.ref_given) ci.ref = ci.levels[0];
    if (std::find(ci.levels.begin(), ci.levels.end(), ci.ref) == ci.levels.end()) {
      std::string shown;
      for (size_t i = 0; i < ci.levels.size() && i < 20; ++i) shown += (i ? ", " : "") + ci.levels[i];
      fail(std::string(op) + ": reference level '" + ci.ref + "' of " + kv.first + " is not among its levels: " + shown);
    }
    auto sub = std::make_shared<Res>();
    sub->str("reference", ci.ref).strs("levels", ci.levels);
    d.coding->sub(kv.first, sub);
  }

  // Columns.
  std::vector<std::pair<std::string, VectorXd>> columns;
  auto factor_columns = [&](const Factor& fa) {
    std::vector<std::pair<std::string, VectorXd>> fc;
    if (fa.kind == Factor::NUM) {
      const Column& c = f.get(fa.col);
      VectorXd v(m);
      for (size_t r = 0; r < m; ++r) v(r) = c.num[d.rows[r]];
      fc.push_back({fa.col, v});
    } else if (fa.kind == Factor::EXPR) {
      VectorXd v(m);
      for (size_t r = 0; r < m; ++r) v(r) = expr_vals[fa.text][d.rows[r]];
      fc.push_back({"I(" + fa.text + ")", v});
    } else {
      const CatInfo& ci = cats[fa.col];
      for (auto& lv : ci.levels) {
        if (lv == ci.ref) continue;
        VectorXd v(m);
        for (size_t r = 0; r < m; ++r) v(r) = ci.cell[r] == lv ? 1.0 : 0.0;
        fc.push_back({fa.col + "[T." + lv + "]", v});
      }
    }
    return fc;
  };
  for (auto& t : terms) {
    if (t.is_const) { columns.push_back({"const", VectorXd::Ones(m)}); continue; }
    if (t.poly) {
      auto base = factor_columns(t.factors[0])[0];
      for (int k = 1; k <= t.poly; ++k)
        columns.push_back({k == 1 ? base.first : base.first + "^" + std::to_string(k), base.second.array().pow(double(k)).matrix()});
      continue;
    }
    std::vector<std::pair<std::string, VectorXd>> combos = {{"", VectorXd::Ones(m)}};
    for (auto& fa : t.factors) {
      auto fc = factor_columns(fa);
      std::vector<std::pair<std::string, VectorXd>> next;
      for (auto& c : combos)
        for (auto& x : fc) next.push_back({c.first.empty() ? x.first : c.first + ":" + x.first, c.second.cwiseProduct(x.second)});
      combos.swap(next);
    }
    for (auto& c : combos) columns.push_back(c);
  }
  if (columns.empty()) fail(std::string(op) + ": the design has no columns");
  std::set<std::string> seen;
  for (auto& c : columns)
    if (!seen.insert(c.first).second) fail(std::string(op) + ": duplicate design column " + c.first);
  d.X.resize(m, columns.size());
  for (size_t j = 0; j < columns.size(); ++j) { d.X.col(j) = columns[j].second; d.names.push_back(columns[j].first); }
  if (!d.X.allFinite()) fail(std::string(op) + ": the design matrix contains non-finite values");
  return d;
}

VectorXd take_col(const Frame& f, const std::string& name, const std::vector<size_t>& rows, const char* op) {
  const Column& c = f.get(name);
  if (!c.numeric) fail(std::string(op) + ": column " + name + " is text, not numeric");
  VectorXd v(rows.size());
  for (size_t r = 0; r < rows.size(); ++r) v(r) = c.num[rows[r]];
  return v;
}
std::vector<int> group_codes(const Frame& f, const std::string& name, const std::vector<size_t>& rows, int& n_groups) {
  const Column& c = f.get(name);
  std::map<std::string, int> code;
  for (size_t r : rows) code.emplace(c.cell_text(r), 0);
  int k = 0;
  for (auto& kv : code) kv.second = k++;
  n_groups = k;
  std::vector<int> g;
  for (size_t r : rows) g.push_back(code[c.cell_text(r)]);
  return g;
}

// Sandwich H S H^T with S = sum of outer products of the rows of xu (per cluster when groups are given).
MatrixXd sandwich(const MatrixXd& bread, const MatrixXd& xu, const std::vector<int>* groups, int n_groups) {
  MatrixXd meat;
  if (groups) {
    MatrixXd sums = MatrixXd::Zero(n_groups, xu.cols());
    for (Eigen::Index i = 0; i < xu.rows(); ++i) sums.row((*groups)[size_t(i)]) += xu.row(i);
    meat = sums.transpose() * sums;
  } else meat = xu.transpose() * xu;
  return bread * meat * bread.transpose();
}

// Normal-theory skewness/kurtosis tests (scipy.stats.skewtest, kurtosistest, normaltest; D'Agostino-Pearson).
double moment(const VectorXd& r, int k) { double m = r.mean(); return (r.array() - m).pow(double(k)).mean(); }
void normality(const VectorXd& resid, Res& out) {
  double n = double(resid.size());
  double m2 = moment(resid, 2), m3 = moment(resid, 3), m4 = moment(resid, 4);
  double skew = m3 / std::pow(m2, 1.5), kurt = m4 / (m2 * m2);
  double jb = n / 6.0 * (skew * skew + 0.25 * (kurt - 3) * (kurt - 3));
  out.num("jarque_bera", jb).num("jarque_bera_p_value", chi2_sf(jb, 2)).num("skew", skew).num("kurtosis", kurt);
  if (n < 8) { out.num("omnibus", NAN).num("omnibus_p_value", NAN); return; }
  double y = skew * std::sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)));
  double beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9));
  double w2 = -1 + std::sqrt(2 * (beta2 - 1));
  double delta = 1 / std::sqrt(0.5 * std::log(w2));
  double alpha = std::sqrt(2.0 / (w2 - 1));
  if (y == 0) y = 1;
  double zs = delta * std::log(y / alpha + std::sqrt((y / alpha) * (y / alpha) + 1));
  double e = 3.0 * (n - 1) / (n + 1);
  double varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
  double x = (kurt - e) / std::sqrt(varb2);
  double sb1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * std::sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)));
  double a = 6.0 + 8.0 / sb1 * (2.0 / sb1 + std::sqrt(1 + 4.0 / (sb1 * sb1)));
  double term1 = 1 - 2 / (9.0 * a);
  double denom = 1 + x * std::sqrt(2 / (a - 4.0));
  double term2 = denom == 0 ? NAN : (denom > 0 ? 1.0 : -1.0) * std::cbrt((1 - 2.0 / a) / std::fabs(denom));
  double zk = (term1 - term2) / std::sqrt(2 / (9.0 * a));
  double k2 = zs * zs + zk * zk;
  out.num("omnibus", k2).num("omnibus_p_value", chi2_sf(k2, 2));
}

std::vector<double> stdv(const VectorXd& v) { return std::vector<double>(v.data(), v.data() + v.size()); }

void include_outputs(Res& r, const Opts& o, const VectorXd& fitted, const VectorXd& resid, const char* op) {
  if (const auto* inc = o.strs("include"))
    for (auto& k : *inc) {
      if (k == "fitted") r.vec("fitted", fitted);
      else if (k == "residuals") r.vec("residuals", resid);
      else fail(std::string(op) + ": include accepts fitted and residuals; got '" + k + "'");
    }
}

std::string cluster_col(const Opts& o) {
  std::string g = o.str("groups", "");
  std::string c = o.str("cluster", "");
  if (!g.empty() && !c.empty() && g != c) fail("cluster column given twice (groups and cluster differ)");
  return g.empty() ? c : g;
}

// ---- OLS / WLS ------------------------------------------------------------------------------------------------------
Res* ols(const Frame& f, const std::string& yname, const std::vector<std::string>& terms, const Opts& o) {
  const char* op = "ols";
  std::string cov_type = o.str("cov_type", "nonrobust");
  if (cov_type != "nonrobust" && cov_type != "HC0" && cov_type != "HC1" && cov_type != "HC2" && cov_type != "HC3" && cov_type != "cluster")
    fail("ols: cov_type must be nonrobust, HC0, HC1, HC2, HC3 or cluster; got '" + cov_type + "'");
  std::string wcol = o.str("weights", "");
  std::string gcol = cluster_col(o);
  if (cov_type == "cluster" && gcol.empty()) fail("ols: cov_type cluster needs the `groups` column");
  if (cov_type != "cluster" && !gcol.empty()) fail("ols: `groups` is only used with cov_type cluster");
  bool use_t = o.flag("use_t", cov_type == "nonrobust");
  double alpha = o.num("alpha", 0.05);
  need_alpha(alpha, op);
  std::vector<std::string> extra = {yname};
  if (!wcol.empty()) extra.push_back(wcol);
  if (!gcol.empty()) extra.push_back(gcol);
  bool has_const_term;
  Design d = build_design(f, terms, extra, o, has_const_term, op);
  const MatrixXd& X = d.X;
  Eigen::Index n = X.rows(), k = X.cols();
  VectorXd y = take_col(f, yname, d.rows, op);
  VectorXd w = VectorXd::Ones(n);
  if (!wcol.empty()) {
    w = take_col(f, wcol, d.rows, op);
    if (!(w.allFinite() && (w.array() > 0).all())) fail("ols: weights must be positive and finite");
  }
  int n_groups = 0;
  std::vector<int> groups;
  if (cov_type == "cluster") {
    groups = group_codes(f, gcol, d.rows, n_groups);
    if (n_groups < 2) fail("ols: cluster needs at least two groups");
  }
  VectorXd sw = w.array().sqrt();
  MatrixXd wX = sw.asDiagonal() * X;
  VectorXd wy = sw.cwiseProduct(y);
  Pinv pv = pinv_extended(wX);
  VectorXd beta = pv.pinv * wy;
  int rank = matrix_rank(wX);
  int k_constant, const_idx;
  detect_constant(X, k_constant, const_idx);
  VectorXd fitted = X * beta, resid = y - fitted, wresid = sw.cwiseProduct(resid);
  double ssr = wresid.squaredNorm();
  double df_model = double(rank - k_constant), df_resid = double(n - rank);
  double wmean = w.dot(y) / w.sum();
  double centered_tss = (w.array() * (y.array() - wmean).square()).sum();
  double uncentered_tss = (w.array() * y.array().square()).sum();
  double tss = k_constant ? centered_tss : uncentered_tss;
  double r2 = 1 - ssr / tss;
  double r2_adj = 1 - double(n - k_constant) / df_resid * (1 - r2);
  double nobs2 = double(n) / 2.0;
  double llf = -std::log(ssr) * nobs2 - (1 + std::log(M_PI / nobs2)) * nobs2;
  if (!wcol.empty()) llf += 0.5 * w.array().log().sum();
  double kp = df_model + k_constant;
  double aic = -2 * llf + 2 * kp, bic = -2 * llf + std::log(double(n)) * kp;
  double scale = ssr / df_resid;
  bool no_dof = df_resid <= 0;
  bool usable = rank == k && !no_dof;

  MatrixXd cov;
  if (cov_type == "nonrobust") cov = pv.normcov * scale;
  else if (cov_type == "cluster") {
    MatrixXd xu = wX.array().colwise() * wresid.array();
    cov = sandwich(pv.normcov, xu, &groups, n_groups);
    cov *= double(n_groups) / (n_groups - 1.0) * (double(n) - 1.0) / double(n - k);
  } else {
    VectorXd het = wresid.array().square();
    if (cov_type == "HC1") het *= double(n) / df_resid;
    if (cov_type == "HC2" || cov_type == "HC3") {
      VectorXd h = ((wX * pv.normcov).cwiseProduct(wX)).rowwise().sum();
      if (cov_type == "HC2") het = het.array() / (1 - h.array());
      else het = (wresid.array() / (1 - h.array())).square();
    }
    MatrixXd ps = pv.pinv.array().rowwise() * het.transpose().array().sqrt();
    cov = ps * ps.transpose();
  }
  VectorXd se = cov.diagonal().array().sqrt();
  double df_inf = cov_type == "cluster" ? double(n_groups - 1) : df_resid;

  double fstat = NAN, fp = NAN;
  if (cov_type == "nonrobust") {
    if (df_model > 0) {
      fstat = ((tss - ssr) / df_model) / (ssr / df_resid);
      fp = f_sf(fstat, df_model, df_resid);
    }
  } else if (rank == k) {
    std::vector<int> keep;
    for (int j = 0; j < int(k); ++j) if (j != const_idx) keep.push_back(j);
    if (!keep.empty()) {
      size_t q = keep.size();
      VectorXd b(q);
      MatrixXd c(q, q);
      for (size_t i = 0; i < q; ++i) { b(i) = beta(keep[i]); for (size_t j = 0; j < q; ++j) c(i, j) = cov(keep[i], keep[j]); }
      if (c.allFinite() && matrix_rank(c) == int(q)) {
        fstat = b.dot(c.fullPivLu().solve(b)) / double(q);
        fp = f_sf(fstat, double(q), df_resid);
      }
    }
  }

  auto* r = make<Res>();
  std::string status = rank < k ? "RANK_DEFICIENT" : "OK";
  if (usable && !(std::isfinite(ssr) && std::isfinite(llf) && se.allFinite())) {
    usable = false;
    status = "NUMERICAL_FAILURE";
  }
  r->str("status", status).integer("n_obs", n).integer("n_dropped", int64_t(d.n_dropped)).integer("rank", rank).str("cov_type", cov_type)
      .flag("use_t", use_t).num("alpha", alpha).sub("coding", d.coding).integer("df_model", int64_t(df_model)).integer("df_resid", int64_t(df_resid));
  if (no_dof) r->flag("no_dof", true);
  r->num("r2", r2).num("r2_adj", r2_adj).num("f_statistic", fstat).num("f_p_value", fp).integer("k_params", k).num("llf", llf).num("aic", aic)
      .num("bic", bic).num("ssr", ssr).num("sigma", df_resid > 0 ? std::sqrt(scale) : NAN)
      .num("condition_number", pv.sv(0) / pv.sv(pv.sv.size() - 1));
  VectorXd dr = resid.tail(n - 1) - resid.head(n - 1);
  r->num("durbin_watson", dr.squaredNorm() / resid.squaredNorm());
  normality(wresid, *r);
  if (status == "NUMERICAL_FAILURE") r->str("message", "the sums of squares overflowed; rescale the response and the regressors");
  r->sub("coefficients", coef_table(d.names, stdv(beta), stdv(se), use_t, df_inf, alpha, usable));
  include_outputs(*r, o, fitted, resid, op);
  o.finish(op);
  return r;
}

// ---- GLM ----------------------------------------------------------------------------------------------------------
struct Link {
  std::string name;
  static double clip01(double p) { return std::min(std::max(p, EPS), 1 - EPS); }
  static double clip0(double p) { return std::max(p, EPS); }
  double power() const { return name == "identity" ? 1.0 : name == "inverse_power" ? -1.0 : name == "sqrt" ? 0.5 : name == "inverse_squared" ? -2.0 : 0.0; }
  bool is_power() const { return power() != 0.0; }
  double link(double mu) const {
    if (name == "log") return std::log(clip0(mu));
    if (name == "logit") { double p = clip01(mu); return std::log(p / (1 - p)); }
    if (name == "probit") return norm_ppf(clip01(mu));
    if (name == "cloglog") return std::log(-std::log(1 - clip01(mu)));
    double pw = power();
    return pw == 1.0 ? mu : std::pow(mu, pw);
  }
  double inverse(double z) const {
    if (name == "log") return std::exp(z);
    if (name == "logit") return 1.0 / (1.0 + std::exp(-z));
    if (name == "probit") return norm_cdf(z);
    if (name == "cloglog") return 1 - std::exp(-std::exp(z));
    double pw = power();
    return pw == 1.0 ? z : std::pow(z, 1.0 / pw);
  }
  double deriv(double mu) const {
    if (name == "log") return 1.0 / clip0(mu);
    if (name == "logit") { double p = clip01(mu); return 1.0 / (p * (1 - p)); }
    if (name == "probit") { double p = clip01(mu); double q = norm_ppf(p); return 1.0 / (std::exp(-0.5 * q * q) / std::sqrt(2 * M_PI)); }
    if (name == "cloglog") { double p = clip01(mu); return 1.0 / ((p - 1) * std::log(1 - p)); }
    double pw = power();
    return pw == 1.0 ? 1.0 : pw * std::pow(mu, pw - 1);
  }
  double deriv2(double mu) const {
    if (name == "log") { double p = clip0(mu); return -1.0 / (p * p); }
    if (name == "logit") { double v = mu * (1 - mu); return (2 * mu - 1) / (v * v); }
    if (name == "probit") { double p = clip01(mu); double q = norm_ppf(p); double d = std::exp(-0.5 * q * q) / std::sqrt(2 * M_PI); return q / (d * d); }
    if (name == "cloglog") { double p = clip01(mu); double fl = std::log(1 - p); double d2 = -1 / ((1 - p) * (1 - p) * fl); return d2 * (1 + 1 / fl); }
    double pw = power();
    return pw == 1.0 ? 0.0 : pw * (pw - 1) * std::pow(mu, pw - 2);
  }
};

struct Family {
  std::string name;
  double nb_alpha = 1.0;
  double variance(double mu) const {
    if (name == "gaussian") return 1.0;
    if (name == "poisson") return std::fabs(mu);
    if (name == "gamma") return mu * mu;
    if (name == "inverse_gaussian") return std::fabs(mu) * mu * mu;
    if (name == "binomial") { double p = Link::clip01(mu); return p * (1 - p); }
    double p = Link::clip0(mu);  // negative_binomial
    return p + nb_alpha * p * p;
  }
  double variance_deriv(double mu) const {
    if (name == "gaussian") return 0.0;
    if (name == "poisson") return mu < 0 ? -1.0 : 1.0;
    if (name == "gamma") return 2 * std::fabs(mu) * (mu < 0 ? -1 : 1);
    if (name == "inverse_gaussian") return 3 * mu * mu * (mu < 0 ? -1 : 1);
    if (name == "binomial") return 1 - 2 * mu;
    return 1 + 2 * nb_alpha * Link::clip0(mu);
  }
  bool fixed_scale() const { return name == "poisson" || name == "binomial" || name == "negative_binomial"; }
  double starting_mu(double y, double ybar) const { return name == "binomial" ? (y + 0.5) / 2 : (y + ybar) / 2; }
  double resid_dev(double y, double mu) const {
    if (name == "gaussian") return (y - mu) * (y - mu);
    if (name == "poisson") return 2 * (y * std::log(Link::clip0(y / mu)) - (y - mu));
    if (name == "gamma") return 2 * (-std::log(Link::clip0(y / mu)) + (y - mu) / mu);
    if (name == "inverse_gaussian") return 1.0 / (y * mu * mu) * (y - mu) * (y - mu);
    if (name == "binomial") {
      double a = Link::clip0(y / (mu + 1e-20)), b = Link::clip0((1 - y) / (1 - mu + 1e-20));
      return 2 * (y * std::log(a) + (1 - y) * std::log(b));
    }
    double ya = y + 1 / nb_alpha, ma = mu + 1 / nb_alpha;
    return 2 * (y * std::log(Link::clip0(y / mu)) - ya * std::log(ya / ma));
  }
  double loglike_obs(double y, double mu, double vw, double scale) const {
    if (name == "gaussian") return (-vw * (y - mu) * (y - mu) / scale - std::log(scale / vw) - std::log(2 * M_PI)) / 2;
    if (name == "poisson") return vw / scale * (y * std::log(mu) - mu - lgam(y + 1));
    if (name == "gamma") {
      double em = Link::clip0(y / mu), ws = vw / scale;
      return ws * std::log(ws * em) - ws * em - lgam(ws) - std::log(y);
    }
    if (name == "inverse_gaussian") return (-vw * (y - mu) * (y - mu) / (scale * y * mu * mu) - std::log(scale * y * y * y / vw) - std::log(2 * M_PI)) / 2;
    if (name == "binomial") return (lgam(2) - lgam(y + 1) - lgam(2 - y) + y * std::log(mu / (1 - mu + 1e-20)) + std::log(1 - mu + 1e-20)) * vw;
    double a = nb_alpha;
    double ll = y * std::log(a * mu) - (y + 1 / a) * std::log(1 + a * mu) + lgam(y + 1 / a) - lgam(1 / a) - lgam(y + 1);
    return vw / scale * ll;
  }
};

const std::map<std::string, std::vector<std::string>>& family_links() {
  static const std::map<std::string, std::vector<std::string>> m = {
      {"gaussian", {"identity", "log", "inverse_power"}},
      {"poisson", {"log", "identity", "sqrt"}},
      {"binomial", {"logit", "probit", "cloglog", "log", "identity"}},
      {"gamma", {"inverse_power", "log", "identity"}},
      {"inverse_gaussian", {"inverse_squared", "inverse_power", "identity", "log"}},
      {"negative_binomial", {"log", "identity", "sqrt", "cloglog"}}};
  return m;
}

struct GlmData {
  MatrixXd X;
  VectorXd y, offset, vw, fw;
  Family fam;
  Link link;
  VectorXd iw() const { return vw.cwiseProduct(fw); }
  double wnobs() const { return fw.sum(); }
};

struct IrlsOut {
  VectorXd beta, mu, weights;
  double scale = 1;
  int n_iter = 0;
  bool converged = false, separation = false;
};

double deviance(const GlmData& g, const VectorXd& mu, double scale) {
  double s = 0;
  for (Eigen::Index i = 0; i < mu.size(); ++i) s += g.fam.resid_dev(g.y(i), mu(i)) * g.fw(i) * g.vw(i) / scale;
  return s;
}
double estimate_scale(const GlmData& g, const VectorXd& mu, double df_resid) {
  if (g.fam.fixed_scale()) return 1.0;
  double s = 0;
  for (Eigen::Index i = 0; i < mu.size(); ++i) s += (g.y(i) - mu(i)) * (g.y(i) - mu(i)) * g.vw(i) * g.fw(i) / g.fam.variance(mu(i));
  return s / df_resid;
}

// GLM._fit_irls: weighted least squares on the working response until |dev_k - dev_{k-1}| <= tol.
IrlsOut irls(const GlmData& g, double df_resid, int maxiter, double tol, const VectorXd* start) {
  Eigen::Index n = g.X.rows();
  IrlsOut out;
  VectorXd mu(n), eta(n);
  if (!start) {
    double ybar = g.y.mean();
    for (Eigen::Index i = 0; i < n; ++i) { mu(i) = g.fam.starting_mu(g.y(i), ybar); eta(i) = g.link.link(mu(i)); }
  } else {
    eta = g.X * *start + g.offset;
    for (Eigen::Index i = 0; i < n; ++i) mu(i) = g.link.inverse(eta(i));
  }
  out.scale = estimate_scale(g, mu, df_resid);
  double dev_prev = deviance(g, mu, out.scale);
  if (std::isnan(dev_prev)) fail("glm: the first guess on the deviance is NaN (a boundary problem: check the response against the family)");
  VectorXd iw = g.iw();
  VectorXd beta = VectorXd::Zero(g.X.cols()), w(n), z(n);
  for (int it = 0; it < maxiter; ++it) {
    for (Eigen::Index i = 0; i < n; ++i) {
      double d = g.link.deriv(mu(i));
      w(i) = iw(i) / (d * d * g.fam.variance(mu(i)));
      z(i) = eta(i) + d * (g.y(i) - mu(i)) - g.offset(i);
    }
    if (!w.allFinite() || !z.allFinite()) { out.n_iter = it + 1; break; }
    VectorXd sw = w.array().sqrt();
    MatrixXd wx = sw.asDiagonal() * g.X;
    Eigen::JacobiSVD<MatrixXd> svd(wx, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd s = svd.singularValues();
    double cut = EPS * double(std::max(wx.rows(), wx.cols())) * (s.size() ? s(0) : 0);
    VectorXd utb = svd.matrixU().transpose() * sw.cwiseProduct(z);
    for (Eigen::Index i = 0; i < s.size(); ++i) utb(i) = s(i) > cut ? utb(i) / s(i) : 0.0;
    beta = svd.matrixV() * utb;
    eta = g.X * beta + g.offset;
    for (Eigen::Index i = 0; i < n; ++i) mu(i) = g.link.inverse(eta(i));
    double dev = deviance(g, mu, out.scale);
    out.scale = estimate_scale(g, mu, df_resid);
    if ((mu - g.y).cwiseAbs().maxCoeff() <= 1e-8) out.separation = true;
    out.n_iter = it + 1;
    if (std::fabs(dev_prev - dev) <= tol) { out.converged = true; break; }
    dev_prev = dev;
  }
  out.beta = beta;
  out.mu = mu;
  out.weights = w;
  return out;
}

Res* glm(const Frame& f, const std::string& yname, const std::vector<std::string>& terms, const std::string& family, const Opts& o) {
  const char* op = "glm";
  auto fl = family_links().find(family);
  if (fl == family_links().end()) fail("glm: family must be gaussian, poisson, binomial, gamma, inverse_gaussian or negative_binomial; got '" + family + "'");
  std::string link = o.str("link", fl->second[0]);
  if (std::find(fl->second.begin(), fl->second.end(), link) == fl->second.end()) {
    std::string ok;
    for (auto& l : fl->second) ok += (ok.empty() ? "" : ", ") + l;
    fail("glm: link '" + link + "' is not available for the " + family + " family (" + ok + ")");
  }
  std::string cov_type = o.str("cov_type", "nonrobust");
  if (cov_type != "nonrobust" && cov_type != "HC0" && cov_type != "HC1" && cov_type != "HC2" && cov_type != "HC3" && cov_type != "cluster")
    fail("glm: cov_type must be nonrobust, HC0, HC1, HC2, HC3 or cluster; got '" + cov_type + "'");
  std::string gcol = cluster_col(o);
  if (cov_type == "cluster" && gcol.empty()) fail("glm: cov_type cluster needs the `groups` column");
  if (cov_type != "cluster" && !gcol.empty()) fail("glm: `groups` is only used with cov_type cluster");
  bool scaled = family == "gaussian" || family == "gamma" || family == "inverse_gaussian";
  bool use_t = o.flag("use_t", scaled && cov_type == "nonrobust");
  double alpha = o.num("alpha", 0.05);
  need_alpha(alpha, op);
  int64_t maxiter = o.integer("maxiter", 100);
  double tol = o.num("tol", 1e-8);
  if (maxiter < 1 || maxiter > 1000) fail("glm: maxiter must be 1..1000");
  if (!(tol > 0)) fail("glm: tol must be positive");
  double nb_alpha = o.num("nb_alpha", 1.0);
  if (family != "negative_binomial" && o.has("nb_alpha")) fail("glm: nb_alpha applies to the negative_binomial family only");
  if (!(nb_alpha > 0)) fail("glm: nb_alpha must be positive");
  std::string off = o.str("offset", ""), expo = o.str("exposure", ""), fwcol = o.str("freq_weights", "");
  std::string vwcol = o.str("var_weights", ""), wcol = o.str("weights", "");
  if (!vwcol.empty() && !wcol.empty()) fail("glm: give var_weights or weights (the same thing), not both");
  if (vwcol.empty()) vwcol = wcol;
  if (!expo.empty() && link != "log") fail("glm: exposure needs the log link");
  std::vector<std::string> extra = {yname};
  for (auto* c : {&off, &expo, &fwcol, &vwcol, &gcol}) if (!c->empty()) extra.push_back(*c);
  bool has_const_term;
  Design d = build_design(f, terms, extra, o, has_const_term, op);
  Eigen::Index n = d.X.rows(), k = d.X.cols();
  GlmData g;
  g.X = d.X;
  g.y = take_col(f, yname, d.rows, op);
  g.fam.name = family;
  g.fam.nb_alpha = nb_alpha;
  g.link.name = link;
  g.offset = VectorXd::Zero(n);
  g.vw = VectorXd::Ones(n);
  g.fw = VectorXd::Ones(n);
  if (!off.empty()) g.offset += take_col(f, off, d.rows, op);
  if (!expo.empty()) {
    VectorXd e = take_col(f, expo, d.rows, op);
    if (!(e.array() > 0).all()) fail("glm: exposure must be positive and finite");
    g.offset += e.array().log().matrix();
  }
  if (!vwcol.empty()) { g.vw = take_col(f, vwcol, d.rows, op); if (!(g.vw.array() > 0).all()) fail("glm: var_weights must be positive and finite"); }
  if (!fwcol.empty()) { g.fw = take_col(f, fwcol, d.rows, op); if (!(g.fw.array() > 0).all()) fail("glm: freq_weights must be positive and finite"); }
  const VectorXd& y = g.y;
  if ((family == "poisson" || family == "negative_binomial") && !((y.array() >= 0).all() && (y.array() == y.array().floor()).all()))
    fail("glm: " + family + " responses must be non-negative integers");
  if ((family == "gamma" || family == "inverse_gaussian") && !(y.array() > 0).all()) fail("glm: " + family + " responses must be positive");
  if (family == "binomial" && !((y.array() >= 0).all() && (y.array() <= 1).all())) fail("glm: binomial responses must lie in [0, 1]");
  int n_groups = 0;
  std::vector<int> groups;
  if (cov_type == "cluster") {
    groups = group_codes(f, gcol, d.rows, n_groups);
    if (n_groups < 2) fail("glm: cluster needs at least two groups");
  }
  int rank = matrix_rank(d.X);
  double df_model = rank - 1, df_resid = g.wnobs() - df_model - 1;

  IrlsOut fit = irls(g, df_resid, int(maxiter), tol, nullptr);
  auto* r = make<Res>();
  std::string status = "OK";
  if (fit.separation) status = "PERFECT_SEPARATION";
  else if (!fit.converged) status = "NO_CONVERGENCE";
  if (rank < k && status == "OK") status = "RANK_DEFICIENT";
  VectorXd beta = fit.beta, mu = fit.mu;
  bool invalid_mean = !mu.allFinite();
  for (Eigen::Index i = 0; i < n && !invalid_mean; ++i) {
    if ((family == "poisson" || family == "gamma" || family == "inverse_gaussian" || family == "negative_binomial") && !(mu(i) > 0)) invalid_mean = true;
    if (family == "binomial" && !(mu(i) >= 0 && mu(i) <= 1)) invalid_mean = true;
  }
  double scale = fit.scale;
  double dev = deviance(g, mu, 1.0);
  // llf: the profile likelihood for gaussian/identity (scale = ssr / nobs), otherwise at the estimated scale
  double llf_scale = scale;
  if (family == "gaussian" && link == "identity") llf_scale = ((y - mu).array().square() * g.iw().array()).sum() / g.wnobs();
  double llf = 0;
  for (Eigen::Index i = 0; i < n; ++i) llf += g.fw(i) * g.fam.loglike_obs(y(i), mu(i), g.vw(i), llf_scale);
  if (!beta.allFinite() || invalid_mean || !std::isfinite(llf) || !std::isfinite(dev)) status = "NUMERICAL_FAILURE";
  // null deviance: intercept-only fit (an IRLS fit when there is an offset, else the weighted mean)
  VectorXd mu0(n);
  bool has_offset = !off.empty() || !expo.empty();
  if (has_offset) {
    GlmData g0 = g;
    g0.X = MatrixXd::Ones(n, 1);
    VectorXd st(1);
    st(0) = g.link.link(y.mean());
    IrlsOut f0 = irls(g0, g.wnobs() - 1, 100, 1e-8, &st);
    mu0 = f0.mu;
  } else {
    VectorXd w = g.iw();
    mu0.setConstant(w.dot(y) / w.sum());
  }
  double null_dev = deviance(g, mu0, 1.0);
  double pearson = 0;
  for (Eigen::Index i = 0; i < n; ++i) pearson += (y(i) - mu(i)) * (y(i) - mu(i)) / g.fam.variance(mu(i)) * g.vw(i) * g.fw(i);

  // covariance: scale * (X' W X)^-1 with the working weights of the last IRLS step; robust sandwiches use the observed Hessian
  VectorXd sw = fit.weights.array().sqrt();
  Pinv pv = pinv_extended(sw.asDiagonal() * d.X);
  MatrixXd cov;
  if (cov_type == "nonrobust") cov = pv.normcov * scale;
  else {
    VectorXd score(n), hfac(n);
    VectorXd iw = g.iw();
    for (Eigen::Index i = 0; i < n; ++i) {
      double dv = g.link.deriv(mu(i)), var = g.fam.variance(mu(i));
      double sf1 = (y(i) - mu(i)) / dv / var * iw(i);
      score(i) = scale == 1 ? sf1 : sf1 / scale;
      double eim = iw(i) / (dv * dv * var);
      double tmp = sf1 * (var * g.link.deriv2(mu(i)) + g.fam.variance_deriv(mu(i)) * dv) / iw(i);
      hfac(i) = eim * (1 + tmp);
      if (scale != 1) hfac(i) /= scale;
    }
    MatrixXd hess = -(d.X.transpose() * hfac.asDiagonal() * d.X);
    MatrixXd hinv = hess.inverse();
    MatrixXd xu = d.X.array().colwise() * score.array();
    if (cov_type == "cluster") {
      cov = sandwich(hinv, xu, &groups, n_groups);
      cov *= double(n_groups) / (n_groups - 1.0) * (double(n) - 1.0) / double(n - k);
    } else {
      xu = xu.array().colwise() / g.fw.array().sqrt();
      cov = sandwich(hinv, xu, nullptr, 0);
    }
  }
  VectorXd se = cov.diagonal().array().sqrt();
  bool usable = rank == k && status != "NUMERICAL_FAILURE" && se.allFinite();
  double df_inf = cov_type == "cluster" ? double(n_groups - 1) : df_resid;

  r->str("status", status).str("family", family).str("link", link).integer("n_obs", n).integer("n_dropped", int64_t(d.n_dropped)).integer("rank", rank)
      .str("cov_type", cov_type).flag("use_t", use_t).num("alpha", alpha).sub("coding", d.coding).flag("converged", fit.converged).integer("n_iter", fit.n_iter);
  if (rank < k) r->flag("rank_deficient", true);
  if (invalid_mean) r->str("message", "a fitted mean lies outside the support of the " + family + " family (choose the canonical link or a different model)");
  else if (status == "PERFECT_SEPARATION") r->str("message", "perfect separation detected; the maximum likelihood estimate does not exist");
  r->integer("df_model", int64_t(df_model)).integer("df_resid", int64_t(std::llround(df_resid))).num("deviance", dev).num("null_deviance", null_dev)
      .num("pearson_chi2", pearson).num("scale", scale).num("llf", llf).integer("k_params", rank).num("aic", -2 * llf + 2 * rank)
      .num("bic", -2 * llf + std::log(g.wnobs()) * rank);
  if (family == "negative_binomial") r->num("nb_alpha", nb_alpha);
  r->sub("coefficients", coef_table(d.names, stdv(beta), stdv(se), use_t, df_inf, alpha, usable));
  include_outputs(*r, o, mu, y - mu, op);
  o.finish(op);
  return r;
}

// "x1 + x2 + C(g, a) + I(x^2)": split on top-level `+`.
std::vector<std::string> rhs_terms(const std::string& rhs) {
  auto t = split_top(rhs, '+');
  for (auto& x : t) if (x.empty()) fail("model terms: empty term in '" + rhs + "'");
  return t;
}

}  // namespace

void register_models(Api& api) {
  api.section("Statistical models",
              "Regression models and time-series statistics after statsmodels 0.14 (pycli `sm_ols`, `sm_glm`, `sm_arima`, `sm_tsa_acf`, "
              "`sm_tsa_tools`). Model terms are strings: a numeric column `x1`, an interaction `x1:x2`, a categorical `C(g)` or "
              "`C(g, ref)` (treatment coding; the reference defaults to the first sorted level; columns are named `g[T.level]`; a text "
              "column is categorical without `C`), an expression `I(x1^2)` (pycli expression grammar over the numeric columns and `$` "
              "scalar bindings), raw powers `poly(x, 3)` (named `x`, `x^2`, `x^3`), `1`/`const` to place the intercept and `0`/`-1` to "
              "drop it. Results keep pycli's field names; undefined uncertainty (rank deficiency, no residual degrees of freedom) is NaN. "
              "Common options of `ols`/`glm`: `cov_type` (nonrobust, HC0-HC3, cluster with `groups`), `use_t`, `alpha` (interval level "
              "1-alpha), `na_action` (error or drop), `include` (StrVec: fitted, residuals), `intercept` (bool).");
  const std::string Rt = "Res", Ft = "Frame", St = "StrVec";
  api.fn("ols", {Ft, "string", St}, Rt,
         "OLS of column y on the terms (pycli sm_ols): coefficients {names, estimate, std_err, statistic, p_value, ci_lower, ci_upper}, n_obs, r2, r2_adj, "
         "sigma, f_statistic, f_p_value, llf, aic, bic, ssr, df_model, df_resid, rank, condition_number, durbin_watson, jarque_bera, omnibus, skew, kurtosis, coding, status",
         [](Args& a) { return O(ols(obj<Frame>(a[0]), a[1].s, obj<StrVec>(a[2]).v, no_opts())); });
  api.fn("ols", {Ft, "string", St, "Opts"}, Rt,
         "OLS/WLS with options: `weights` (column: WLS), `cov_type`, `groups` (cluster column), `use_t`, `alpha`, `na_action`, `include`, `intercept`",
         [](Args& a) { return O(ols(obj<Frame>(a[0]), a[1].s, obj<StrVec>(a[2]).v, obj<Opts>(a[3]))); });
  api.fn("glm", {Ft, "string", St, "string"}, Rt,
         "GLM by IRLS (pycli sm_glm) with family gaussian, poisson, binomial, gamma, inverse_gaussian or negative_binomial and its default link: "
         "coefficients, deviance, null_deviance, pearson_chi2, scale, llf, aic, bic, df_model, df_resid, converged, n_iter, status",
         [](Args& a) { return O(glm(obj<Frame>(a[0]), a[1].s, obj<StrVec>(a[2]).v, a[3].s, no_opts())); });
  api.fn("glm", {Ft, "string", St, "string", "Opts"}, Rt,
         "GLM with options: `link`, `offset` / `exposure` (columns; exposure needs the log link), `var_weights` (alias `weights`), `freq_weights`, "
         "`nb_alpha` (negative binomial dispersion, default 1), `maxiter` (100), `tol` (1e-8, absolute deviance change), `cov_type`, `groups`, `use_t`, `alpha`, `na_action`, `include`, `intercept`",
         [](Args& a) { return O(glm(obj<Frame>(a[0]), a[1].s, obj<StrVec>(a[2]).v, a[3].s, obj<Opts>(a[4]))); });
  api.fn("ols", {Ft, "string", "string"}, Rt, "OLS with the terms written as one right-hand side joined by `+`: `ols(f, \"y\", \"x1 + x2 + C(g, a) + x1:x2\")`",
         [](Args& a) { return O(ols(obj<Frame>(a[0]), a[1].s, rhs_terms(a[2].s), no_opts())); });
  api.fn("ols", {Ft, "string", "string", "Opts"}, Rt, "OLS/WLS with a `+`-joined right-hand side and options",
         [](Args& a) { return O(ols(obj<Frame>(a[0]), a[1].s, rhs_terms(a[2].s), obj<Opts>(a[3]))); });
  api.fn("glm", {Ft, "string", "string", "string"}, Rt, "GLM with a `+`-joined right-hand side: `glm(f, \"count\", \"C(group)\", \"poisson\")`",
         [](Args& a) { return O(glm(obj<Frame>(a[0]), a[1].s, rhs_terms(a[2].s), a[3].s, no_opts())); });
  api.fn("glm", {Ft, "string", "string", "string", "Opts"}, Rt, "GLM with a `+`-joined right-hand side and options",
         [](Args& a) { return O(glm(obj<Frame>(a[0]), a[1].s, rhs_terms(a[2].s), a[3].s, obj<Opts>(a[4]))); });
  models::register_tsa(api);
  models::register_arima(api);
}

}  // namespace lmpy
