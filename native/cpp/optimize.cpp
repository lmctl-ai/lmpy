// Optimisation family: minimize, least_squares, root_scalar, root, quad, linprog.
// pycli ops covered: opt_minimize, opt_least_squares, num_root, opt_root, num_quad, opt_linprog(_to).
// Objectives, models, residuals, equations and integrands are expression strings in pycli's grammar (expr.hpp); option
// keys starting with `$` bind named constants and data vectors. Every expression evaluation is counted against
// `max_evals` (EVAL_LIMIT) and an optional `time_limit` in seconds (TIME_LIMIT); a stopped run reports the best point
// found so far, as pycli does. The numerical kernels live in optimize_min.cpp, optimize_lsq.cpp, optimize_roots.cpp,
// optimize_quad.cpp and optimize_lp.cpp.
#include "optimize_impl.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace lmnum {
namespace {

using namespace opt;
const std::string Vt = "Vec", St = "StrVec", Ot = "Opts", Rt = "Res", Mt = "Mat";
const double NaN = std::numeric_limits<double>::quiet_NaN();

bool is_ident(const std::string& s) {
  if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
  for (char c : s) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
  return true;
}

// Names of the unknowns plus the `$` bindings of one operation.
struct Space {
  std::string op;
  std::vector<std::string> names;
  ExprBindings b;
  std::vector<std::string> scalars;  // names, then bound scalars
  std::vector<VecRef> refs;
  Space(std::string o, std::vector<std::string> n, ExprBindings bind) : op(std::move(o)), names(std::move(n)), b(std::move(bind)) {
    std::set<std::string> seen;
    for (auto& s : names) {
      if (!is_ident(s)) fail(op + ": '" + s + "' is not a valid unknown name");
      if (!seen.insert(s).second) fail(op + ": unknown '" + s + "' is listed twice");
    }
    for (auto& s : b.scalar_names) if (seen.count(s)) fail(op + ": '$" + s + "' binds the name of an unknown");
    for (auto& s : b.vector_names) if (seen.count(s)) fail(op + ": '$" + s + "' binds the name of an unknown");
    scalars = names;
    scalars.insert(scalars.end(), b.scalar_names.begin(), b.scalar_names.end());
    refs = b.refs();
  }
  ExprProgram compile(const std::string& text, const char* what) const {
    try {
      return ExprCompiler(scalars, b.vector_names).compile(text);
    } catch (const dslvm::Error& e) {
      fail(op + ": " + what + ": " + e.message);
    }
  }
  // slots: unknowns then bound scalars
  std::vector<double> slots(const VectorXd& x) const {
    std::vector<double> s(scalars.size());
    for (size_t i = 0; i < names.size(); ++i) s[i] = x(Eigen::Index(i));
    for (size_t i = 0; i < b.scalar_values.size(); ++i) s[names.size() + i] = b.scalar_values[i];
    return s;
  }
  double eval(const ExprProgram& p, const VectorXd& x) const {
    auto s = slots(x);
    return expr_eval(p, s.data(), refs.data());
  }
};

void read_budget(Budget& bud, const Opts& o, const std::string& op) {
  bud.max_evals = o.integer("max_evals", 500000);
  if (bud.max_evals < 1 || bud.max_evals > 2000000) fail(op + ": max_evals must be in 1..2000000");
  bud.time_limit = o.num("time_limit", INF);
  if (!(bud.time_limit > 0)) fail(op + ": time_limit must be positive (seconds)");
  bud.start = std::chrono::steady_clock::now();
}

std::shared_ptr<Res> named(const std::vector<std::string>& names, const VectorXd& v) {
  auto r = std::make_shared<Res>();
  for (size_t i = 0; i < names.size(); ++i) r->num(names[i], v(Eigen::Index(i)));
  return r;
}

void read_bounds(const Opts& o, const std::string& op, Eigen::Index n, VectorXd& lb, VectorXd& ub) {
  const std::vector<double>* l = o.vec("lower");
  const std::vector<double>* u = o.vec("upper");
  if (!l && !u) { lb.resize(0); ub.resize(0); return; }
  lb = VectorXd::Constant(n, -INF);
  ub = VectorXd::Constant(n, INF);
  if (l) {
    if (Eigen::Index(l->size()) != n) fail(op + ": lower needs one value per unknown (" + std::to_string(n) + ")");
    for (Eigen::Index i = 0; i < n; ++i) lb(i) = std::isnan((*l)[size_t(i)]) ? -INF : (*l)[size_t(i)];
  }
  if (u) {
    if (Eigen::Index(u->size()) != n) fail(op + ": upper needs one value per unknown (" + std::to_string(n) + ")");
    for (Eigen::Index i = 0; i < n; ++i) ub(i) = std::isnan((*u)[size_t(i)]) ? INF : (*u)[size_t(i)];
  }
  for (Eigen::Index i = 0; i < n; ++i) {
    if (lb(i) == INF || ub(i) == -INF) fail(op + ": bound " + std::to_string(i) + " is an infinite value on the wrong side");
    if (!(lb(i) < ub(i))) fail(op + ": lower must be < upper (unknown " + std::to_string(i) + ")");
  }
}

void need_x0(const std::vector<std::string>& names, const std::vector<double>& x0, const std::string& op) {
  if (names.empty()) fail(op + ": no unknowns");
  if (names.size() != x0.size()) fail(op + ": " + std::to_string(names.size()) + " names but " + std::to_string(x0.size()) + " start values");
  for (double v : x0) if (!std::isfinite(v)) fail(op + ": start values must be finite");
}

// =============================================================================================== minimize
Res* minimize_op(const std::string& objective, const std::vector<std::string>& names, const std::vector<double>& x0v, const Opts& o) {
  const std::string op = "minimize";
  std::string method = o.str("method", "");
  double tol = o.num("tol", NaN);
  int64_t maxiter = o.integer("maxiter", 1000);
  std::vector<std::string> grads = o.strs("gradient") ? *o.strs("gradient") : std::vector<std::string>{};
  std::vector<std::string> eqs = o.strs("eq") ? *o.strs("eq") : std::vector<std::string>{};
  std::vector<std::string> ineqs = o.strs("ineq") ? *o.strs("ineq") : std::vector<std::string>{};
  std::string nonfinite = o.str("nonfinite", "inf");
  Budget bud;
  read_budget(bud, o, op);
  need_x0(names, x0v, op);
  Eigen::Index n = Eigen::Index(names.size());
  VectorXd lb, ub;
  read_bounds(o, op, n, lb, ub);
  Space sp(op, names, expr_bindings(o));
  o.finish(op);
  bool has_b = lb.size() > 0, has_c = !eqs.empty() || !ineqs.empty();
  if (method.empty()) method = has_c ? "SLSQP" : has_b ? "L-BFGS-B" : "BFGS";
  if (method == "trust-constr") fail("minimize: method trust-constr is not available in lmnum; use SLSQP for constrained problems");
  static const std::set<std::string> methods = {"Nelder-Mead", "Powell", "BFGS", "L-BFGS-B", "SLSQP"};
  if (!methods.count(method)) fail("minimize: method must be Nelder-Mead, Powell, BFGS, L-BFGS-B or SLSQP; got '" + method + "'");
  if (has_b && method == "BFGS") fail("minimize: BFGS does not take bounds; use L-BFGS-B, SLSQP, Powell or Nelder-Mead");
  if (has_c && method != "SLSQP") fail("minimize: constraints (eq, ineq) need method SLSQP");
  if (!grads.empty() && (method == "Nelder-Mead" || method == "Powell")) fail("minimize: gradient expressions are not used by " + method);
  if (!grads.empty() && Eigen::Index(grads.size()) != n) fail("minimize: gradient needs one expression per unknown (" + std::to_string(n) + ")");
  if (!std::isnan(tol) && !(tol > 0 && tol <= 1)) fail("minimize: tol must be in (0, 1]");
  if (maxiter < 1 || maxiter > 10000) fail("minimize: maxiter must be in 1..10000");
  if (nonfinite != "inf" && nonfinite != "error") fail("minimize: nonfinite must be \"inf\" or \"error\"");
  VectorXd x0 = to_eigen(x0v);
  if (has_b)
    for (Eigen::Index i = 0; i < n; ++i)
      if (x0(i) < lb(i) || x0(i) > ub(i)) fail("minimize: x0 is outside the bounds for '" + names[size_t(i)] + "'");
  ExprProgram fprog = sp.compile(objective, "objective");
  std::vector<ExprProgram> gprog, eprog, iprog;
  for (size_t i = 0; i < grads.size(); ++i) gprog.push_back(sp.compile(grads[i], "gradient"));
  for (auto& e : eqs) eprog.push_back(sp.compile(e, "eq constraint"));
  for (auto& e : ineqs) iprog.push_back(sp.compile(e, "ineq constraint"));

  bool at_start = true, policy_error = nonfinite == "error";
  double best_f = INF;
  VectorXd best_x = x0;
  auto f = [&](const VectorXd& x) {
    bud.tick();
    double v = sp.eval(fprog, x);
    if (!std::isfinite(v)) {
      if (at_start) fail("minimize: the objective is not finite at x0");
      ++bud.nonfinite;
      if (policy_error) throw Stop{"NON_FINITE", "non-finite objective value; the best parameters so far are reported"};
      return INF;
    }
    if (v < best_f) { best_f = v; best_x = x; }
    return v;
  };
  auto gradient_values = [&](const VectorXd& x) {
    bud.tick();
    VectorXd g(n);
    for (Eigen::Index i = 0; i < n; ++i) g(i) = sp.eval(gprog[size_t(i)], x);
    return g;
  };
  MinProblem P;
  P.f = f;
  if (!gprog.empty())
    P.grad = [&](const VectorXd& x) {
      VectorXd g = gradient_values(x);
      if (!g.allFinite()) { ++bud.nonfinite; throw Stop{"NON_FINITE", "non-finite gradient value"}; }
      return g;
    };
  P.lb = lb;
  P.ub = ub;
  for (auto& pr : eprog) P.eq.push_back([&, pp = &pr](const VectorXd& x) { bud.tick(); return sp.eval(*pp, x); });
  for (auto& pr : iprog) P.ineq.push_back([&, pp = &pr](const VectorXd& x) { bud.tick(); return sp.eval(*pp, x); });
  P.tol = tol;
  P.maxiter = int(maxiter);

  std::vector<std::string> warnings;
  std::string status = "OK", message;
  MinResult mr;
  bool stopped = false;
  VectorXd xv = x0;
  double fun = NaN;
  try {
    double f0 = f(x0);
    at_start = false;
    if (!gprog.empty()) {
      VectorXd an = gradient_values(x0);
      if (!an.allFinite()) fail("minimize: a gradient expression is not finite at x0");
      VectorXd num = VectorXd::Constant(n, NaN);
      for (Eigen::Index i = 0; i < n; ++i) {
        VectorXd up = x0;
        double h = 1.4901161193847656e-8 * std::max(1.0, std::fabs(up(i)));
        up(i) += h;
        bud.tick();
        double v = sp.eval(fprog, up);
        if (std::isfinite(v)) num(i) = (v - f0) / h;
      }
      double scale = 1, worst = 0;
      bool any = false;
      for (Eigen::Index i = 0; i < n; ++i)
        if (std::isfinite(num(i))) { any = true; scale = std::max(scale, std::fabs(num(i))); worst = std::max(worst, std::fabs(an(i) - num(i))); }
      if (any && !(worst <= 1e-4 * scale))
        warnings.push_back("gradient expressions disagree with finite differences at x0 (max difference " + jnum(worst) + ")");
    }
    if (method == "Nelder-Mead") mr = nelder_mead(P, x0);
    else if (method == "Powell") mr = powell(P, x0);
    else if (method == "BFGS") mr = bfgs(P, x0);
    else if (method == "L-BFGS-B") mr = lbfgsb(P, x0);
    else mr = slsqp(P, x0);
    xv = mr.x;
    fun = mr.fun;
    message = mr.message;
    status = mr.success ? "OK" : "NO_CONVERGENCE";
  } catch (const Stop& s) {
    stopped = true;
    xv = best_x;
    fun = best_f;
    status = s.status;
    message = s.message;
  }
  if (!stopped && !(xv.allFinite() && std::isfinite(fun))) { status = "NUMERICAL_FAILURE"; message = "the solver returned a non-finite point"; }
  double violation = NaN;
  if (has_c) {
    violation = 0;
    for (auto& pr : eprog) violation = std::max(violation, std::fabs(sp.eval(pr, xv)));
    for (auto& pr : iprog) violation = std::max(violation, std::max(0.0, -sp.eval(pr, xv)));
    if (status == "OK" && !(violation <= 1e-6)) { status = "NO_CONVERGENCE"; message = "constraint violation " + jnum(violation) + " exceeds 1e-6; " + message; }
  }
  if (bud.nonfinite) warnings.push_back(std::to_string(bud.nonfinite) + " evaluations of the objective were non-finite");
  auto* r = make<Res>();
  r->str("status", status).str("message", message).str("method", method);
  r->vec("x", xv).sub("par", named(names, xv)).num("fun", fun);
  r->integer("nit", stopped ? -1 : mr.nit);
  r->integer("nfev", stopped ? bud.count : mr.nfev);
  if (!stopped && mr.njev >= 0) r->integer("njev", mr.njev);
  r->integer("evals", bud.count).integer("nonfinite_evals", bud.nonfinite);
  if (has_c) r->num("constraint_violation", violation);
  r->strs("warnings", warnings);
  return r;
}

// =========================================================================================== least squares
// pycli's per-parameter adaptive central-difference Jacobian (step ladder 2^-2..2^6); returns columns disagreeing with J.
int adaptive_jacobian(const VectorFn& residual, const VectorXd& x, const VectorXd& r_ref, const MatrixXd& J0, double scale, const VectorXd& lb,
                      const VectorXd& ub, MatrixXd& J) {
  Eigen::Index n = x.size();
  J.resize(r_ref.size(), n);
  int disagree = 0;
  const double factors[9] = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  for (Eigen::Index j = 0; j < n; ++j) {
    double norm = J0.col(j).norm();
    double natural = norm > 0 ? scale / norm : 1.0;
    double base = 6.0554544523933395e-6 * std::max(std::fabs(x(j)), 1e-4 * natural);
    std::vector<VectorXd> est;
    for (double fct : factors) {
      double h = base * fct;
      VectorXd up = x, down = x;
      up(j) += h;
      down(j) -= h;
      double hi = ub.size() ? ub(j) : INF, lo = lb.size() ? lb(j) : -INF;
      if (up(j) > hi) est.push_back((r_ref - residual(down)) / h);
      else if (down(j) < lo) est.push_back((residual(up) - r_ref) / h);
      else est.push_back((residual(up) - residual(down)) / (2.0 * h));
    }
    size_t best = 0;
    double be = INF;
    for (size_t i = 0; i + 1 < est.size(); ++i) {
      double e = (est[i] - est[i + 1]).norm();
      if (e < be) { be = e; best = i; }
    }
    J.col(j) = est[best];
    if ((J.col(j) - J0.col(j)).norm() > 1e-3 * std::max(norm, J.col(j).norm())) ++disagree;
  }
  return disagree;
}

// curve_fit convention: pinv(J'J) from the SVD of the column-equilibrated J, times 2 cost / (m - n) unless absolute.
std::string covariance(const MatrixXd& jac, double cost, Eigen::Index m, Eigen::Index n, bool absolute, MatrixXd& cov, double& condition) {
  VectorXd norms = jac.colwise().norm().transpose();
  condition = INF;
  if (!(norms.array() > 0).all()) return "SINGULAR_JACOBIAN";
  MatrixXd Js = jac * norms.cwiseInverse().asDiagonal();
  Eigen::JacobiSVD<MatrixXd> svd(Js, Eigen::ComputeThinV);
  VectorXd s = svd.singularValues();
  condition = s(s.size() - 1) > 0 ? s(0) / s(s.size() - 1) : INF;
  for (Eigen::Index i = 0; i < s.size(); ++i) if (!(s(i) > 1e-8 * s(0))) return "SINGULAR_JACOBIAN";
  if (s.size() < n) return "SINGULAR_JACOBIAN";
  MatrixXd V = svd.matrixV();
  cov = V * s.array().square().inverse().matrix().asDiagonal() * V.transpose();
  cov = cov.cwiseQuotient(norms * norms.transpose());
  if (!absolute) {
    if (m <= n) return "NO_DOF";
    cov *= 2.0 * cost / double(m - n);
  }
  return "OK";
}

struct LsqOptions {
  std::string method, loss;
  double f_scale, tol;
  bool absolute_sigma;
  int64_t max_nfev;
  VectorXd lb, ub;
};

LsqOptions read_lsq_options(const Opts& o, const std::string& op, Eigen::Index n) {
  LsqOptions L;
  L.method = o.str("method", "trf");
  L.loss = o.str("loss", "linear");
  L.f_scale = o.num("f_scale", 1.0);
  L.absolute_sigma = o.flag("absolute_sigma", false);
  L.tol = o.num("tol", 1e-10);
  L.max_nfev = o.integer("max_nfev", 1000);
  read_bounds(o, op, n, L.lb, L.ub);
  if (L.method != "trf" && L.method != "dogbox" && L.method != "lm") fail(op + ": method must be trf, dogbox or lm; got '" + L.method + "'");
  static const std::set<std::string> losses = {"linear", "soft_l1", "huber", "cauchy", "arctan"};
  if (!losses.count(L.loss)) fail(op + ": loss must be linear, soft_l1, huber, cauchy or arctan; got '" + L.loss + "'");
  if (L.method == "lm" && L.lb.size()) fail(op + ": method lm does not support bounds; use trf or dogbox");
  if (L.method == "lm" && L.loss != "linear") fail(op + ": method lm supports only the linear loss");
  if (!(L.f_scale > 0 && std::isfinite(L.f_scale))) fail(op + ": f_scale must be positive");
  if (!(L.tol >= 2.3e-16 && L.tol <= 0.1)) fail(op + ": tol must be in [2.3e-16, 0.1]");
  if (L.max_nfev < 1 || L.max_nfev > 100000) fail(op + ": max_nfev must be in 1..100000");
  return L;
}

Res* run_least_squares(const std::string& op, const std::vector<std::string>& names, const VectorXd& x0, const LsqOptions& L, Budget& bud,
                       const std::function<VectorXd(const VectorXd&)>& eval_residuals) {
  Eigen::Index n = x0.size();
  if (L.lb.size())
    for (Eigen::Index i = 0; i < n; ++i)
      if (x0(i) < L.lb(i) || x0(i) > L.ub(i)) fail(op + ": x0 is outside the bounds for '" + names[size_t(i)] + "'");
  bool at_start = true;
  Eigen::Index rows = -1;
  double best_cost = INF;
  VectorXd best_x = x0;
  VectorFn residual = [&](const VectorXd& x) -> VectorXd {
    bud.tick();
    VectorXd r = eval_residuals(x);
    if (rows < 0) {
      rows = r.size();
      if (rows * n > 20000000) fail(op + ": " + std::to_string(rows) + " residuals x " + std::to_string(n) + " unknowns exceeds the Jacobian cap of 20000000 entries");
    } else if (r.size() != rows) fail(op + ": the residual length changed between evaluations");
    if (!r.allFinite()) {
      if (at_start) fail(op + ": the residuals are not finite at x0");
      ++bud.nonfinite;
      throw Stop{"NON_FINITE", "non-finite residual; the best parameters so far are reported"};
    }
    double c = 0.5 * r.dot(r);
    if (c < best_cost) { best_cost = c; best_x = x; }
    return r;
  };
  std::string status = "OK", message, cov_status = "UNAVAILABLE";
  std::vector<std::string> warnings;
  LsqResult res;
  bool solved = false;
  VectorXd xv = x0;
  Eigen::Index m = 0;
  VectorXd r0;
  try {
    r0 = residual(x0);
    at_start = false;
    m = r0.size();
    if (m == 0) fail(op + ": there are no residuals");
    if (L.method == "lm" && m < n) fail(op + ": method lm needs at least as many residuals as unknowns");
    LsqProblem P;
    P.residual = residual;
    P.lb = L.lb; P.ub = L.ub;
    P.method = L.method; P.loss = L.loss; P.f_scale = L.f_scale;
    P.ftol = P.xtol = P.gtol = L.tol;
    P.max_nfev = std::min<int64_t>(L.max_nfev, bud.max_evals);
    res = least_squares_solve(P, x0);
    solved = true;
    xv = res.x;
    message = res.message;
    status = res.status > 0 ? "OK" : "NO_CONVERGENCE";
  } catch (const Stop& s) {
    xv = best_x;
    status = s.status;
    message = s.message;
  }
  bool finite = xv.allFinite();
  if (!finite) { status = "NUMERICAL_FAILURE"; message = "the solver returned a non-finite point"; }
  auto* out = make<Res>();
  out->str("status", status).str("message", message).str("method", L.method).str("loss", L.loss);
  out->vec("x", xv).sub("par", named(names, xv));
  out->integer("n_residuals", int64_t(m)).integer("n_parameters", int64_t(n));
  MatrixXd cov = MatrixXd::Constant(n, n, NaN);
  VectorXd se = VectorXd::Constant(n, NaN), final_r;
  double condition = NaN;
  if (solved && finite) {
    double cost = res.cost;
    try {
      final_r = residual(xv);
      MatrixXd jac = res.jac;
      int disagree = 0;
      if (L.loss == "linear") {
        double scale = std::max({r0.norm(), final_r.norm(), 1e-300});
        MatrixXd J;
        disagree = adaptive_jacobian(residual, xv, final_r, res.jac, scale, L.lb, L.ub, J);
        if (J.allFinite()) jac = J;
        else disagree = 0;
      }
      MatrixXd c;
      cov_status = covariance(jac, cost, m, n, L.absolute_sigma, c, condition);
      if (cov_status == "OK") {
        if (disagree) {
          cov_status = "JACOBIAN_UNCERTAIN";
          warnings.push_back("the finite-difference Jacobian disagrees with the solver's on " + std::to_string(disagree) + " column(s); standard errors are uncertain");
        }
        if (condition > 1e6) warnings.push_back("the column-equilibrated Jacobian is ill-conditioned (condition number " + jnum(condition) + "): standard errors may be inaccurate");
        cov = c;
        se = c.diagonal().cwiseSqrt();
      }
      if (status != "OK") warnings.push_back("standard errors come from a non-converged iterate (" + status + ") and are unreliable");
      out->num("rss", final_r.dot(final_r));
    } catch (const Stop& s) {
      cov_status = "UNAVAILABLE_BUDGET";
      warnings.push_back("standard errors skipped: " + s.message);
      final_r = res.fun;
      out->num("rss", res.fun.dot(res.fun));
    }
    out->num("cost", cost).integer("nfev", res.nfev).num("optimality", res.optimality);
    if (res.njev >= 0) out->integer("njev", res.njev);
  } else {
    out->num("cost", best_cost).num("rss", 2.0 * best_cost).integer("nfev", bud.count);
  }
  out->sub("std_err", named(names, se)).vec("se", se).mat("cov", cov).str("cov_status", cov_status);
  if (!std::isnan(condition)) out->num("jacobian_condition", condition);
  out->vec("residuals", final_r.size() ? final_r : VectorXd(0));
  out->integer("evals", bud.count).integer("nonfinite_evals", bud.nonfinite);
  if (bud.nonfinite) warnings.push_back(std::to_string(bud.nonfinite) + " evaluations of the residuals were non-finite");
  out->strs("warnings", warnings);
  return out;
}

// model + response: residual_i = (model(row i) - response_i) / sigma_i
Res* least_squares_model(const std::string& model, const std::string& response, const std::vector<std::string>& names, const std::vector<double>& x0v, const Opts& o) {
  const std::string op = "least_squares";
  need_x0(names, x0v, op);
  Eigen::Index n = Eigen::Index(names.size());
  LsqOptions L = read_lsq_options(o, op, n);
  Budget bud;
  read_budget(bud, o, op);
  const Item* sig = o.find("sigma");
  Space sp(op, names, expr_bindings(o));
  o.finish(op);
  auto at = std::find(sp.b.vector_names.begin(), sp.b.vector_names.end(), response);
  if (at == sp.b.vector_names.end()) fail(op + ": response '" + response + "' must name a data vector bound with .set(\"$" + response + "\", values)");
  const std::vector<double>& y = sp.b.vector_values[size_t(at - sp.b.vector_names.begin())];
  size_t m = y.size();
  if (m == 0) fail(op + ": the response is empty");
  for (size_t k = 0; k < sp.b.vector_names.size(); ++k)
    if (sp.b.vector_values[k].size() != m) fail(op + ": data vector '" + sp.b.vector_names[k] + "' has " + std::to_string(sp.b.vector_values[k].size()) + " values; the response has " + std::to_string(m));
  for (double v : y) if (!std::isfinite(v)) fail(op + ": the response contains NaN or infinity");
  std::vector<double> sigma(m, 1.0);
  bool has_sigma = sig != nullptr;
  if (sig) {
    if (sig->kind == Item::NUM || sig->kind == Item::INT) {
      double s = sig->kind == Item::NUM ? sig->f : double(sig->i);
      if (!(s > 0 && std::isfinite(s))) fail(op + ": sigma must be positive");
      std::fill(sigma.begin(), sigma.end(), s);
    } else if (sig->kind == Item::VEC) {
      if (sig->vec.size() != m) fail(op + ": sigma needs one value per data row (" + std::to_string(m) + ")");
      for (size_t i = 0; i < m; ++i) { if (!(sig->vec[i] > 0 && std::isfinite(sig->vec[i]))) fail(op + ": sigma values must be positive"); sigma[i] = sig->vec[i]; }
    } else fail(op + ": sigma must be a positive float or a Vec of positive values");
  }
  (void)has_sigma;
  // per-row evaluation: data vectors become scalars holding the row's value
  std::vector<std::string> scal = sp.scalars;
  for (auto& v : sp.b.vector_names) scal.push_back(v);
  ExprProgram prog;
  try {
    prog = ExprCompiler(scal, {}).compile(model);
  } catch (const dslvm::Error& e) {
    fail(op + ": model: " + e.message);
  }
  size_t base = sp.scalars.size(), nv = sp.b.vector_names.size();
  auto eval = [&, prog](const VectorXd& x) {
    std::vector<double> s = sp.slots(x);
    s.resize(base + nv);
    VectorXd r = VectorXd::Zero(Eigen::Index(m));
    for (size_t i = 0; i < m; ++i) {
      for (size_t k = 0; k < nv; ++k) s[base + k] = sp.b.vector_values[k][i];
      r(Eigen::Index(i)) = (expr_eval(prog, s.data(), nullptr) - y[i]) / sigma[i];
    }
    return r;
  };
  return run_least_squares(op, names, to_eigen(x0v), L, bud, eval);
}

// residual expressions: an expression that names a data vector is evaluated per row (m values), others once.
Res* least_squares_residuals(const std::vector<std::string>& exprs, const std::vector<std::string>& names, const std::vector<double>& x0v, const Opts& o) {
  const std::string op = "least_squares";
  need_x0(names, x0v, op);
  if (exprs.empty()) fail(op + ": no residual expressions");
  Eigen::Index n = Eigen::Index(names.size());
  LsqOptions L = read_lsq_options(o, op, n);
  Budget bud;
  read_budget(bud, o, op);
  Space sp(op, names, expr_bindings(o));
  o.finish(op);
  struct Term { ExprProgram p; bool rows; };
  std::vector<Term> terms;
  size_t m_rows = 0;
  std::vector<std::string> scal = sp.scalars;
  for (auto& v : sp.b.vector_names) scal.push_back(v);
  for (auto& e : exprs) {
    std::vector<std::string> used = expr_names(e);
    bool rows = false;
    for (auto& u : used) rows = rows || std::find(sp.b.vector_names.begin(), sp.b.vector_names.end(), u) != sp.b.vector_names.end();
    bool reducer = e.find("sum(") != std::string::npos || e.find("mean(") != std::string::npos || e.find("prod(") != std::string::npos || e.find('[') != std::string::npos;
    if (rows && !reducer) {
      for (auto& u : used) {
        auto it = std::find(sp.b.vector_names.begin(), sp.b.vector_names.end(), u);
        if (it == sp.b.vector_names.end()) continue;
        size_t len = sp.b.vector_values[size_t(it - sp.b.vector_names.begin())].size();
        if (m_rows == 0) m_rows = len;
        else if (len != m_rows) fail(op + ": data vectors used row by row must have the same length");
      }
      try {
        terms.push_back({ExprCompiler(scal, {}).compile(e), true});
      } catch (const dslvm::Error& err) {
        fail(op + ": residual: " + err.message);
      }
    } else {
      terms.push_back({sp.compile(e, "residual"), false});
    }
  }
  size_t base = sp.scalars.size(), nv = sp.b.vector_names.size();
  auto eval = [&](const VectorXd& x) {
    std::vector<double> s = sp.slots(x);
    std::vector<double> out;
    for (auto& t : terms) {
      if (!t.rows) { out.push_back(expr_eval(t.p, s.data(), sp.refs.data())); continue; }
      std::vector<double> sr = s;
      sr.resize(base + nv);
      for (size_t i = 0; i < m_rows; ++i) {
        for (size_t k = 0; k < nv; ++k) {
          const auto& vv = sp.b.vector_values[k];
          sr[base + k] = i < vv.size() ? vv[i] : NaN;
        }
        out.push_back(expr_eval(t.p, sr.data(), nullptr));
      }
    }
    return to_eigen(out);
  };
  return run_least_squares(op, names, to_eigen(x0v), L, bud, eval);
}

// ============================================================================================ root_scalar
Res* root_scalar_op(const std::string& fexpr, const std::string& var, double lo, double hi, const Opts& o) {
  const std::string op = "root_scalar";
  std::string method = o.str("method", "brentq");
  double xtol = o.num("xtol", 2e-12), rtol = o.num("rtol", 8.881784197001252e-16);
  int64_t maxiter = o.integer("maxiter", 100);
  Budget bud;
  read_budget(bud, o, op);
  Space sp(op, {var}, expr_bindings(o));
  o.finish(op);
  if (!(std::isfinite(lo) && std::isfinite(hi) && lo < hi)) fail(op + ": need finite lower < upper");
  if (!(xtol > 0 && std::isfinite(xtol))) fail(op + ": xtol must be positive");
  if (!(rtol >= 8.881784197001252e-16 && rtol < 1)) fail(op + ": rtol must be in [4*eps, 1)");
  if (maxiter < 1 || maxiter > 1000) fail(op + ": maxiter must be in 1..1000");
  if (method != "brentq" && method != "bisect" && method != "ridder" && method != "toms748") fail(op + ": method must be brentq, bisect, ridder or toms748; got '" + method + "'");
  ExprProgram prog = sp.compile(fexpr, "f");
  auto raw = [&](double x) {
    bud.tick();
    VectorXd v(1);
    v(0) = x;
    return sp.eval(prog, v);
  };
  auto f = [&](double x) {
    double v = raw(x);
    if (!std::isfinite(v)) { ++bud.nonfinite; throw Stop{"NON_FINITE", "f(" + jnum(x) + ") is not finite"}; }
    return v;
  };
  auto* r = make<Res>();
  r->str("method", method);
  double fa = raw(lo), fb = raw(hi);
  if (!std::isfinite(fa) || !std::isfinite(fb)) fail(op + ": f is not finite at a bracket end");
  for (auto [val, pt] : {std::make_pair(fa, lo), std::make_pair(fb, hi)})
    if (val == 0.0) {
      r->str("status", "OK").str("message", "").num("root", pt).num("f_root", 0.0).integer("iterations", 0).integer("function_calls", 2).integer("evals", bud.count).flag("converged", true);
      return r;
    }
  if (std::signbit(fa) == std::signbit(fb)) fail(op + ": f(lower) and f(upper) must have opposite signs (or one must be zero)");
  try {
    ScalarRoot sr = bracket_root(method, f, lo, hi, xtol, rtol, int(maxiter));
    double froot = NaN;
    try { froot = raw(sr.root); } catch (const Stop&) {}
    r->str("status", sr.converged ? "OK" : "NO_CONVERGENCE").str("message", sr.converged ? "" : sr.flag);
    r->num("root", sr.root).num("f_root", froot).integer("iterations", std::max(0, sr.iterations)).integer("function_calls", std::max(0, sr.function_calls));
    r->flag("converged", sr.converged);
  } catch (const Stop& s) {
    r->str("status", s.status).str("message", s.message).num("root", NaN).num("f_root", NaN).flag("converged", false);
  }
  r->integer("evals", bud.count);
  return r;
}

// =================================================================================================== root
Res* root_op(const std::vector<std::string>& eqs, const std::vector<std::string>& names, const std::vector<double>& x0v, const Opts& o) {
  const std::string op = "root";
  bool explicit_method = o.has("method");
  const std::vector<double>* bracket = o.vec("bracket");
  std::string method = o.str("method", bracket ? "brentq" : "hybr");
  double xtol = o.num("xtol", 1e-12), residual_tol = o.num("residual_tol", 1e-8);
  bool has_maxiter = o.has("maxiter");
  int64_t maxiter = o.integer("maxiter", 100);
  Budget bud;
  read_budget(bud, o, op);
  Space sp(op, names, expr_bindings(o));
  o.finish(op);
  if (names.empty()) fail(op + ": no unknowns");
  if (eqs.size() != names.size()) fail(op + ": needs as many equations (" + std::to_string(eqs.size()) + ") as unknowns (" + std::to_string(names.size()) + ")");
  if (!bracket && x0v.size() != names.size()) fail(op + ": " + std::to_string(names.size()) + " names but " + std::to_string(x0v.size()) + " start values");
  if (bracket && explicit_method && method != "brentq") fail(op + ": a bracket requires method brentq (or omit method)");
  if (!bracket && method == "brentq") fail(op + ": method brentq needs a bracket");
  if (method != "hybr" && method != "lm" && method != "brentq") fail(op + ": method must be hybr, lm or brentq; got '" + method + "'");
  if (!bracket && has_maxiter) fail(op + ": maxiter is brentq-only; use max_evals for hybr/lm");
  if (!(xtol > 0 && xtol < 1)) fail(op + ": xtol must be in (0, 1)");
  if (!(residual_tol > 0 && std::isfinite(residual_tol))) fail(op + ": residual_tol must be positive");
  if (maxiter < 1 || maxiter > 1000) fail(op + ": maxiter must be in 1..1000");
  Eigen::Index n = Eigen::Index(names.size());
  std::vector<ExprProgram> progs;
  for (auto& e : eqs) progs.push_back(sp.compile(e, "equation"));
  bool at_start = true;
  double best_norm = INF;
  VectorXd best_x = bracket ? VectorXd::Zero(1) : to_eigen(x0v), best_fun;
  auto fun = [&](const VectorXd& x) {
    bud.tick();
    VectorXd r(n);
    for (Eigen::Index i = 0; i < n; ++i) r(i) = sp.eval(progs[size_t(i)], x);
    if (!r.allFinite()) {
      if (at_start) fail(op + (bracket ? ": the equation is not finite at a bracket endpoint" : ": the equations are not finite at the start point"));
      ++bud.nonfinite;
      throw Stop{"NON_FINITE", "non-finite equation value; the best point so far is reported"};
    }
    double nm = r.cwiseAbs().maxCoeff();
    if (nm < best_norm) { best_norm = nm; best_x = x; best_fun = r; }
    return r;
  };
  auto* res = make<Res>();
  res->str("method", method).integer("n_equations", int64_t(n));
  std::string status = "OK", message;
  VectorXd xv = best_x;
  try {
    if (bracket) {
      if (n != 1) fail(op + ": a bracket needs exactly one equation in one unknown");
      if (bracket->size() != 2 || !std::isfinite((*bracket)[0]) || !std::isfinite((*bracket)[1]) || !((*bracket)[0] < (*bracket)[1]))
        fail(op + ": bracket needs finite (lower, upper) with lower < upper");
      double lo = (*bracket)[0], hi = (*bracket)[1];
      auto f1 = [&](double t) { VectorXd v(1); v(0) = t; return fun(v)(0); };
      double fa = f1(lo), fb = f1(hi);
      at_start = false;
      if (fa == 0.0 || fb == 0.0) {
        xv = VectorXd::Constant(1, fa == 0.0 ? lo : hi);
        res->integer("iterations", 0);
      } else if (std::signbit(fa) == std::signbit(fb)) {
        fail(op + ": f(lower) and f(upper) must have opposite signs (or one must be zero)");
      } else {
        ScalarRoot sr = bracket_root("brentq", f1, lo, hi, xtol, 8.881784197001252e-16, int(maxiter));
        xv = VectorXd::Constant(1, sr.root);
        res->integer("iterations", std::max(0, sr.iterations));
        if (!sr.converged) { status = "NO_CONVERGENCE"; message = sr.flag; }
      }
    } else {
      VectorXd x0 = to_eigen(x0v);
      for (Eigen::Index i = 0; i < n; ++i) if (!std::isfinite(x0(i))) fail(op + ": start values must be finite");
      fun(x0);
      at_start = false;
      int64_t budget = std::max<int64_t>(1, bud.max_evals - bud.count);
      SystemRoot sr = method == "hybr" ? root_hybr(fun, x0, xtol, budget) : root_lm(fun, x0, xtol, budget);
      xv = sr.x;
      res->integer("nfev", sr.nfev);
      if (!sr.success) { status = "NO_CONVERGENCE"; message = sr.message; }
    }
  } catch (const Stop& s) {
    status = s.status;
    message = s.message;
    xv = best_x;
  }
  if (!xv.allFinite()) { status = "NUMERICAL_FAILURE"; message = "the solver returned a non-finite point"; xv = best_x; }
  VectorXd fin = best_fun;
  try {
    at_start = false;
    fin = fun(xv);
  } catch (const Stop&) {
  }
  if (fin.size() != n) fin = VectorXd::Constant(n, NaN);
  double residual_max = fin.allFinite() ? fin.cwiseAbs().maxCoeff() : INF;
  if (status == "OK" && !(residual_max <= residual_tol)) {
    status = "NO_CONVERGENCE";
    message = "the solver stopped at a point where max|f| = " + jnum(residual_max) + " exceeds residual_tol " + jnum(residual_tol) +
              " (a local minimum of the residual norm is not a root)";
  }
  res->str("status", status).str("message", message).vec("x", xv).sub("par", named(names, xv)).vec("fun", fin);
  res->num("residual_norm", fin.allFinite() ? fin.norm() : INF).num("residual_max", residual_max);
  res->integer("evals", bud.count).integer("nonfinite_evals", bud.nonfinite);
  return res;
}

// =================================================================================================== quad
Res* quad_op(const std::string& fexpr, const std::string& var, double a, double b, const Opts& o) {
  const std::string op = "quad";
  double epsabs = o.num("epsabs", 1.49e-8), epsrel = o.num("epsrel", 1.49e-8);
  int64_t limit = o.integer("limit", 50);
  std::vector<double> points = o.vec("points") ? *o.vec("points") : std::vector<double>{};
  Budget bud;
  read_budget(bud, o, op);
  Space sp(op, {var}, expr_bindings(o));
  o.finish(op);
  if (std::isnan(a) || std::isnan(b)) fail(op + ": the limits must not be NaN");
  if (std::isinf(a) && a == b) fail(op + ": a and b are the same infinity");
  if (!(epsabs >= 0 && epsabs <= 1) || !(epsrel >= 0 && epsrel < 1) || (epsabs == 0 && epsrel < 1e-13))
    fail(op + ": need 0 <= epsabs <= 1, 0 <= epsrel < 1 and epsrel >= 1e-13 when epsabs is 0");
  if (limit < 1 || limit > 1000) fail(op + ": limit must be in 1..1000");
  if (!points.empty()) {
    if (std::isinf(a) || std::isinf(b)) fail(op + ": points need finite limits");
    if (points.size() > 64) fail(op + ": at most 64 points");
    double lo = std::min(a, b), hi = std::max(a, b);
    for (double p : points) if (!(p > lo && p < hi)) fail(op + ": points must lie strictly between the limits");
  }
  ExprProgram prog = sp.compile(fexpr, "f");
  auto f = [&](double x) {
    bud.tick();
    VectorXd v(1);
    v(0) = x;
    double y = sp.eval(prog, v);
    if (!std::isfinite(y)) {
      ++bud.nonfinite;
      throw Stop{"NON_FINITE", "f(" + jnum(x) + ") is not finite; a singularity sitting on a Gauss-Kronrod node (for example the midpoint of a symmetric "
                               "interval) is evaluated exactly: pass points to isolate it"};
    }
    return y;
  };
  auto* r = make<Res>();
  if (a == b) {
    r->str("status", "OK").str("message", "").num("value", 0.0).num("abserr", 0.0).integer("neval", 0).integer("subdivisions", 0).integer("ier", 0).integer("evals", 0);
    return r;
  }
  try {
    QuadResult q = quad_integrate(f, a, b, epsabs, epsrel, int(limit), points);
    if (q.ier == 6) fail(op + ": QUADPACK reports invalid input");
    r->str("status", q.ier == 0 ? "OK" : "NO_CONVERGENCE").str("message", quad_message(q.ier, int(limit)));
    r->num("value", q.value).num("abserr", q.abserr).integer("neval", q.neval).integer("subdivisions", q.last).integer("ier", q.ier);
  } catch (const Stop& s) {
    r->str("status", s.status).str("message", s.message).num("value", NaN).num("abserr", NaN);
  }
  r->integer("evals", bud.count);
  return r;
}

}  // namespace

void register_optimize(Api& api) {
  api.section("Optimisation, roots, quadrature and linear programming",
              "Minimisation (`minimize`, scipy.optimize.minimize), nonlinear least squares (`least_squares`, scipy.optimize.least_squares "
              "with curve_fit covariances), bracketed and multivariate roots (`root_scalar`, `root`), adaptive quadrature (`quad`, "
              "QUADPACK) and LP/MILP (`linprog`, HiGHS). Objectives, models and equations are expression strings in pycli's "
              "grammar; the unknowns are named by a StrVec, and option keys starting with `$` bind constants and data vectors "
              "(`opts().set(\"$x\", xs)`). Numerical outcomes are a `status` (OK, NO_CONVERGENCE, EVAL_LIMIT, TIME_LIMIT, NON_FINITE, "
              "NUMERICAL_FAILURE, ...); bad input is a run-time error. Every expression evaluation counts against `max_evals` "
              "(default 500000) and the optional `time_limit` (seconds); a stopped run reports the best point so far.");
  // minimize
  const char* min_opts =
      "options: method (Nelder-Mead, Powell, BFGS, L-BFGS-B, SLSQP; default SLSQP with constraints, L-BFGS-B with bounds, else BFGS), "
      "tol, maxiter (default 1000), max_evals, time_limit, gradient (StrVec, one expression per unknown; checked against finite "
      "differences at x0), lower/upper (Vec; NaN or +-inf = open), eq/ineq (StrVec; ineq means expr >= 0; SLSQP), nonfinite (inf|error), "
      "$name bindings";
  api.fn("minimize", {"string", St, Vt}, Rt,
         "minimise an expression over the named unknowns from x0 (pycli opt_minimize, BFGS): x, par, fun, status, message, nit, nfev, njev, evals",
         [](Args& a) { return O(minimize_op(a[0].s, obj<StrVec>(a[1]).v, obj<Vec>(a[2]).v, no_opts())); });
  api.fn("minimize", {"string", St, Vt, Ot}, Rt, std::string("minimize with ") + min_opts + "; result adds constraint_violation with constraints",
         [](Args& a) { return O(minimize_op(a[0].s, obj<StrVec>(a[1]).v, obj<Vec>(a[2]).v, obj<Opts>(a[3]))); });
  // least squares
  const char* lsq_opts =
      "options: method (trf, dogbox, lm), loss (linear, soft_l1, huber, cauchy, arctan), f_scale, tol (ftol = xtol = gtol, default 1e-10), "
      "max_nfev (default 1000), lower/upper, absolute_sigma, max_evals, time_limit, $name bindings";
  api.fn("least_squares", {"string", "string", St, Vt, Ot}, Rt,
         std::string("fit model(row) to the response data vector (pycli opt_least_squares): residual_i = (model - response_i) / sigma_i, every `$` "
                     "data vector read row by row; ") + lsq_opts + ", sigma (float or Vec). Result: x, par, std_err, se, cov (curve_fit convention), "
                     "cov_status, cost, rss, residuals, status, nfev, optimality, jacobian_condition",
         [](Args& a) { return O(least_squares_model(a[0].s, a[1].s, obj<StrVec>(a[2]).v, obj<Vec>(a[3]).v, obj<Opts>(a[4]))); });
  api.fn("least_squares", {St, St, Vt, Ot}, Rt,
         std::string("minimise the sum of squared residual expressions (concatenated; one naming a data vector yields one residual per row, "
                     "one with a reducer or none yields one); ") + lsq_opts + ". Result as the model form",
         [](Args& a) { return O(least_squares_residuals(obj<StrVec>(a[0]).v, obj<StrVec>(a[1]).v, obj<Vec>(a[2]).v, obj<Opts>(a[3]))); });
  // roots
  api.fn("root_scalar", {"string", "string", "float", "float"}, Rt,
         "bracketed root of f(var) on [lower, upper] with brentq (pycli num_root): root, f_root, iterations, function_calls, status, evals",
         [](Args& a) { return O(root_scalar_op(a[0].s, a[1].s, a[2].f, a[3].f, no_opts())); });
  api.fn("root_scalar", {"string", "string", "float", "float", Ot}, Rt,
         "root_scalar with options method (brentq, bisect, ridder, toms748), xtol (default 2e-12), rtol (default 4 eps), maxiter (default 100), "
         "max_evals, time_limit, $name bindings",
         [](Args& a) { return O(root_scalar_op(a[0].s, a[1].s, a[2].f, a[3].f, obj<Opts>(a[4]))); });
  api.fn("root", {St, St, Vt}, Rt,
         "solve n equations = 0 in n named unknowns from x0 with MINPACK hybr (pycli opt_root); success is verified by max|f| <= residual_tol: "
         "x, par, fun, residual_norm, residual_max, nfev, status, evals",
         [](Args& a) { return O(root_op(obj<StrVec>(a[0]).v, obj<StrVec>(a[1]).v, obj<Vec>(a[2]).v, no_opts())); });
  api.fn("root", {St, St, Vt, Ot}, Rt,
         "root with options method (hybr, lm; brentq with bracket), bracket (Vec lower, upper: one equation, x0 ignored), xtol (default 1e-12), "
         "residual_tol (default 1e-8), maxiter (brentq only), max_evals, time_limit, $name bindings",
         [](Args& a) { return O(root_op(obj<StrVec>(a[0]).v, obj<StrVec>(a[1]).v, obj<Vec>(a[2]).v, obj<Opts>(a[3]))); });
  // quadrature
  api.fn("quad", {"string", "string", "float", "float"}, Rt,
         "integral of f(var) from a to b (pycli num_quad; QUADPACK QAGS, QAGI for inf()/-inf() limits): value, abserr, neval, subdivisions, status",
         [](Args& a) { return O(quad_op(a[0].s, a[1].s, a[2].f, a[3].f, no_opts())); });
  api.fn("quad", {"string", "string", "float", "float", Ot}, Rt,
         "quad with options epsabs, epsrel (default 1.49e-8), limit (subintervals, default 50), points (Vec of break points, QAGP), max_evals, "
         "time_limit, $name bindings",
         [](Args& a) { return O(quad_op(a[0].s, a[1].s, a[2].f, a[3].f, obj<Opts>(a[4]))); });
  // linear programming
  api.fn("linprog", {Vt}, Rt, "minimise c.x subject to x >= 0 (HiGHS; see the Opts form)", [](Args& a) { return O(linprog_run(obj<Vec>(a[0]).v, no_opts())); });
  api.fn("linprog", {Vt, Ot}, Rt,
         "LP/MILP with HiGHS (pycli opt_linprog, scipy linprog method highs): options A_ub, b_ub, A_eq, b_eq (Mat/Vec), lower/upper (Vec; default "
         "0 and +inf), integrality (Vec of 0/1), maximize, time_limit (default 10 s), node_limit, maxiter, mip_rel_gap. Result: x, fun "
         "(= objective), slack, con, status, message, nit, mip, ineqlin_marginals, eqlin_marginals, lower_marginals, upper_marginals, duals "
         "{ub, eq, lower, upper}, mip_gap, mip_node_count, primal_feasible, optimality_verified",
         [](Args& a) { return O(linprog_run(obj<Vec>(a[0]).v, obj<Opts>(a[1]))); });
}

}  // namespace lmnum
