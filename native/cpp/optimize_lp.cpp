// Linear and mixed-integer programming behind `linprog`: HiGHS (MIT) with scipy.optimize.linprog(method="highs")
// conventions for the result (x, fun, slack, con, marginals with scipy's signs) plus pycli's opt_linprog safeguards:
// rows and objective scaled by their largest coefficient before the solve, marginals restored to original units, and an
// original-data certificate (primal feasibility, dual signs, stationarity, complementarity, duality gap) that turns an
// unverifiable "optimal" into NUMERICAL_FAILURE.
#include "optimize_impl.hpp"
#include <Highs.h>
#include <algorithm>
#include <cmath>

namespace lmnum::opt {
namespace {

using LD = long double;

std::string status_of(HighsModelStatus s) {
  switch (s) {
    case HighsModelStatus::kOptimal: return "OK";
    case HighsModelStatus::kInfeasible: return "INFEASIBLE";
    case HighsModelStatus::kUnboundedOrInfeasible: return "INFEASIBLE_OR_UNBOUNDED";
    case HighsModelStatus::kUnbounded: return "UNBOUNDED";
    case HighsModelStatus::kModelError: return "MODEL_ERROR";
    case HighsModelStatus::kTimeLimit: return "TIME_LIMIT";
    case HighsModelStatus::kIterationLimit: return "ITERATION_LIMIT";
    case HighsModelStatus::kSolutionLimit: return "NODE_LIMIT";
    default: return "NUMERICAL_FAILURE";
  }
}

double rel_error(LD value, LD scale) {
  LD r = std::fabs(value) / std::max(scale, std::numeric_limits<LD>::min());
  return std::isfinite(double(r)) ? double(r) : INF;
}

}  // namespace

Res* linprog_run(const std::vector<double>& cin, const Opts& o) {
  const std::string op = "linprog";
  int n = int(cin.size());
  if (n == 0) fail("linprog: c is empty");
  auto mat_opt = [&](const char* key) -> MatrixXd {
    const MatrixXd* m = o.mat(key);
    return m ? *m : MatrixXd(0, n);
  };
  auto vec_opt = [&](const char* key) -> VectorXd {
    const std::vector<double>* v = o.vec(key);
    return v ? to_eigen(*v) : VectorXd(0);
  };
  MatrixXd Aub = mat_opt("A_ub"), Aeq = mat_opt("A_eq");
  VectorXd bub = vec_opt("b_ub"), beq = vec_opt("b_eq");
  bool has_lower = o.has("lower"), has_upper = o.has("upper");
  VectorXd lower = has_lower ? vec_opt("lower") : VectorXd::Zero(n);
  VectorXd upper = has_upper ? vec_opt("upper") : VectorXd::Constant(n, INF);
  VectorXd integ = o.has("integrality") ? vec_opt("integrality") : VectorXd::Zero(n);
  bool maximize = o.flag("maximize", false);
  double time_limit = o.num("time_limit", 10.0);
  int64_t node_limit = o.integer("node_limit", 10000);
  int64_t maxiter = o.integer("maxiter", 100000);
  double mip_rel_gap = o.num("mip_rel_gap", 0.0);
  o.finish(op);
  VectorXd c = to_eigen(cin);
  if (!c.allFinite()) fail("linprog: c must be finite");
  if (Aub.cols() != n || Aeq.cols() != n) fail("linprog: A_ub and A_eq need " + std::to_string(n) + " columns (one per variable)");
  if (bub.size() != Aub.rows()) fail("linprog: b_ub has " + std::to_string(bub.size()) + " values; A_ub has " + std::to_string(Aub.rows()) + " rows");
  if (beq.size() != Aeq.rows()) fail("linprog: b_eq has " + std::to_string(beq.size()) + " values; A_eq has " + std::to_string(Aeq.rows()) + " rows");
  if (lower.size() != n || upper.size() != n || integ.size() != n) fail("linprog: lower, upper and integrality need one value per variable");
  if (!Aub.allFinite() || !Aeq.allFinite() || !bub.allFinite() || !beq.allFinite()) fail("linprog: constraint matrices and right-hand sides must be finite");
  for (int i = 0; i < n; ++i) {
    if (std::isnan(lower(i))) lower(i) = -INF;
    if (std::isnan(upper(i))) upper(i) = INF;
    if (lower(i) > upper(i)) fail("linprog: lower > upper for variable " + std::to_string(i));
    if (integ(i) != 0 && integ(i) != 1) fail("linprog: integrality entries must be 0 (continuous) or 1 (integer)");
  }
  if (!(time_limit > 0)) fail("linprog: time_limit must be positive");
  bool mixed = (integ.array() != 0).any();
  double sign = maximize ? -1 : 1;
  double cost_scale = c.cwiseAbs().maxCoeff();
  if (cost_scale == 0) cost_scale = 1;
  auto row_scale = [](const MatrixXd& A) {
    VectorXd s(A.rows());
    for (Eigen::Index i = 0; i < A.rows(); ++i) { double m = A.row(i).cwiseAbs().maxCoeff(); s(i) = m == 0 ? 1 : m; }
    return s;
  };
  VectorXd us = row_scale(Aub), es = row_scale(Aeq);
  int mu = int(Aub.rows()), me = int(Aeq.rows()), m = mu + me;
  HighsLp lp;
  lp.num_col_ = n;
  lp.num_row_ = m;
  lp.col_cost_.resize(size_t(n));
  for (int j = 0; j < n; ++j) lp.col_cost_[size_t(j)] = sign * c(j) / cost_scale;
  for (int j = 0; j < n; ++j) {
    lp.col_lower_.push_back(std::isinf(lower(j)) ? -kHighsInf : lower(j));
    lp.col_upper_.push_back(std::isinf(upper(j)) ? kHighsInf : upper(j));
  }
  for (int i = 0; i < mu; ++i) { lp.row_lower_.push_back(-kHighsInf); lp.row_upper_.push_back(bub(i) / us(i)); }
  for (int i = 0; i < me; ++i) { lp.row_lower_.push_back(beq(i) / es(i)); lp.row_upper_.push_back(beq(i) / es(i)); }
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.num_col_ = n;
  lp.a_matrix_.num_row_ = m;
  lp.a_matrix_.start_.assign(1, 0);  // the default matrix already holds start_ = {0}
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < mu; ++i) {
      double v = Aub(i, j) / us(i);
      if (v != 0) { lp.a_matrix_.index_.push_back(i); lp.a_matrix_.value_.push_back(v); }
    }
    for (int i = 0; i < me; ++i) {
      double v = Aeq(i, j) / es(i);
      if (v != 0) { lp.a_matrix_.index_.push_back(mu + i); lp.a_matrix_.value_.push_back(v); }
    }
    lp.a_matrix_.start_.push_back(HighsInt(lp.a_matrix_.index_.size()));
  }
  if (mixed)
    for (int j = 0; j < n; ++j) lp.integrality_.push_back(integ(j) != 0 ? HighsVarType::kInteger : HighsVarType::kContinuous);
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("log_to_console", false);
  highs.setOptionValue("threads", 1);
  highs.setOptionValue("presolve", "on");
  highs.setOptionValue("time_limit", time_limit);
  highs.setOptionValue("simplex_iteration_limit", HighsInt(maxiter));
  highs.setOptionValue("ipm_iteration_limit", HighsInt(maxiter));
  highs.setOptionValue("mip_max_nodes", HighsInt(node_limit));
  highs.setOptionValue("mip_rel_gap", mip_rel_gap);
  highs.setOptionValue("small_matrix_value", 1e-12);
  highs.setOptionValue("primal_feasibility_tolerance", 1e-9);
  highs.setOptionValue("dual_feasibility_tolerance", 1e-9);
  highs.setOptionValue("ipm_optimality_tolerance", 1e-10);
  highs.setOptionValue("simplex_strategy", 1);  // dual simplex, as scipy
  auto* res = make<Res>();
  if (highs.passModel(lp) == HighsStatus::kError) {
    res->str("status", "MODEL_ERROR").str("message", "HiGHS refused the model").flag("mip", mixed).flag("has_primal", false);
    return res;
  }
  highs.run();
  HighsModelStatus ms = highs.getModelStatus();
  const HighsInfo& info = highs.getInfo();
  std::string status = status_of(ms);
  res->str("status", status);
  res->str("message", highs.modelStatusToString(ms) + " (HiGHS Status " + std::to_string(int(ms)) + ")");
  res->integer("highs_model_status", int(ms));
  res->integer("nit", info.simplex_iteration_count > 0 ? info.simplex_iteration_count : std::max(0, int(info.ipm_iteration_count)));
  res->flag("mip", mixed);
  bool lp_ok = ms == HighsModelStatus::kOptimal;
  bool mip_has = mixed && (ms == HighsModelStatus::kOptimal || ms == HighsModelStatus::kTimeLimit || ms == HighsModelStatus::kIterationLimit ||
                           ms == HighsModelStatus::kSolutionLimit) && info.objective_function_value < kHighsInf && info.primal_solution_status == 2;
  if (mixed) {
    if (std::isfinite(info.mip_gap)) res->num("mip_gap", info.mip_gap);
    res->integer("mip_node_count", info.mip_node_count);
    if (std::isfinite(info.mip_dual_bound)) res->num("mip_dual_bound", sign * cost_scale * info.mip_dual_bound);
  }
  if (!(mixed ? mip_has : lp_ok)) {
    res->flag("has_primal", false);
    return res;
  }
  const HighsSolution& sol = highs.getSolution();
  VectorXd x(n);
  for (int j = 0; j < n; ++j) x(j) = sol.col_value[size_t(j)];
  double integrality_error = 0;
  for (int j = 0; j < n; ++j)
    if (integ(j) != 0) integrality_error = std::max(integrality_error, std::fabs(x(j) - std::nearbyint(x(j))));
  if (integrality_error <= 1e-6)
    for (int j = 0; j < n; ++j) if (integ(j) != 0) x(j) = std::nearbyint(x(j));
  for (int j = 0; j < n; ++j) if (x(j) == 0) x(j) = 0.0;
  if (!x.allFinite()) {
    res->str("status", "NUMERICAL_FAILURE").str("validation_message", "non-finite primal result").flag("has_primal", false);
    return res;
  }
  double objective = c.dot(x);
  VectorXd slack = mu ? VectorXd(bub - Aub * x) : VectorXd(0), con = me ? VectorXd(beq - Aeq * x) : VectorXd(0);
  // duals in original units, for the original objective sense (scipy's marginals for a minimisation)
  bool have_duals = !mixed && lp_ok && sol.dual_valid;
  VectorXd y_ub(mu), y_eq(me), z_lo = VectorXd::Zero(n), z_up = VectorXd::Zero(n);
  if (have_duals) {
    const HighsBasis& basis = highs.getBasis();
    for (int i = 0; i < mu; ++i) y_ub(i) = cost_scale * sol.row_dual[size_t(i)] / us(i);
    for (int i = 0; i < me; ++i) y_eq(i) = cost_scale * sol.row_dual[size_t(mu + i)] / es(i);
    for (int j = 0; j < n; ++j) {
      double d = cost_scale * sol.col_dual[size_t(j)];
      if (basis.valid && basis.col_status[size_t(j)] == HighsBasisStatus::kLower) z_lo(j) = d;
      else if (basis.valid && basis.col_status[size_t(j)] == HighsBasisStatus::kUpper) z_up(j) = d;
      else if (!basis.valid) { if (d > 0) z_lo(j) = d; else z_up(j) = d; }
    }
  }
  // certificate in original units (long double), pycli's lp_certificate
  double primal = 0, raw = 0;
  for (int i = 0; i < mu; ++i) {
    LD prod = 0, sc = std::fabs(bub(i));
    for (int j = 0; j < n; ++j) { prod += LD(Aub(i, j)) * x(j); sc += std::fabs(LD(Aub(i, j)) * x(j)); }
    LD v = std::max<LD>(prod - bub(i), 0);
    primal = std::max(primal, rel_error(v, sc));
    raw = std::max(raw, double(v));
  }
  for (int i = 0; i < me; ++i) {
    LD prod = 0, sc = std::fabs(beq(i));
    for (int j = 0; j < n; ++j) { prod += LD(Aeq(i, j)) * x(j); sc += std::fabs(LD(Aeq(i, j)) * x(j)); }
    LD v = std::fabs(prod - beq(i));
    primal = std::max(primal, rel_error(v, sc));
    raw = std::max(raw, double(v));
  }
  for (int j = 0; j < n; ++j) {
    if (std::isfinite(lower(j))) { LD v = std::max<LD>(LD(lower(j)) - x(j), 0); primal = std::max(primal, rel_error(v, std::fabs(lower(j)) + std::fabs(x(j)))); raw = std::max(raw, double(v)); }
    if (std::isfinite(upper(j))) { LD v = std::max<LD>(LD(x(j)) - upper(j), 0); primal = std::max(primal, rel_error(v, std::fabs(upper(j)) + std::fabs(x(j)))); raw = std::max(raw, double(v)); }
  }
  bool primal_feasible = primal <= 1e-7 && integrality_error <= 1e-6;
  res->vec("x", x).num("fun", objective).num("objective", objective).vec("slack", slack).vec("con", con);
  res->num("primal_relative_violation", primal).num("primal_violation", raw).num("integrality_error", integrality_error).flag("primal_feasible", primal_feasible);
  res->flag("has_primal", true);
  bool dual_feasible = false;
  if (have_duals) {
    // stationarity of the scaled-back problem: sign*c = A_ub' y_ub + A_eq' y_eq + z_lo + z_up
    double dual_error = 0;
    // sign conditions up to rounding: HiGHS reports degenerate duals like -1e-16 (relative to the largest cost)
    double sign_tol = 1e-9 * std::max(1.0, c.cwiseAbs().maxCoeff());
    for (int i = 0; i < mu; ++i) if (y_ub(i) > sign_tol) dual_error = INF;
    for (int j = 0; j < n; ++j) {
      if (z_lo(j) < -sign_tol || z_up(j) > sign_tol) dual_error = INF;
      if ((!std::isfinite(lower(j)) && std::fabs(z_lo(j)) > sign_tol) || (!std::isfinite(upper(j)) && std::fabs(z_up(j)) > sign_tol)) dual_error = INF;
    }
    LD dual_obj = 0, gap_scale = 0;
    for (int j = 0; j < n; ++j) gap_scale += std::fabs(LD(sign) * c(j) * x(j));
    for (int j = 0; j < n; ++j) {
      LD st = LD(sign) * c(j), ss = std::fabs(c(j));
      for (int i = 0; i < mu; ++i) { st -= LD(Aub(i, j)) * y_ub(i); ss += std::fabs(LD(Aub(i, j)) * y_ub(i)); }
      for (int i = 0; i < me; ++i) { st -= LD(Aeq(i, j)) * y_eq(i); ss += std::fabs(LD(Aeq(i, j)) * y_eq(i)); }
      st -= z_lo(j) + z_up(j);
      ss += std::fabs(z_lo(j)) + std::fabs(z_up(j));
      dual_error = std::max(dual_error, rel_error(st, ss));
    }
    for (int i = 0; i < mu; ++i) {
      dual_obj += LD(bub(i)) * y_ub(i);
      gap_scale += std::fabs(LD(bub(i)) * y_ub(i));
      LD sc = std::fabs(bub(i));
      for (int j = 0; j < n; ++j) sc += std::fabs(LD(Aub(i, j)) * x(j));
      dual_error = std::max(dual_error, rel_error(LD(y_ub(i)) * slack(i), std::fabs(y_ub(i)) * sc));
    }
    for (int i = 0; i < me; ++i) { dual_obj += LD(beq(i)) * y_eq(i); gap_scale += std::fabs(LD(beq(i)) * y_eq(i)); }
    for (int j = 0; j < n; ++j) {
      if (std::isfinite(lower(j))) {
        dual_obj += LD(lower(j)) * z_lo(j);
        gap_scale += std::fabs(LD(lower(j)) * z_lo(j));
        dual_error = std::max(dual_error, rel_error(LD(z_lo(j)) * (x(j) - lower(j)), std::fabs(z_lo(j)) * (std::fabs(lower(j)) + std::fabs(x(j)))));
      }
      if (std::isfinite(upper(j))) {
        dual_obj += LD(upper(j)) * z_up(j);
        gap_scale += std::fabs(LD(upper(j)) * z_up(j));
        dual_error = std::max(dual_error, rel_error(LD(z_up(j)) * (upper(j) - x(j)), std::fabs(z_up(j)) * (std::fabs(upper(j)) + std::fabs(x(j)))));
      }
    }
    double gap = rel_error(LD(sign) * objective - dual_obj, gap_scale);
    dual_feasible = dual_error <= 1e-7 && gap <= 1e-7;
    res->num("dual_relative_violation", dual_error).num("relative_duality_gap", gap).flag("dual_feasible", dual_feasible);
  }
  std::string kind = status == "OK" && (!mixed || (res->find("mip_gap") && res->get("mip_gap", Item::NUM).f == 0)) ? "optimal" : "incumbent";
  bool verified = have_duals && dual_feasible && primal <= 1e-7;
  if (!primal_feasible || (have_duals && !dual_feasible)) {
    res->str("status", "NUMERICAL_FAILURE").str("validation_message", "original-data primal/KKT certificate failed");
    kind = "candidate";
  }
  res->str("solution_kind", kind).flag("optimality_verified", verified);
  if (have_duals && res->find("status")->s != "NUMERICAL_FAILURE") {
    // pycli duals follow the original objective (maximize flips them); scipy marginals are those of the minimised problem
    auto scaled = [&](const VectorXd& v) { return VectorXd(sign * v); };
    res->vec("ineqlin_marginals", y_ub).vec("eqlin_marginals", y_eq).vec("lower_marginals", z_lo).vec("upper_marginals", z_up);
    auto d = std::make_shared<Res>();
    d->vec("ub", scaled(y_ub)).vec("eq", scaled(y_eq)).vec("lower", scaled(z_lo)).vec("upper", scaled(z_up));
    res->sub("duals", d);
  }
  return res;
}

}  // namespace lmnum::opt
