// Internal interface of the optimisation family (optimize.cpp registers the DSL members; optimize_*.cpp hold the
// numerical kernels). Nothing here is visible to the DSL.
#pragma once
#include "expr.hpp"
#include "host.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace lmpy::opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;
constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double INF = std::numeric_limits<double>::infinity();

// Raised by an evaluation (budget, wall clock, non-finite policy) to stop a solver; the caller reports the best point.
struct Stop {
  std::string status, message;
};

// Evaluation counter shared by every expression call of one operation (pycli's Guard): max_evals and an optional
// wall-clock limit, both checked before the evaluation they admit.
struct Budget {
  int64_t max_evals = 500000;
  int64_t count = 0;
  int64_t nonfinite = 0;
  double time_limit = INF;  // seconds; INF = none
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  void tick();
};

using ScalarFn = std::function<double(const VectorXd&)>;
using VectorFn = std::function<VectorXd(const VectorXd&)>;

// ---- minimize ---------------------------------------------------------------------------------------------------
struct MinProblem {
  ScalarFn f;
  VectorFn grad;                  // empty: finite differences
  VectorXd lb, ub;                // size 0: no bounds; otherwise +-inf for open ends
  std::vector<ScalarFn> eq, ineq; // SLSQP only; ineq means g(x) >= 0
  double tol = std::numeric_limits<double>::quiet_NaN();  // NaN: the method's default
  int maxiter = 1000;
};
struct MinResult {
  VectorXd x;
  double fun = std::numeric_limits<double>::quiet_NaN();
  int64_t nit = -1, nfev = 0, njev = -1;
  bool success = false;
  std::string message;
};
MinResult nelder_mead(const MinProblem& p, const VectorXd& x0);
MinResult powell(const MinProblem& p, const VectorXd& x0);
MinResult bfgs(const MinProblem& p, const VectorXd& x0);
MinResult lbfgsb(const MinProblem& p, const VectorXd& x0);
MinResult slsqp(const MinProblem& p, const VectorXd& x0);

// scipy.optimize._numdiff 2-point forward differences of a scalar function at x (f0 = f(x)). abs_step > 0 uses that
// absolute step (minimize's eps); abs_step <= 0 uses the relative step sqrt(eps) * max(1, |x|). Bounds (may be empty)
// flip or shorten the step as scipy's _adjust_scheme_to_bounds does.
VectorXd fd_gradient(const ScalarFn& f, const VectorXd& x, double f0, double abs_step, const VectorXd& lb, const VectorXd& ub);
MatrixXd fd_jacobian(const VectorFn& f, const VectorXd& x, const VectorXd& f0, double abs_step, const VectorXd& lb, const VectorXd& ub);

// ---- least squares ------------------------------------------------------------------------------------------------
struct LsqProblem {
  VectorFn residual;              // m residuals
  VectorXd lb, ub;                // size 0: none
  std::string method = "trf";     // trf, dogbox, lm
  std::string loss = "linear";    // linear, soft_l1, huber, cauchy, arctan
  double f_scale = 1.0;
  double ftol = 1e-10, xtol = 1e-10, gtol = 1e-10;
  int64_t max_nfev = 1000;
  bool x_scale_jac = true;        // pycli runs with x_scale="jac"
};
struct LsqResult {
  VectorXd x, fun;  // fun: residuals at x (unscaled by the loss)
  MatrixXd jac;     // the solver's last Jacobian (loss-scaled for robust losses)
  double cost = 0, optimality = 0;
  int64_t nfev = 0, njev = -1;
  int status = 0;   // scipy: -1 improper input, 0 max_nfev, 1 gtol, 2 ftol, 3 xtol, 4 ftol and xtol
  std::string message;
};
LsqResult least_squares_solve(const LsqProblem& p, const VectorXd& x0);

// ---- roots -------------------------------------------------------------------------------------------------------
struct ScalarRoot {
  double root = std::numeric_limits<double>::quiet_NaN();
  int iterations = 0, function_calls = 0;
  bool converged = false;
  std::string flag;  // "converged", "convergence error", "sign error"
};
// Bracketed scalar roots (scipy.optimize brentq/bisect/ridder/toms748 semantics, including their counters).
ScalarRoot bracket_root(const std::string& method, const std::function<double(double)>& f, double a, double b, double xtol, double rtol, int maxiter);

struct SystemRoot {
  VectorXd x;
  int64_t nfev = 0;
  bool success = false;
  std::string message;
};
// Square systems: MINPACK hybrd (Powell hybrid) or lmdif (Levenberg-Marquardt), forward-difference Jacobians.
SystemRoot root_hybr(const VectorFn& f, const VectorXd& x0, double xtol, int64_t maxfev);
SystemRoot root_lm(const VectorFn& f, const VectorXd& x0, double xtol, int64_t maxfev);

// ---- quadrature --------------------------------------------------------------------------------------------------
struct QuadResult {
  double value = 0, abserr = 0;
  int64_t neval = 0, last = 0;
  int ier = 0;  // QUADPACK ier: 0 ok, 1 limit, 2 roundoff, 3 bad integrand, 4 no convergence, 5 divergent, 6 invalid input
};
// QUADPACK QAGS (finite), QAGI (one or two infinite limits) and QAGP (finite with break points), as scipy.integrate.quad.
QuadResult quad_integrate(const std::function<double(double)>& f, double a, double b, double epsabs, double epsrel, int limit, const std::vector<double>& points);
std::string quad_message(int ier, int limit);

// ---- linear programming (HiGHS) -----------------------------------------------------------------------------------
Res* linprog_run(const std::vector<double>& c, const Opts& o);

}  // namespace lmpy::opt
