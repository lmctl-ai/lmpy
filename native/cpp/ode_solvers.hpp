// Initial value problem solvers: a C++ port of scipy.integrate._ivp (scipy 1.16.2, BSD-3-Clause, Copyright (c) 2001-2002
// Enthought, Inc. 2003-2024 SciPy Developers): the explicit Runge-Kutta pairs RK23, RK45 (Dormand-Prince 5(4)) and DOP853
// with their dense outputs, the implicit Radau IIA (order 5) and the variable-order BDF/NDF solver, with scipy's step-size
// control, initial step selection and finite-difference Jacobian (num_jac). The arithmetic follows scipy's order of
// operations so that trajectories agree with solve_ivp to rounding. LSODA is not a port of ODEPACK: it is an automatic
// stiffness switch from RK45 (Hairer's stiffness test) to BDF, documented in ode.cpp.
#pragma once
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lmpy::ode {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// f(t, y) -> dy/dt. May throw to stop the integration (budget, non-finite values).
using Rhs = std::function<void(double t, const VectorXd& y, VectorXd& out)>;

// Local interpolant over one accepted step [t_old, t] (scipy DenseOutput).
struct Dense {
  double t_old = 0, t = 0;
  virtual ~Dense() = default;
  virtual void eval(double at, VectorXd& out) const = 0;
  VectorXd operator()(double at) const { VectorXd r; eval(at, r); return r; }
};
using DensePtr = std::shared_ptr<const Dense>;

struct Settings {
  double rtol = 1e-3;
  VectorXd atol;          // one per state (a scalar is broadcast by the caller)
  double max_step = INFINITY;
  double first_step = 0;  // 0: select_initial_step
};

class Solver {
 public:
  Solver(Rhs f, double t0, const VectorXd& y0, double t_bound);
  virtual ~Solver() = default;
  // One accepted step (scipy OdeSolver.step). status becomes "finished" or "failed"; the return value is the message.
  std::string step();
  // Interpolant over the last step; a zero-length step gives a constant.
  DensePtr dense_output();
  virtual const char* name() const = 0;
  virtual bool alt_segment() const { return false; }  // OdeSolution breakpoint side (BDF and LSODA: true)
  virtual double switch_time() const { return NAN; }   // LSODA: time of the switch to BDF (NaN: none)
  virtual long switch_steps() const { return 0; }     // LSODA: accepted RK45 steps before the switch

  double t, t_bound;
  double t_old = NAN;
  bool has_old = false;
  VectorXd y;
  int n;
  double direction;
  std::string status = "running";
  long nfev = 0, njev = 0, nlu = 0;

 protected:
  VectorXd fun(double at, const VectorXd& state) { ++nfev; VectorXd r(n); rhs_(at, state, r); return r; }
  void fun_uncounted(double at, const VectorXd& state, VectorXd& out) { rhs_(at, state, out); }
  virtual bool step_impl(std::string& message) = 0;
  virtual DensePtr dense_impl() = 0;
  Rhs rhs_;
  static constexpr const char* TOO_SMALL_STEP = "Required step size is less than spacing between numbers.";
};

// scipy.integrate._ivp.common.norm: RMS norm.
double rms(const VectorXd& x);
// Clamp rtol below 100 eps as scipy's validate_tol does.
double valid_rtol(double rtol);

// method: RK23, RK45, DOP853, Radau, BDF, LSODA.
std::unique_ptr<Solver> make_solver(const std::string& method, Rhs f, double t0, const VectorXd& y0, double t_bound, const Settings& s);

// Piecewise solution over all steps (scipy OdeSolution): segments in integration order.
struct Solution {
  std::vector<double> ts;  // n_segments + 1 breakpoints
  std::vector<DensePtr> pieces;
  bool alt_segment = false;
  void eval(double at, VectorXd& out) const;
};

// scipy.optimize.brentq (scipy/optimize/Zeros/brentq.c, BSD-3-Clause). Returns the root; `converged` false after maxiter.
double brentq(const std::function<double(double)>& f, double xa, double xb, double xtol, double rtol, int maxiter, bool& converged);

}  // namespace lmpy::ode
