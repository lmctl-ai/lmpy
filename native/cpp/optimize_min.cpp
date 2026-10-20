// Unconstrained and constrained minimisers behind `minimize`.
//
// Nelder-Mead, Powell (with Brent / bounded scalar line searches), BFGS (with the MINPACK-2 dcsrch line search and the
// Wolfe zoom fallback) and the finite-difference rules are C++ ports of scipy.optimize 1.16 (_optimize.py,
// _linesearch.py, _dcsrch.py, _numdiff.py; BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. and 2003-2024 SciPy
// Developers), so iteration and evaluation counts follow scipy's. L-BFGS-B is LBFGSpp (MIT). SLSQP follows the
// structure of Kraft's algorithm (D. Kraft, "A software package for sequential quadratic programming", DFVLR-FB 88-28,
// 1988): damped BFGS on the Lagrangian Hessian, an L1 merit function with Kraft's multiplier-based penalty update and
// line search, relaxation of inconsistent linearisations; the QP subproblem is solved here by the Goldfarb-Idnani dual
// active-set method (Math. Programming 27, 1983) instead of Kraft's LSQ/NNLS chain, so results agree with scipy's SLSQP
// at the solution but iteration counts can differ.
#include "optimize_impl.hpp"
#include <LBFGSB.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace lmnum::opt {

void Budget::tick() {
  if (count >= max_evals) throw Stop{"EVAL_LIMIT", "expression evaluation cap of " + std::to_string(max_evals) + " reached"};
  if (time_limit < INF && (count & 15) == 0) {
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (el > time_limit) throw Stop{"TIME_LIMIT", "wall-clock limit exhausted after " + std::to_string(count) + " evaluations"};
  }
  ++count;
}

namespace {

const char* MSG_SUCCESS = "Optimization terminated successfully.";
const char* MSG_MAXFEV = "Maximum number of function evaluations has been exceeded.";
const char* MSG_MAXITER = "Maximum number of iterations has been exceeded.";
const char* MSG_PRLOSS = "Desired error not necessarily achieved due to precision loss.";
const char* MSG_NAN = "NaN result encountered.";
const char* MSG_OOB = "The result is outside of the provided bounds.";

bool has_bounds(const VectorXd& lb, const VectorXd& ub) {
  for (Eigen::Index i = 0; i < lb.size(); ++i)
    if (lb(i) != -INF || ub(i) != INF) return true;
  return false;
}

VectorXd clip(const VectorXd& x, const VectorXd& lb, const VectorXd& ub) {
  if (lb.size() == 0) return x;
  return x.cwiseMax(lb).cwiseMin(ub);
}

double sign_of(double v) { return v > 0 ? 1.0 : v < 0 ? -1.0 : (v == 0 ? 0.0 : v); }

// scipy _numdiff: step for one component.
double fd_step(double x, double abs_step) {
  double sgn = x >= 0 ? 1.0 : -1.0;
  double rel = std::sqrt(EPS) * sgn * std::max(1.0, std::fabs(x));
  if (abs_step > 0) {
    double h = abs_step;
    return ((x + h) - x) == 0 ? rel : h;
  }
  return rel;
}

void adjust_one_sided(const VectorXd& x, VectorXd& h, const VectorXd& lb, const VectorXd& ub) {
  if (lb.size() == 0 || !has_bounds(lb, ub)) return;
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    double lower = x(i) - lb(i), upper = ub(i) - x(i), xi = x(i) + h(i);
    bool violated = xi < lb(i) || xi > ub(i);
    bool fitting = std::fabs(h(i)) <= std::max(lower, upper);
    if (violated && fitting) h(i) = -h(i);
    else if (!fitting) h(i) = upper >= lower ? upper : -lower;
  }
}

}  // namespace

VectorXd fd_gradient(const ScalarFn& f, const VectorXd& x, double f0, double abs_step, const VectorXd& lb, const VectorXd& ub) {
  Eigen::Index n = x.size();
  VectorXd h(n);
  for (Eigen::Index i = 0; i < n; ++i) h(i) = fd_step(x(i), abs_step);
  adjust_one_sided(x, h, lb, ub);
  VectorXd g(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    VectorXd x1 = x;
    x1(i) = x(i) + h(i);
    double dx = x1(i) - x(i);
    g(i) = (f(x1) - f0) / dx;
  }
  return g;
}

MatrixXd fd_jacobian(const VectorFn& f, const VectorXd& x, const VectorXd& f0, double abs_step, const VectorXd& lb, const VectorXd& ub) {
  Eigen::Index n = x.size();
  VectorXd h(n);
  for (Eigen::Index i = 0; i < n; ++i) h(i) = fd_step(x(i), abs_step);
  adjust_one_sided(x, h, lb, ub);
  MatrixXd J(f0.size(), n);
  for (Eigen::Index i = 0; i < n; ++i) {
    VectorXd x1 = x;
    x1(i) = x(i) + h(i);
    double dx = x1(i) - x(i);
    VectorXd fi = f(x1);
    if (fi.size() != f0.size()) fail("the number of residuals changed between evaluations");
    J.col(i) = (fi - f0) / dx;
  }
  return J;
}

namespace {

// scipy's ScalarFunction: caches f and grad at the last x, counts objective evaluations including finite differences.
struct SF {
  const MinProblem& p;
  double eps;
  VectorXd lb, ub;
  VectorXd x;
  double f = 0;
  VectorXd g;
  bool fu = false, gu = false;
  int64_t nfev = 0, ngev = 0;
  SF(const MinProblem& prob, const VectorXd& x0, double step, VectorXd l, VectorXd u) : p(prob), eps(step), lb(std::move(l)), ub(std::move(u)), x(x0) {
    update_fun();
    update_grad();
  }
  void set_x(const VectorXd& y) {
    bool same = y.size() == x.size();
    for (Eigen::Index i = 0; same && i < y.size(); ++i) same = y(i) == x(i);
    if (!same) { x = y; fu = gu = false; }
  }
  void update_fun() {
    if (!fu) { f = p.f(x); ++nfev; fu = true; }
  }
  void update_grad() {
    if (gu) return;
    if (!p.grad) {
      update_fun();
      int64_t* counter = &nfev;
      const ScalarFn& fn = p.f;
      g = fd_gradient([&](const VectorXd& v) { ++*counter; return fn(v); }, x, f, eps, lb, ub);
    } else {
      g = p.grad(x);
    }
    ++ngev;
    gu = true;
  }
  double fun(const VectorXd& y) { set_x(y); update_fun(); return f; }
  VectorXd grad(const VectorXd& y) { set_x(y); update_grad(); return g; }
};

// ------------------------------------------------------------------------------------------- Nelder-Mead
template <class F> void sort_simplex(MatrixXd& sim, VectorXd& fsim, F) {
  std::vector<int> idx(size_t(fsim.size()));
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return fsim(a) < fsim(b) || (std::isnan(fsim(b)) && !std::isnan(fsim(a))); });
  MatrixXd s2 = sim;
  VectorXd f2 = fsim;
  for (size_t i = 0; i < idx.size(); ++i) { s2.row(Eigen::Index(i)) = sim.row(idx[i]); f2(Eigen::Index(i)) = fsim(idx[i]); }
  sim = s2;
  fsim = f2;
}

}  // namespace

MinResult nelder_mead(const MinProblem& p, const VectorXd& x0in) {
  const double rho = 1, chi = 2, psi = 0.5, sigma = 0.5, nonzdelt = 0.05, zdelt = 0.00025;
  double xatol = std::isnan(p.tol) ? 1e-4 : p.tol, fatol = xatol;
  bool bounded = p.lb.size() > 0;
  VectorXd x0 = bounded ? clip(x0in, p.lb, p.ub) : x0in;
  Eigen::Index N = x0.size();
  MatrixXd sim(N + 1, N);
  sim.row(0) = x0.transpose();
  for (Eigen::Index k = 0; k < N; ++k) {
    VectorXd y = x0;
    y(k) = y(k) != 0 ? (1 + nonzdelt) * y(k) : zdelt;
    sim.row(k + 1) = y.transpose();
  }
  int64_t maxiter = p.maxiter;
  if (bounded) {
    for (Eigen::Index i = 0; i <= N; ++i)
      for (Eigen::Index j = 0; j < N; ++j) {
        if (sim(i, j) > p.ub(j)) sim(i, j) = 2 * p.ub(j) - sim(i, j);
        sim(i, j) = std::min(std::max(sim(i, j), p.lb(j)), p.ub(j));
      }
  }
  int64_t fcalls = 0;
  auto func = [&](const VectorXd& x) { ++fcalls; return p.f(x); };
  auto cl = [&](VectorXd v) { return bounded ? clip(v, p.lb, p.ub) : v; };
  VectorXd fsim = VectorXd::Constant(N + 1, INF);
  for (Eigen::Index k = 0; k <= N; ++k) fsim(k) = func(sim.row(k).transpose());
  sort_simplex(sim, fsim, 0);
  int64_t iterations = 1;
  while (iterations < maxiter) {
    double xd = 0, fd = 0;
    for (Eigen::Index i = 1; i <= N; ++i) {
      xd = std::max(xd, (sim.row(i) - sim.row(0)).cwiseAbs().maxCoeff());
      fd = std::max(fd, std::fabs(fsim(0) - fsim(i)));
    }
    if (xd <= xatol && fd <= fatol) break;
    VectorXd xbar = sim.topRows(N).colwise().sum().transpose() / double(N);
    VectorXd last = sim.row(N).transpose();
    VectorXd xr = cl((1 + rho) * xbar - rho * last);
    double fxr = func(xr);
    bool doshrink = false;
    if (fxr < fsim(0)) {
      VectorXd xe = cl((1 + rho * chi) * xbar - rho * chi * last);
      double fxe = func(xe);
      if (fxe < fxr) { sim.row(N) = xe.transpose(); fsim(N) = fxe; }
      else { sim.row(N) = xr.transpose(); fsim(N) = fxr; }
    } else if (fxr < fsim(N - 1)) {
      sim.row(N) = xr.transpose();
      fsim(N) = fxr;
    } else {
      if (fxr < fsim(N)) {
        VectorXd xc = cl((1 + psi * rho) * xbar - psi * rho * last);
        double fxc = func(xc);
        if (fxc <= fxr) { sim.row(N) = xc.transpose(); fsim(N) = fxc; }
        else doshrink = true;
      } else {
        VectorXd xcc = cl((1 - psi) * xbar + psi * last);
        double fxcc = func(xcc);
        if (fxcc < fsim(N)) { sim.row(N) = xcc.transpose(); fsim(N) = fxcc; }
        else doshrink = true;
      }
      if (doshrink)
        for (Eigen::Index j = 1; j <= N; ++j) {
          VectorXd v = cl((sim.row(0) + sigma * (sim.row(j) - sim.row(0))).transpose());
          sim.row(j) = v.transpose();
          fsim(j) = func(v);
        }
    }
    ++iterations;
    sort_simplex(sim, fsim, 0);
  }
  MinResult r;
  r.x = sim.row(0).transpose();
  r.fun = fsim.minCoeff();
  r.nit = iterations;
  r.nfev = fcalls;
  r.success = iterations < maxiter;
  r.message = r.success ? MSG_SUCCESS : MSG_MAXITER;
  return r;
}

// ------------------------------------------------------------------------------------------------- Powell
namespace {

struct BracketFail { double x, f; int64_t calls; };

// scipy.optimize.bracket
void bracket(const std::function<double(double)>& func, double& xa, double& xb, double& xc, double& fa, double& fb, double& fc, int64_t& calls) {
  const double gold = 1.618034, verysmall = 1e-21, grow_limit = 110.0;
  const int maxiter = 1000;
  xa = 0.0; xb = 1.0;
  fa = func(xa);
  fb = func(xb);
  if (fa < fb) { std::swap(xa, xb); std::swap(fa, fb); }
  xc = xb + gold * (xb - xa);
  fc = func(xc);
  calls = 3;
  int iter = 0;
  while (fc < fb) {
    double tmp1 = (xb - xa) * (fb - fc), tmp2 = (xb - xc) * (fb - fa), val = tmp2 - tmp1;
    double denom = std::fabs(val) < verysmall ? 2.0 * verysmall : 2.0 * val;
    double w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
    double wlim = xb + grow_limit * (xc - xb), fw;
    if (iter > maxiter) fail("minimize(Powell): no valid line-search bracket was found before the iteration limit");
    ++iter;
    if ((w - xc) * (xb - w) > 0.0) {
      fw = func(w); ++calls;
      if (fw < fc) { xa = xb; xb = w; fa = fb; fb = fw; break; }
      else if (fw > fb) { xc = w; fc = fw; break; }
      w = xc + gold * (xc - xb);
      fw = func(w); ++calls;
    } else if ((w - wlim) * (wlim - xc) >= 0.0) {
      w = wlim;
      fw = func(w); ++calls;
    } else if ((w - wlim) * (xc - w) > 0.0) {
      fw = func(w); ++calls;
      if (fw < fc) {
        xb = xc; xc = w; w = xc + gold * (xc - xb);
        fb = fc; fc = fw;
        fw = func(w); ++calls;
      }
    } else {
      w = xc + gold * (xc - xb);
      fw = func(w); ++calls;
    }
    xa = xb; xb = xc; xc = w;
    fa = fb; fb = fc; fc = fw;
  }
  bool c1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
  bool c2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
  bool c3 = std::isfinite(xa) && std::isfinite(xb) && std::isfinite(xc);
  if (!(c1 && c2 && c3)) {
    double xs[3] = {xa, xb, xc}, fs[3] = {fa, fb, fc};
    bool anynan = false;
    for (int i = 0; i < 3; ++i) anynan = anynan || std::isnan(xs[i]) || std::isnan(fs[i]);
    if (anynan) throw BracketFail{NAN, NAN, calls};
    int im = 0;
    for (int i = 1; i < 3; ++i) if (fs[i] < fs[im]) im = i;
    throw BracketFail{xs[im], fs[im], calls};
  }
}

// scipy Brent.optimize with the automatic bracket; returns (x, f).
std::pair<double, double> brent_min(const std::function<double(double)>& func, double tol) {
  double xa, xb, xc, fa, fb, fc;
  int64_t calls;
  try {
    bracket(func, xa, xb, xc, fa, fb, fc, calls);
  } catch (const BracketFail& e) {
    return {e.x, e.f};
  }
  const double mintol = 1.0e-11, cg = 0.3819660;
  const int maxiter = 500;
  double x = xb, w = xb, v = xb, fw = fb, fv = fb, fx = fb;
  double a = xa < xc ? xa : xc, b = xa < xc ? xc : xa;
  double deltax = 0.0, rat = 0.0;
  for (int iter = 0; iter < maxiter; ++iter) {
    double tol1 = tol * std::fabs(x) + mintol, tol2 = 2.0 * tol1, xmid = 0.5 * (a + b);
    if (std::fabs(x - xmid) < (tol2 - 0.5 * (b - a))) break;
    if (std::fabs(deltax) <= tol1) {
      deltax = x >= xmid ? a - x : b - x;
      rat = cg * deltax;
    } else {
      double tmp1 = (x - w) * (fx - fv), tmp2 = (x - v) * (fx - fw);
      double pp = (x - v) * tmp2 - (x - w) * tmp1;
      tmp2 = 2.0 * (tmp2 - tmp1);
      if (tmp2 > 0.0) pp = -pp;
      tmp2 = std::fabs(tmp2);
      double dx_temp = deltax;
      deltax = rat;
      if (pp > tmp2 * (a - x) && pp < tmp2 * (b - x) && std::fabs(pp) < std::fabs(0.5 * tmp2 * dx_temp)) {
        rat = pp * 1.0 / tmp2;
        double u = x + rat;
        if ((u - a) < tol2 || (b - u) < tol2) rat = xmid - x >= 0 ? tol1 : -tol1;
      } else {
        deltax = x >= xmid ? a - x : b - x;
        rat = cg * deltax;
      }
    }
    double u = std::fabs(rat) < tol1 ? (rat >= 0 ? x + tol1 : x - tol1) : x + rat;
    double fu = func(u);
    if (fu > fx) {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) { v = w; w = u; fv = fw; fw = fu; }
      else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
    } else {
      if (u >= x) a = x; else b = x;
      v = w; w = x; x = u;
      fv = fw; fw = fx; fx = fu;
    }
  }
  return {x, fx};
}

// scipy _minimize_scalar_bounded (fminbound); returns (x, f).
std::pair<double, double> bounded_min(const std::function<double(double)>& func, double x1, double x2, double xatol) {
  const int maxfun = 500;
  const double sqrt_eps = std::sqrt(2.2e-16), golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
  double a = x1, b = x2, fulc = a + golden_mean * (b - a), nfc = fulc, xf = fulc;
  double rat = 0.0, e = 0.0, x = xf, fx = func(x);
  int num = 1;
  double ffulc = fx, fnfc = fx, xm = 0.5 * (a + b);
  double tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0, tol2 = 2.0 * tol1;
  while (std::fabs(xf - xm) > (tol2 - 0.5 * (b - a))) {
    bool golden = true;
    if (std::fabs(e) > tol1) {
      golden = false;
      double r = (xf - nfc) * (fx - ffulc), q = (xf - fulc) * (fx - fnfc);
      double pp = (xf - fulc) * q - (xf - nfc) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) pp = -pp;
      q = std::fabs(q);
      r = e;
      e = rat;
      if (std::fabs(pp) < std::fabs(0.5 * q * r) && pp > q * (a - xf) && pp < q * (b - xf)) {
        rat = (pp + 0.0) / q;
        x = xf + rat;
        if ((x - a) < tol2 || (b - x) < tol2) {
          double si = sign_of(xm - xf) + ((xm - xf) == 0 ? 1 : 0);
          rat = tol1 * si;
        }
      } else {
        golden = true;
      }
    }
    if (golden) {
      e = xf >= xm ? a - xf : b - xf;
      rat = golden_mean * e;
    }
    double si = sign_of(rat) + (rat == 0 ? 1 : 0);
    x = xf + si * std::max(std::fabs(rat), tol1);
    double fu = func(x);
    ++num;
    if (fu <= fx) {
      if (x >= xf) a = xf; else b = xf;
      fulc = nfc; ffulc = fnfc;
      nfc = xf; fnfc = fx;
      xf = x; fx = fu;
    } else {
      if (x < xf) a = x; else b = x;
      if (fu <= fnfc || nfc == xf) { fulc = nfc; ffulc = fnfc; nfc = x; fnfc = fu; }
      else if (fu <= ffulc || fulc == xf || fulc == nfc) { fulc = x; ffulc = fu; }
    }
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
    tol2 = 2.0 * tol1;
    if (num >= maxfun) break;
  }
  return {xf, fx};
}

std::pair<double, double> line_for_search(const VectorXd& x0, const VectorXd& alpha, const VectorXd& lb, const VectorXd& ub) {
  double lmin = -INF, lmax = INF;
  bool any = false;
  for (Eigen::Index i = 0; i < x0.size(); ++i) {
    if (alpha(i) == 0) continue;
    double low = (lb(i) - x0(i)) / alpha(i), high = (ub(i) - x0(i)) / alpha(i);
    double mn = alpha(i) > 0 ? low : high, mx = alpha(i) > 0 ? high : low;
    if (!any) { lmin = mn; lmax = mx; any = true; }
    else { lmin = std::max(lmin, mn); lmax = std::min(lmax, mx); }
  }
  return lmax >= lmin ? std::make_pair(lmin, lmax) : std::make_pair(0.0, 0.0);
}

// scipy _linesearch_powell: returns (fval, new x, new direction)
struct PowellLS { double f; VectorXd x, dir; };
PowellLS linesearch_powell(const std::function<double(const VectorXd&)>& func, const VectorXd& p, const VectorXd& xi, double tol,
                           const VectorXd* lb, const VectorXd* ub, double fval) {
  auto my = [&](double alpha) { return func(p + alpha * xi); };
  if (!(xi.array() != 0).any()) return {fval, p, xi};
  if (!lb) {
    auto [a, f] = brent_min(my, tol);
    VectorXd d = a * xi;
    return {f, p + d, d};
  }
  auto bound = line_for_search(p, xi, *lb, *ub);
  if (std::isinf(bound.first) && bound.first < 0 && std::isinf(bound.second) && bound.second > 0) return linesearch_powell(func, p, xi, tol, nullptr, nullptr, fval);
  if (!std::isinf(bound.first) && !std::isinf(bound.second)) {
    auto [a, f] = bounded_min(my, bound.first, bound.second, tol / 100);
    VectorXd d = a * xi;
    return {f, p + d, d};
  }
  auto [t, f] = bounded_min([&](double s) { return my(std::tan(s)); }, std::atan(bound.first), std::atan(bound.second), tol / 100);
  VectorXd d = std::tan(t) * xi;
  return {f, p + d, d};
}

}  // namespace

MinResult powell(const MinProblem& p, const VectorXd& x0) {
  double xtol = std::isnan(p.tol) ? 1e-4 : p.tol, ftol = xtol;
  Eigen::Index N = x0.size();
  int64_t maxiter = p.maxiter, fcalls = 0;
  auto func = [&](const VectorXd& x) { ++fcalls; double v = p.f(x); return v; };
  MatrixXd direc = MatrixXd::Identity(N, N);
  bool bounded = p.lb.size() > 0;
  const VectorXd* lb = bounded ? &p.lb : nullptr;
  const VectorXd* ub = bounded ? &p.ub : nullptr;
  VectorXd x = x0;
  double fval = func(x);
  VectorXd x1 = x;
  int64_t iter = 0;
  for (;;) {
    double fx = fval, delta = 0.0;
    Eigen::Index bigind = 0;
    for (Eigen::Index i = 0; i < N; ++i) {
      VectorXd direc1 = direc.row(i).transpose();
      double fx2 = fval;
      auto r = linesearch_powell(func, x, direc1, xtol * 100, lb, ub, fval);
      fval = r.f; x = r.x;
      if ((fx2 - fval) > delta) { delta = fx2 - fval; bigind = i; }
    }
    ++iter;
    double bnd = ftol * (std::fabs(fx) + std::fabs(fval)) + 1e-20;
    if (2.0 * (fx - fval) <= bnd) break;
    if (iter >= maxiter) break;
    if (std::isnan(fx) && std::isnan(fval)) break;
    VectorXd direc1 = x - x1;
    x1 = x;
    double lmax = 1;
    if (bounded) lmax = line_for_search(x, direc1, p.lb, p.ub).second;
    VectorXd x2 = x + std::min(lmax, 1.0) * direc1;
    double fx2 = func(x2);
    if (fx > fx2) {
      double t = 2.0 * (fx + fx2 - 2.0 * fval);
      double temp = fx - fval - delta;
      t *= temp * temp;
      temp = fx - fx2;
      t -= delta * temp * temp;
      if (t < 0.0) {
        auto r = linesearch_powell(func, x, direc1, xtol * 100, lb, ub, fval);
        fval = r.f; x = r.x;
        if ((r.dir.array() != 0).any()) {
          direc.row(bigind) = direc.row(N - 1);
          direc.row(N - 1) = r.dir.transpose();
        }
      }
    }
  }
  MinResult r;
  r.x = x; r.fun = fval; r.nit = iter; r.nfev = fcalls;
  if (bounded && ((p.lb.array() > x.array()).any() || (x.array() > p.ub.array()).any())) r.message = MSG_OOB;
  else if (iter >= maxiter) r.message = MSG_MAXITER;
  else if (std::isnan(fval) || x.hasNaN()) r.message = MSG_NAN;
  else { r.message = MSG_SUCCESS; r.success = true; }
  return r;
}

// ---------------------------------------------------------------------------------------------------- BFGS
namespace {

struct DCSRCH {
  std::function<double(double)> phi, derphi;
  double ftol, gtol, xtol, stpmin, stpmax;
  int stage = 0;
  bool brackt = false;
  double ginit = 0, gtest = 0, gx = 0, gy = 0, finit = 0, fx = 0, fy = 0, stx = 0, sty = 0, stmin = 0, stmax = 0, width = 0, width1 = 0;

  static void dcstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy, double& stp, double fp, double dp, bool& brackt, double stpmin, double stpmax) {
    double sgnd = sign_of(dp) * sign_of(dx), stpf;
    if (fp > fx) {
      double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
      double s = std::max({std::fabs(theta), std::fabs(dx), std::fabs(dp)});
      double gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp < stx) gamma = -gamma;
      double pp = (gamma - dx) + theta, q = ((gamma - dx) + gamma) + dp, r = pp / q;
      double stpc = stx + r * (stp - stx);
      double stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
      stpf = std::fabs(stpc - stx) <= std::fabs(stpq - stx) ? stpc : stpc + (stpq - stpc) / 2.0;
      brackt = true;
    } else if (sgnd < 0.0) {
      double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      double s = std::max({std::fabs(theta), std::fabs(dx), std::fabs(dp)});
      double gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp > stx) gamma = -gamma;
      double pp = (gamma - dp) + theta, q = ((gamma - dp) + gamma) + dx, r = pp / q;
      double stpc = stp + r * (stx - stp), stpq = stp + (dp / (dp - dx)) * (stx - stp);
      stpf = std::fabs(stpc - stp) > std::fabs(stpq - stp) ? stpc : stpq;
      brackt = true;
    } else if (std::fabs(dp) < std::fabs(dx)) {
      double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      double s = std::max({std::fabs(theta), std::fabs(dx), std::fabs(dp)});
      double gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
      if (stp > stx) gamma = -gamma;
      double pp = (gamma - dp) + theta, q = (gamma + (dx - dp)) + gamma, r = pp / q, stpc;
      if (r < 0 && gamma != 0) stpc = stp + r * (stx - stp);
      else if (stp > stx) stpc = stpmax;
      else stpc = stpmin;
      double stpq = stp + (dp / (dp - dx)) * (stx - stp);
      if (brackt) {
        stpf = std::fabs(stpc - stp) < std::fabs(stpq - stp) ? stpc : stpq;
        stpf = stp > stx ? std::min(stp + 0.66 * (sty - stp), stpf) : std::max(stp + 0.66 * (sty - stp), stpf);
      } else {
        stpf = std::fabs(stpc - stp) > std::fabs(stpq - stp) ? stpc : stpq;
        stpf = std::min(std::max(stpf, stpmin), stpmax);
      }
    } else {
      if (brackt) {
        double theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
        double s = std::max({std::fabs(theta), std::fabs(dy), std::fabs(dp)});
        double gamma = s * std::sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
        if (stp > sty) gamma = -gamma;
        double pp = (gamma - dp) + theta, q = ((gamma - dp) + gamma) + dy, r = pp / q;
        stpf = stp + r * (sty - stp);
      } else if (stp > stx) stpf = stpmax;
      else stpf = stpmin;
    }
    if (fp > fx) { sty = stp; fy = fp; dy = dp; }
    else {
      if (sgnd < 0) { sty = stx; fy = fx; dy = dx; }
      stx = stp; fx = fp; dx = dp;
    }
    stp = stpf;
  }

  // returns task: 0 FG, 1 CONVERGENCE, 2 WARNING, 3 ERROR
  int iterate(double& stp, double f, double g, bool start) {
    const double p5 = 0.5, p66 = 0.66, xtrapl = 1.1, xtrapu = 4.0;
    if (start) {
      if (stp < stpmin || stp > stpmax || g >= 0 || ftol < 0 || gtol < 0 || xtol < 0 || stpmin < 0 || stpmax < stpmin) return 3;
      brackt = false; stage = 1; finit = f; ginit = g; gtest = ftol * ginit;
      width = stpmax - stpmin; width1 = width / p5;
      stx = 0.0; fx = finit; gx = ginit; sty = 0.0; fy = finit; gy = ginit;
      stmin = 0; stmax = stp + xtrapu * stp;
      return 0;
    }
    double ftest = finit + stp * gtest;
    if (stage == 1 && f <= ftest && g >= 0) stage = 2;
    int task = -1;
    if (brackt && (stp <= stmin || stp >= stmax)) task = 2;
    if (brackt && stmax - stmin <= xtol * stmax) task = 2;
    if (stp == stpmax && f <= ftest && g <= gtest) task = 2;
    if (stp == stpmin && (f > ftest || g >= gtest)) task = 2;
    if (f <= ftest && std::fabs(g) <= gtol * -ginit) task = 1;
    if (task >= 0) return task;
    if (stage == 1 && f <= fx && f > ftest) {
      double fm = f - stp * gtest, fxm = fx - stx * gtest, fym = fy - sty * gtest;
      double gm = g - gtest, gxm = gx - gtest, gym = gy - gtest;
      dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);
      fx = fxm + stx * gtest; fy = fym + sty * gtest; gx = gxm + gtest; gy = gym + gtest;
    } else {
      dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
    }
    if (brackt) {
      if (std::fabs(sty - stx) >= p66 * width1) stp = stx + p5 * (sty - stx);
      width1 = width;
      width = std::fabs(sty - stx);
    }
    if (brackt) { stmin = std::min(stx, sty); stmax = std::max(stx, sty); }
    else { stmin = stp + xtrapl * (stp - stx); stmax = stp + xtrapu * (stp - stx); }
    stp = std::min(std::max(stp, stpmin), stpmax);
    if ((brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax)) stp = stx;
    return 0;
  }

  // returns false on failure; stp, phi1 set on success
  bool run(double alpha1, double phi0, double derphi0, double& stp, double& phi1) {
    phi1 = phi0;
    double derphi1 = derphi0;
    stp = alpha1;
    bool start = true;
    for (int i = 0; i < 100; ++i) {
      int task = iterate(stp, phi1, derphi1, start);
      start = false;
      if (!std::isfinite(stp)) return false;
      if (task == 0) {
        phi1 = phi(stp);
        derphi1 = derphi(stp);
      } else {
        return task == 1;
      }
    }
    return false;
  }
};

double cubicmin(double a, double fa, double fpa, double b, double fb, double c, double fc, bool& ok) {
  double C = fpa, db = b - a, dc = c - a;
  double denom = (db * dc) * (db * dc) * (db - dc);
  double d00 = dc * dc, d01 = -db * db, d10 = -dc * dc * dc, d11 = db * db * db;
  double v0 = fb - fa - C * db, v1 = fc - fa - C * dc;
  double A = d00 * v0 + d01 * v1, B = d10 * v0 + d11 * v1;
  if (denom == 0) { ok = false; return 0; }
  A /= denom; B /= denom;
  double radical = B * B - 3 * A * C;
  if (radical < 0 || A == 0) { ok = false; return 0; }
  double xmin = a + (-B + std::sqrt(radical)) / (3 * A);
  ok = std::isfinite(xmin) && std::isfinite(A) && std::isfinite(B);
  return xmin;
}
double quadmin(double a, double fa, double fpa, double b, double fb, bool& ok) {
  double D = fa, C = fpa, db = b - a * 1.0;
  if (db == 0) { ok = false; return 0; }
  double B = (fb - D - C * db) / (db * db);
  if (B == 0) { ok = false; return 0; }
  double xmin = a - C / (2.0 * B);
  ok = std::isfinite(xmin) && std::isfinite(B);
  return xmin;
}

bool zoom(double a_lo, double a_hi, double phi_lo, double phi_hi, double derphi_lo, const std::function<double(double)>& phi,
          const std::function<double(double)>& derphi, double phi0, double derphi0, double c1, double c2, double& a_star, double& val_star) {
  const int maxiter = 10;
  int i = 0;
  const double delta1 = 0.2, delta2 = 0.1;
  double phi_rec = phi0, a_rec = 0;
  for (;;) {
    double dalpha = a_hi - a_lo, a, b;
    if (dalpha < 0) { a = a_hi; b = a_lo; } else { a = a_lo; b = a_hi; }
    double a_j = 0, cchk = 0;
    bool ok = false;
    if (i > 0) {
      cchk = delta1 * dalpha;
      a_j = cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec, ok);
    }
    if (i == 0 || !ok || a_j > b - cchk || a_j < a + cchk) {
      double qchk = delta2 * dalpha;
      a_j = quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, ok);
      if (!ok || a_j > b - qchk || a_j < a + qchk) a_j = a_lo + 0.5 * dalpha;
    }
    double phi_aj = phi(a_j);
    if (phi_aj > phi0 + c1 * a_j * derphi0 || phi_aj >= phi_lo) {
      phi_rec = phi_hi; a_rec = a_hi; a_hi = a_j; phi_hi = phi_aj;
    } else {
      double derphi_aj = derphi(a_j);
      if (std::fabs(derphi_aj) <= -c2 * derphi0) { a_star = a_j; val_star = phi_aj; return true; }
      if (derphi_aj * (a_hi - a_lo) >= 0) { phi_rec = phi_hi; a_rec = a_hi; a_hi = a_lo; phi_hi = phi_lo; }
      else { phi_rec = phi_lo; a_rec = a_lo; }
      a_lo = a_j; phi_lo = phi_aj; derphi_lo = derphi_aj;
    }
    ++i;
    if (i > maxiter) return false;
  }
}

// scalar_search_wolfe2; returns alpha (NaN on failure), phi_star, and whether derphi_star is available.
bool wolfe2(const std::function<double(double)>& phi, const std::function<double(double)>& derphi, double phi0, double old_phi0, double derphi0,
            double c1, double c2, double amax, double& alpha_star, double& phi_star, bool& have_derphi) {
  double alpha0 = 0, alpha1 = derphi0 != 0 ? std::min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0) : 1.0;
  if (alpha1 < 0) alpha1 = 1.0;
  alpha1 = std::min(alpha1, amax);
  double phi_a1 = phi(alpha1), phi_a0 = phi0, derphi_a0 = derphi0;
  have_derphi = true;
  for (int i = 0; i < 10; ++i) {
    if (alpha1 == 0 || alpha0 > amax) return false;
    bool not_first = i > 0;
    if (phi_a1 > phi0 + c1 * alpha1 * derphi0 || (phi_a1 >= phi_a0 && not_first))
      return zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2, alpha_star, phi_star);
    double derphi_a1 = derphi(alpha1);
    if (std::fabs(derphi_a1) <= -c2 * derphi0) { alpha_star = alpha1; phi_star = phi_a1; return true; }
    if (derphi_a1 >= 0) return zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2, alpha_star, phi_star);
    double alpha2 = std::min(2 * alpha1, amax);
    alpha0 = alpha1; alpha1 = alpha2; phi_a0 = phi_a1;
    phi_a1 = phi(alpha1);
    derphi_a0 = derphi_a1;
  }
  alpha_star = alpha1; phi_star = phi_a1; have_derphi = false;
  return true;
}

// Plain left-to-right sums for the small products of BFGS: Eigen's vectorised kernels fuse multiply-adds on aarch64
// (NEON), numpy's do not here, and the one-ulp differences change the iterates of an ill-conditioned problem.
double fdot(const VectorXd& a, const VectorXd& b) {
  double s = 0;
  for (Eigen::Index i = 0; i < a.size(); ++i) s += a(i) * b(i);
  return s;
}
VectorXd fmatvec(const MatrixXd& A, const VectorXd& x) {
  VectorXd y(A.rows());
  for (Eigen::Index i = 0; i < A.rows(); ++i) {
    double s = 0;
    for (Eigen::Index k = 0; k < A.cols(); ++k) s += A(i, k) * x(k);
    y(i) = s;
  }
  return y;
}
MatrixXd fmatmul(const MatrixXd& A, const MatrixXd& B) {
  MatrixXd C(A.rows(), B.cols());
  for (Eigen::Index i = 0; i < A.rows(); ++i)
    for (Eigen::Index j = 0; j < B.cols(); ++j) {
      double s = 0;
      for (Eigen::Index k = 0; k < A.cols(); ++k) s = std::fma(A(i, k), B(k, j), s);
      C(i, j) = s;
    }
  return C;
}

}  // namespace

MinResult bfgs(const MinProblem& p, const VectorXd& x0) {
  double gtol = std::isnan(p.tol) ? 1e-5 : p.tol;
  const double c1 = 1e-4, c2 = 0.9;
  int64_t maxiter = p.maxiter;
  Eigen::Index N = x0.size();
  SF sf(p, x0, std::sqrt(EPS), VectorXd(), VectorXd());
  double old_fval = sf.fun(x0);
  VectorXd gfk = sf.grad(x0);
  int64_t k = 0;
  MatrixXd Hk = MatrixXd::Identity(N, N);
  double old_old_fval = old_fval + std::sqrt(fdot(gfk, gfk)) / 2;
  VectorXd xk = x0;
  int warnflag = 0;
  double gnorm = gfk.cwiseAbs().maxCoeff();
  while (gnorm > gtol && k < maxiter) {
    VectorXd pk = -fmatvec(Hk, gfk);
    // line search: wolfe1 (dcsrch), then wolfe2
    VectorXd gval = gfk;
    bool gval_set = false;
    auto phi = [&](double s) { return sf.fun(xk + s * pk); };
    auto derphi = [&](double s) { gval = sf.grad(xk + s * pk); gval_set = true; return fdot(gval, pk); };
    double derphi0 = fdot(gfk, pk);
    double alpha1 = 1.0;
    if (derphi0 != 0) {
      alpha1 = std::min(1.0, 1.01 * 2 * (old_fval - old_old_fval) / derphi0);
      if (alpha1 < 0) alpha1 = 1.0;
    }
    DCSRCH ls{phi, derphi, c1, c2, 1e-14, 1e-100, 1e100};
    double stp = 0, phi1 = 0;
    bool ok = ls.run(alpha1, old_fval, derphi0, stp, phi1);
    double new_fval = phi1, alpha_k = stp;
    VectorXd gfkp1;
    bool have_g = false;
    if (ok) { gfkp1 = gval; have_g = true; }
    else {
      gval_set = false;
      double a_star = 0, p_star = 0;
      bool have_derphi = false;
      ok = wolfe2(phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, 1e100, a_star, p_star, have_derphi);
      if (ok) {
        alpha_k = a_star;
        new_fval = p_star;
        if (have_derphi && gval_set) { gfkp1 = gval; have_g = true; }
      }
    }
    if (!ok) { warnflag = 2; break; }
    old_old_fval = old_fval;
    old_fval = new_fval;
    VectorXd sk = alpha_k * pk;
    VectorXd xkp1 = xk + sk;
    xk = xkp1;
    if (!have_g) gfkp1 = sf.grad(xkp1);
    VectorXd yk = gfkp1 - gfk;
    gfk = gfkp1;
    ++k;
    gnorm = gfk.cwiseAbs().maxCoeff();
    if (gnorm <= gtol) break;
    if (alpha_k * pk.norm() <= 0) break;
    if (!std::isfinite(old_fval)) { warnflag = 2; break; }
    double rhok_inv = fdot(yk, sk), rhok = rhok_inv == 0 ? 1000.0 : 1.0 / rhok_inv;
    MatrixXd A1(N, N), A2(N, N), S(N, N);
    for (Eigen::Index i = 0; i < N; ++i)
      for (Eigen::Index j = 0; j < N; ++j) {
        double id = i == j ? 1.0 : 0.0;
        A1(i, j) = id - (sk(i) * yk(j)) * rhok;
        A2(i, j) = id - (yk(i) * sk(j)) * rhok;
        S(i, j) = (rhok * sk(i)) * sk(j);
      }
    Hk = fmatmul(A1, fmatmul(Hk, A2)) + S;
  }
  MinResult r;
  r.x = xk; r.fun = old_fval; r.nit = k; r.nfev = sf.nfev; r.njev = sf.ngev;
  if (warnflag == 2) r.message = MSG_PRLOSS;
  else if (k >= maxiter) r.message = MSG_MAXITER;
  else if (std::isnan(gnorm) || std::isnan(old_fval) || xk.hasNaN()) r.message = MSG_NAN;
  else { r.message = MSG_SUCCESS; r.success = true; }
  return r;
}

// ------------------------------------------------------------------------------------------------ L-BFGS-B
MinResult lbfgsb(const MinProblem& p, const VectorXd& x0) {
  Eigen::Index n = x0.size();
  VectorXd lb = p.lb.size() ? p.lb : VectorXd::Constant(n, -INF);
  VectorXd ub = p.ub.size() ? p.ub : VectorXd::Constant(n, INF);
  LBFGSpp::LBFGSBParam<double> param;
  param.m = 10;
  param.epsilon = std::isnan(p.tol) ? 1e-5 : p.tol;
  param.epsilon_rel = 0;
  param.past = 1;
  param.delta = std::isnan(p.tol) ? 2.2204460492503131e-09 : p.tol;
  param.max_iterations = p.maxiter;
  param.max_linesearch = 20;
  LBFGSpp::LBFGSBSolver<double> solver(param);
  int64_t nfev = 0, njev = 0;
  auto fg = [&](const VectorXd& x, VectorXd& g) -> double {
    double f = p.f(x);
    ++nfev;
    if (p.grad) g = p.grad(x);
    else g = fd_gradient([&](const VectorXd& v) { ++nfev; return p.f(v); }, x, f, 1e-8, lb, ub);
    ++njev;
    return f;
  };
  VectorXd x = x0.cwiseMax(lb).cwiseMin(ub);
  double fx = 0;
  MinResult r;
  try {
    int k = solver.minimize(fg, x, fx, lb, ub);
    r.nit = k;
    r.success = k < p.maxiter || p.maxiter == 0;
    r.message = r.success ? "CONVERGENCE: NORM OF PROJECTED GRADIENT OR RELATIVE REDUCTION OF F" : "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT";
    if (r.success && !(solver.final_grad_norm() <= param.epsilon)) r.message = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH";
    else if (r.success) r.message = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL";
  } catch (const Stop&) {
    throw;
  } catch (const dslvm::Error&) {
    throw;
  } catch (const std::exception& e) {
    r.success = false;
    r.message = std::string("ABNORMAL_TERMINATION_IN_LNSRCH (") + e.what() + ")";
    fx = p.f(x);
  }
  r.x = x; r.fun = fx; r.nfev = nfev; r.njev = njev;
  return r;
}

// ---------------------------------------------------------------------------------------------------- SLSQP
namespace {

// Goldfarb-Idnani dual active-set method for the strictly convex QP
//   min 0.5 d'Gd + g'd   s.t.  CE d + ce = 0,  CI d + ci >= 0.
// Returns false when the constraints are inconsistent. lam_e/lam_i are the multipliers (L = q - lam'(c(d))).
struct QP {
  int n;
  MatrixXd J, R;
  int q = 0;
  explicit QP(int n_) : n(n_) {}

  static void givens(double a, double b, double& c, double& s, double& r) {
    if (b == 0) { c = 1; s = 0; r = a; return; }
    r = std::hypot(a, b);
    c = a / r;
    s = b / r;
  }
  // add the constraint whose transformed normal is d = J'n; false when linearly dependent on the active set
  bool add(VectorXd d) {
    for (int j = n - 1; j > q; --j) {
      double c, s, r;
      givens(d(j - 1), d(j), c, s, r);
      if (s == 0) continue;
      d(j - 1) = r;
      d(j) = 0;
      for (int i = 0; i < n; ++i) {
        double a = J(i, j - 1), b = J(i, j);
        J(i, j - 1) = c * a + s * b;
        J(i, j) = -s * a + c * b;
      }
    }
    if (std::fabs(d(q)) <= 1e3 * EPS * std::max(1.0, d.head(q + 1).norm())) return false;
    for (int i = 0; i <= q; ++i) R(i, q) = d(i);
    ++q;
    return true;
  }
  void drop(int k) {
    for (int j = k; j < q - 1; ++j) R.col(j) = R.col(j + 1);
    R.col(q - 1).setZero();
    for (int i = k; i < q - 1; ++i) {
      double c, s, r;
      givens(R(i, i), R(i + 1, i), c, s, r);
      if (s != 0) {
        for (int j = i; j < q - 1; ++j) {
          double a = R(i, j), b = R(i + 1, j);
          R(i, j) = c * a + s * b;
          R(i + 1, j) = -s * a + c * b;
        }
        for (int t = 0; t < n; ++t) {
          double a = J(t, i), b = J(t, i + 1);
          J(t, i) = c * a + s * b;
          J(t, i + 1) = -s * a + c * b;
        }
      }
      R(i + 1, i) = 0;
    }
    --q;
  }
};

bool solve_qp(const MatrixXd& G, const VectorXd& g, const MatrixXd& CE, const VectorXd& ce, const MatrixXd& CI, const VectorXd& ci,
              VectorXd& x, VectorXd& lam_e, VectorXd& lam_i) {
  int n = int(G.rows()), me = int(CE.rows()), mi = int(CI.rows());
  Eigen::LLT<MatrixXd> llt(G);
  if (llt.info() != Eigen::Success) return false;
  MatrixXd L = llt.matrixL();
  QP qp(n);
  qp.J = L.transpose().triangularView<Eigen::Upper>().solve(MatrixXd::Identity(n, n));  // L^{-T}
  qp.R = MatrixXd::Zero(n, n);
  x = -llt.solve(g);
  std::vector<int> active;   // constraint id: 0..me-1 equality, me.. inequality
  std::vector<double> sgn;   // +-1: equality rows may enter with their sign flipped
  std::vector<double> u;     // multipliers of the active set
  lam_e = VectorXd::Zero(me);
  lam_i = VectorXd::Zero(mi);
  auto normal = [&](int id) -> VectorXd { return id < me ? VectorXd(CE.row(id).transpose()) : VectorXd(CI.row(id - me).transpose()); };
  auto value = [&](int id) { return id < me ? CE.row(id).dot(x) + ce(id) : CI.row(id - me).dot(x) + ci(id - me); };
  auto drop = [&](int k) {
    qp.drop(k);
    active.erase(active.begin() + k);
    sgn.erase(sgn.begin() + k);
    u.erase(u.begin() + k);
  };
  // Goldfarb-Idnani step 2 for constraint id with sign s (s * value(id) < 0 on entry).
  auto make_active = [&](int id, double s) -> bool {
    VectorXd np = s * normal(id);
    double scale = std::max(1.0, np.norm());
    double upl = 0;
    for (int guard = 0; guard < 10 * (n + me + mi) + 50; ++guard) {
      double sp = s * value(id);
      VectorXd d = qp.J.transpose() * np;
      VectorXd z = qp.J.rightCols(n - qp.q) * d.tail(n - qp.q);
      VectorXd r = qp.q ? VectorXd(qp.R.topLeftCorner(qp.q, qp.q).triangularView<Eigen::Upper>().solve(d.head(qp.q))) : VectorXd();
      double t1 = INF;
      int k = -1;
      for (int j = 0; j < qp.q; ++j)
        if (active[size_t(j)] >= me && r(j) > 0) {
          double tj = u[size_t(j)] / r(j);
          if (tj < t1) { t1 = tj; k = j; }
        }
      double zn = z.dot(np);
      bool zero_z = !(z.norm() > 1e-12 * scale);
      double t2 = zero_z ? INF : std::max(0.0, -sp / zn);
      if (zero_z && std::fabs(sp) <= 1e-12 * scale * std::max(1.0, x.cwiseAbs().maxCoeff())) return true;  // dependent and satisfied
      double t = std::min(t1, t2);
      if (t == INF) return false;
      if (!zero_z) x += t * z;
      for (int j = 0; j < qp.q; ++j) u[size_t(j)] -= t * r(j);
      upl += t;
      if (t == t2) {
        if (!qp.add(d)) return std::fabs(value(id)) <= 1e-9 * scale * std::max(1.0, x.cwiseAbs().maxCoeff());
        active.push_back(id);
        sgn.push_back(s);
        u.push_back(upl);
        return true;
      }
      drop(k);
    }
    return false;
  };
  for (int i = 0; i < me; ++i)
    if (!make_active(i, value(i) > 0 ? -1.0 : 1.0)) return false;
  for (int iter = 0; iter < 20 * (n + mi + 1); ++iter) {
    int p = -1;
    double worst = 0;
    for (int i = 0; i < mi; ++i) {
      if (std::find(active.begin(), active.end(), me + i) != active.end()) continue;
      double s = value(me + i);
      double tolv = 1e-12 * std::max(1.0, CI.row(i).cwiseAbs().maxCoeff() * std::max(1.0, x.cwiseAbs().maxCoeff()) + std::fabs(ci(i)));
      if (s < -tolv && s < worst) { worst = s; p = me + i; }
    }
    if (p < 0) {
      for (size_t j = 0; j < active.size(); ++j) {
        int id = active[j];
        if (id < me) lam_e(id) = sgn[j] * u[j];
        else lam_i(id - me) = u[j];
      }
      return x.allFinite();
    }
    if (!make_active(p, 1.0)) return false;
  }
  return false;
}

double violation_sum(const VectorXd& c, int meq) {
  double h = 0;
  for (Eigen::Index j = 0; j < c.size(); ++j) h += j < meq ? std::fabs(c(j)) : std::max(-c(j), 0.0);
  return h;
}

}  // namespace

MinResult slsqp(const MinProblem& p, const VectorXd& x0in) {
  const double acc = std::isnan(p.tol) ? 1e-6 : p.tol, tol = 10 * acc, alfmin = 0.1;
  const int maxiter = p.maxiter;
  int n = int(x0in.size()), meq = int(p.eq.size()), m = meq + int(p.ineq.size());
  bool bounded = p.lb.size() > 0;
  VectorXd lb = bounded ? p.lb : VectorXd::Constant(n, -INF), ub = bounded ? p.ub : VectorXd::Constant(n, INF);
  VectorXd x = x0in.cwiseMax(lb).cwiseMin(ub);
  auto cons_value = [&](const VectorXd& v) {
    VectorXd c(m);
    for (int j = 0; j < meq; ++j) c(j) = p.eq[size_t(j)](v);
    for (int j = meq; j < m; ++j) c(j) = p.ineq[size_t(j - meq)](v);
    return c;
  };
  auto clipx = [&](const VectorXd& v) { return VectorXd(v.cwiseMax(lb).cwiseMin(ub)); };
  auto cons_jac = [&](const VectorXd& v0, const VectorXd& c0) {
    VectorXd v = clipx(v0);
    MatrixXd A(m, n);
    for (int j = 0; j < m; ++j) {
      const ScalarFn& cj = j < meq ? p.eq[size_t(j)] : p.ineq[size_t(j - meq)];
      A.row(j) = fd_gradient(cj, v, c0(j), std::sqrt(EPS), lb, ub).transpose();
    }
    return A;
  };
  SF sf(p, x, std::sqrt(EPS), lb, ub);
  double f = sf.fun(x);
  VectorXd g = sf.grad(x);
  VectorXd c = cons_value(x);
  MatrixXd A = m ? cons_jac(x, c) : MatrixXd(0, n);
  MatrixXd B = MatrixXd::Identity(n, n);
  VectorXd mu = VectorXd::Zero(m), r = VectorXd::Zero(m), s = VectorXd::Zero(n), v(n);
  double f0 = f;
  int iter = 0, ireset = 0, mode = 0;
  // bound rows of the QP: d_i >= lb_i - x_i,  -d_i >= x_i - ub_i
  auto search_direction = [&](double& h4) -> bool {
    std::vector<int> lo, hi;
    for (int i = 0; i < n; ++i) {
      if (std::isfinite(lb(i))) lo.push_back(i);
      if (std::isfinite(ub(i))) hi.push_back(i);
    }
    auto build = [&](int nv, MatrixXd& CE, VectorXd& ce, MatrixXd& CI, VectorXd& ci, bool aug) {
      int nb = int(lo.size() + hi.size()) + (aug ? 2 : 0);
      CE = MatrixXd::Zero(meq, nv); ce = VectorXd::Zero(meq);
      CI = MatrixXd::Zero(m - meq + nb, nv); ci = VectorXd::Zero(m - meq + nb);
      for (int j = 0; j < meq; ++j) {
        CE.row(j).head(n) = A.row(j);
        ce(j) = c(j);
        if (aug) CE(j, n) = -c(j);
      }
      for (int j = meq; j < m; ++j) {
        CI.row(j - meq).head(n) = A.row(j);
        ci(j - meq) = c(j);
        if (aug) CI(j - meq, n) = std::max(-c(j), 0.0);
      }
      int row = m - meq;
      for (int i : lo) { CI(row, i) = 1; ci(row) = x(i) - lb(i); ++row; }
      for (int i : hi) { CI(row, i) = -1; ci(row) = ub(i) - x(i); ++row; }
      if (aug) {
        CI(row, n) = 1; ci(row) = 0; ++row;       // delta >= 0
        CI(row, n) = -1; ci(row) = 1; ++row;      // delta <= 1
      }
    };
    MatrixXd CE, CI;
    VectorXd ce, ci, d, le, li;
    build(n, CE, ce, CI, ci, false);
    h4 = 1.0;
    if (solve_qp(B, g, CE, ce, CI, ci, d, le, li)) {
      s = d;
      for (int j = 0; j < meq; ++j) r(j) = le(j);
      for (int j = meq; j < m; ++j) r(j) = li(j - meq);
      return true;
    }
    // augmented problem for an inconsistent linearisation (Kraft): constraints relaxed by (1 - delta)
    double rho = 100.0;
    for (int incons = 0; incons <= 5; ++incons) {
      MatrixXd Ba = MatrixXd::Zero(n + 1, n + 1);
      Ba.topLeftCorner(n, n) = B;
      Ba(n, n) = rho;
      VectorXd ga = VectorXd::Zero(n + 1);
      ga.head(n) = g;
      build(n + 1, CE, ce, CI, ci, true);
      if (solve_qp(Ba, ga, CE, ce, CI, ci, d, le, li)) {
        s = d.head(n);
        h4 = 1.0 - d(n);
        for (int j = 0; j < meq; ++j) r(j) = le(j);
        for (int j = meq; j < m; ++j) r(j) = li(j - meq);
        return true;
      }
      rho *= 10;
    }
    return false;
  };
  for (;;) {
    // reset of the quasi-Newton matrix (label 110)
    ++ireset;
    if (ireset > 5) {
      double h3 = violation_sum(c, meq);
      mode = ((std::fabs(f - f0) < tol || s.norm() < tol) && h3 < tol) ? 0 : 8;
      break;
    }
    B.setIdentity();
    bool restart = false;
    for (;;) {
      ++iter;
      if (iter > maxiter) { mode = 9; break; }
      double h4 = 1.0;
      if (!search_direction(h4)) { mode = 4; break; }
      v = g - A.transpose() * r;
      f0 = f;
      VectorXd x0 = x;
      double gs = g.dot(s), h1 = std::fabs(gs), h2 = 0;
      for (int j = 0; j < m; ++j) {
        double h3 = j < meq ? c(j) : 0.0;
        h2 += std::max(-c(j), h3);
        h3 = std::fabs(r(j));
        mu(j) = std::max(h3, (mu(j) + h3) / 2);
        h1 += h3 * std::fabs(c(j));
      }
      if (h1 < acc && h2 < acc) { mode = 0; break; }
      h1 = 0;
      for (int j = 0; j < m; ++j) h1 += mu(j) * std::max(-c(j), j < meq ? c(j) : 0.0);
      double t0 = f + h1, h3 = gs - h1 * h4;
      if (h3 >= 0) { mode = 8; restart = true; break; }
      double alpha = 1.0;
      int line = 0;
      for (;;) {
        ++line;
        h3 = alpha * h3;
        s *= alpha;
        x = clipx(x0 + s);
        f = sf.fun(x);
        c = cons_value(x);
        double t = f;
        for (int j = 0; j < m; ++j) t += mu(j) * std::max(-c(j), j < meq ? c(j) : 0.0);
        h1 = t - t0;
        if (h1 <= h3 / 10 || line > 10) break;
        alpha = std::max(h3 / (2 * (h3 - h1)), alfmin);
      }
      double hv = violation_sum(c, meq);
      if ((std::fabs(f - f0) < acc || s.norm() < acc) && hv < acc) { mode = 0; break; }
      if (!std::isfinite(f)) { mode = 8; break; }
      g = sf.grad(x);
      if (m) A = cons_jac(x, c);
      // damped BFGS update of the Lagrangian Hessian (Powell's modification, as in Kraft's code)
      VectorXd u = g - A.transpose() * r - v;
      VectorXd Bs = B * s;
      double sBs = s.dot(Bs), sy = s.dot(u), h3b = 0.2 * sBs;
      if (sy < h3b) {
        double th = (sBs - h3b) / (sBs - sy);
        sy = h3b;
        u = th * u + (1 - th) * Bs;
      }
      if (sBs > 0 && sy > 0) B += u * u.transpose() / sy - Bs * Bs.transpose() / sBs;
    }
    if (!restart) break;
  }
  MinResult res;
  static const char* modes[] = {"Optimization terminated successfully", "Function evaluation required (f & c)",
                                "More equality constraints than independent variables", "More than 3*n iterations in LSQ subproblem",
                                "Inequality constraints incompatible", "Singular matrix E in LSQ subproblem", "Singular matrix C in LSQ subproblem",
                                "Rank-deficient equality constraint subproblem HFTI", "Positive directional derivative for linesearch",
                                "Iteration limit reached"};
  res.x = x; res.fun = f; res.nit = std::min(iter, maxiter); res.nfev = sf.nfev; res.njev = sf.ngev;
  res.success = mode == 0;
  res.message = modes[mode];
  return res;
}

}  // namespace lmnum::opt
