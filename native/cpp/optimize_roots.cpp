// Root finding behind `root_scalar` and `root`.
// brentq, bisect and ridder follow scipy's C implementations (scipy/optimize/Zeros/*.c: Brent 1973 with Charles
// Harris's bracketing bookkeeping; Ridders 1979) and toms748 is a port of scipy's TOMS748Solver (Alefeld, Potra, Shi,
// ACM TOMS 21 (1995) 327, re-implemented in scipy's _zeros_py.py; BSD-3-Clause, Copyright (c) 2001-2002 Enthought,
// Inc. and 2003-2024 SciPy Developers), so iteration and call counts follow scipy's.
// Systems: MINPACK hybrd and lmdif (Garbow, Hillstrom, More, Argonne National Laboratory; MINPACK licence) through
// Eigen's unsupported NonLinearOptimization translation (MPL-2.0).
#include "optimize_impl.hpp"
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <algorithm>
#include <cmath>

namespace lmnum::opt {
namespace {

using F1 = std::function<double(double)>;

ScalarRoot brentq(const F1& f, double xa, double xb, double xtol, double rtol, int iter) {
  ScalarRoot R;
  double xpre = xa, xcur = xb, xblk = 0, fpre, fcur, fblk = 0, spre = 0, scur = 0, sbis;
  fpre = f(xpre);
  fcur = f(xcur);
  R.function_calls = 2;
  if (fpre == 0) { R.root = xpre; R.converged = true; return R; }
  if (fcur == 0) { R.root = xcur; R.converged = true; return R; }
  if (std::signbit(fpre) == std::signbit(fcur)) { R.flag = "sign error"; return R; }
  for (int i = 0; i < iter; ++i) {
    ++R.iterations;
    if (fpre != 0 && fcur != 0 && std::signbit(fpre) != std::signbit(fcur)) {
      xblk = xpre; fblk = fpre;
      spre = scur = xcur - xpre;
    }
    if (std::fabs(fblk) < std::fabs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    double delta = (xtol + rtol * std::fabs(xcur)) / 2;
    sbis = (xblk - xcur) / 2;
    if (fcur == 0 || std::fabs(sbis) < delta) { R.root = xcur; R.converged = true; return R; }
    if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
      double stry;
      if (xpre == xblk) stry = -fcur * (xcur - xpre) / (fcur - fpre);
      else {
        double dpre = (fpre - fcur) / (xpre - xcur), dblk = (fblk - fcur) / (xblk - xcur);
        stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
      }
      if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) { spre = scur; scur = stry; }
      else { spre = sbis; scur = sbis; }
    } else {
      spre = sbis; scur = sbis;
    }
    xpre = xcur; fpre = fcur;
    if (std::fabs(scur) > delta) xcur += scur;
    else xcur += (sbis > 0 ? delta : -delta);
    fcur = f(xcur);
    ++R.function_calls;
  }
  R.root = xcur;
  R.flag = "convergence error";
  return R;
}

ScalarRoot bisect(const F1& f, double xa, double xb, double xtol, double rtol, int iter) {
  ScalarRoot R;
  double fa = f(xa), fb = f(xb);
  R.function_calls = 2;
  if (fa * fb > 0) { R.flag = "sign error"; return R; }
  if (fa == 0) { R.root = xa; R.converged = true; return R; }
  if (fb == 0) { R.root = xb; R.converged = true; return R; }
  double dm = xb - xa, xm = xa;
  for (int i = 0; i < iter; ++i) {
    ++R.iterations;
    dm *= 0.5;
    xm = xa + dm;
    double fm = f(xm);
    ++R.function_calls;
    if (fm * fa >= 0) xa = xm;
    if (fm == 0 || std::fabs(dm) < xtol + rtol * std::fabs(xm)) { R.root = xm; R.converged = true; return R; }
  }
  R.root = xa;
  R.flag = "convergence error";
  return R;
}

ScalarRoot ridder(const F1& f, double xa, double xb, double xtol, double rtol, int iter) {
  ScalarRoot R;
  double fa = f(xa), fb = f(xb);
  R.function_calls = 2;
  if (fa * fb > 0) { R.flag = "sign error"; return R; }
  if (fa == 0) { R.root = xa; R.converged = true; return R; }
  if (fb == 0) { R.root = xb; R.converged = true; return R; }
  double tol = xtol + rtol * std::max(std::fabs(xa), std::fabs(xb)) * 0, xn = 0;
  (void)tol;
  for (int i = 0; i < iter; ++i) {
    ++R.iterations;
    double dm = 0.5 * (xb - xa), xm = xa + dm, fm = f(xm);
    double dn = (fb - fa >= 0 ? 1.0 : -1.0) * dm * fm / std::sqrt(fm * fm - fa * fb);
    double t = xtol + rtol * std::fabs(xm);
    xn = xm - (dn >= 0 ? 1.0 : -1.0) * std::min(std::fabs(dn), std::fabs(dm) - .5 * t);
    double fn = f(xn);
    R.function_calls += 2;
    if (fn * fm < 0.0) { xa = xn; fa = fn; xb = xm; fb = fm; }
    else if (fn * fa < 0.0) { xb = xn; fb = fn; }
    else { xa = xn; fa = fn; }
    double tol2 = xtol + rtol * xn;
    if (fn == 0.0 || std::fabs(xb - xa) < tol2) { R.root = xn; R.converged = true; return R; }
  }
  R.root = xn;
  R.flag = "convergence error";
  return R;
}

// ---- TOMS 748 (scipy port) ----------------------------------------------------------------------------------------
bool isclose(double a, double b, double rtol, double atol) { return std::fabs(a - b) <= atol + rtol * std::fabs(b); }

double interpolated_poly(const double* xv, const double* fv, int N, double x) {
  double Q[4][4] = {}, D[4][4] = {};
  for (int i = 0; i < N; ++i) Q[i][0] = D[i][0] = fv[i];
  for (int k = 1; k < N; ++k)
    for (int i = k; i < N; ++i) {
      double alpha = D[i][k - 1] - Q[i - 1][k - 1];
      double diffik = xv[i - k] - xv[i];
      Q[i][k] = (xv[i] - x) / diffik * alpha;
      D[i][k] = (xv[i - k] - x) / diffik * alpha;
    }
  double s = 0;
  for (int k = 1; k < N; ++k) s += Q[N - 1][k];
  return s + Q[N - 1][0];
}

// divided differences, non-full form: returns dd[0..n-1]
void divided_differences(const double* x, const double* fvals, int n, bool forward, double* dd) {
  double row[3];
  for (int i = 0; i < n; ++i) row[i] = fvals[i];
  dd[0] = forward ? fvals[0] : fvals[n - 1];
  int len = n;
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < len - 1; ++j) row[j] = (row[j + 1] - row[j]) / (x[i + j] - x[j]);
    --len;
    dd[i] = forward ? row[0] : row[len - 1];
  }
}

struct Toms748 {
  const F1& f;
  double xtol, rtol;
  int maxiter, k = 1;
  int calls = 0, iterations = 0;
  double ab[2], fab[2];
  double d = NAN, fd = NAN, e = NAN, fe = NAN;
  bool have_e = false;
  double callf(double x) {
    double v = f(x);
    ++calls;
    if (!std::isfinite(v)) fail("root_scalar(toms748): f(" + jnum(x) + ") is not finite");
    return v;
  }
  std::pair<double, double> update(double c, double fc) {
    int idx = (sign(fab[0]) * sign(fc) > 0) ? 0 : 1;
    std::pair<double, double> r{ab[idx], fab[idx]};
    fab[idx] = fc;
    ab[idx] = c;
    return r;
  }
  static double sign(double v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }
  double newton_quadratic(int kk) {
    double a = ab[0], b = ab[1], fa = fab[0];
    double xs[3] = {a, b, d}, fs[3] = {fab[0], fab[1], fd}, dd[3];
    divided_differences(xs, fs, 3, true, dd);
    double B = dd[1], A = dd[2];
    auto P = [&](double x) { return (A * (x - b) + B) * (x - a) + fa; };
    double r;
    if (A == 0) r = a - fa / B;
    else {
      r = sign(A) * sign(fa) > 0 ? a : b;
      for (int i = 0; i < kk; ++i) {
        double r1 = r - P(r) / (B + A * (2 * r - a - b));
        if (!(ab[0] < r1 && r1 < ab[1])) {
          if (ab[0] < r && r < ab[1]) return r;
          r = (ab[0] + ab[1]) / 2.0;
          break;
        }
        r = r1;
      }
    }
    return r;
  }
  bool notclose(const double* fs, int n) {
    for (int i = 0; i < n; ++i) if (fs[i] == 0 || !std::isfinite(fs[i])) return false;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (isclose(fs[i], fs[j], 0, 32 * EPS)) return false;
    return true;
  }
  // returns 1 converged, 0 in progress, -1 iteration limit; x set
  int iterate(double& x) {
    ++iterations;
    double ab_width = ab[1] - ab[0];
    double c = NAN;
    bool have_c = false;
    for (int nsteps = 2; nsteps < k + 2; ++nsteps) {
      double fs[4] = {fab[0], fab[1], fd, fe};
      if (have_e && notclose(fs, 4)) {
        double xs4[4] = {fab[0], fab[1], fd, fe}, ys4[4] = {ab[0], ab[1], d, e};
        double c0 = interpolated_poly(xs4, ys4, 4, 0);
        if (ab[0] < c0 && c0 < ab[1]) { c = c0; have_c = true; }
      }
      if (!have_c) { c = newton_quadratic(nsteps); have_c = true; }
      double fc = callf(c);
      if (fc == 0) { x = c; return 1; }
      e = d; fe = fd; have_e = true;
      auto r = update(c, fc);
      d = r.first; fd = r.second;
    }
    int uix = std::fabs(fab[0]) < std::fabs(fab[1]) ? 0 : 1;
    double u = ab[uix], fu = fab[uix];
    double dd[2];
    divided_differences(ab, fab, 2, uix == 0, dd);
    double A = dd[1];
    c = u - 2 * fu / A;
    if (std::fabs(c - u) > 0.5 * (ab[1] - ab[0])) c = (ab[0] + ab[1]) / 2.0;
    else {
      if (isclose(c, u, EPS, 0)) {
        int e0, e1;
        std::frexp(fab[uix], &e0);
        std::frexp(fab[1 - uix], &e1);
        if (e0 < e1 - 50) c = (31 * ab[uix] + ab[1 - uix]) / 32;
        else {
          double mm = uix == 0 ? 1 : -1;
          c = u + mm * std::fabs(c) * rtol + mm * xtol;
        }
        if (!(ab[0] < c && c < ab[1])) c = (ab[0] + ab[1]) / 2.0;
      }
    }
    double fc = callf(c);
    if (fc == 0) { x = c; return 1; }
    e = d; fe = fd; have_e = true;
    auto r = update(c, fc);
    d = r.first; fd = r.second;
    if (ab[1] - ab[0] > 0.5 * ab_width) {
      e = d; fe = fd;
      double z = (ab[0] + ab[1]) / 2.0, fz = callf(z);
      if (fz == 0) { x = z; return 1; }
      auto r2 = update(z, fz);
      d = r2.first; fd = r2.second;
    }
    x = (ab[0] + ab[1]) / 2.0;
    if (isclose(ab[0], ab[1], rtol, xtol)) return 1;
    if (iterations >= maxiter) return -1;
    return 0;
  }
  ScalarRoot solve(double a, double b) {
    ScalarRoot R;
    ab[0] = a; ab[1] = b;
    double fa = callf(a);
    if (fa == 0) { R.root = a; R.converged = true; R.function_calls = calls; return R; }
    double fb = callf(b);
    if (fb == 0) { R.root = b; R.converged = true; R.function_calls = calls; return R; }
    if (sign(fb) * sign(fa) > 0) { R.flag = "sign error"; R.function_calls = calls; return R; }
    fab[0] = fa; fab[1] = fb;
    double c;
    if (fa == fb) c = NAN;
    else if (std::fabs(fb) > std::fabs(fa)) c = (-fa / fb * b + a) / (1 - fa / fb);
    else c = (-fb / fa * a + b) / (1 - fb / fa);
    if (!(ab[0] < c && c < ab[1])) c = (ab[0] + ab[1]) / 2.0;
    double fc = callf(c);
    if (fc == 0) { R.root = c; R.converged = true; R.function_calls = calls; return R; }
    auto r = update(c, fc);
    d = r.first; fd = r.second;
    have_e = false;
    ++iterations;
    for (;;) {
      double x;
      int st = iterate(x);
      if (st != 0) {
        R.root = x; R.converged = st == 1; R.iterations = iterations; R.function_calls = calls;
        if (!R.converged) R.flag = "convergence error";
        return R;
      }
    }
  }
};

}  // namespace

ScalarRoot bracket_root(const std::string& method, const std::function<double(double)>& f, double a, double b, double xtol, double rtol, int maxiter) {
  ScalarRoot r;
  if (method == "brentq") r = brentq(f, a, b, xtol, rtol, maxiter);
  else if (method == "bisect") r = bisect(f, a, b, xtol, rtol, maxiter);
  else if (method == "ridder") r = ridder(f, a, b, xtol, rtol, maxiter);
  else if (method == "toms748") {
    Toms748 t{f, xtol, rtol, maxiter};
    r = t.solve(a, b);
  } else fail("root_scalar: method must be brentq, bisect, ridder or toms748; got '" + method + "'");
  if (r.converged) r.flag = "converged";
  return r;
}

// ---- systems ----------------------------------------------------------------------------------------------------
namespace {
struct SysFunctor {
  typedef double Scalar;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
  typedef VectorXd InputType;
  typedef VectorXd ValueType;
  typedef MatrixXd JacobianType;
  const VectorFn& f;
  int n;
  int64_t* count;
  int inputs() const { return n; }
  int values() const { return n; }
  int operator()(const VectorXd& x, VectorXd& fv) const {
    ++*count;
    fv = f(x);
    return 0;
  }
};
}  // namespace

SystemRoot root_hybr(const VectorFn& f, const VectorXd& x0, double xtol, int64_t maxfev) {
  int64_t count = 0;
  SysFunctor fn{f, int(x0.size()), &count};
  Eigen::HybridNonLinearSolver<SysFunctor> solver(fn);
  solver.parameters.xtol = xtol;
  solver.parameters.maxfev = Eigen::Index(maxfev);
  solver.parameters.factor = 100.0;
  solver.parameters.epsfcn = 0.0;
  VectorXd x = x0;
  int info = int(solver.solveNumericalDiff(x));
  SystemRoot r;
  r.x = x;
  r.nfev = count;
  r.success = info == 1;
  static const char* msgs[] = {"Improper input parameters were entered.", "The solution converged.",
                               "The number of calls to function has reached maxfev.",
                               "xtol is too small, no further improvement in the approximate solution is possible.",
                               "The iteration is not making good progress, as measured by the improvement from the last five Jacobian evaluations.",
                               "The iteration is not making good progress, as measured by the improvement from the last ten iterations.",
                               "Stopped by the caller."};
  r.message = info >= 0 && info <= 6 ? msgs[info] : "unknown MINPACK status";
  return r;
}

SystemRoot root_lm(const VectorFn& f, const VectorXd& x0, double xtol, int64_t maxfev) {
  int64_t count = 0;
  SysFunctor base{f, int(x0.size()), &count};
  Eigen::NumericalDiff<SysFunctor> fn(base, 0.0);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<SysFunctor>> solver(fn);
  solver.parameters.xtol = xtol;
  solver.parameters.ftol = 1.49012e-08;
  solver.parameters.gtol = 0.0;
  solver.parameters.maxfev = Eigen::Index(maxfev);
  solver.parameters.factor = 100.0;
  VectorXd x = x0;
  int info = int(solver.minimize(x));
  SystemRoot r;
  r.x = x;
  r.nfev = count;
  r.success = info >= 1 && info <= 4;
  static const char* msgs[] = {"Improper input parameters.", "Both actual and predicted relative reductions in the sum of squares are at most ftol.",
                               "The relative error between two consecutive iterates is at most xtol.",
                               "Both actual and predicted relative reductions in the sum of squares are at most ftol and the relative error between two consecutive iterates is at most xtol.",
                               "The cosine of the angle between func(x) and any column of the Jacobian is at most gtol in absolute value.",
                               "Number of calls to func has reached maxfev.",
                               "ftol is too small: no further reduction in the sum of squares is possible.",
                               "xtol is too small: no further improvement in the approximate solution is possible.",
                               "gtol is too small: func(x) is orthogonal to the columns of the Jacobian to machine precision.", "Stopped by the caller."};
  r.message = info >= 0 && info <= 9 ? msgs[info] : "unknown MINPACK status";
  return r;
}

}  // namespace lmnum::opt
