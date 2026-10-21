// SARIMAX(p,d,q)(P,D,Q,s) with trend and exogenous regressors by exact Gaussian maximum likelihood (Kalman filter),
// after statsmodels.tsa.statespace.sarimax.SARIMAX 0.14 (BSD-3-Clause). pycli op covered: sm_arima(_to).
//
// What is reproduced exactly: the state-space layout (Harvey representation, differencing kept inside the state with an
// approximate-diffuse prior of variance 1e6 and the first d + sD observations burned), the stationary initialisation of
// the ARMA block (mean (I - T)^-1 c, covariance from the discrete Lyapunov equation), the Monahan (1984) / Jones (1980)
// parameter transformation, statsmodels' starting values (conditional least squares), and the three covariance
// estimators (opg, oim, approx) computed with complex-step derivatives exactly as statsmodels does. The optimiser is
// different: BFGS on the unconstrained parameters with complex-step (exact) gradients instead of scipy's L-BFGS-B with
// finite differences plus pycli's Nelder-Mead polish; it converges to the same maximum, usually closer to it.
#include "models_common.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>

namespace lmpy {
namespace models {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using cd = std::complex<double>;
template <class T> using Mx = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T> using Vx = Eigen::Matrix<T, Eigen::Dynamic, 1>;
const double EPS = std::numeric_limits<double>::epsilon();

inline double re(double x) { return x; }
inline double re(cd x) { return x.real(); }

struct Spec {
  int p = 0, d = 0, q = 0, P = 0, D = 0, Q = 0, s = 0;
  std::vector<int> trend_powers;  // nonzero powers of the trend polynomial (0 = constant, 1 = t, ...)
  int k_trend = 0, k_exog = 0;
  bool enforce_stationarity = true, enforce_invertibility = true;
  int k_ar = 0, k_ma = 0, k_sar = 0, k_sma = 0;  // largest lags: p, q, P*s, Q*s
  int k_order = 0, k_diff_states = 0, k_states = 0, burn = 0;
  int n = 0;
  std::vector<double> y;  // NaN = missing
  MatrixXd exog;          // n x k_exog
  MatrixXd trend_data;    // n x k_trend, t = 1..n
  int k_params() const { return k_trend + k_exog + p + q + P + Q + 1; }
  int i_exog() const { return k_trend; }
  int i_ar() const { return k_trend + k_exog; }
  int i_ma() const { return i_ar() + p; }
  int i_sar() const { return i_ma() + q; }
  int i_sma() const { return i_sar() + P; }
  int i_sig() const { return i_sma() + Q; }
  double trend_at(int t, int j) const {  // t 0-based, possibly beyond the sample (forecasting)
    int pw = trend_powers[size_t(j)];
    return pw == 0 ? 1.0 : std::pow(double(t + 1), double(pw));
  }
};

// Monahan (1984) map from R^n to stationary AR coefficients (statsmodels constrain_stationary_univariate) and back.
template <class T> std::vector<T> constrain_stationary(const std::vector<T>& u) {
  size_t n = u.size();
  std::vector<std::vector<T>> y(n, std::vector<T>(n, T(0)));
  std::vector<T> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = u[i] / std::sqrt(T(1) + u[i] * u[i]);
  for (size_t k = 0; k < n; ++k) {
    for (size_t i = 0; i < k; ++i) y[k][i] = y[k - 1][i] + r[k] * y[k - 1][k - i - 1];
    y[k][k] = r[k];
  }
  std::vector<T> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = -y[n - 1][i];
  return out;
}
std::vector<double> unconstrain_stationary(const std::vector<double>& c) {
  size_t n = c.size();
  if (!n) return {};
  std::vector<std::vector<double>> y(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) y[n - 1][i] = -c[i];
  for (size_t k = n - 1; k > 0; --k)
    for (size_t i = 0; i < k; ++i) y[k - 1][i] = (y[k][i] - y[k][k] * y[k][k - i - 1]) / (1 - y[k][k] * y[k][k]);
  std::vector<double> x(n);
  for (size_t i = 0; i < n; ++i) { double r = y[i][i]; x[i] = r / std::sqrt(1 - r * r); }
  return x;
}

template <class T> std::vector<T> slice(const std::vector<T>& v, int a, int n) { return std::vector<T>(v.begin() + a, v.begin() + a + n); }
template <class T> void put(std::vector<T>& v, int a, const std::vector<T>& x) { for (size_t i = 0; i < x.size(); ++i) v[size_t(a) + i] = x[i]; }

template <class T> std::vector<T> transform(const Spec& m, const std::vector<T>& u) {
  std::vector<T> c = u;
  auto stat = [&](int a, int n, bool enforce, bool neg) {
    if (!n || !enforce) return;
    auto x = constrain_stationary(slice(u, a, n));
    if (neg) for (auto& v : x) v = -v;
    put(c, a, x);
  };
  stat(m.i_ar(), m.p, m.enforce_stationarity, false);
  stat(m.i_ma(), m.q, m.enforce_invertibility, true);
  stat(m.i_sar(), m.P, m.enforce_stationarity, false);
  stat(m.i_sma(), m.Q, m.enforce_invertibility, true);
  c[size_t(m.i_sig())] = u[size_t(m.i_sig())] * u[size_t(m.i_sig())];
  return c;
}
std::vector<double> untransform(const Spec& m, const std::vector<double>& c) {
  std::vector<double> u = c;
  auto stat = [&](int a, int n, bool enforce, bool neg) {
    if (!n || !enforce) return;
    auto x = slice(c, a, n);
    if (neg) for (auto& v : x) v = -v;
    put(u, a, unconstrain_stationary(x));
  };
  stat(m.i_ar(), m.p, m.enforce_stationarity, false);
  stat(m.i_ma(), m.q, m.enforce_invertibility, true);
  stat(m.i_sar(), m.P, m.enforce_stationarity, false);
  stat(m.i_sma(), m.Q, m.enforce_invertibility, true);
  u[size_t(m.i_sig())] = std::sqrt(c[size_t(m.i_sig())]);
  return u;
}

// statsmodels is_invertible: all eigenvalues of the companion matrix of c[0] + c[1] L + ... inside 1 - 1e-10.
bool is_invertible(const std::vector<double>& c) {
  size_t n = c.size() - 1;
  if (n == 0) return true;
  MatrixXd cm = MatrixXd::Zero(n, n);
  for (size_t i = 0; i < n; ++i) cm(i, 0) = -c[i + 1] / c[0];
  for (size_t i = 0; i + 1 < n; ++i) cm(i, i + 1) = 1;
  Eigen::EigenSolver<MatrixXd> es(cm, false);
  for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i)
    if (!(std::abs(es.eigenvalues()(i)) < 1 - 1e-10)) return false;
  return true;
}

// ---- state space ----------------------------------------------------------------------------------------------------
template <class T> struct System {
  Mx<T> Tm;
  Vx<T> R, Z;
  T sigma2;
  std::vector<T> trend_params, exog_params;
  Vx<T> a0;
  Mx<T> P0;
};

// Stationary covariance P = A P A' + Q by the doubling algorithm (sums of A^j Q A^j'); analytic, so complex-step safe.
template <class T> Mx<T> lyapunov(const Mx<T>& A, const Mx<T>& Q) {
  Mx<T> a = A, p = Q;
  for (int it = 0; it < 200; ++it) {
    Mx<T> add = a * p * a.transpose();
    p += add;
    a = a * a;
    double na = 0, nadd = 0, np = 0;
    for (Eigen::Index i = 0; i < a.size(); ++i) na = std::max(na, std::abs(re(a.data()[i])));
    for (Eigen::Index i = 0; i < p.size(); ++i) { nadd = std::max(nadd, std::abs(re(add.data()[i]))); np = std::max(np, std::abs(re(p.data()[i]))); }
    if (na < 1e-300 || (nadd <= 1e-18 * np && na < 1e-8)) break;
  }
  return p;
}

template <class T> System<T> build(const Spec& m, const std::vector<T>& c) {
  System<T> sys;
  int k = m.k_states, kd = m.k_diff_states, ko = m.k_order;
  sys.Tm = Mx<T>::Zero(k, k);
  sys.R = Vx<T>::Zero(k);
  sys.Z = Vx<T>::Zero(k);
  // reduced-form AR (1 - phi(L))(1 - Phi(L^s)) and MA (1 + theta(L))(1 + Theta(L^s)) polynomials
  std::vector<T> ar(size_t(m.k_ar + m.k_sar + 1), T(0)), ma(size_t(m.k_ma + m.k_sma + 1), T(0));
  std::vector<T> a1(size_t(m.k_ar + 1), T(0)), a2(size_t(m.k_sar + 1), T(0)), m1(size_t(m.k_ma + 1), T(0)), m2(size_t(m.k_sma + 1), T(0));
  a1[0] = a2[0] = m1[0] = m2[0] = T(1);
  for (int i = 0; i < m.p; ++i) a1[size_t(i + 1)] = -c[size_t(m.i_ar() + i)];
  for (int i = 0; i < m.q; ++i) m1[size_t(i + 1)] = c[size_t(m.i_ma() + i)];
  for (int i = 0; i < m.P; ++i) a2[size_t((i + 1) * m.s)] = -c[size_t(m.i_sar() + i)];
  for (int i = 0; i < m.Q; ++i) m2[size_t((i + 1) * m.s)] = c[size_t(m.i_sma() + i)];
  for (size_t i = 0; i < a1.size(); ++i) for (size_t j = 0; j < a2.size(); ++j) ar[i + j] += a1[i] * a2[j];
  for (size_t i = 0; i < m1.size(); ++i) for (size_t j = 0; j < m2.size(); ++j) ma[i + j] += m1[i] * m2[j];
  // ARMA block: companion with the AR coefficients in the first column
  for (int i = 0; i + 1 < ko; ++i) sys.Tm(kd + i, kd + i + 1) = T(1);
  for (size_t i = 1; i < ar.size(); ++i) sys.Tm(kd + int(i) - 1, kd) = -ar[i];
  sys.R(kd) = T(1);
  for (size_t i = 1; i < ma.size(); ++i) sys.R(kd + int(i)) = ma[i];
  // seasonal differencing blocks
  for (int dd = 0; dd < m.D; ++dd) {
    int st = m.d + dd * m.s;
    for (int i = 0; i + 1 < m.s; ++i) sys.Tm(st + i + 1, st + i) = T(1);
    sys.Tm(st, st + m.s - 1) = T(1);
    if (dd < m.D - 1) sys.Tm(st, st + m.s + m.s - 1) = T(1);
    sys.Tm(st, kd) = T(1);
  }
  // simple differencing
  for (int i = 0; i < m.d; ++i) {
    for (int j = i; j < m.d; ++j) sys.Tm(i, j) = T(1);
    for (int dd = 0; dd < m.D; ++dd) sys.Tm(i, m.d + dd * m.s + m.s - 1) = T(1);
    sys.Tm(i, kd) = T(1);
  }
  for (int i = 0; i < m.d; ++i) sys.Z(i) = T(1);
  for (int dd = 0; dd < m.D; ++dd) sys.Z(m.d + dd * m.s + m.s - 1) = T(1);
  sys.Z(kd) = T(1);
  sys.sigma2 = c[size_t(m.i_sig())];
  sys.trend_params = slice(c, 0, m.k_trend);
  sys.exog_params = slice(c, m.i_exog(), m.k_exog);
  // initial state: approximate diffuse (variance 1e6) for the differencing states, stationary for the ARMA block
  sys.a0 = Vx<T>::Zero(k);
  sys.P0 = Mx<T>::Zero(k, k);
  if (m.enforce_stationarity) {
    for (int i = 0; i < kd; ++i) sys.P0(i, i) = T(1e6);
    Mx<T> Tb = sys.Tm.block(kd, kd, ko, ko);
    Vx<T> Rb = sys.R.segment(kd, ko);
    Mx<T> Qb = Rb * Rb.transpose() * sys.sigma2;
    sys.P0.block(kd, kd, ko, ko) = lyapunov(Tb, Qb);
    if (m.k_trend) {
      Vx<T> cb = Vx<T>::Zero(ko);
      for (int j = 0; j < m.k_trend; ++j) cb(0) += T(m.trend_at(0, j)) * sys.trend_params[size_t(j)];
      Mx<T> IA = Mx<T>::Identity(ko, ko) - Tb;
      sys.a0.segment(kd, ko) = IA.partialPivLu().solve(cb);
    }
  } else {
    for (int i = 0; i < k; ++i) sys.P0(i, i) = T(1e6);
  }
  return sys;
}

template <class T> struct FilterOut {
  std::vector<T> llf_obs, v, F;
  Vx<T> a;  // predicted state for time n (after the last observation)
  Mx<T> P;
  T llf = T(0);
};

template <class T> T state_intercept(const Spec& m, const System<T>& sys, int t) {
  T c = T(0);
  for (int j = 0; j < m.k_trend; ++j) c += T(m.trend_at(t, j)) * sys.trend_params[size_t(j)];
  return c;
}
template <class T> T obs_intercept(const Spec& m, const System<T>& sys, int t, const MatrixXd* xf = nullptr) {
  T d = T(0);
  for (int j = 0; j < m.k_exog; ++j) d += T(xf ? (*xf)(t, j) : m.exog(t, j)) * sys.exog_params[size_t(j)];
  return d;
}

// Conventional Kalman filter (statsmodels): v = y - d - Z a, F = Z P Z', a|t = a + P Z' v / F, then a+ = c + T a|t.
template <class T> FilterOut<T> kfilter(const Spec& m, const System<T>& sys) {
  int k = m.k_states;
  FilterOut<T> o;
  o.llf_obs.assign(size_t(m.n), T(0));
  o.v.assign(size_t(m.n), T(NAN));
  o.F.assign(size_t(m.n), T(NAN));
  Vx<T> a = sys.a0;
  Mx<T> P = sys.P0;
  Mx<T> RQR = sys.R * sys.R.transpose() * sys.sigma2;
  const double l2pi = std::log(2 * M_PI);
  for (int t = 0; t < m.n; ++t) {
    Vx<T> af = a;
    Mx<T> Pf = P;
    if (!std::isnan(m.y[size_t(t)])) {
      Vx<T> PZ = P * sys.Z;
      T F = sys.Z.dot(PZ);
      T v = T(m.y[size_t(t)]) - obs_intercept(m, sys, t) - sys.Z.dot(a);
      o.v[size_t(t)] = v;
      o.F[size_t(t)] = F;
      af = a + PZ * (v / F);
      Pf = P - PZ * PZ.transpose() / F;
      T ll = T(-0.5) * (T(l2pi) + std::log(F) + v * v / F);
      o.llf_obs[size_t(t)] = t < m.burn ? T(0) : ll;
      if (t >= m.burn) o.llf += ll;
    }
    a = sys.Tm * af;
    a(m.k_diff_states) += state_intercept(m, sys, t);
    P = sys.Tm * Pf * sys.Tm.transpose() + RQR;
    P = (P + P.transpose()) * T(0.5);
    (void)k;
  }
  o.a = a;
  o.P = P;
  return o;
}

template <class T> T loglike(const Spec& m, const std::vector<T>& constrained) { return kfilter(m, build(m, constrained)).llf; }

// ---- starting values (SARIMAX.start_params) -------------------------------------------------------------------------
std::vector<double> pinv_solve(const MatrixXd& X, const VectorXd& y) {
  if (X.cols() == 0) return {};
  Pinv pv = pinv_extended(X);
  VectorXd b = pv.pinv * y;
  return std::vector<double>(b.data(), b.data() + b.size());
}

struct Css { std::vector<double> trend, ar, ma; double variance = NAN; bool has_variance = false; };
// _conditional_sum_squares: AR(k) residuals as proxies for the innovations, then OLS of y on its lags and their lags.
Css conditional_ss(const std::vector<double>& y, int k_ar, const std::vector<int>& ar_cols, int k_ma, const std::vector<int>& ma_cols, int k_trend,
                   const MatrixXd* trend_data) {
  Css out;
  int n = int(y.size());
  int k = 2 * k_ma, r = std::max(k + k_ma, k_ar);
  int kpa = int(ar_cols.size()), kpm = int(ma_cols.size());
  if (k_ar + k_ma + k_trend == 0) return out;
  std::vector<double> resid;
  bool ok = n - r > 0 && n - k > 0;
  std::vector<double> params;
  if (ok) {
    std::vector<double> res1;
    if (k_ma > 0) {
      MatrixXd X(n - k, k);
      VectorXd Y(n - k);
      for (int t = k; t < n; ++t) { Y(t - k) = y[size_t(t)]; for (int j = 0; j < k; ++j) X(t - k, j) = y[size_t(t - j - 1)]; }
      auto b = pinv_solve(X, Y);
      VectorXd bb = Eigen::Map<VectorXd>(b.data(), Eigen::Index(b.size()));
      VectorXd e = Y - X * bb;
      res1.assign(e.data(), e.data() + e.size());
    }
    int rows = n - r;
    MatrixXd X(rows, k_trend + kpa + kpm);
    VectorXd Y(rows);
    for (int i = 0; i < rows; ++i) {
      int t = i + r;
      Y(i) = y[size_t(t)];
      int c = 0;
      for (int j = 0; j < k_trend; ++j) X(i, c++) = (*trend_data)(i, j);  // trend_data[:-r]: the first rows (statsmodels quirk)
      for (int lag : ar_cols) X(i, c++) = t - lag >= 0 ? y[size_t(t - lag)] : 0.0;
      for (int lag : ma_cols) { int u = t - k - lag; X(i, c++) = u >= 0 ? res1[size_t(u)] : 0.0; }
    }
    params = pinv_solve(X, Y);
    VectorXd bb = Eigen::Map<VectorXd>(params.data(), Eigen::Index(params.size()));
    VectorXd e = Y - X * bb;
    resid.assign(e.data(), e.data() + e.size());
  } else {
    params.assign(size_t(k_trend + kpa + kpm), 0.0);
    double mean = 0;
    for (double v : y) mean += v / double(n);
    resid.assign(size_t(kpm * 2), 0.0);
    for (double v : y) resid.push_back(v - mean);
  }
  out.trend = slice(params, 0, k_trend);
  out.ar = slice(params, k_trend, kpa);
  out.ma = slice(params, k_trend + kpa, kpm);
  out.has_variance = true;
  if (int(resid.size()) > std::max(1, kpm)) {
    double s = 0;
    for (size_t i = size_t(kpm); i < resid.size(); ++i) s += resid[i] * resid[i];
    out.variance = s / double(resid.size() - size_t(kpm));
  } else {
    double mean = 0, s = 0;
    for (double v : y) mean += v / double(n);
    for (double v : y) s += (v - mean) * (v - mean);
    out.variance = s / double(n);
  }
  return out;
}

std::vector<double> start_params(const Spec& m) {
  // difference: seasonal first, then simple (statsmodels tools.diff)
  std::vector<double> y = m.y;
  MatrixXd X = m.exog;
  for (int dd = 0; dd < m.D; ++dd) {
    std::vector<double> z;
    for (size_t t = size_t(m.s); t < y.size(); ++t) z.push_back(y[t] - y[t - size_t(m.s)]);
    if (m.k_exog) X = (X.bottomRows(X.rows() - m.s) - X.topRows(X.rows() - m.s)).eval();
    y.swap(z);
  }
  for (int dd = 0; dd < m.d; ++dd) {
    std::vector<double> z;
    for (size_t t = 1; t < y.size(); ++t) z.push_back(y[t] - y[t - 1]);
    if (m.k_exog) X = (X.bottomRows(X.rows() - 1) - X.topRows(X.rows() - 1)).eval();
    y.swap(z);
  }
  MatrixXd td = m.trend_data.topRows(Eigen::Index(y.size()));
  // drop missing values
  std::vector<double> yy;
  std::vector<Eigen::Index> keep;
  for (size_t t = 0; t < y.size(); ++t) if (!std::isnan(y[t])) { yy.push_back(y[t]); keep.push_back(Eigen::Index(t)); }
  MatrixXd tdk(Eigen::Index(keep.size()), td.cols()), Xk(Eigen::Index(keep.size()), X.cols());
  for (size_t i = 0; i < keep.size(); ++i) { tdk.row(Eigen::Index(i)) = td.row(keep[i]); if (m.k_exog) Xk.row(Eigen::Index(i)) = X.row(keep[i]); }
  std::vector<double> pexog;
  if (m.k_exog) {
    VectorXd Y = Eigen::Map<VectorXd>(yy.data(), Eigen::Index(yy.size()));
    pexog = pinv_solve(Xk, Y);
    VectorXd b = Eigen::Map<VectorXd>(pexog.data(), Eigen::Index(pexog.size()));
    VectorXd e = Y - Xk * b;
    yy.assign(e.data(), e.data() + e.size());
  }
  std::vector<int> ar_cols, ma_cols, sar_cols, sma_cols;
  for (int i = 1; i <= m.p; ++i) ar_cols.push_back(i);
  for (int i = 1; i <= m.q; ++i) ma_cols.push_back(i);
  for (int i = 1; i <= m.P; ++i) sar_cols.push_back(i * m.s);
  for (int i = 1; i <= m.Q; ++i) sma_cols.push_back(i * m.s);
  Css ns = conditional_ss(yy, m.k_ar, ar_cols, m.k_ma, ma_cols, m.k_trend, &tdk);
  Css se = conditional_ss(yy, m.k_sar, sar_cols, m.k_sma, sma_cols, 0, nullptr);
  auto invertible_ar = [](const std::vector<double>& c) { std::vector<double> p = {1}; for (double v : c) p.push_back(-v); return is_invertible(p); };
  auto invertible_ma = [](const std::vector<double>& c) { std::vector<double> p = {1}; for (double v : c) p.push_back(v); return is_invertible(p); };
  if (m.p && m.enforce_stationarity && !invertible_ar(ns.ar)) std::fill(ns.ar.begin(), ns.ar.end(), 0.0);
  if (m.q && m.enforce_invertibility && !invertible_ma(ns.ma)) std::fill(ns.ma.begin(), ns.ma.end(), 0.0);
  if (m.P && m.enforce_stationarity && !invertible_ar(se.ar)) std::fill(se.ar.begin(), se.ar.end(), 0.0);
  if (m.Q && m.enforce_invertibility && !invertible_ma(se.ma)) std::fill(se.ma.begin(), se.ma.end(), 0.0);
  double var;
  if (ns.has_variance) var = ns.variance;
  else if (se.has_variance) var = se.variance;
  else {
    double ss = 0;
    for (double v : yy) ss += v * v;
    var = m.k_exog ? ss : ss / double(m.n);
  }
  var = std::max(var, 1e-10);
  std::vector<double> c;
  for (double v : ns.trend) c.push_back(v);
  for (double v : pexog) c.push_back(v);
  for (double v : ns.ar) c.push_back(v);
  for (double v : ns.ma) c.push_back(v);
  for (double v : se.ar) c.push_back(v);
  for (double v : se.ma) c.push_back(v);
  c.push_back(var);
  if (int(c.size()) != m.k_params()) c.assign(size_t(m.k_params()), 0.0), c.back() = var;
  return c;
}

// ---- derivatives (complex step, as statsmodels) --------------------------------------------------------------------
double get_eps(double x, double s) { return std::pow(EPS, 1.0 / s) * std::max(std::fabs(x), 0.1); }

// Gradient of f(x) = -llf(transform(u)) / n with respect to the unconstrained u.
double objective(const Spec& m, const std::vector<double>& u, std::vector<double>* grad) {
  double f = -re(loglike(m, transform(m, u))) / double(m.n);
  if (grad) {
    grad->assign(u.size(), 0.0);
    for (size_t i = 0; i < u.size(); ++i) {
      double h = 1e-20 * std::max(1.0, std::fabs(u[i]));
      std::vector<cd> z(u.begin(), u.end());
      z[i] += cd(0, h);
      cd l = loglike(m, transform(m, z));
      (*grad)[i] = -l.imag() / h / double(m.n);
    }
  }
  return std::isfinite(f) ? f : INFINITY;
}

struct OptResult { std::vector<double> u; double f; int iterations; bool converged; };
// BFGS with backtracking (Armijo) line search on the unconstrained parameters; deterministic, bounded by maxiter.
OptResult bfgs(const Spec& m, std::vector<double> x, int maxiter) {
  size_t k = x.size();
  std::vector<double> g;
  double f = objective(m, x, &g);
  if (!std::isfinite(f)) fail("arima: the likelihood is not finite at the starting parameters");
  MatrixXd H = MatrixXd::Identity(Eigen::Index(k), Eigen::Index(k));
  OptResult r{x, f, 0, false};
  bool first = true;
  int stall = 0;
  for (int it = 0; it < maxiter; ++it) {
    r.iterations = it + 1;
    VectorXd gv = Eigen::Map<VectorXd>(g.data(), Eigen::Index(k));
    if (gv.cwiseAbs().maxCoeff() < 1e-10) { r.converged = true; break; }
    VectorXd p = -H * gv;
    if (first) p /= std::max(1.0, gv.norm());  // first step of length <= 1, as L-BFGS-B
    if (p.dot(gv) >= 0) { H.setIdentity(); p = -gv; }
    double step = 1.0, slope = p.dot(gv);
    std::vector<double> xn(k), gn;
    double fn = INFINITY;
    bool accepted = false;
    for (int ls = 0; ls < 60; ++ls) {
      for (size_t i = 0; i < k; ++i) xn[i] = x[i] + step * p(Eigen::Index(i));
      fn = objective(m, xn, nullptr);
      if (std::isfinite(fn) && fn <= f + 1e-4 * step * slope) { accepted = true; break; }
      step *= 0.5;
    }
    if (!accepted) { r.converged = gv.cwiseAbs().maxCoeff() < 1e-5; break; }
    objective(m, xn, &gn);
    VectorXd s(static_cast<Eigen::Index>(k)), yv(static_cast<Eigen::Index>(k));
    for (size_t i = 0; i < k; ++i) { s(Eigen::Index(i)) = xn[i] - x[i]; yv(Eigen::Index(i)) = gn[i] - g[i]; }
    double sy = s.dot(yv);
    double fprev = f;
    x = xn; g = gn; f = fn;
    if (sy > 1e-14 * s.norm() * yv.norm()) {
      if (first) { H *= sy / yv.squaredNorm(); first = false; }
      double rho = 1.0 / sy;
      MatrixXd I = MatrixXd::Identity(Eigen::Index(k), Eigen::Index(k));
      H = (I - rho * s * yv.transpose()) * H * (I - rho * yv * s.transpose()) + rho * s * s.transpose();
    }
    // stop when the objective no longer moves (relative change below 1e-14 twice in a row) and the gradient is small
    stall = (fprev - f) <= 1e-14 * std::max({1.0, std::fabs(f), std::fabs(fprev)}) ? stall + 1 : 0;
    if (stall >= 2 && Eigen::Map<VectorXd>(g.data(), Eigen::Index(k)).cwiseAbs().maxCoeff() < 1e-5) { r.converged = true; break; }
  }
  r.u = x;
  r.f = f;
  return r;
}

MatrixXd pinv_sym(const MatrixXd& a) { return pinv_extended(a).pinv; }

// Covariance of the constrained parameters: opg (outer product of per-observation scores), oim (Harvey's information
// matrix from the derivatives of v_t and F_t), approx (complex-step Hessian of the log-likelihood).
MatrixXd arima_cov(const Spec& m, const std::vector<double>& c, const std::string& type) {
  size_t k = c.size();
  if (type == "none") return MatrixXd::Constant(Eigen::Index(k), Eigen::Index(k), NAN);
  if (type == "opg") {
    MatrixXd S(m.n, Eigen::Index(k));
    for (size_t i = 0; i < k; ++i) {
      double h = get_eps(c[i], 2);
      std::vector<cd> z(c.begin(), c.end());
      z[i] += cd(0, h);
      auto fo = kfilter(m, build(m, z));
      for (int t = 0; t < m.n; ++t) S(t, Eigen::Index(i)) = fo.llf_obs[size_t(t)].imag() / h;
    }
    return pinv_sym(S.transpose() * S);
  }
  if (type == "oim") {
    auto base = kfilter(m, build(m, c));
    std::vector<std::vector<double>> dv(k), dF(k);
    for (size_t i = 0; i < k; ++i) {
      double h = get_eps(c[i], 2);
      std::vector<cd> z(c.begin(), c.end());
      z[i] += cd(0, h);
      auto fo = kfilter(m, build(m, z));
      dv[i].resize(size_t(m.n));
      dF[i].resize(size_t(m.n));
      for (int t = 0; t < m.n; ++t) { dv[i][size_t(t)] = fo.v[size_t(t)].imag() / h; dF[i][size_t(t)] = fo.F[size_t(t)].imag() / h; }
    }
    MatrixXd info = MatrixXd::Zero(Eigen::Index(k), Eigen::Index(k));
    for (int t = m.burn; t < m.n; ++t) {
      double F = base.F[size_t(t)];
      if (std::isnan(F)) continue;
      for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < k; ++j)
          info(Eigen::Index(i), Eigen::Index(j)) += 0.5 * (dF[i][size_t(t)] / F) * (dF[j][size_t(t)] / F) + dv[i][size_t(t)] * dv[j][size_t(t)] / F;
    }
    return pinv_sym(info);
  }
  // approx: statsmodels approx_hess_cs
  std::vector<double> h(k);
  for (size_t i = 0; i < k; ++i) h[i] = get_eps(c[i], 3);
  MatrixXd H(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(k));
  for (size_t i = 0; i < k; ++i)
    for (size_t j = i; j < k; ++j) {
      std::vector<cd> zp(c.begin(), c.end()), zm(c.begin(), c.end());
      zp[i] += cd(0, h[i]); zm[i] += cd(0, h[i]);
      zp[j] += h[j]; zm[j] -= h[j];
      double v = (loglike(m, zp) - loglike(m, zm)).imag() / 2.0 / (h[i] * h[j]);
      H(Eigen::Index(i), Eigen::Index(j)) = H(Eigen::Index(j), Eigen::Index(i)) = v;
    }
  return -pinv_sym(H);
}

std::vector<std::string> param_names(const Spec& m, const std::vector<std::string>& exog_names) {
  std::vector<std::string> nm;
  for (int pw : m.trend_powers) nm.push_back(pw == 0 ? "intercept" : pw == 1 ? "drift" : "trend." + std::to_string(pw));
  for (int j = 0; j < m.k_exog; ++j) nm.push_back(j < int(exog_names.size()) ? exog_names[size_t(j)] : "x" + std::to_string(j + 1));
  for (int i = 1; i <= m.p; ++i) nm.push_back("ar.L" + std::to_string(i));
  for (int i = 1; i <= m.q; ++i) nm.push_back("ma.L" + std::to_string(i));
  for (int i = 1; i <= m.P; ++i) nm.push_back("ar.S.L" + std::to_string(i * m.s));
  for (int i = 1; i <= m.Q; ++i) nm.push_back("ma.S.L" + std::to_string(i * m.s));
  nm.push_back("sigma2");
  return nm;
}

Res* arima(const std::vector<double>& y, int64_t p, int64_t d, int64_t q, const Opts& o) {
  const char* op = "arima";
  Spec m;
  if (p < 0 || p > 10 || q < 0 || q > 10 || d < 0 || d > 2) fail("arima: order needs p, q in 0..10 and d in 0..2");
  m.p = int(p); m.d = int(d); m.q = int(q);
  if (const auto* so = o.vec("seasonal_order")) {
    if (so->size() != 4) fail("arima: seasonal_order is [P, D, Q, s]");
    for (double v : *so) if (v != std::floor(v) || v < 0) fail("arima: seasonal_order entries must be non-negative integers");
    m.P = int((*so)[0]); m.D = int((*so)[1]); m.Q = int((*so)[2]); m.s = int((*so)[3]);
    if (m.P > 3 || m.Q > 3 || m.D > 2) fail("arima: seasonal P, Q must be 0..3 and D 0..2");
    if ((m.P || m.D || m.Q) && (m.s < 2 || m.s > 400)) fail("arima: the seasonal period s must be 2..400");
    if (!(m.P || m.D || m.Q)) m.s = 0;
  }
  std::string trend = o.str("trend", "n");
  if (trend == "c") m.trend_powers = {0};
  else if (trend == "t") m.trend_powers = {1};
  else if (trend == "ct") m.trend_powers = {0, 1};
  else if (trend != "n") fail("arima: trend must be n, c, t or ct; got '" + trend + "'");
  m.k_trend = int(m.trend_powers.size());
  m.enforce_stationarity = o.flag("enforce_stationarity", true);
  m.enforce_invertibility = o.flag("enforce_invertibility", true);
  std::string method = o.str("method", "lbfgs");
  if (method != "lbfgs" && method != "bfgs") fail("arima: method must be lbfgs or bfgs (both run BFGS with exact complex-step gradients); got '" + method + "'");
  int64_t maxiter = o.integer("maxiter", 200);
  if (maxiter < 1 || maxiter > 10000) fail("arima: maxiter must be 1..10000");
  std::string cov_type = o.str("cov_type", "opg");
  if (cov_type != "opg" && cov_type != "oim" && cov_type != "approx" && cov_type != "none") fail("arima: cov_type must be opg, oim, approx or none; got '" + cov_type + "'");
  int64_t steps = o.integer("forecast_steps", 0);
  if (steps < 0 || steps > 10000) fail("arima: forecast_steps must be 0..10000");
  double alpha = o.num("alpha", 0.05);
  need_alpha(alpha, op);
  const MatrixXd* exog = o.mat("exog");
  const MatrixXd* exog_future = o.mat("exog_future");
  std::vector<std::string> exog_names;
  if (const auto* en = o.strs("exog_names")) exog_names = *en;
  const std::vector<double>* sp = o.vec("start_params");
  o.finish(op);

  m.y = y;
  m.n = int(y.size());
  int n_missing = 0;
  for (double v : y) {
    if (std::isinf(v)) fail("arima: the series contains infinite values");
    n_missing += std::isnan(v);
  }
  if (m.n < 10 || m.n - n_missing < 5) fail("arima: the series needs at least 10 observations (5 non-missing)");
  if (exog) {
    if (exog->rows() != m.n) fail("arima: exog has " + std::to_string(exog->rows()) + " rows; the series has " + std::to_string(m.n));
    if (!exog->allFinite()) fail("arima: exog contains NaN or infinite values");
    m.exog = *exog;
    m.k_exog = int(exog->cols());
  } else m.exog = MatrixXd(m.n, 0);
  if (!exog_names.empty() && int(exog_names.size()) != m.k_exog) fail("arima: exog_names has " + std::to_string(exog_names.size()) + " names for " + std::to_string(m.k_exog) + " exog columns");
  m.k_ar = m.p; m.k_ma = m.q; m.k_sar = m.P * m.s; m.k_sma = m.Q * m.s;
  m.k_order = std::max(m.k_ar + m.k_sar, m.k_ma + m.k_sma + 1);
  m.k_diff_states = m.d + m.s * m.D;
  m.k_states = m.k_order + m.k_diff_states;
  if (m.k_states > 160) fail("arima: the state dimension " + std::to_string(m.k_states) + " exceeds 160");
  m.burn = m.enforce_stationarity ? m.k_diff_states : m.k_states;
  m.trend_data.resize(m.n, m.k_trend);
  for (int t = 0; t < m.n; ++t) for (int j = 0; j < m.k_trend; ++j) m.trend_data(t, j) = m.trend_at(t, j);
  int kp = m.k_params();
  std::vector<std::string> names = param_names(m, exog_names);

  std::vector<double> c0;
  if (sp) {
    if (int(sp->size()) != kp) {
      std::string all;
      for (auto& s : names) all += (all.empty() ? "" : ", ") + s;
      fail("arima: start_params has " + std::to_string(sp->size()) + " value(s) but this model has " + std::to_string(kp) + " parameters (" + all + ")");
    }
    c0 = *sp;
    if (!(c0.back() > 0)) fail("arima: the starting sigma2 must be positive");
  } else c0 = start_params(m);
  if (m.enforce_stationarity) {
    auto chk = [&](int a, int n, int sgn) {
      if (!n) return;
      std::vector<double> pl = {1};
      for (int i = 0; i < n; ++i) pl.push_back(sgn * c0[size_t(a + i)]);
      if (!is_invertible(pl)) fail("arima: the starting autoregressive parameters are not stationary");
    };
    chk(m.i_ar(), m.p, -1);
    chk(m.i_sar(), m.P, -1);
  }
  if (m.enforce_invertibility) {
    auto chk = [&](int a, int n) {
      if (!n) return;
      std::vector<double> pl = {1};
      for (int i = 0; i < n; ++i) pl.push_back(c0[size_t(a + i)]);
      if (!is_invertible(pl)) fail("arima: the starting moving-average parameters are not invertible");
    };
    chk(m.i_ma(), m.q);
    chk(m.i_sma(), m.Q);
  }
  OptResult opt = bfgs(m, untransform(m, c0), int(maxiter));
  std::vector<double> c = transform(m, opt.u);
  auto fo = kfilter(m, build(m, c));
  double llf = fo.llf;
  std::string status = opt.converged ? "OK" : "NO_CONVERGENCE";
  bool finite = std::isfinite(llf);
  for (double v : c) finite &= std::isfinite(v);
  if (!finite) status = "NUMERICAL_FAILURE";
  MatrixXd cov = finite ? arima_cov(m, c, cov_type) : MatrixXd::Constant(kp, kp, NAN);
  std::vector<double> bse(static_cast<size_t>(kp));
  for (int i = 0; i < kp; ++i) { double v = cov(i, i); bse[size_t(i)] = v >= 0 ? std::sqrt(v) : NAN; }
  int neff = m.n - m.burn;
  auto* r = make<Res>();
  std::vector<double> order = {double(p), double(d), double(q)}, sorder = {double(m.P), double(m.D), double(m.Q), double(m.s)};
  r->str("status", status).str("cov_type", cov_type).integer("nobs", m.n).integer("n_missing", n_missing).vec("order", order).vec("seasonal_order", sorder)
      .str("trend", trend).num("alpha", alpha).flag("converged", opt.converged).integer("n_iter", opt.iterations);
  r->sub("coefficients", coef_table(names, c, bse, false, 0, alpha, true));
  r->vec("params", c).strs("param_names", names).vec("bse", bse);
  r->num("sigma2", c.back()).num("llf", llf).num("aic", -2 * llf + 2 * kp).num("bic", -2 * llf + kp * std::log(double(neff)))
      .num("hqic", -2 * llf + 2 * kp * std::log(std::log(double(neff)))).integer("nobs_effective", neff);
  std::vector<double> resid(size_t(m.n));
  for (int t = 0; t < m.n; ++t) resid[size_t(t)] = fo.v[size_t(t)];
  r->vec("residuals", resid);
  if (steps > 0 && (status == "OK" || status == "NO_CONVERGENCE")) {
    if (m.k_exog) {
      if (!exog_future || exog_future->rows() != steps || exog_future->cols() != m.k_exog)
        fail("arima: forecasting with exog needs exog_future with forecast_steps rows and " + std::to_string(m.k_exog) + " columns");
    } else if (exog_future) fail("arima: exog_future given without exog");
    System<double> sys = build(m, c);
    Eigen::VectorXd a = fo.a;
    MatrixXd P = fo.P;
    MatrixXd RQR = sys.R * sys.R.transpose() * sys.sigma2;
    double z = norm_ppf(1 - alpha / 2);
    std::vector<double> mean, lo, hi;
    for (int h = 0; h < steps; ++h) {
      double dt = m.k_exog ? obs_intercept(m, sys, h, exog_future) : 0.0;
      double mu = dt + sys.Z.dot(a), var = sys.Z.dot(P * sys.Z);
      double sd = std::sqrt(var);
      mean.push_back(mu); lo.push_back(mu - z * sd); hi.push_back(mu + z * sd);
      a = sys.Tm * a;
      a(m.k_diff_states) += state_intercept(m, sys, m.n + h);
      P = sys.Tm * P * sys.Tm.transpose() + RQR;
    }
    auto fc = std::make_shared<Res>();
    fc->vec("mean", mean).vec("lower", lo).vec("upper", hi);
    r->sub("forecast", fc);
  }
  return r;
}

}  // namespace

void register_arima(Api& api) {
  const std::string Rt = "Res", Vt = "Vec";
  api.fn("arima", {Vt, "int", "int", "int"}, Rt,
         "ARIMA(p, d, q) by exact maximum likelihood (pycli sm_arima; statsmodels SARIMAX state space): coefficients {names, estimate, std_err (opg), statistic, "
         "p_value, ci_lower, ci_upper}, params, param_names, bse, sigma2, llf, aic, bic, hqic, nobs_effective, residuals, converged, status (NaN = missing)",
         [](Args& a) { return O(arima(obj<Vec>(a[0]).v, a[1].i, a[2].i, a[3].i, no_opts())); });
  api.fn("arima", {Vt, "int", "int", "int", "Opts"}, Rt,
         "ARIMA with options: `trend` (n, c, t, ct; c is the regression intercept), `seasonal_order` (Vec [P, D, Q, s]), `exog` (Mat), `exog_names`, "
         "`exog_future` (Mat), `forecast_steps` (adds forecast {mean, lower, upper}), `alpha`, `cov_type` (opg, oim, approx, none), `method` (lbfgs/bfgs), "
         "`maxiter` (200), `start_params` (Vec, statsmodels order), `enforce_stationarity`, `enforce_invertibility`",
         [](Args& a) { return O(arima(obj<Vec>(a[0]).v, a[1].i, a[2].i, a[3].i, obj<Opts>(a[4]))); });
}

}  // namespace models
}  // namespace lmpy
