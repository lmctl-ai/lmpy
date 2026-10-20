// Port of scipy.integrate._ivp (scipy 1.16.2, BSD-3-Clause; see ode_solvers.hpp). Function and variable names follow the
// Python originals (rk.py, radau.py, bdf.py, common.py, base.py) so the two can be read side by side.
#include "ode_solvers.hpp"
#include "ode_tableaux.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace lmnum::ode {
namespace {

constexpr double EPS = std::numeric_limits<double>::epsilon();
using cd = std::complex<double>;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;

double min_step_at(double t, double direction) {
  return 10 * std::fabs(std::nextafter(t, direction * INFINITY) - t);
}

// common.select_initial_step
template <class F>
double select_initial_step(F&& fun, double t0, const VectorXd& y0, double t_bound, double max_step, const VectorXd& f0, double direction,
                           int order, double rtol, const VectorXd& atol) {
  if (y0.size() == 0) return INFINITY;
  double interval_length = std::fabs(t_bound - t0);
  if (interval_length == 0.0) return 0.0;
  VectorXd scale = atol.array() + y0.array().abs() * rtol;
  double d0 = rms(y0.cwiseQuotient(scale)), d1 = rms(f0.cwiseQuotient(scale));
  double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
  h0 = std::min(h0, interval_length);
  VectorXd y1 = y0 + (h0 * direction) * f0;
  VectorXd f1 = fun(t0 + h0 * direction, y1);
  double d2 = rms((f1 - f0).cwiseQuotient(scale)) / h0;
  double h1 = (d1 <= 1e-15 && d2 <= 1e-15) ? std::max(1e-6, h0 * 1e-3) : std::pow(0.01 / std::max(d1, d2), 1.0 / (order + 1));
  return std::min({100 * h0, h1, interval_length, max_step});
}

struct ConstantDense : Dense {
  VectorXd value;
  void eval(double, VectorXd& out) const override { out = value; }
};

// ------------------------------------------------------------------------------------------------- explicit Runge-Kutta
struct RkDense : Dense {  // rk.RkDenseOutput
  double h; MatrixXd Q; VectorXd y_old;
  void eval(double at, VectorXd& out) const override {
    double x = (at - t_old) / h;
    Eigen::Index order = Q.cols() - 1;
    VectorXd p(order + 1);
    double acc = 1;
    for (Eigen::Index i = 0; i <= order; ++i) { acc *= x; p(i) = acc; }
    out = h * (Q * p);
    out += y_old;
  }
};

struct Dop853Dense : Dense {  // rk.Dop853DenseOutput
  double h; std::vector<VectorXd> F; VectorXd y_old;
  void eval(double at, VectorXd& out) const override {
    double x = (at - t_old) / h;
    out = VectorXd::Zero(y_old.size());
    int i = 0;
    for (auto it = F.rbegin(); it != F.rend(); ++it, ++i) {
      out += *it;
      if (i % 2 == 0) out *= x; else out *= (1 - x);
    }
    out += y_old;
  }
};

struct Tableau {
  int n_stages, order, error_estimator_order;
  std::vector<std::vector<double>> A;  // A[s][j], j < s
  std::vector<double> B, C, E;
  std::vector<std::vector<double>> P;  // (n_stages + 1) x interpolant order
};

template <size_t R, size_t Cc> std::vector<std::vector<double>> rows(const double (&a)[R][Cc]) {
  std::vector<std::vector<double>> r(R);
  for (size_t i = 0; i < R; ++i) r[i].assign(a[i], a[i] + Cc);
  return r;
}
template <size_t N> std::vector<double> vecof(const double (&a)[N]) { return std::vector<double>(a, a + N); }

const Tableau& rk23() {
  static const Tableau t{3, 3, 2, rows(tab::RK23_A), vecof(tab::RK23_B), vecof(tab::RK23_C), vecof(tab::RK23_E), rows(tab::RK23_P)};
  return t;
}
const Tableau& rk45() {
  static const Tableau t{6, 5, 4, rows(tab::RK45_A), vecof(tab::RK45_B), vecof(tab::RK45_C), vecof(tab::RK45_E), rows(tab::RK45_P)};
  return t;
}
const Tableau& dop853() {
  static const Tableau t = [] {
    Tableau x;
    x.n_stages = tab::DOP_N_STAGES; x.order = 8; x.error_estimator_order = 7;
    x.A = rows(tab::DOP_A);
    x.B = vecof(tab::DOP_B);
    x.C = vecof(tab::DOP_C);
    return x;
  }();
  return t;
}

// sum_j K[j] * a[j] for j < s (numpy np.dot(K[:s].T, a[:s]))
VectorXd combo(const std::vector<VectorXd>& K, const std::vector<double>& a, size_t s) {
  VectorXd r = VectorXd::Zero(K[0].size());
  for (size_t j = 0; j < s; ++j) r += K[j] * a[j];
  return r;
}

class RungeKutta : public Solver {
 public:
  RungeKutta(const Tableau& tb, bool is_dop853, Rhs f, double t0, const VectorXd& y0, double tb_, const Settings& s, const char* nm)
      : Solver(std::move(f), t0, y0, tb_), T(tb), dop(is_dop853), nm_(nm) {
    max_step = s.max_step;
    rtol = valid_rtol(s.rtol);
    atol = s.atol;
    fcur = fun(t, y);
    if (s.first_step <= 0)
      h_abs = select_initial_step([this](double a, const VectorXd& b) { return fun(a, b); }, t, y, t_bound, max_step, fcur, direction, T.error_estimator_order, rtol, atol);
    else h_abs = s.first_step;
    K.assign(dop ? tab::DOP_N_STAGES_EXTENDED : T.n_stages + 1, VectorXd::Zero(n));
    error_exponent = -1.0 / (T.error_estimator_order + 1);
  }
  const char* name() const override { return nm_; }

  // LSODA stiffness detection after Hairer & Wanner, Solving ODEs II, section IV.2: rho = |h| |f(y_new) - f(y6)| / |y_new - y6|
  // estimates |h lambda|; RK45 is stability-limited near 3.3, and step control keeps h lambda just below that, so a
  // threshold of 2.5 on 15 accepted steps (reset after 6 consecutive non-stiff steps) marks the problem stiff. RK45 only.
  bool detect_stiffness = false;
  bool stiff = false;

 protected:
  const Tableau& T;
  bool dop;
  const char* nm_;
  double max_step, rtol, h_abs, error_exponent, h_previous = 0;
  VectorXd atol, fcur, y_old;
  std::vector<VectorXd> K;
  int stiff_hits = 0, nonstiff_run = 0;

  // rk.rk_step
  void rk_step(double t0, const VectorXd& y0, const VectorXd& f, double h, VectorXd& y_new, VectorXd& f_new) {
    K[0] = f;
    for (int s = 1; s < T.n_stages; ++s) {
      VectorXd dy = combo(K, T.A[s], s) * h;
      K[s] = fun(t0 + T.C[s] * h, y0 + dy);
    }
    VectorXd acc = VectorXd::Zero(n);
    for (int j = 0; j < T.n_stages; ++j) acc += K[j] * T.B[j];
    y_new = y0 + h * acc;
    f_new = fun(t0 + h, y_new);
    K[T.n_stages] = f_new;
  }

  double error_norm(double h, const VectorXd& scale) const {
    if (!dop) {
      VectorXd e = VectorXd::Zero(n);
      for (int j = 0; j <= T.n_stages; ++j) e += K[j] * T.E[j];
      return rms((e * h).cwiseQuotient(scale));
    }
    VectorXd err5 = VectorXd::Zero(n), err3 = VectorXd::Zero(n);
    for (int j = 0; j <= T.n_stages; ++j) { err5 += K[j] * tab::DOP_E5[j]; err3 += K[j] * tab::DOP_E3[j]; }
    err5 = err5.cwiseQuotient(scale);
    err3 = err3.cwiseQuotient(scale);
    double e5 = err5.squaredNorm(), e3 = err3.squaredNorm();
    if (e5 == 0 && e3 == 0) return 0.0;
    double denom = e5 + 0.01 * e3;
    return std::fabs(h) * e5 / std::sqrt(denom * double(scale.size()));
  }

  bool step_impl(std::string& message) override {
    const double SAFETY = 0.9, MIN_FACTOR = 0.2, MAX_FACTOR = 10;
    double t0 = t;
    VectorXd y0 = y;
    double min_step = min_step_at(t0, direction);
    double h_abs_ = h_abs > max_step ? max_step : (h_abs < min_step ? min_step : h_abs);
    bool accepted = false, rejected = false;
    double h = 0, t_new = 0;
    VectorXd y_new, f_new;
    while (!accepted) {
      if (h_abs_ < min_step) { message = TOO_SMALL_STEP; return false; }
      h = h_abs_ * direction;
      t_new = t0 + h;
      if (direction * (t_new - t_bound) > 0) t_new = t_bound;
      h = t_new - t0;
      h_abs_ = std::fabs(h);
      rk_step(t0, y0, fcur, h, y_new, f_new);
      VectorXd scale = atol.array() + y0.array().abs().max(y_new.array().abs()) * rtol;
      double en = error_norm(h, scale);
      if (en < 1) {
        double factor = en == 0 ? MAX_FACTOR : std::min(MAX_FACTOR, SAFETY * std::pow(en, error_exponent));
        if (rejected) factor = std::min(1.0, factor);
        h_abs_ *= factor;
        accepted = true;
      } else {
        h_abs_ *= std::max(MIN_FACTOR, SAFETY * std::pow(en, error_exponent));
        rejected = true;
      }
    }
    if (detect_stiffness) {
      // stage 6 of Dormand-Prince is evaluated at t + h (c6 = 1): compare its slope with the slope at the new point
      VectorXd y6 = y0 + combo(K, T.A[5], 5) * h;
      double den = (y_new - y6).norm();
      double hl = den > 0 ? std::fabs(h) * (K[6] - K[5]).norm() / den : 0.0;
      if (hl > 2.5) { nonstiff_run = 0; if (++stiff_hits >= 15) stiff = true; }
      else if (++nonstiff_run >= 6) stiff_hits = 0;
    }
    h_previous = h;
    y_old = y0;
    t = t_new;
    y = y_new;
    h_abs = h_abs_;
    fcur = f_new;
    return true;
  }

  DensePtr dense_impl() override {
    if (!dop) {
      auto d = std::make_shared<RkDense>();
      d->t_old = t_old; d->t = t; d->h = t - t_old; d->y_old = y_old;
      size_t cols = T.P[0].size();
      d->Q = MatrixXd::Zero(n, Eigen::Index(cols));
      for (size_t k = 0; k < cols; ++k)
        for (int j = 0; j <= T.n_stages; ++j) d->Q.col(Eigen::Index(k)) += K[j] * T.P[j][k];
      return d;
    }
    double h = h_previous;
    for (int s = tab::DOP_N_STAGES + 1; s < tab::DOP_N_STAGES_EXTENDED; ++s) {
      VectorXd dy = combo(K, T.A[s], s) * h;
      K[s] = fun(t_old + T.C[s] * h, y_old + dy);
    }
    auto d = std::make_shared<Dop853Dense>();
    d->t_old = t_old; d->t = t; d->h = t - t_old; d->y_old = y_old;
    const VectorXd& f_old = K[0];
    VectorXd delta_y = y - y_old;
    d->F.push_back(delta_y);
    d->F.push_back(h * f_old - delta_y);
    d->F.push_back(2 * delta_y - h * (fcur + f_old));
    for (int r = 0; r < 4; ++r) {
      VectorXd acc = VectorXd::Zero(n);
      for (int j = 0; j < tab::DOP_N_STAGES_EXTENDED; ++j) acc += K[j] * tab::DOP_D[r][j];
      d->F.push_back(h * acc);
    }
    return d;
  }
};

// ------------------------------------------------------------------------------------------------- num_jac (common.py)
const double NUM_JAC_DIFF_REJECT = std::pow(EPS, 0.875);
const double NUM_JAC_DIFF_SMALL = std::pow(EPS, 0.75);
const double NUM_JAC_DIFF_BIG = std::pow(EPS, 0.25);
const double NUM_JAC_MIN_FACTOR = 1000 * EPS;

template <class F>
MatrixXd num_jac(F&& fun, double t, const VectorXd& y, const VectorXd& f, const VectorXd& threshold, VectorXd& factor) {
  Eigen::Index n = y.size();
  if (factor.size() != n) factor = VectorXd::Constant(n, std::sqrt(EPS));
  VectorXd y_scale(n), h(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    double f_sign = f(i) >= 0 ? 1.0 : -1.0;
    y_scale(i) = f_sign * std::max(threshold(i), std::fabs(y(i)));
    h(i) = (y(i) + factor(i) * y_scale(i)) - y(i);
    while (h(i) == 0) { factor(i) *= 10; h(i) = (y(i) + factor(i) * y_scale(i)) - y(i); }
  }
  MatrixXd f_new(n, n);
  VectorXd col(n), yp;
  for (Eigen::Index i = 0; i < n; ++i) { yp = y; yp(i) += h(i); fun(t, yp, col); f_new.col(i) = col; }
  MatrixXd diff = f_new.colwise() - f;
  std::vector<Eigen::Index> max_ind(n);
  VectorXd max_diff(n), scale(n);
  for (Eigen::Index j = 0; j < n; ++j) {
    diff.col(j).cwiseAbs().maxCoeff(&max_ind[j]);
    max_diff(j) = std::fabs(diff(max_ind[j], j));
    scale(j) = std::max(std::fabs(f(max_ind[j])), std::fabs(f_new(max_ind[j], j)));
  }
  for (Eigen::Index j = 0; j < n; ++j) {
    if (!(max_diff(j) < NUM_JAC_DIFF_REJECT * scale(j))) continue;
    double new_factor = 10 * factor(j);
    double h_new = (y(j) + new_factor * y_scale(j)) - y(j);
    yp = y; yp(j) += h_new;
    fun(t, yp, col);
    VectorXd diff_new = col - f;
    Eigen::Index mi; diff_new.cwiseAbs().maxCoeff(&mi);
    double max_diff_new = std::fabs(diff_new(mi));
    double scale_new = std::max(std::fabs(f(mi)), std::fabs(col(mi)));
    if (max_diff(j) * scale_new < max_diff_new * scale(j)) {
      factor(j) = new_factor; h(j) = h_new; diff.col(j) = diff_new; scale(j) = scale_new; max_diff(j) = max_diff_new;
    }
  }
  for (Eigen::Index j = 0; j < n; ++j) {
    diff.col(j) /= h(j);
    if (max_diff(j) < NUM_JAC_DIFF_SMALL * scale(j)) factor(j) *= 10;
    else if (max_diff(j) > NUM_JAC_DIFF_BIG * scale(j)) factor(j) *= 0.1;
    factor(j) = std::max(factor(j), NUM_JAC_MIN_FACTOR);
  }
  return diff;
}

// ------------------------------------------------------------------------------------------------- Radau IIA (radau.py)
struct RadauDense : Dense {
  MatrixXd Q; VectorXd y_old; double h;
  void eval(double at, VectorXd& out) const override {
    double x = (at - t_old) / h;
    VectorXd p(Q.cols()); double acc = 1;
    for (Eigen::Index i = 0; i < Q.cols(); ++i) { acc *= x; p(i) = acc; }
    out = Q * p;
    out += y_old;
  }
};

class Radau : public Solver {
 public:
  Radau(Rhs f, double t0, const VectorXd& y0, double tb, const Settings& s) : Solver(std::move(f), t0, y0, tb) {
    max_step = s.max_step;
    rtol = valid_rtol(s.rtol);
    atol = s.atol;
    fcur = fun(t, y);
    if (s.first_step <= 0)
      h_abs = select_initial_step([this](double a, const VectorXd& b) { return fun(a, b); }, t, y, t_bound, max_step, fcur, direction, 3, rtol, atol);
    else h_abs = s.first_step;
    newton_tol = std::max(10 * EPS / rtol, std::min(0.03, std::sqrt(rtol)));
    J = jac(t, y, fcur);
  }
  const char* name() const override { return "Radau"; }

 private:
  static constexpr int NEWTON_MAXITER = 6;
  double max_step, rtol, h_abs, newton_tol;
  double h_abs_old = NAN, error_norm_old = NAN;  // NaN = None
  VectorXd atol, fcur, jac_factor;
  MatrixXd J, Z;
  std::shared_ptr<RadauDense> sol;
  bool current_jac = true;
  std::unique_ptr<Eigen::PartialPivLU<MatrixXd>> LU_real;
  std::unique_ptr<Eigen::PartialPivLU<MatrixXcd>> LU_complex;

  MatrixXd jac(double tt, const VectorXd& yy, const VectorXd& ff) {
    ++njev;
    return num_jac([this](double a, const VectorXd& b, VectorXd& o) { fun_uncounted(a, b, o); }, tt, yy, ff, atol, jac_factor);
  }

  static double predict_factor(double h_abs, double h_abs_old, double error_norm, double error_norm_old) {
    double multiplier = (std::isnan(error_norm_old) || std::isnan(h_abs_old) || error_norm == 0)
                            ? 1.0 : h_abs / h_abs_old * std::pow(error_norm_old / error_norm, 0.25);
    return std::min(1.0, multiplier) * std::pow(error_norm, -0.25);
  }

  bool solve_collocation(double t0, const VectorXd& y0, double h, const MatrixXd& Z0, const VectorXd& scale, int& n_iter, MatrixXd& Zout, double& rate) {
    double M_real = tab::RADAU_MU_REAL / h;
    cd M_complex = cd(tab::RADAU_MU_COMPLEX_RE, tab::RADAU_MU_COMPLEX_IM) / h;
    Eigen::Matrix3d TI, TT;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { TI(i, j) = tab::RADAU_TI[i][j]; TT(i, j) = tab::RADAU_T[i][j]; }
    MatrixXd W = TI * Z0;  // 3 x n
    MatrixXd Zc = Z0;
    MatrixXd F(3, n), dW(3, n);
    double dW_norm_old = NAN;
    bool converged = false;
    rate = NAN;
    int k = 0;
    for (k = 0; k < NEWTON_MAXITER; ++k) {
      bool finite = true;
      for (int i = 0; i < 3; ++i) {
        VectorXd yi = y0 + Zc.row(i).transpose();
        F.row(i) = fun(t0 + h * tab::RADAU_C[i], yi).transpose();
        finite = finite && F.row(i).allFinite();
      }
      if (!finite) break;
      VectorXd f_real = F.row(0).transpose() * TI(0, 0) + F.row(1).transpose() * TI(0, 1) + F.row(2).transpose() * TI(0, 2);
      f_real -= M_real * W.row(0).transpose();
      VectorXcd f_complex(n);
      for (int c = 0; c < n; ++c) {
        cd acc = F(0, c) * cd(TI(1, 0), TI(2, 0)) + F(1, c) * cd(TI(1, 1), TI(2, 1)) + F(2, c) * cd(TI(1, 2), TI(2, 2));
        f_complex(c) = acc - M_complex * cd(W(1, c), W(2, c));
      }
      VectorXd dW_real = LU_real->solve(f_real);
      VectorXcd dW_complex = LU_complex->solve(f_complex);
      dW.row(0) = dW_real.transpose();
      dW.row(1) = dW_complex.real().transpose();
      dW.row(2) = dW_complex.imag().transpose();
      double ss = 0;
      for (int i = 0; i < 3; ++i) for (int c = 0; c < n; ++c) { double v = dW(i, c) / scale(c); ss += v * v; }
      double dW_norm = std::sqrt(ss) / std::sqrt(3.0 * n);
      if (!std::isnan(dW_norm_old)) rate = dW_norm / dW_norm_old;
      if (!std::isnan(rate) && (rate >= 1 || std::pow(rate, NEWTON_MAXITER - k) / (1 - rate) * dW_norm > newton_tol)) break;
      W += dW;
      Zc = TT * W;
      if (dW_norm == 0 || (!std::isnan(rate) && rate / (1 - rate) * dW_norm < newton_tol)) { converged = true; break; }
      dW_norm_old = dW_norm;
    }
    n_iter = std::min(k, NEWTON_MAXITER - 1) + 1;
    Zout = Zc;
    return converged;
  }

  void factor_lu(double h) {
    MatrixXd I = MatrixXd::Identity(n, n);
    ++nlu; LU_real = std::make_unique<Eigen::PartialPivLU<MatrixXd>>(MatrixXd(tab::RADAU_MU_REAL / h * I - J));
    cd mc = cd(tab::RADAU_MU_COMPLEX_RE, tab::RADAU_MU_COMPLEX_IM) / h;
    ++nlu; LU_complex = std::make_unique<Eigen::PartialPivLU<MatrixXcd>>(MatrixXcd(mc * I.cast<cd>() - J.cast<cd>()));
  }

  bool step_impl(std::string& message) override {
    const double MIN_FACTOR = 0.2, MAX_FACTOR = 10;
    double t0 = t;
    VectorXd y0 = y, f0 = fcur;
    double min_step = min_step_at(t0, direction);
    double h_abs_, h_abs_old_, error_norm_old_;
    if (h_abs > max_step) { h_abs_ = max_step; h_abs_old_ = NAN; error_norm_old_ = NAN; }
    else if (h_abs < min_step) { h_abs_ = min_step; h_abs_old_ = NAN; error_norm_old_ = NAN; }
    else { h_abs_ = h_abs; h_abs_old_ = h_abs_old; error_norm_old_ = error_norm_old; }
    bool rejected = false, accepted = false;
    double h = 0, t_new = 0, error_norm = 0, safety = 0, rate = NAN;
    int n_iter = 0;
    VectorXd y_new;
    MatrixXd Zn;
    while (!accepted) {
      if (h_abs_ < min_step) { message = TOO_SMALL_STEP; return false; }
      h = h_abs_ * direction;
      t_new = t0 + h;
      if (direction * (t_new - t_bound) > 0) t_new = t_bound;
      h = t_new - t0;
      h_abs_ = std::fabs(h);
      MatrixXd Z0(3, n);
      if (!sol) Z0.setZero();
      else for (int i = 0; i < 3; ++i) Z0.row(i) = ((*sol)(t0 + h * tab::RADAU_C[i]) - y0).transpose();
      VectorXd scale = atol.array() + y0.array().abs() * rtol;
      bool converged = false;
      while (!converged) {
        if (!LU_real || !LU_complex) factor_lu(h);
        converged = solve_collocation(t0, y0, h, Z0, scale, n_iter, Zn, rate);
        if (!converged) {
          if (current_jac) break;
          J = jac(t0, y0, f0);
          current_jac = true;
          LU_real.reset(); LU_complex.reset();
        }
      }
      if (!converged) { h_abs_ *= 0.5; LU_real.reset(); LU_complex.reset(); continue; }
      y_new = y0 + Zn.row(2).transpose();
      VectorXd ZE = (Zn.row(0).transpose() * tab::RADAU_E[0] + Zn.row(1).transpose() * tab::RADAU_E[1] + Zn.row(2).transpose() * tab::RADAU_E[2]) / h;
      VectorXd error = LU_real->solve(VectorXd(f0 + ZE));
      scale = atol.array() + y0.array().abs().max(y_new.array().abs()) * rtol;
      error_norm = rms(error.cwiseQuotient(scale));
      safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter);
      if (rejected && error_norm > 1) {
        error = LU_real->solve(VectorXd(fun(t0, y0 + error) + ZE));
        error_norm = rms(error.cwiseQuotient(scale));
      }
      if (error_norm > 1) {
        double factor = predict_factor(h_abs_, h_abs_old_, error_norm, error_norm_old_);
        h_abs_ *= std::max(MIN_FACTOR, safety * factor);
        LU_real.reset(); LU_complex.reset();
        rejected = true;
      } else accepted = true;
    }
    bool recompute_jac = n_iter > 2 && rate > 1e-3;
    double factor = predict_factor(h_abs_, h_abs_old_, error_norm, error_norm_old_);
    factor = std::min(MAX_FACTOR, safety * factor);
    if (!recompute_jac && factor < 1.2) factor = 1;
    else { LU_real.reset(); LU_complex.reset(); }
    VectorXd f_new = fun(t_new, y_new);
    if (recompute_jac) { J = jac(t_new, y_new, f_new); current_jac = true; }
    else current_jac = false;
    h_abs_old = h_abs;
    error_norm_old = error_norm;
    h_abs = h_abs_ * factor;
    t = t_new;
    y = y_new;
    fcur = f_new;
    Z = Zn;
    auto d = std::make_shared<RadauDense>();
    d->t_old = t0; d->t = t_new; d->h = t_new - t0; d->y_old = y0;
    d->Q = MatrixXd::Zero(n, 3);
    for (int k = 0; k < 3; ++k)
      for (int i = 0; i < 3; ++i) d->Q.col(k) += Z.row(i).transpose() * tab::RADAU_P[i][k];
    sol = d;
    return true;
  }

  DensePtr dense_impl() override { return sol; }
};

// ------------------------------------------------------------------------------------------------- BDF / NDF (bdf.py)
constexpr int MAX_ORDER = 5;

MatrixXd compute_R(int order, double factor) {
  MatrixXd M = MatrixXd::Zero(order + 1, order + 1);
  for (int i = 1; i <= order; ++i)
    for (int j = 1; j <= order; ++j) M(i, j) = (double(i) - 1 - factor * double(j)) / double(i);
  M.row(0).setOnes();
  for (int i = 1; i <= order; ++i) M.row(i) = M.row(i).cwiseProduct(M.row(i - 1));
  return M;
}

void change_D(MatrixXd& D, int order, double factor) {
  MatrixXd R = compute_R(order, factor), U = compute_R(order, 1);
  MatrixXd RU = R * U;
  D.topRows(order + 1) = RU.transpose() * D.topRows(order + 1);
}

struct BdfDense : Dense {
  int order; VectorXd t_shift, denom; MatrixXd D;  // D: (order + 1) x n
  void eval(double at, VectorXd& out) const override {
    VectorXd p(order); double acc = 1;
    for (int i = 0; i < order; ++i) { acc *= (at - t_shift(i)) / denom(i); p(i) = acc; }
    out = D.bottomRows(order).transpose() * p;
    out += D.row(0).transpose();
  }
};

class Bdf : public Solver {
 public:
  Bdf(Rhs f, double t0, const VectorXd& y0, double tb, const Settings& s) : Solver(std::move(f), t0, y0, tb) {
    max_step = s.max_step;
    rtol = valid_rtol(s.rtol);
    atol = s.atol;
    VectorXd f0 = fun(t, y);
    if (s.first_step <= 0)
      h_abs = select_initial_step([this](double a, const VectorXd& b) { return fun(a, b); }, t, y, t_bound, max_step, f0, direction, 1, rtol, atol);
    else h_abs = s.first_step;
    newton_tol = std::max(10 * EPS / rtol, std::min(0.03, std::sqrt(rtol)));
    J = jac(t, y);
    const double kappa[6] = {0, -0.185, -1.0 / 9, -0.0823, -0.0415, 0};
    gamma[0] = 0;
    for (int i = 1; i <= MAX_ORDER; ++i) gamma[i] = gamma[i - 1] + 1.0 / i;
    for (int i = 0; i <= MAX_ORDER; ++i) { alpha[i] = (1 - kappa[i]) * gamma[i]; error_const[i] = kappa[i] * gamma[i] + 1.0 / (i + 1); }
    D = MatrixXd::Zero(MAX_ORDER + 3, n);
    D.row(0) = y.transpose();
    D.row(1) = (f0 * h_abs * direction).transpose();
  }
  const char* name() const override { return "BDF"; }
  bool alt_segment() const override { return true; }

 private:
  static constexpr int NEWTON_MAXITER = 4;
  double max_step, rtol, h_abs, newton_tol;
  double gamma[MAX_ORDER + 1], alpha[MAX_ORDER + 1], error_const[MAX_ORDER + 1];
  VectorXd atol, jac_factor;
  MatrixXd J, D;
  int order = 1, n_equal_steps = 0;
  std::unique_ptr<Eigen::PartialPivLU<MatrixXd>> LU;

  MatrixXd jac(double tt, const VectorXd& yy) {
    ++njev;
    VectorXd f(n);
    fun_uncounted(tt, yy, f);
    return num_jac([this](double a, const VectorXd& b, VectorXd& o) { fun_uncounted(a, b, o); }, tt, yy, f, atol, jac_factor);
  }

  bool solve_bdf_system(double t_new, const VectorXd& y_predict, double c, const VectorXd& psi, const VectorXd& scale, int& n_iter, VectorXd& y_out, VectorXd& d) {
    d = VectorXd::Zero(n);
    VectorXd yv = y_predict;
    double dy_norm_old = NAN;
    bool converged = false;
    int k = 0;
    for (k = 0; k < NEWTON_MAXITER; ++k) {
      VectorXd f = fun(t_new, yv);
      if (!f.allFinite()) break;
      VectorXd dy = LU->solve(VectorXd(c * f - psi - d));
      double dy_norm = rms(dy.cwiseQuotient(scale));
      double rate = std::isnan(dy_norm_old) ? NAN : dy_norm / dy_norm_old;
      if (!std::isnan(rate) && (rate >= 1 || std::pow(rate, NEWTON_MAXITER - k) / (1 - rate) * dy_norm > newton_tol)) break;
      yv += dy;
      d += dy;
      if (dy_norm == 0 || (!std::isnan(rate) && rate / (1 - rate) * dy_norm < newton_tol)) { converged = true; break; }
      dy_norm_old = dy_norm;
    }
    n_iter = std::min(k, NEWTON_MAXITER - 1) + 1;
    y_out = yv;
    return converged;
  }

  bool step_impl(std::string& message) override {
    const double MIN_FACTOR = 0.2, MAX_FACTOR = 10;
    double t0 = t;
    double min_step = min_step_at(t0, direction);
    double h_abs_;
    if (h_abs > max_step) { h_abs_ = max_step; change_D(D, order, max_step / h_abs); n_equal_steps = 0; }
    else if (h_abs < min_step) { h_abs_ = min_step; change_D(D, order, min_step / h_abs); n_equal_steps = 0; }
    else h_abs_ = h_abs;
    bool current_jac = false;  // the Jacobian is always numerical here (scipy: current_jac = self.jac is None, False)
    bool accepted = false;
    double t_new = 0, error_norm = 0, safety = 0;
    VectorXd y_new, d, scale;
    while (!accepted) {
      if (h_abs_ < min_step) { message = TOO_SMALL_STEP; return false; }
      double h = h_abs_ * direction;
      t_new = t0 + h;
      if (direction * (t_new - t_bound) > 0) {
        t_new = t_bound;
        change_D(D, order, std::fabs(t_new - t0) / h_abs_);
        n_equal_steps = 0;
        LU.reset();
      }
      h = t_new - t0;
      h_abs_ = std::fabs(h);
      VectorXd y_predict = D.topRows(order + 1).colwise().sum().transpose();
      scale = atol.array() + rtol * y_predict.array().abs();
      VectorXd psi = VectorXd::Zero(n);
      for (int i = 1; i <= order; ++i) psi += D.row(i).transpose() * gamma[i];
      psi /= alpha[order];
      bool converged = false;
      double c = h / alpha[order];
      int n_iter = 0;
      while (!converged) {
        if (!LU) { ++nlu; LU = std::make_unique<Eigen::PartialPivLU<MatrixXd>>(MatrixXd(MatrixXd::Identity(n, n) - c * J)); }
        converged = solve_bdf_system(t_new, y_predict, c, psi, scale, n_iter, y_new, d);
        if (!converged) {
          if (current_jac) break;
          J = jac(t_new, y_predict);
          LU.reset();
          current_jac = true;
        }
      }
      if (!converged) {
        double factor = 0.5;
        h_abs_ *= factor;
        change_D(D, order, factor);
        n_equal_steps = 0;
        LU.reset();
        continue;
      }
      safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter);
      scale = atol.array() + rtol * y_new.array().abs();
      VectorXd error = error_const[order] * d;
      error_norm = rms(error.cwiseQuotient(scale));
      if (error_norm > 1) {
        double factor = std::max(MIN_FACTOR, safety * std::pow(error_norm, -1.0 / (order + 1)));
        h_abs_ *= factor;
        change_D(D, order, factor);
        n_equal_steps = 0;
      } else accepted = true;
    }
    n_equal_steps += 1;
    t = t_new;
    y = y_new;
    h_abs = h_abs_;
    D.row(order + 2) = d.transpose() - D.row(order + 1);
    D.row(order + 1) = d.transpose();
    for (int i = order; i >= 0; --i) D.row(i) += D.row(i + 1);
    if (n_equal_steps < order + 1) return true;
    double error_m_norm = INFINITY, error_p_norm = INFINITY;
    if (order > 1) error_m_norm = rms(VectorXd(error_const[order - 1] * D.row(order).transpose()).cwiseQuotient(scale));
    if (order < MAX_ORDER) error_p_norm = rms(VectorXd(error_const[order + 1] * D.row(order + 2).transpose()).cwiseQuotient(scale));
    double norms[3] = {error_m_norm, error_norm, error_p_norm};
    double factors[3];
    int best = 0;
    for (int i = 0; i < 3; ++i) {
      factors[i] = std::pow(norms[i], -1.0 / (order + i));
      if (factors[i] > factors[best]) best = i;
    }
    order += best - 1;
    double factor = std::min(MAX_FACTOR, safety * factors[best]);
    h_abs *= factor;
    change_D(D, order, factor);
    n_equal_steps = 0;
    LU.reset();
    return true;
  }

  DensePtr dense_impl() override {
    auto d = std::make_shared<BdfDense>();
    double h = h_abs * direction;
    d->t_old = t_old; d->t = t; d->order = order;
    d->t_shift.resize(order); d->denom.resize(order);
    for (int i = 0; i < order; ++i) { d->t_shift(i) = t - h * i; d->denom(i) = h * (1 + i); }
    d->D = D.topRows(order + 1);
    return d;
  }
};

// ------------------------------------------------------------------------------------------------- LSODA substitute
// Automatic method switching in the spirit of LSODA (Petzold 1983): integrate with the non-stiff Dormand-Prince pair until
// the stiffness test (RK45 above) fires on 15 steps, then continue with BDF from the current state. No switch back.
class Switching : public Solver {
 public:
  Switching(Rhs f, double t0, const VectorXd& y0, double tb, const Settings& s) : Solver(f, t0, y0, tb), settings(s) {
    auto rk = std::make_unique<RungeKutta>(rk45(), false, f, t0, y0, tb, s, "RK45");
    rk->detect_stiffness = true;
    rkp = rk.get();
    inner = std::move(rk);
    sync();
  }
  const char* name() const override { return "LSODA"; }
  bool alt_segment() const override { return true; }
  bool switched = false;
  double switched_at = NAN;
  long rk_steps = 0;
  double switch_time() const override { return switched_at; }
  long switch_steps() const override { return rk_steps; }

 private:
  Settings settings;
  std::unique_ptr<Solver> inner;
  RungeKutta* rkp = nullptr;
  long base_nfev = 0, base_njev = 0, base_nlu = 0;
  void sync() { nfev = base_nfev + inner->nfev; njev = base_njev + inner->njev; nlu = base_nlu + inner->nlu; }
  bool step_impl(std::string& message) override {
    if (rkp && rkp->stiff) {
      base_nfev += inner->nfev; base_njev += inner->njev; base_nlu += inner->nlu;
      Settings s = settings; s.first_step = 0;
      inner = std::make_unique<Bdf>(rhs_, t, y, t_bound, s);
      rkp = nullptr;
      switched = true;
      switched_at = t;
    }
    message = inner->step();
    if (rkp) ++rk_steps;
    sync();
    if (inner->status == "failed") return false;
    t = inner->t;
    y = inner->y;
    return true;
  }
  DensePtr dense_impl() override { return inner->dense_output(); }
};

}  // namespace

// ------------------------------------------------------------------------------------------------- public pieces
double rms(const VectorXd& x) { return x.size() ? std::sqrt(x.squaredNorm()) / std::sqrt(double(x.size())) : 0.0; }
double valid_rtol(double rtol) { return rtol < 100 * EPS ? 100 * EPS : rtol; }

Solver::Solver(Rhs f, double t0, const VectorXd& y0, double tb) : t(t0), t_bound(tb), y(y0), n(int(y0.size())), rhs_(std::move(f)) {
  direction = tb != t0 ? (tb > t0 ? 1.0 : -1.0) : 1.0;
}

std::string Solver::step() {
  if (status != "running") throw std::logic_error("Attempt to step on a failed or finished solver.");
  std::string message;
  if (n == 0 || t == t_bound) {
    t_old = t; has_old = true;
    t = t_bound;
    status = "finished";
    return message;
  }
  double t0 = t;
  if (!step_impl(message)) { status = "failed"; return message; }
  t_old = t0; has_old = true;
  if (direction * (t - t_bound) >= 0) status = "finished";
  return message;
}

DensePtr Solver::dense_output() {
  if (!has_old) throw std::logic_error("Dense output is available after a successful step was made.");
  if (n == 0 || t == t_old) {
    auto c = std::make_shared<ConstantDense>();
    c->t_old = t_old; c->t = t; c->value = y;
    return c;
  }
  return dense_impl();
}

std::unique_ptr<Solver> make_solver(const std::string& method, Rhs f, double t0, const VectorXd& y0, double t_bound, const Settings& s) {
  if (method == "RK45") return std::make_unique<RungeKutta>(rk45(), false, std::move(f), t0, y0, t_bound, s, "RK45");
  if (method == "RK23") return std::make_unique<RungeKutta>(rk23(), false, std::move(f), t0, y0, t_bound, s, "RK23");
  if (method == "DOP853") return std::make_unique<RungeKutta>(dop853(), true, std::move(f), t0, y0, t_bound, s, "DOP853");
  if (method == "Radau") return std::make_unique<Radau>(std::move(f), t0, y0, t_bound, s);
  if (method == "BDF") return std::make_unique<Bdf>(std::move(f), t0, y0, t_bound, s);
  if (method == "LSODA") return std::make_unique<Switching>(std::move(f), t0, y0, t_bound, s);
  throw std::invalid_argument("unknown method");
}

void Solution::eval(double at, VectorXd& out) const {
  size_t nseg = pieces.size();
  bool ascending = ts.back() >= ts.front();
  // searchsorted on the ascending breakpoints with scipy's side rule
  bool right = ascending ? alt_segment : !alt_segment;
  std::vector<double> sorted = ts;
  if (!ascending) std::reverse(sorted.begin(), sorted.end());
  size_t ind = right ? size_t(std::upper_bound(sorted.begin(), sorted.end(), at) - sorted.begin())
                     : size_t(std::lower_bound(sorted.begin(), sorted.end(), at) - sorted.begin());
  long seg = std::min<long>(std::max<long>(long(ind) - 1, 0), long(nseg) - 1);
  if (!ascending) seg = long(nseg) - 1 - seg;
  pieces[size_t(seg)]->eval(at, out);
}

double brentq(const std::function<double(double)>& f, double xa, double xb, double xtol, double rtol, int maxiter, bool& converged) {
  double xpre = xa, xcur = xb, xblk = 0, fblk = 0, spre = 0, scur = 0;
  double fpre = f(xpre), fcur = f(xcur);
  converged = true;
  if (fpre == 0) return xpre;
  if (fcur == 0) return xcur;
  if (std::signbit(fpre) == std::signbit(fcur)) { converged = false; return std::fabs(fpre) < std::fabs(fcur) ? xpre : xcur; }
  for (int i = 0; i < maxiter; ++i) {
    if (fpre != 0 && fcur != 0 && std::signbit(fpre) != std::signbit(fcur)) {
      xblk = xpre; fblk = fpre;
      spre = scur = xcur - xpre;
    }
    if (std::fabs(fblk) < std::fabs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    double delta = (xtol + rtol * std::fabs(xcur)) / 2;
    double sbis = (xblk - xcur) / 2;
    if (fcur == 0 || std::fabs(sbis) < delta) return xcur;
    if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
      double stry;
      if (xpre == xblk) stry = -fcur * (xcur - xpre) / (fcur - fpre);  // interpolate
      else {                                                           // extrapolate
        double dpre = (fpre - fcur) / (xpre - xcur);
        double dblk = (fblk - fcur) / (xblk - xcur);
        stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
      }
      if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) { spre = scur; scur = stry; }
      else { spre = sbis; scur = sbis; }
    } else { spre = sbis; scur = sbis; }
    xpre = xcur; fpre = fcur;
    if (std::fabs(scur) > delta) xcur += scur;
    else xcur += (sbis > 0 ? delta : -delta);
    fcur = f(xcur);
  }
  converged = false;
  return xcur;
}

}  // namespace lmnum::ode
