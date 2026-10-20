// Shared pieces of the statistical-model family (models.cpp: OLS/GLM, models_tsa.cpp: ACF/PACF and unit-root tests,
// models_arima.cpp: SARIMAX). Reference distributions come from Boost.Math with a double-only policy (no long double
// promotion, NaN instead of exceptions), which is what scipy.stats uses underneath for t, F and chi-square.
#pragma once
#include "host.hpp"
#include <Eigen/SVD>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace lmnum {
namespace models {

using namespace boost::math::policies;
using Pol = policy<promote_double<false>, domain_error<ignore_error>, overflow_error<ignore_error>, evaluation_error<ignore_error>,
                   pole_error<ignore_error>, rounding_error<ignore_error>>;

inline double norm_cdf(double x) { return std::isnan(x) ? NAN : 0.5 * std::erfc(-x / std::sqrt(2.0)); }
inline double norm_sf(double x) { return std::isnan(x) ? NAN : 0.5 * std::erfc(x / std::sqrt(2.0)); }
inline double norm_ppf(double p) {
  if (!(p > 0 && p < 1)) return p == 0 ? -INFINITY : p == 1 ? INFINITY : NAN;
  return boost::math::quantile(boost::math::normal_distribution<double, Pol>(0, 1), p);
}
inline double t_sf(double x, double df) {
  if (std::isnan(x) || !(df > 0)) return NAN;
  if (std::isinf(x)) return x > 0 ? 0.0 : 1.0;
  return boost::math::cdf(boost::math::complement(boost::math::students_t_distribution<double, Pol>(df), x));
}
inline double t_ppf(double p, double df) {
  if (!(df > 0) || !(p > 0 && p < 1)) return NAN;
  return boost::math::quantile(boost::math::students_t_distribution<double, Pol>(df), p);
}
inline double f_sf(double x, double d1, double d2) {
  if (std::isnan(x) || !(d1 > 0) || !(d2 > 0)) return NAN;
  if (x <= 0) return 1.0;
  if (std::isinf(x)) return 0.0;
  return boost::math::cdf(boost::math::complement(boost::math::fisher_f_distribution<double, Pol>(d1, d2), x));
}
inline double chi2_sf(double x, double df) {
  if (std::isnan(x) || !(df > 0)) return NAN;
  if (x <= 0) return 1.0;
  if (std::isinf(x)) return 0.0;
  return boost::math::cdf(boost::math::complement(boost::math::chi_squared_distribution<double, Pol>(df), x));
}
inline double lgam(double x) { return boost::math::lgamma(x, Pol()); }

// Coefficient table as pycli returns it: names, estimate, std_err, statistic, p_value, ci_lower, ci_upper. Undefined
// uncertainty (rank deficiency, no residual degrees of freedom) is NaN (JSON null; pycli wrote the string "NA").
inline std::shared_ptr<Res> coef_table(const std::vector<std::string>& names, const std::vector<double>& est, const std::vector<double>& se,
                                       bool use_t, double df, double alpha, bool usable) {
  size_t k = names.size();
  std::vector<double> stat(k, NAN), p(k, NAN), lo(k, NAN), hi(k, NAN), s(k, NAN);
  if (usable) {
    double q = use_t ? t_ppf(1 - alpha / 2, df) : norm_ppf(1 - alpha / 2);
    for (size_t i = 0; i < k; ++i) {
      s[i] = se[i];
      if (!std::isfinite(se[i])) { s[i] = NAN; continue; }
      stat[i] = est[i] / se[i];
      p[i] = use_t ? 2 * t_sf(std::fabs(stat[i]), df) : 2 * norm_sf(std::fabs(stat[i]));
      lo[i] = est[i] - q * se[i];
      hi[i] = est[i] + q * se[i];
    }
  }
  auto t = std::make_shared<Res>();
  t->strs("names", names).vec("estimate", est).vec("std_err", s).vec("statistic", stat).vec("p_value", p).vec("ci_lower", lo).vec("ci_upper", hi);
  return t;
}

inline void need_alpha(double alpha, const char* op) {
  if (!(alpha > 0 && alpha < 1)) fail(std::string(op) + ": alpha must lie in (0, 1); got " + jnum(alpha));
}

inline int matrix_rank(const Eigen::MatrixXd& x) {
  if (x.size() == 0) return 0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(x);
  Eigen::VectorXd s = svd.singularValues();
  double tol = s(0) * double(std::max(x.rows(), x.cols())) * std::numeric_limits<double>::epsilon();
  int r = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i) r += s(i) > tol;
  return r;
}

// statsmodels ModelData._handle_constant: an explicit constant column, or an implicit one (a set of dummies spanning it).
inline void detect_constant(const Eigen::MatrixXd& x, int& k_constant, int& const_idx) {
  const_idx = -1;
  std::vector<int> idx;
  for (Eigen::Index j = 0; j < x.cols(); ++j)
    if (x.col(j).maxCoeff() == x.col(j).minCoeff()) idx.push_back(int(j));
  bool check_implicit = false;
  k_constant = int(idx.size());
  if (idx.size() == 1) {
    if (x.col(idx[0]).mean() != 0) const_idx = idx[0];
    else check_implicit = true;
  } else if (idx.size() > 1) {
    bool found = false;
    for (int j : idx) if (x.col(j).mean() == 1.0) { k_constant = 1; const_idx = j; found = true; break; }
    if (!found) {
      for (int j : idx) if (x.col(j).mean() != 0) { k_constant = 1; const_idx = j; found = true; break; }
      if (!found) check_implicit = true;
    }
  } else check_implicit = true;
  if (check_implicit) {
    Eigen::MatrixXd aug(x.rows(), x.cols() + 1);
    aug << Eigen::VectorXd::Ones(x.rows()), x;
    k_constant = matrix_rank(aug) == matrix_rank(x) ? 1 : 0;
    const_idx = -1;
  }
}

// Pseudo-inverse pieces of A (statsmodels pinv_extended, rcond 1e-15): pinv = V S^-1 U^T, normalized covariance V S^-2 V^T.
struct Pinv { Eigen::MatrixXd pinv, normcov; Eigen::VectorXd sv; };
inline Pinv pinv_extended(const Eigen::MatrixXd& a) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Pinv p;
  p.sv = svd.singularValues();
  double cut = 1e-15 * (p.sv.size() ? p.sv(0) : 0.0);
  Eigen::VectorXd inv(p.sv.size());
  for (Eigen::Index i = 0; i < p.sv.size(); ++i) inv(i) = p.sv(i) > cut ? 1.0 / p.sv(i) : 0.0;
  p.pinv = svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
  p.normcov = p.pinv * p.pinv.transpose();
  return p;
}

// Registration of the time-series and ARIMA parts (models_tsa.cpp, models_arima.cpp); called from register_models.
void register_tsa(Api& api);
void register_arima(Api& api);

}  // namespace models
}  // namespace lmnum
