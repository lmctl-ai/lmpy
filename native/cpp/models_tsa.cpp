// Time-series statistics after statsmodels 0.14 (BSD-3-Clause): acf, pacf, adfuller, kpss and the Ljung-Box test.
// pycli ops covered: sm_tsa_acf, sm_tsa_tools. As in pycli, every series is first divided by its largest magnitude
// (correlations and test statistics are scale-free; this keeps products away from overflow) and the scale is reported.
// The MacKinnon (1994, 2010) response-surface coefficients for the ADF p-value and critical values are the published
// numbers as tabulated in statsmodels.tsa.adfvalues; the KPSS critical values are Kwiatkowski et al. (1992), Table 1.
#include "models_common.hpp"
#include <Eigen/Dense>
#include <algorithm>

namespace lmnum {
namespace models {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using V = std::vector<double>;

// ---- MacKinnon tables (statsmodels.tsa.adfvalues, N = 1 rows) -----------------------------------------------------
struct AdfTable {
  double tau_max, tau_min, tau_star;
  double smallp[3];   // polynomial in t (increasing powers), already scaled
  double largep[4];
  double crit[3][4];  // 1 %, 5 %, 10 %: polynomial in 1/nobs (increasing powers)
};
const AdfTable* adf_table(const std::string& reg) {
  static const AdfTable n = {INFINITY, -19.04, -1.04, {0.6344, 1.2378, 3.2496e-2}, {0.4797, 9.3557e-1, -0.6999e-1, 3.3066e-2},
                             {{-2.56574, -2.2358, -3.627, 0}, {-1.94100, -0.2686, -3.365, 31.223}, {-1.61682, 0.2656, -2.714, 25.364}}};
  static const AdfTable c = {2.74, -18.83, -1.61, {2.1659, 1.4412, 3.8269e-2}, {1.7339, 9.3202e-1, -1.2745e-1, -1.0368e-2},
                             {{-3.43035, -6.5393, -16.786, -79.433}, {-2.86154, -2.8903, -4.234, -40.040}, {-2.56677, -1.5384, -2.809, 0}}};
  static const AdfTable ct = {0.7, -16.18, -2.89, {3.2512, 1.6047, 4.9588e-2}, {2.5261, 6.1654e-1, -3.7956e-1, -6.0285e-2},
                              {{-3.95877, -9.0531, -28.428, -134.155}, {-3.41049, -4.3904, -9.036, -45.374}, {-3.12705, -2.5856, -3.925, -22.380}}};
  static const AdfTable ctt = {0.54, -17.17, -3.21, {4.0003, 1.658, 4.8288e-2}, {3.0778, 4.9529e-1, -4.1477e-1, -5.9359e-2},
                               {{-4.37113, -11.5882, -35.819, -334.047}, {-3.83239, -5.9057, -12.490, -118.284}, {-3.55326, -3.6596, -5.293, -63.559}}};
  if (reg == "n") return &n;
  if (reg == "c") return &c;
  if (reg == "ct") return &ct;
  if (reg == "ctt") return &ctt;
  return nullptr;
}
double polyval_inc(const double* c, int k, double x) {  // c[0] + c[1] x + ... (Horner from the top)
  double v = 0;
  for (int i = k - 1; i >= 0; --i) v = v * x + c[i];
  return v;
}
double mackinnonp(double stat, const AdfTable& t) {
  if (stat > t.tau_max) return 1.0;
  if (stat < t.tau_min) return 0.0;
  if (stat <= t.tau_star) return norm_cdf(polyval_inc(t.smallp, 3, stat));
  return norm_cdf(polyval_inc(t.largep, 4, stat));
}

// ---- helpers --------------------------------------------------------------------------------------------------------
struct Series { V x; double scale; };
Series prepare(const std::vector<double>& raw, size_t min_n, const char* op) {
  for (double v : raw) if (!std::isfinite(v)) fail(std::string(op) + ": the series contains NaN or infinite values");
  if (raw.size() < min_n) fail(std::string(op) + ": the series needs at least " + std::to_string(min_n) + " observations; got " + std::to_string(raw.size()));
  double mx = *std::max_element(raw.begin(), raw.end()), mn = *std::min_element(raw.begin(), raw.end());
  if (mx == mn) fail(std::string(op) + ": the series is constant: the statistic is undefined");
  double scale = 0;
  for (double v : raw) scale = std::max(scale, std::fabs(v));
  Series s{raw, scale};
  for (double& v : s.x) v /= scale;
  return s;
}

double mean_of(const V& x) { double s = 0; for (double v : x) s += v; return s / double(x.size()); }

// statsmodels acovf(fft=False): demeaned lagged dot products over n (or n - k when adjusted).
V acovf(const V& x, size_t nlag, bool adjusted) {
  size_t n = x.size();
  double m = mean_of(x);
  V xo(n);
  for (size_t i = 0; i < n; ++i) xo[i] = x[i] - m;
  V acov(nlag + 1);
  for (size_t k = 0; k <= nlag; ++k) {
    double s = 0;
    for (size_t t = k; t < n; ++t) s += xo[t] * xo[t - k];
    acov[k] = s / (adjusted ? double(n - k) : double(n));
  }
  return acov;
}

V acf_values(const V& x, size_t nlags, bool adjusted) {
  V av = acovf(x, nlags, adjusted), r(nlags + 1);
  for (size_t k = 0; k <= nlags; ++k) r[k] = av[k] / av[0];
  return r;
}

// Ljung-Box Q over lags 1..m: n (n + 2) cumsum(r_k^2 / (n - k)).
V ljung_q(const V& r, size_t n, size_t m) {
  V q(m);
  double s = 0;
  for (size_t k = 1; k <= m; ++k) { s += r[k] * r[k] / double(n - k); q[k - 1] = double(n) * double(n + 2) * s; }
  return q;
}

Res* acf_op(const std::vector<double>& raw, int64_t nlags, const Opts& o) {
  const char* op = "acf";
  bool adjusted = o.flag("adjusted", false);
  o.flag("fft", false);  // accepted for statsmodels compatibility; the sums are computed directly either way
  double alpha = o.num("alpha", 0.05);
  bool qstat = o.flag("qstat", false);
  need_alpha(alpha, op);
  o.finish(op);
  Series s = prepare(raw, 3, op);
  size_t n = s.x.size();
  if (nlags < 1 || size_t(nlags) >= n) fail("acf: nlags must be 1.." + std::to_string(n - 1) + " for a series of length " + std::to_string(n));
  V r = acf_values(s.x, size_t(nlags), adjusted);
  // Bartlett's variance: 1/n at lag 1, (1 + 2 sum_{j<k} r_j^2) / n beyond
  V lo(r.size()), hi(r.size());
  double z = norm_ppf(1 - alpha / 2), cum = 0;
  for (size_t k = 0; k < r.size(); ++k) {
    double var = k == 0 ? 0.0 : (1 + 2 * cum) / double(n);
    if (k >= 1) cum += r[k] * r[k];
    double h = z * std::sqrt(var);
    lo[k] = r[k] - h;
    hi[k] = r[k] + h;
  }
  auto* res = make<Res>();
  bool finite = true;
  for (double v : r) finite &= std::isfinite(v);
  res->str("status", finite ? "OK" : "NUMERICAL_FAILURE").str("kind", "acf").integer("n", int64_t(n));
  V lags(r.size());
  for (size_t k = 0; k < r.size(); ++k) lags[k] = double(k);
  res->vec("lags", lags).vec("values", r).vec("ci_lower", lo).vec("ci_upper", hi).num("alpha", alpha).flag("adjusted", adjusted);
  if (qstat) {
    V q = ljung_q(r, n, size_t(nlags)), p(q.size());
    for (size_t k = 0; k < q.size(); ++k) p[k] = chi2_sf(q[k], double(k + 1));
    res->vec("qstat", q).vec("pvalues", p);
  }
  return res;
}

// Partial autocorrelations.
V yule_walker_last(const V& x, size_t order, bool adjusted) {
  size_t n = x.size();
  double m = mean_of(x);
  V xo(n);
  for (size_t i = 0; i < n; ++i) xo[i] = x[i] - m;
  V r(order + 1);
  double s0 = 0;
  for (double v : xo) s0 += v * v;
  r[0] = s0 / double(n);
  for (size_t k = 1; k <= order; ++k) {
    double s = 0;
    for (size_t t = 0; t + k < n; ++t) s += xo[t] * xo[t + k];
    r[k] = s / (double(n) - double(k) * (adjusted ? 1.0 : 0.0));
  }
  MatrixXd R(order, order);
  VectorXd b(order);
  for (size_t i = 0; i < order; ++i) {
    b(i) = r[i + 1];
    for (size_t j = 0; j < order; ++j) R(i, j) = r[i > j ? i - j : j - i];
  }
  VectorXd rho = R.partialPivLu().solve(b);
  return V(rho.data(), rho.data() + rho.size());
}

V lstsq_last(const MatrixXd& a, const VectorXd& b) {
  Eigen::JacobiSVD<MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd s = svd.singularValues();
  double cut = std::numeric_limits<double>::epsilon() * double(std::max(a.rows(), a.cols())) * (s.size() ? s(0) : 0);
  VectorXd utb = svd.matrixU().transpose() * b;
  for (Eigen::Index i = 0; i < s.size(); ++i) utb(i) = s(i) > cut ? utb(i) / s(i) : 0.0;
  VectorXd x = svd.matrixV() * utb;
  return V(x.data(), x.data() + x.size());
}

V levinson_pacf(const V& acov, size_t nlags) {
  std::vector<V> phi(nlags + 1, V(nlags + 1, 0.0));
  V sig(nlags + 1, 0.0), out(nlags + 1);
  phi[1][1] = acov[1] / acov[0];
  sig[1] = acov[0] - phi[1][1] * acov[1];
  for (size_t k = 2; k <= nlags; ++k) {
    double dot = 0;
    for (size_t j = 1; j < k; ++j) dot += phi[j][k - 1] * acov[k - j];
    phi[k][k] = (acov[k] - dot) / sig[k - 1];
    for (size_t j = 1; j < k; ++j) phi[j][k] = phi[j][k - 1] - phi[k][k] * phi[k - j][k - 1];
    sig[k] = sig[k - 1] * (1 - phi[k][k] * phi[k][k]);
  }
  out[0] = 1.0;
  for (size_t k = 1; k <= nlags; ++k) out[k] = phi[k][k];
  return out;
}

Res* pacf_op(const std::vector<double>& raw, int64_t nlags, const Opts& o) {
  const char* op = "pacf";
  std::string method = o.str("method", "ywm");
  double alpha = o.num("alpha", 0.05);
  need_alpha(alpha, op);
  o.finish(op);
  Series s = prepare(raw, 3, op);
  size_t n = s.x.size();
  if (nlags < 1 || size_t(nlags) >= n / 2) fail("pacf: nlags must be 1.." + std::to_string(n / 2 - 1) + " (below half the series length " + std::to_string(n) + ")");
  size_t L = size_t(nlags);
  const V& x = s.x;
  V ret(L + 1);
  ret[0] = 1.0;
  if (method == "ywm" || method == "ywmle" || method == "yw_mle" || method == "ywadjusted" || method == "yw" || method == "ywa" || method == "yw_adjusted") {
    bool adj = method == "ywadjusted" || method == "yw" || method == "ywa" || method == "yw_adjusted";
    for (size_t k = 1; k <= L; ++k) ret[k] = yule_walker_last(x, k, adj).back();
  } else if (method == "ols" || method == "ols-adjusted") {
    // lagmat(x, L, original="sep") with a leading constant; regression on rows k.. for each k
    MatrixXd xl = MatrixXd::Zero(n, L + 1);
    VectorXd x0(n);
    for (size_t t = 0; t < n; ++t) {
      xl(t, 0) = 1.0;
      x0(t) = x[t];
      for (size_t j = 0; j < L; ++j) xl(t, j + 1) = t >= j + 1 ? x[t - j - 1] : 0.0;
    }
    for (size_t k = 1; k <= L; ++k) ret[k] = lstsq_last(xl.block(k, 0, n - k, k + 1), x0.tail(n - k)).back();
    if (method == "ols-adjusted") for (size_t k = 0; k <= L; ++k) ret[k] *= double(n) / double(n - k);
  } else if (method == "ols-inefficient") {
    double m = mean_of(x);
    size_t rows = n - L;
    MatrixXd xl(rows, L);
    VectorXd x0(rows);
    for (size_t t = L; t < n; ++t) {
      x0(t - L) = x[t] - m;
      for (size_t j = 0; j < L; ++j) xl(t - L, j) = x[t - j - 1] - m;
    }
    for (size_t k = 1; k <= L; ++k) ret[k] = lstsq_last(xl.leftCols(k), x0).back();
  } else if (method == "ld" || method == "lda" || method == "ldadjusted" || method == "ld_adjusted") {
    ret = levinson_pacf(acovf(x, n - 1, true), L);
  } else if (method == "ldb" || method == "ldbiased" || method == "ld_biased") {
    ret = levinson_pacf(acovf(x, n - 1, false), L);
  } else fail("pacf: method must be ywm, ywadjusted, ols, ols-inefficient, ols-adjusted, ldadjusted or ldb; got '" + method + "'");
  double h = norm_ppf(1 - alpha / 2) * std::sqrt(1.0 / double(n));
  V lo(L + 1), hi(L + 1), lags(L + 1);
  for (size_t k = 0; k <= L; ++k) { lo[k] = k ? ret[k] - h : ret[k]; hi[k] = k ? ret[k] + h : ret[k]; lags[k] = double(k); }
  bool finite = true;
  for (double v : ret) finite &= std::isfinite(v);
  auto* res = make<Res>();
  res->str("status", finite ? "OK" : "NUMERICAL_FAILURE").str("kind", "pacf").integer("n", int64_t(n)).vec("lags", lags).vec("values", ret)
      .vec("ci_lower", lo).vec("ci_upper", hi).num("alpha", alpha).str("method", method);
  return res;
}

// OLS pieces needed by the ADF regression: t-value of one column and the information criteria (statsmodels OLS: pinv).
struct OlsFit { VectorXd t; double aic, bic; };
OlsFit ols_small(const MatrixXd& X, const VectorXd& y) {
  Pinv pv = pinv_extended(X);
  VectorXd b = pv.pinv * y;
  VectorXd e = y - X * b;
  double ssr = e.squaredNorm();
  int rank = matrix_rank(X);
  double n = double(X.rows()), df_resid = n - rank;
  int kc, ci;
  detect_constant(X, kc, ci);
  double nobs2 = n / 2.0;
  double llf = -std::log(ssr) * nobs2 - (1 + std::log(M_PI / nobs2)) * nobs2;
  double kp = double(rank);  // df_model + k_constant
  OlsFit f;
  f.aic = -2 * llf + 2 * kp;
  f.bic = -2 * llf + std::log(n) * kp;
  VectorXd se = (pv.normcov.diagonal() * (ssr / df_resid)).array().sqrt();
  f.t = b.cwiseQuotient(se);
  (void)kc;
  return f;
}

// [trend columns] for t = 1..nobs: const, t, t^2 (statsmodels add_trend)
MatrixXd trend_cols(const std::string& reg, size_t nobs) {
  size_t k = reg == "n" ? 0 : reg.size();
  MatrixXd tr(nobs, k);
  for (size_t i = 0; i < nobs; ++i) {
    double t = double(i + 1);
    if (k >= 1) tr(i, 0) = 1.0;
    if (k >= 2) tr(i, 1) = t;
    if (k >= 3) tr(i, 2) = t * t;
  }
  return tr;
}

// Columns [level x_{t-1}, dx_{t-1}, ..., dx_{t-lags}] and the response dx_t on the rows where all lags exist.
void adf_data(const V& x, size_t lags, MatrixXd& xd, VectorXd& yd) {
  size_t m = x.size() - 1;  // length of the differences
  V dx(m);
  for (size_t i = 0; i < m; ++i) dx[i] = x[i + 1] - x[i];
  size_t nobs = m - lags;
  xd.resize(nobs, lags + 1);
  yd.resize(nobs);
  for (size_t r = 0; r < nobs; ++r) {
    size_t t = r + lags;  // index into dx
    yd(r) = dx[t];
    xd(r, 0) = x[t];      // the level x_{t} that precedes dx_t = x_{t+1} - x_t
    for (size_t j = 1; j <= lags; ++j) xd(r, j) = dx[t - j];
  }
}

Res* adfuller_op(const std::vector<double>& raw, const Opts& o) {
  const char* op = "adfuller";
  std::string reg = o.str("regression", "c");
  std::string autolag = o.str("autolag", "AIC");
  const AdfTable* tab = adf_table(reg);
  if (!tab) fail("adfuller: regression must be c, ct, ctt or n; got '" + reg + "'");
  std::string al = autolag;
  std::transform(al.begin(), al.end(), al.begin(), ::tolower);
  if (al != "aic" && al != "bic" && al != "t-stat" && al != "none") fail("adfuller: autolag must be AIC, BIC, t-stat or none; got '" + autolag + "'");
  bool has_maxlag = o.has("maxlag");
  int64_t maxlag_opt = o.integer("maxlag", 0);
  o.finish(op);
  Series s = prepare(raw, 8, op);
  const V& x = s.x;
  size_t n = x.size();
  int ntrend = reg == "n" ? 0 : int(reg.size());
  int64_t limit = int64_t(n / 2) - ntrend - 1;
  int64_t maxlag;
  if (!has_maxlag) {
    maxlag = std::min<int64_t>(int64_t(std::ceil(12.0 * std::pow(double(n) / 100.0, 0.25))), limit);
    if (maxlag < 0) fail("adfuller: the series is too short for the selected regression");
  } else {
    maxlag = maxlag_opt;
    if (maxlag < 0 || maxlag > limit) fail("adfuller: maxlag must be 0.." + std::to_string(limit) + " (n/2 - deterministic terms - 1) for this series");
  }
  size_t used = size_t(maxlag);
  double icbest = NAN;
  if (al != "none") {
    MatrixXd xd;
    VectorXd yd;
    adf_data(x, size_t(maxlag), xd, yd);
    size_t nobs = size_t(yd.size());
    MatrixXd tr = trend_cols(reg, nobs);
    MatrixXd full(nobs, tr.cols() + xd.cols());
    full << tr, xd;
    size_t startlag = size_t(tr.cols()) + 1;
    std::vector<OlsFit> fits;
    for (size_t lag = startlag; lag <= startlag + size_t(maxlag); ++lag) fits.push_back(ols_small(full.leftCols(lag), yd));
    size_t best = 0;
    if (al == "aic" || al == "bic") {
      icbest = INFINITY;
      for (size_t i = 0; i < fits.size(); ++i) {  // min over (ic, lag): the first minimum
        double ic = al == "aic" ? fits[i].aic : fits[i].bic;
        if (ic < icbest) { icbest = ic; best = i; }
      }
      icbest += 2.0 * double(nobs) * std::log(s.scale) * 1.0;  // back to the units of the data
    } else {
      best = fits.size() - 1;
      for (size_t i = fits.size(); i-- > 0;) {
        icbest = std::fabs(fits[i].t(fits[i].t.size() - 1));
        best = i;
        if (icbest >= 1.6448536269514722) break;
      }
    }
    used = best;
  }
  MatrixXd xd;
  VectorXd yd;
  adf_data(x, used, xd, yd);
  size_t nobs = size_t(yd.size());
  MatrixXd tr = trend_cols(reg, nobs);
  MatrixXd X(nobs, xd.cols() + tr.cols());
  X << xd, tr;
  OlsFit fit = ols_small(X, yd);
  double stat = fit.t(0);
  double p = mackinnonp(stat, *tab);
  auto crit = std::make_shared<Res>();
  const char* labels[3] = {"1%", "5%", "10%"};
  for (int i = 0; i < 3; ++i) crit->num(labels[i], polyval_inc(tab->crit[i], 4, 1.0 / double(nobs)));
  auto* res = make<Res>();
  bool finite = std::isfinite(stat) && std::isfinite(p) && (al == "none" || std::isfinite(icbest));
  res->str("status", finite ? "OK" : "NUMERICAL_FAILURE").str("test", "adfuller").integer("n", int64_t(n)).num("normalisation_scale", s.scale)
      .num("statistic", stat).num("p_value", p).integer("used_lag", int64_t(used)).integer("nobs", int64_t(nobs)).sub("critical_values", crit)
      .str("regression", reg).str("autolag", autolag).integer("maxlag", maxlag);
  if (al != "none") res->num("icbest", icbest);
  if (!finite) res->str("message", "the test statistic is not finite for this series");
  return res;
}

Res* kpss_op(const std::vector<double>& raw, const Opts& o) {
  const char* op = "kpss";
  std::string reg = o.str("regression", "c");
  if (reg != "c" && reg != "ct") fail("kpss: regression must be c or ct; got '" + reg + "'");
  const Item* nl = o.find("nlags");
  std::string mode = "auto";
  int64_t lags_opt = -1;
  if (nl) {
    if (nl->kind == Item::STR) mode = nl->s;
    else if (nl->kind == Item::INT) { mode = "int"; lags_opt = nl->i; }
    else fail("kpss: nlags must be \"auto\", \"legacy\" or an int");
    if (mode != "auto" && mode != "legacy" && mode != "int") fail("kpss: nlags must be \"auto\", \"legacy\" or an int; got '" + mode + "'");
  }
  o.finish(op);
  Series s = prepare(raw, 8, op);
  const V& x = s.x;
  size_t n = x.size();
  V e(n);
  double crit[4];
  if (reg == "ct") {
    MatrixXd X(n, 2);
    VectorXd y(n);
    for (size_t i = 0; i < n; ++i) { X(i, 0) = 1.0; X(i, 1) = double(i + 1); y(i) = x[i]; }
    Pinv pv = pinv_extended(X);
    VectorXd r = y - X * (pv.pinv * y);
    for (size_t i = 0; i < n; ++i) e[i] = r(i);
    double c[4] = {0.119, 0.146, 0.176, 0.216};
    std::copy(c, c + 4, crit);
  } else {
    double m = mean_of(x);
    for (size_t i = 0; i < n; ++i) e[i] = x[i] - m;
    double c[4] = {0.347, 0.463, 0.574, 0.739};
    std::copy(c, c + 4, crit);
  }
  auto lagdot = [&](size_t i) { double d = 0; for (size_t t = i; t < n; ++t) d += e[t] * e[t - i]; return d; };
  int64_t lags;
  if (mode == "legacy") lags = std::min<int64_t>(int64_t(std::ceil(12.0 * std::pow(double(n) / 100.0, 0.25))), int64_t(n) - 1);
  else if (mode == "int") {
    if (lags_opt < 0 || lags_opt >= int64_t(n)) fail("kpss: nlags must be 0.." + std::to_string(n - 1));
    lags = lags_opt;
  } else {  // Hobijn et al. (1998)
    size_t covlags = size_t(std::pow(double(n), 2.0 / 9.0));
    double s0 = 0, s1 = 0;
    for (double v : e) s0 += v * v;
    s0 /= double(n);
    for (size_t i = 1; i <= covlags; ++i) {
      double p = lagdot(i) / (double(n) / 2.0);
      s0 += p;
      s1 += double(i) * p;
    }
    double sh = s1 / s0;
    double gamma = 1.1447 * std::pow(sh * sh, 1.0 / 3.0);
    lags = std::min<int64_t>(int64_t(gamma * std::pow(double(n), 1.0 / 3.0)), int64_t(n) - 1);
  }
  double cs = 0, eta = 0;
  for (double v : e) { cs += v; eta += cs * cs; }
  eta /= double(n) * double(n);
  double shat = 0;
  for (double v : e) shat += v * v;
  for (int64_t i = 1; i <= lags; ++i) shat += 2 * lagdot(size_t(i)) * (1.0 - double(i) / (double(lags) + 1.0));
  shat /= double(n);
  double stat = eta / shat;
  const double pv[4] = {0.10, 0.05, 0.025, 0.01};
  double p;
  if (stat <= crit[0]) p = pv[0];
  else if (stat >= crit[3]) p = pv[3];
  else {
    int j = 0;
    while (stat > crit[j + 1]) ++j;
    p = pv[j] + (stat - crit[j]) * (pv[j + 1] - pv[j]) / (crit[j + 1] - crit[j]);
  }
  bool below = p <= 0.01 && stat > crit[3], above = p >= 0.1 && stat < crit[0];
  auto cr = std::make_shared<Res>();
  cr->num("10%", crit[0]).num("5%", crit[1]).num("2.5%", crit[2]).num("1%", crit[3]);
  auto* res = make<Res>();
  res->str("status", std::isfinite(stat) ? "OK" : "NUMERICAL_FAILURE").str("test", "kpss").integer("n", int64_t(n)).num("normalisation_scale", s.scale)
      .num("statistic", stat).num("p_value", p).integer("lags", lags).sub("critical_values", cr).str("regression", reg)
      .str("p_value_bounded", below ? "below" : above ? "above" : "none");
  return res;
}

Res* ljung_op(const std::vector<double>& raw, const std::vector<int64_t>* list, int64_t m, const Opts& o) {
  const char* op = "ljung_box";
  int64_t model_df = o.integer("model_df", 0);
  bool bp = o.flag("boxpierce", false);
  o.finish(op);
  Series s = prepare(raw, 8, op);
  size_t n = s.x.size();
  std::vector<int64_t> lags;
  if (list) {
    lags = *list;
    if (lags.empty() || lags.size() > 100) fail("ljung_box: give 1..100 lags");
    for (size_t i = 0; i < lags.size(); ++i) {
      if (i && lags[i] <= lags[i - 1]) fail("ljung_box: the lags must be strictly increasing");
      if (lags[i] <= model_df) fail("ljung_box: every lag must exceed model_df (" + std::to_string(model_df) + ")");
    }
  } else {
    if (model_df < 0 || model_df >= m) fail("ljung_box: model_df must be 0.." + std::to_string(m - 1));
    for (int64_t k = model_df + 1; k <= m; ++k) lags.push_back(k);
  }
  if (model_df < 0) fail("ljung_box: model_df must be >= 0");
  if (lags.front() < 1 || size_t(lags.back()) >= n) fail("ljung_box: lags must lie in 1.." + std::to_string(n - 1) + " for a series of length " + std::to_string(n));
  size_t maxlag = size_t(lags.back());
  V r = acf_values(s.x, maxlag, false);
  V q = ljung_q(r, n, maxlag);
  V lv, st, pv, bst, bpv;
  double cbp = 0;
  V bpc(maxlag);
  for (size_t k = 1; k <= maxlag; ++k) { cbp += r[k] * r[k]; bpc[k - 1] = double(n) * cbp; }
  bool finite = true;
  for (int64_t L : lags) {
    lv.push_back(double(L));
    st.push_back(q[size_t(L) - 1]);
    pv.push_back(chi2_sf(q[size_t(L) - 1], double(L - model_df)));
    bst.push_back(bpc[size_t(L) - 1]);
    bpv.push_back(chi2_sf(bpc[size_t(L) - 1], double(L - model_df)));
    finite &= std::isfinite(st.back()) && std::isfinite(pv.back());
  }
  auto* res = make<Res>();
  res->str("status", finite ? "OK" : "NUMERICAL_FAILURE").str("test", "ljung_box").integer("n", int64_t(n)).num("normalisation_scale", s.scale)
      .vec("lags", lv).vec("statistic", st).vec("p_value", pv).integer("model_df", model_df);
  if (bp) res->vec("bp_statistic", bst).vec("bp_p_value", bpv);
  return res;
}

std::vector<int64_t> int_lags(const std::vector<double>& v) {
  std::vector<int64_t> out;
  for (double x : v) {
    if (x != std::floor(x)) fail("ljung_box: lags must be integers");
    out.push_back(int64_t(x));
  }
  return out;
}

}  // namespace

void register_tsa(Api& api) {
  const std::string Rt = "Res", Vt = "Vec";
  api.fn("acf", {Vt, "int"}, Rt,
         "sample autocorrelation at lags 0..nlags (pycli sm_tsa_acf kind acf; statsmodels acf, divide by n): n, lags, values, ci_lower, ci_upper (Bartlett, 95%), status",
         [](Args& a) { return O(acf_op(obj<Vec>(a[0]).v, a[1].i, no_opts())); });
  api.fn("acf", {Vt, "int", "Opts"}, Rt, "acf with options: `adjusted` (divide by n - k), `alpha`, `qstat` (adds Ljung-Box `qstat` and `pvalues` for lags 1..nlags), `fft` (accepted; sums are direct)",
         [](Args& a) { return O(acf_op(obj<Vec>(a[0]).v, a[1].i, obj<Opts>(a[2]))); });
  api.fn("pacf", {Vt, "int"}, Rt,
         "partial autocorrelation at lags 0..nlags by Yule-Walker with n denominator (pycli sm_tsa_acf kind pacf, method ywm = R's pacf): values, ci_lower, ci_upper (+-z/sqrt(n))",
         [](Args& a) { return O(pacf_op(obj<Vec>(a[0]).v, a[1].i, no_opts())); });
  api.fn("pacf", {Vt, "int", "Opts"}, Rt, "pacf with options: `method` (ywm, ywadjusted, ols, ols-inefficient, ols-adjusted, ldadjusted, ldb), `alpha`; nlags < n/2",
         [](Args& a) { return O(pacf_op(obj<Vec>(a[0]).v, a[1].i, obj<Opts>(a[2]))); });
  api.fn("adfuller", {Vt}, Rt,
         "augmented Dickey-Fuller unit-root test (pycli sm_tsa_tools adfuller; constant, AIC lag choice): statistic, p_value (MacKinnon), used_lag, nobs, critical_values {1%, 5%, 10%}, icbest, maxlag",
         [](Args& a) { return O(adfuller_op(obj<Vec>(a[0]).v, no_opts())); });
  api.fn("adfuller", {Vt, "Opts"}, Rt, "adfuller with options: `regression` (c, ct, ctt, n), `maxlag` (int; default ceil(12 (n/100)^(1/4)) clipped), `autolag` (AIC, BIC, t-stat, none)",
         [](Args& a) { return O(adfuller_op(obj<Vec>(a[0]).v, obj<Opts>(a[1]))); });
  api.fn("kpss", {Vt}, Rt,
         "KPSS stationarity test (pycli sm_tsa_tools kpss; level, automatic lags): statistic, p_value (table interpolation), p_value_bounded none/above/below, lags, critical_values {10%, 5%, 2.5%, 1%}",
         [](Args& a) { return O(kpss_op(obj<Vec>(a[0]).v, no_opts())); });
  api.fn("kpss", {Vt, "Opts"}, Rt, "kpss with options: `regression` (c or ct), `nlags` (\"auto\", \"legacy\" or an int)",
         [](Args& a) { return O(kpss_op(obj<Vec>(a[0]).v, obj<Opts>(a[1]))); });
  api.fn("ljung_box", {Vt, "int"}, Rt,
         "Ljung-Box test of no autocorrelation for lags 1..m (pycli sm_tsa_tools ljung_box): lags, statistic, p_value, model_df",
         [](Args& a) { return O(ljung_op(obj<Vec>(a[0]).v, nullptr, a[1].i, no_opts())); });
  api.fn("ljung_box", {Vt, "int", "Opts"}, Rt, "ljung_box with options: `model_df` (fitted ARMA parameters; lags model_df+1..m are tested against chi2(lag - model_df)), `boxpierce` (adds bp_statistic, bp_p_value)",
         [](Args& a) { return O(ljung_op(obj<Vec>(a[0]).v, nullptr, a[1].i, obj<Opts>(a[2]))); });
  api.fn("ljung_box", {Vt, Vt}, Rt, "ljung_box at an explicit strictly increasing list of lags",
         [](Args& a) { auto l = int_lags(obj<Vec>(a[1]).v); return O(ljung_op(obj<Vec>(a[0]).v, &l, 0, no_opts())); });
  api.fn("ljung_box", {Vt, Vt, "Opts"}, Rt, "ljung_box at explicit lags with options (`model_df`, `boxpierce`)",
         [](Args& a) { auto l = int_lags(obj<Vec>(a[1]).v); return O(ljung_op(obj<Vec>(a[0]).v, &l, 0, obj<Opts>(a[2]))); });
}

}  // namespace models
}  // namespace lmnum
