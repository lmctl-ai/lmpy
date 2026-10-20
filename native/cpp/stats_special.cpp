// Special functions with scipy.special semantics (pycli special_fn). The implementations are the ones scipy 1.16 itself
// uses wherever that code is permissively licensed:
//   - xsf (scipy's header-only special-function library, BSD-3-Clause; bundles Cephes, BSD-3-Clause, and Faddeeva, MIT):
//     erf, erfc, erfcx, erfcinv, ndtr, log_ndtr, ndtri, gamma, gammaln, digamma, polygamma (via zeta), gammainc and
//     inverses, beta, betaln, zeta, lambertw, ellipk, ellipe, hyp2f1, expit, logit, log_expit, xlogy, xlog1py, owens_t,
//     cosm1, i0e, poch, Airy on [-10, 10], Fresnel integrals;
//   - Boost.Math 1.86 (BSL-1.0) with scipy's policies: erfinv, betainc and inverses (as scipy), and the Bessel functions
//     jv, yv, iv, kv, the Airy functions outside [-10, 10] and the Bessel calls inside struve and spherical_jn, where
//     scipy uses AMOS (not used here because of its ACM TOMS provenance).
//   - scipy's own small kernels, ported: boxcox family (_boxcox.pxd), ndtri_exp (_ndtri_exp.pxd, Albert Steppi),
//     struve_h (Cephes struve.h, BSD-3-Clause), spherical_jn for real arguments (xsf sph_bessel.h).
// scipy: Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers; BSD-3-Clause.
#include "stats_internal.hpp"

#include <xsf/beta.h>
#include <xsf/cephes/airy.h>
#include <xsf/cephes/dd_real.h>
#include <xsf/cephes/erfinv.h>
#include <xsf/cephes/fresnl.h>
#include <xsf/cephes/gamma.h>
#include <xsf/cephes/i0.h>
#include <xsf/cephes/poch.h>
#include <xsf/cephes/polevl.h>
#include <xsf/cephes/rgamma.h>
#include <xsf/digamma.h>
#include <xsf/ellip.h>
#include <xsf/erf.h>
#include <xsf/gamma.h>
#include <xsf/hyp2f1.h>
#include <xsf/lambertw.h>
#include <xsf/log.h>
#include <xsf/log_exp.h>
#include <xsf/stats.h>
#include <xsf/trig.h>
#include <xsf/zeta.h>

#include <boost/math/special_functions/airy.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/erf.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

namespace lmnum { namespace st {

namespace bp = boost::math::policies;
// scipy's SpecialPolicy (no promotion of double, 400 root iterations); errors are reported as IEEE values instead of
// exceptions, which is what scipy's wrappers turn them into.
typedef bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>, bp::overflow_error<bp::ignore_error>,
                   bp::underflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>, bp::rounding_error<bp::ignore_error>,
                   bp::promote_float<false>, bp::promote_double<false>, bp::max_root_iterations<400>>
    SpecPol;

const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();

double ndtr(double x) { return xsf::ndtr(x); }
double log_ndtr(double x) { return xsf::log_ndtr(x); }
double ndtri(double p) { return xsf::ndtri(p); }
double erf(double x) { return xsf::erf(x); }
double erfc(double x) { return xsf::erfc(x); }
double erfcx(double x) { return xsf::erfcx(x); }
double gammaln(double x) { return xsf::gammaln(x); }
double gammafn(double x) { return xsf::gamma(x); }
double xlogy(double x, double y) { return xsf::xlogy(x, y); }
double xlog1py(double x, double y) { return xsf::xlog1py(x, y); }
double expit(double x) { return xsf::expit(x); }
double logit(double p) { return xsf::logit(p); }
double log_expit(double x) { return xsf::log_expit(x); }
double cosm1(double x) { return xsf::cosm1(x); }
double i0e(double x) { return xsf::cephes::i0e(x); }
double poch(double a, double m) { return xsf::cephes::poch(a, m); }
double betaln(double a, double b) { return xsf::betaln(a, b); }
double gammainc(double a, double x) { return xsf::gammainc(a, x); }
double gammaincc(double a, double x) { return xsf::gammaincc(a, x); }
double gammaincinv(double a, double p) { return xsf::gammaincinv(a, p); }
double gammainccinv(double a, double p) { return xsf::gammainccinv(a, p); }
double zeta(double x, double q) { return xsf::zeta(x, q); }
double riemann_zeta(double x) { return xsf::riemann_zeta(x); }
double chdtr(double df, double x) { return xsf::chdtr(df, x); }
double chdtrc(double df, double x) { return xsf::chdtrc(df, x); }
double chdtri(double df, double p) { return xsf::chdtri(df, p); }
double fdtr(double a, double b, double x) { return xsf::fdtr(a, b, x); }
double fdtrc(double a, double b, double x) { return xsf::fdtrc(a, b, x); }
double fdtri(double a, double b, double p) { return xsf::fdtri(a, b, p); }
double pdtr(double k, double m) { return xsf::pdtr(k, m); }
double pdtrc(double k, double m) { return xsf::pdtrc(k, m); }
double fresnel_s(double x) { double s, c; xsf::cephes::fresnl(x, &s, &c); return s; }
double fresnel_c(double x) { double s, c; xsf::cephes::fresnl(x, &s, &c); return c; }

// ---- scipy boost wrappers (boost_special_functions.h) ----------------------------------------------------------------
double betainc(double a, double b, double x) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(x)) return NaN;
  if (a < 0 || b < 0 || x < 0 || x > 1) return NaN;
  if ((a == 0 && b == 0) || (std::isinf(a) && std::isinf(b))) return NaN;
  if (a == 0 || std::isinf(b)) return x > 0 ? 1 : 0;
  if (b == 0 || std::isinf(a)) return x < 1 ? 0 : 1;
  return boost::math::ibeta(a, b, x, SpecPol());
}
double betaincc(double a, double b, double x) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(x)) return NaN;
  if (a < 0 || b < 0 || x < 0 || x > 1) return NaN;
  if ((a == 0 && b == 0) || (std::isinf(a) && std::isinf(b))) return NaN;
  if (a == 0 || std::isinf(b)) return x > 0 ? 0 : 1;
  if (b == 0 || std::isinf(a)) return x < 1 ? 1 : 0;
  return boost::math::ibetac(a, b, x, SpecPol());
}
double betaincinv(double a, double b, double p) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(p)) return NaN;
  if (a <= 0 || b <= 0 || p < 0 || p > 1) return NaN;
  return boost::math::ibeta_inv(a, b, p, SpecPol());
}
double betainccinv(double a, double b, double p) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(p)) return NaN;
  if (a <= 0 || b <= 0 || p < 0 || p > 1) return NaN;
  return boost::math::ibetac_inv(a, b, p, SpecPol());
}
static double erfinv(double x) {
  if (x == -1) return -Inf;
  if (x == 1) return Inf;
  if (std::isnan(x) || x < -1 || x > 1) return NaN;
  return boost::math::erf_inv(x, SpecPol());
}

// ---- _boxcox.pxd --------------------------------------------------------------------------------------------------
double boxcox(double x, double l) {
  if (std::fabs(l) < 1e-19) return std::log(x);
  if (l * std::log(x) < 709.78) return std::expm1(l * std::log(x)) / l;
  return std::copysign(1., l) * std::exp(l * std::log(x) - std::log(std::fabs(l))) - 1 / l;
}
double boxcox1p(double x, double l) {
  double lgx = std::log1p(x);
  if (std::fabs(l) < 1e-19 || (std::fabs(lgx) < 1e-289 && std::fabs(l) < 1e273)) return lgx;
  if (l * lgx < 709.78) return std::expm1(l * lgx) / l;
  return std::copysign(1., l) * std::exp(l * lgx - std::log(std::fabs(l))) - 1 / l;
}
double inv_boxcox(double x, double l) {
  if (l == 0) return std::exp(x);
  if (l * x < 1.79e308) return std::exp(std::log1p(l * x) / l);
  return std::exp((std::log(std::copysign(1., l) * (x + 1 / l)) + std::log(std::fabs(l))) / l);
}
double inv_boxcox1p(double x, double l) {
  if (l == 0) return std::expm1(x);
  if (std::fabs(l * x) < 1e-154) return x;
  if (l * x < 1.79e308) return std::expm1(std::log1p(l * x) / l);
  return std::expm1((std::log(std::copysign(1., l) * (x + 1 / l)) + std::log(std::fabs(l))) / l);
}

// ---- _ndtri_exp.pxd (Albert Steppi, scipy) -------------------------------------------------------------------------
static double ndtri_exp_small_y(double y) {
  static const double P1[] = {4.05544892305962419923, 3.15251094599893866154e1, 5.71628192246421288162e1, 4.40805073893200834700e1,
                              1.46849561928858024014e1, 2.18663306850790267539, -1.40256079171354495875e-1, -3.50424626827848203418e-2,
                              -8.57456785154685413611e-4};
  static const double Q1[] = {1.57799883256466749731e1, 4.53907635128879210584e1, 4.13172038254672030440e1, 1.50425385692907503408e1,
                              2.50464946208309415979, -1.42182922854787788574e-1, -3.80806407691578277194e-2, -9.33259480895457427372e-4};
  static const double P2[] = {3.23774891776946035970, 6.91522889068984211695, 3.93881025292474443415, 1.33303460815807542389,
                              2.01485389549179081538e-1, 1.23716634817820021358e-2, 3.01581553508235416007e-4, 2.65806974686737550832e-6,
                              6.23974539184983293730e-9};
  static const double Q2[] = {6.02427039364742014255, 3.67983563856160859403, 1.37702099489081330271, 2.16236993594496635890e-1,
                              1.34204006088543189037e-2, 3.28014464682127739104e-4, 2.89247864745380683936e-6, 6.79019408009981274425e-9};
  double x = y >= -DBL_MAX * 0.5 ? std::sqrt(-2 * y) : M_SQRT2 * std::sqrt(-y);
  double x0 = x - std::log(x) / x;
  double z = 1 / x, x1;
  if (x < 8.0) x1 = z * xsf::cephes::polevl(z, P1, 8) / xsf::cephes::p1evl(z, Q1, 8);
  else x1 = z * xsf::cephes::polevl(z, P2, 8) / xsf::cephes::p1evl(z, Q2, 8);
  return x1 - x0;
}
double ndtri_exp(double y) {
  if (y < -DBL_MAX) return -Inf;
  if (y < -2.0) return ndtri_exp_small_y(y);
  if (y > -0.14541345786885906) return -xsf::ndtri(-std::expm1(y));
  return xsf::ndtri(std::exp(y));
}

// ---- Bessel functions (Boost; scipy uses AMOS) -----------------------------------------------------------------------
static double jv(double v, double x) {
  if (std::isnan(v) || std::isnan(x)) return NaN;
  if (x < 0 && v != std::floor(v)) return NaN;
  return boost::math::cyl_bessel_j(v, x, SpecPol());
}
static double yv(double v, double x) {
  if (std::isnan(v) || std::isnan(x)) return NaN;
  if (x < 0) return NaN;
  if (x == 0) return -Inf;
  return boost::math::cyl_neumann(v, x, SpecPol());
}
static double iv(double v, double x) {
  if (std::isnan(v) || std::isnan(x)) return NaN;
  if (x < 0 && v != std::floor(v)) return NaN;
  return boost::math::cyl_bessel_i(v, x, SpecPol());
}
static double kv(double v, double x) {
  if (std::isnan(v) || std::isnan(x)) return NaN;
  if (x < 0) return NaN;
  if (x == 0) return Inf;
  return boost::math::cyl_bessel_k(v, x, SpecPol());
}

// xsf sph_bessel_j for real x (the cylindrical Bessel call goes to Boost).
static double spherical_jn(double nd, double x) {
  if (std::isnan(x) || std::isnan(nd)) return NaN;
  long n = long(nd);
  if (double(n) != nd || n < 0) return NaN;
  if (std::isinf(x)) return 0;
  if (x == 0) return n == 0 ? 1 : 0;
  if (n > 0 && n >= x) return std::sqrt(M_PI_2 / x) * jv(n + 0.5, x);
  double s0 = std::sin(x) / x;
  if (n == 0) return s0;
  double s1 = (s0 - std::cos(x)) / x;
  if (n == 1) return s1;
  double sn = 0;
  for (long i = 0; i < n - 1; ++i) {
    sn = (2 * i + 3) * s1 / x - s0;
    s0 = s1;
    s1 = sn;
    if (std::isinf(sn)) return sn;
  }
  return sn;
}

void airy(double x, double& ai, double& aip, double& bi, double& bip) {
  if (std::isnan(x)) { ai = aip = bi = bip = NaN; return; }
  if (x < -10 || x > 10) {
    ai = boost::math::airy_ai(x, SpecPol());
    aip = boost::math::airy_ai_prime(x, SpecPol());
    bi = boost::math::airy_bi(x, SpecPol());
    bip = boost::math::airy_bi_prime(x, SpecPol());
  } else {
    xsf::cephes::airy(x, &ai, &aip, &bi, &bip);
  }
}

// ---- Struve H: Cephes struve.h (scipy), with Boost Bessel functions -------------------------------------------------
namespace struve {
using xsf::cephes::gammasgn;
using xsf::cephes::lgam;
using xsf::cephes::rgamma;
using xsf::cephes::detail::double_double;
const int MAXITER = 10000;
const double SUM_EPS = 1e-16, SUM_TINY = 1e-100, GOOD_EPS = 1e-12, ACCEPTABLE_EPS = 1e-7, ACCEPTABLE_ATOL = 1e-300;

double asymp_large_z(double v, double z, double* err) {
  int n, maxiter;
  double term, sum, maxterm, m = z / 2;
  const int sgn = -1;
  if (m <= 0) maxiter = 0;
  else if (m > MAXITER) maxiter = MAXITER;
  else maxiter = int(m);
  if (maxiter == 0 || z < v) { *err = Inf; return NaN; }
  term = -sgn / std::sqrt(M_PI) * std::exp(-lgam(v + 0.5) + (v - 1) * std::log(z / 2)) * gammasgn(v + 0.5);
  sum = term;
  maxterm = 0;
  for (n = 0; n < maxiter; ++n) {
    term *= sgn * (1 + 2 * n) * (1 + 2 * n - 2 * v) / (z * z);
    sum += term;
    if (std::fabs(term) > maxterm) maxterm = std::fabs(term);
    if (std::fabs(term) < SUM_EPS * std::fabs(sum) || term == 0 || !std::isfinite(sum)) break;
  }
  sum += yv(v, z);
  *err = std::fabs(term) + std::fabs(maxterm) * SUM_EPS;
  return sum;
}

double power_series(double v, double z, double* err) {
  int n;
  double term, sum, maxterm, scaleexp, tmp;
  double_double cterm, csum, cdiv, z2, c2v, ctmp;
  const int sgn = -1;
  tmp = -lgam(v + 1.5) + (v + 1) * std::log(z / 2);
  if (tmp < -600 || tmp > 600) { scaleexp = tmp / 2; tmp -= scaleexp; }
  else scaleexp = 0;
  term = 2 / std::sqrt(M_PI) * std::exp(tmp) * gammasgn(v + 1.5);
  sum = term;
  maxterm = 0;
  cterm = double_double(term);
  csum = double_double(sum);
  z2 = double_double(sgn * z * z);
  c2v = double_double(2 * v);
  for (n = 0; n < MAXITER; ++n) {
    cdiv = double_double(3 + 2 * n);
    ctmp = double_double(3 + 2 * n);
    ctmp = ctmp + c2v;
    cdiv = cdiv * ctmp;
    cterm = cterm * z2;
    cterm = cterm / cdiv;
    csum = csum + cterm;
    term = static_cast<double>(cterm);
    sum = static_cast<double>(csum);
    if (std::fabs(term) > maxterm) maxterm = std::fabs(term);
    if (std::fabs(term) < SUM_TINY * std::fabs(sum) || term == 0 || !std::isfinite(sum)) break;
  }
  *err = std::fabs(term) + std::fabs(maxterm) * 1e-22;
  if (scaleexp != 0) { sum *= std::exp(scaleexp); *err *= std::exp(scaleexp); }
  return sum;
}

double bessel_series(double v, double z, double* err) {
  if (v < 0) { *err = Inf; return NaN; }
  double term = 0, sum = 0, maxterm = 0, cterm = std::sqrt(z / (2 * M_PI));
  for (int n = 0; n < MAXITER; ++n) {
    term = cterm * jv(n + v + 0.5, z) / (n + 0.5);
    cterm *= z / 2 / (n + 1);
    sum += term;
    if (std::fabs(term) > maxterm) maxterm = std::fabs(term);
    if (std::fabs(term) < SUM_EPS * std::fabs(sum) || term == 0 || !std::isfinite(sum)) break;
  }
  *err = std::fabs(term) + std::fabs(maxterm) * 1e-16;
  *err += 1e-300 * std::fabs(cterm);
  return sum;
}

double struve_h(double v, double z) {
  double value[3], err[3], tmp;
  int n;
  if (std::isnan(v) || std::isnan(z)) return NaN;
  if (z < 0) {
    n = int(v);
    if (v == n) return ((n % 2 == 0) ? -1 : 1) * struve_h(v, -z);
    return NaN;
  } else if (z == 0) {
    if (v < -1) return gammasgn(v + 1.5) * Inf;
    if (v == -1) return 2 / std::sqrt(M_PI) * rgamma(0.5);
    return 0;
  }
  n = int(-v - 0.5);
  if (n == -v - 0.5 && n > 0) return (n % 2 == 0 ? 1 : -1) * jv(n + 0.5, z);
  if (z >= 0.7 * v + 12) {
    value[0] = asymp_large_z(v, z, &err[0]);
    if (err[0] < GOOD_EPS * std::fabs(value[0])) return value[0];
  } else {
    err[0] = Inf;
  }
  value[1] = power_series(v, z, &err[1]);
  if (err[1] < GOOD_EPS * std::fabs(value[1])) return value[1];
  if (std::fabs(z) < std::fabs(v) + 20) {
    value[2] = bessel_series(v, z, &err[2]);
    if (err[2] < GOOD_EPS * std::fabs(value[2])) return value[2];
  } else {
    err[2] = Inf;
  }
  n = 0;
  if (err[1] < err[n]) n = 1;
  if (err[2] < err[n]) n = 2;
  if (err[n] < ACCEPTABLE_EPS * std::fabs(value[n]) || err[n] < ACCEPTABLE_ATOL) return value[n];
  tmp = -lgam(v + 1.5) + (v + 1) * std::log(z / 2);
  if (tmp > 700) return Inf * gammasgn(v + 1.5);
  return NaN;
}
}  // namespace struve

// ---- table ----------------------------------------------------------------------------------------------------------
static double lambertw_real(double z, double k) {
  if (std::isnan(z) || std::isnan(k) || k != std::floor(k)) return NaN;
  std::complex<double> w = xsf::lambertw(std::complex<double>(z, 0.0), long(k), 1e-8);
  return w.imag() == 0 ? w.real() : NaN;
}
static double polygamma(double n, double x) {
  // scipy.special.polygamma: (-1)^(n+1) gamma(n+1) zeta(n+1, x); n = 0 is digamma
  if (std::isnan(n) || std::isnan(x)) return NaN;
  if (n != std::floor(n)) return NaN;
  if (n == 0) return xsf::digamma(x);
  return std::pow(-1.0, n + 1) * xsf::gamma(n + 1.0) * xsf::zeta(n + 1, x);
}

const std::vector<SpecialFn>& special_table() {
  static const std::vector<SpecialFn> t = {
      {"erf", 1, [](const double* a) { return xsf::erf(a[0]); }, "error function"},
      {"erfc", 1, [](const double* a) { return xsf::erfc(a[0]); }, "complementary error function 1 - erf(x)"},
      {"erfcx", 1, [](const double* a) { return xsf::erfcx(a[0]); }, "scaled complementary error function exp(x^2) erfc(x)"},
      {"erfinv", 1, [](const double* a) { return erfinv(a[0]); }, "inverse of erf on (-1, 1)"},
      {"erfcinv", 1, [](const double* a) { return xsf::cephes::erfcinv(a[0]); }, "inverse of erfc on (0, 2)"},
      {"expit", 1, [](const double* a) { return xsf::expit(a[0]); }, "logistic sigmoid 1 / (1 + exp(-x))"},
      {"logit", 1, [](const double* a) { return xsf::logit(a[0]); }, "log(p / (1 - p))"},
      {"log_expit", 1, [](const double* a) { return xsf::log_expit(a[0]); }, "log of expit, accurate for large |x|"},
      {"ndtr", 1, [](const double* a) { return xsf::ndtr(a[0]); }, "standard normal cdf"},
      {"log_ndtr", 1, [](const double* a) { return xsf::log_ndtr(a[0]); }, "log of the standard normal cdf, accurate in the far tails"},
      {"ndtri", 1, [](const double* a) { return xsf::ndtri(a[0]); }, "standard normal quantile (inverse of ndtr)"},
      {"ndtri_exp", 1, [](const double* a) { return ndtri_exp(a[0]); }, "inverse of log_ndtr"},
      {"ellipk", 1, [](const double* a) { return xsf::ellipk(a[0]); }, "complete elliptic integral of the first kind K(m), parameter m = k^2"},
      {"ellipe", 1, [](const double* a) { return xsf::ellipe(a[0]); }, "complete elliptic integral of the second kind E(m)"},
      {"gamma", 1, [](const double* a) { return xsf::gamma(a[0]); }, "gamma function"},
      {"gammaln", 1, [](const double* a) { return xsf::gammaln(a[0]); }, "log |gamma(x)|"},
      {"digamma", 1, [](const double* a) { return xsf::digamma(a[0]); }, "digamma (psi) function"},
      {"psi", 1, [](const double* a) { return xsf::digamma(a[0]); }, "alias of digamma"},
      {"zeta", 1, [](const double* a) { return xsf::riemann_zeta(a[0]); }, "Riemann zeta function (scipy zeta(x) with q = None)"},
      {"lambertw", 1, [](const double* a) { return lambertw_real(a[0], 0); }, "principal branch W0 of the Lambert W function (NaN where W0 is complex)"},
      {"i0e", 1, [](const double* a) { return xsf::cephes::i0e(a[0]); }, "exponentially scaled modified Bessel function exp(-|x|) I0(x)"},
      {"cosm1", 1, [](const double* a) { return xsf::cosm1(a[0]); }, "cos(x) - 1 without cancellation"},
      {"gammainc", 2, [](const double* a) { return xsf::gammainc(a[0], a[1]); }, "regularized lower incomplete gamma P(a, x)"},
      {"gammaincc", 2, [](const double* a) { return xsf::gammaincc(a[0], a[1]); }, "regularized upper incomplete gamma Q(a, x)"},
      {"gammaincinv", 2, [](const double* a) { return xsf::gammaincinv(a[0], a[1]); }, "x with gammainc(a, x) = y"},
      {"gammainccinv", 2, [](const double* a) { return xsf::gammainccinv(a[0], a[1]); }, "x with gammaincc(a, x) = y"},
      {"zeta", 2, [](const double* a) { return xsf::zeta(a[0], a[1]); }, "Hurwitz zeta function zeta(x, q) (x >= 1)"},
      {"polygamma", 2, [](const double* a) { return polygamma(a[0], a[1]); }, "polygamma psi^(n)(x) for integral n >= 0"},
      {"lambertw", 2, [](const double* a) { return lambertw_real(a[0], a[1]); }, "Lambert W branch k (second argument) where it is real (k = 0 or -1), NaN otherwise"},
      {"jv", 2, [](const double* a) { return jv(a[0], a[1]); }, "Bessel function of the first kind J_v(x)"},
      {"yv", 2, [](const double* a) { return yv(a[0], a[1]); }, "Bessel function of the second kind Y_v(x)"},
      {"iv", 2, [](const double* a) { return iv(a[0], a[1]); }, "modified Bessel function of the first kind I_v(x)"},
      {"kv", 2, [](const double* a) { return kv(a[0], a[1]); }, "modified Bessel function of the second kind K_v(x)"},
      {"spherical_jn", 2, [](const double* a) { return spherical_jn(a[0], a[1]); }, "spherical Bessel function j_n(x), integral n >= 0"},
      {"struve", 2, [](const double* a) { return struve::struve_h(a[0], a[1]); }, "Struve function H_v(x)"},
      {"owens_t", 2, [](const double* a) { return xsf::owens_t(a[0], a[1]); }, "Owen's T function T(h, a)"},
      {"boxcox", 2, [](const double* a) { return boxcox(a[0], a[1]); }, "Box-Cox transform (x^l - 1) / l, log(x) for l = 0"},
      {"inv_boxcox", 2, [](const double* a) { return inv_boxcox(a[0], a[1]); }, "inverse Box-Cox transform"},
      {"boxcox1p", 2, [](const double* a) { return boxcox1p(a[0], a[1]); }, "Box-Cox transform of 1 + x"},
      {"inv_boxcox1p", 2, [](const double* a) { return inv_boxcox1p(a[0], a[1]); }, "inverse of boxcox1p"},
      {"xlogy", 2, [](const double* a) { return xsf::xlogy(a[0], a[1]); }, "x * log(y), 0 when x = 0"},
      {"xlog1py", 2, [](const double* a) { return xsf::xlog1py(a[0], a[1]); }, "x * log1p(y), 0 when x = 0"},
      {"beta", 2, [](const double* a) { return xsf::beta(a[0], a[1]); }, "beta function B(a, b)"},
      {"betaln", 2, [](const double* a) { return xsf::betaln(a[0], a[1]); }, "log |B(a, b)|"},
      {"poch", 2, [](const double* a) { return xsf::cephes::poch(a[0], a[1]); }, "Pochhammer symbol (a)_m = gamma(a + m) / gamma(a)"},
      {"betainc", 3, [](const double* a) { return betainc(a[0], a[1], a[2]); }, "regularized incomplete beta I_x(a, b)"},
      {"betaincc", 3, [](const double* a) { return betaincc(a[0], a[1], a[2]); }, "complement 1 - I_x(a, b)"},
      {"betaincinv", 3, [](const double* a) { return betaincinv(a[0], a[1], a[2]); }, "x with betainc(a, b, x) = y"},
      {"betainccinv", 3, [](const double* a) { return betainccinv(a[0], a[1], a[2]); }, "x with betaincc(a, b, x) = y"},
      {"hyp2f1", 4, [](const double* a) { return xsf::hyp2f1(a[0], a[1], a[2], a[3]); }, "Gauss hypergeometric function 2F1(a, b; c; z)"},
  };
  return t;
}

const SpecialFn& find_special(const std::string& name, int arity) {
  std::string same_name, all;
  for (auto& f : special_table()) {
    if (f.name == name && f.arity == arity) return f;
    if (f.name == name) same_name += (same_name.empty() ? "" : " or ") + std::to_string(f.arity);
  }
  if (name == "airy") fail("special: airy has four outputs; use airy(x), which returns Res {ai, aip, bi, bip}");
  if (name == "logsumexp") fail("special: logsumexp reduces a vector; use logsumexp(v)");
  if (!same_name.empty()) fail("special: " + name + " takes " + same_name + " argument(s), not " + std::to_string(arity));
  std::vector<std::string> names;
  for (auto& f : special_table())
    if (std::find(names.begin(), names.end(), f.name) == names.end()) names.push_back(f.name);
  for (auto& n : names) all += (all.empty() ? "" : ", ") + n;
  fail("special: unknown function '" + name + "'; known: " + all + ", plus airy(x) and logsumexp(v)");
}

double logsumexp(const std::vector<double>& v) {
  // scipy.special.logsumexp (1-D, no weights): a_max + log(sum(exp(a - a_max))), -inf for an empty or all -inf input
  if (v.empty()) return -Inf;
  double m = -Inf;
  for (double x : v) {
    if (std::isnan(x)) return NaN;
    if (x > m) m = x;
  }
  if (std::isinf(m)) return m;
  // scipy 1.16 _logsumexp: the maxima count separately, log1p(sum of the rest / count) + log(count) + max
  double cnt = 0, s = 0;
  for (double x : v) {
    if (x == m) cnt += 1;
    else s += std::exp(x - m);
  }
  if (s != 0) s /= cnt;
  return std::log1p(s) + std::log(cnt) + m;
}

}}  // namespace lmnum::st
