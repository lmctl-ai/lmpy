// Probability distributions with scipy.stats semantics (pycli stats_dist, 31 families). Each family is a port of the
// corresponding scipy 1.16 `*_gen` class (scipy/stats/_continuous_distns.py, _discrete_distns.py, BSD-3-Clause): the same
// formulas for _pdf/_logpdf/_cdf/_sf/_ppf/_isf/_stats and the same rv_continuous / rv_discrete defaults (sf = 1 - cdf,
// log cdf switching at the median, generic brentq quantiles with xtol 1e-14, discrete bisection search, moments from
// _munp). The special functions underneath are scipy's own (xsf/Cephes) or Boost.Math with scipy's policies, as scipy
// calls them (stats_special.cpp). Deviations are listed in native/API.md (section Statistics):
//   - Student t cdf/ppf use Boost students_t (scipy: CDFLIB, an ACM TOMS 708 derivative that lmnum does not use);
//   - Poisson ppf uses the generic discrete search (scipy: CDFLIB pdtrik + correction; same definition);
//   - levy_stable integrates Nolan's representation with Boost Gauss-Kronrod instead of QUADPACK.
// Random variates use the numpy Generator(PCG64(seed)) streams pycli used (stats_rng_numpy.cpp).
#include "stats_internal.hpp"

#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/cauchy.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/non_central_f.hpp>
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/math/distributions/skew_normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <functional>
#include <map>

namespace lmnum { namespace st {

namespace bp = boost::math::policies;
// scipy's StatsPolicy (boost_special_functions.h), with errors reported as IEEE values.
typedef bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>, bp::overflow_error<bp::ignore_error>,
                   bp::underflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>, bp::rounding_error<bp::ignore_error>,
                   bp::promote_float<false>, bp::promote_double<false>, bp::discrete_quantile<bp::integer_round_up>>
    StatPol;
typedef bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>, bp::overflow_error<bp::ignore_error>,
                   bp::underflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>, bp::rounding_error<bp::ignore_error>,
                   bp::promote_float<false>, bp::promote_double<false>, bp::max_root_iterations<400>>
    SpecPol;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();
const double EULER = 0.577215664901532860606512090082402431042;
const double ZETA3 = 1.202056903159594285399738161511449990765;
const double LOG_PI = 1.1447298858494002;
const double SQRT_2_OVER_PI = 0.7978845608028654;
const double XMIN = DBL_MIN;
const double NORM_PDF_C = std::sqrt(2 * M_PI);
const double NORM_PDF_LOGC = std::log(std::sqrt(2 * M_PI));

double norm_pdf(double x) { return std::exp(-x * x / 2.0) / NORM_PDF_C; }
double norm_logpdf(double x) { return -x * x / 2.0 - NORM_PDF_LOGC; }

// scipy.optimize brentq (scipy/optimize/Zeros/brentq.c, BSD-3-Clause)
template <class F> double brentq(F f, double xa, double xb, double xtol, double rtol = 4 * DBL_EPSILON, int iter = 100) {
  double xpre = xa, xcur = xb, xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis, delta, stry, dpre, dblk;
  fpre = f(xpre);
  fcur = f(xcur);
  if (fpre == 0) return xpre;
  if (fcur == 0) return xcur;
  if (std::signbit(fpre) == std::signbit(fcur)) return NaN;
  for (int i = 0; i < iter; i++) {
    if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
      xblk = xpre; fblk = fpre; spre = scur = xcur - xpre;
    }
    if (std::fabs(fblk) < std::fabs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    delta = (xtol + rtol * std::fabs(xcur)) / 2;
    sbis = (xblk - xcur) / 2;
    if (fcur == 0 || std::fabs(sbis) < delta) return xcur;
    if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
      if (xpre == xblk) {
        stry = -fcur * (xcur - xpre) / (fcur - fpre);
      } else {
        dpre = (fpre - fcur) / (xpre - xcur);
        dblk = (fblk - fcur) / (xblk - xcur);
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
  }
  return xcur;
}

// Globally adaptive Gauss-Kronrod (21 points) in the manner of QUADPACK qags/qagi: bisect the subinterval with the
// largest error estimate until the summed error meets rtol or `limit` subintervals exist. Infinite ends are mapped
// to (0, 1] by x = a + (1 - t) / t (qagi).
template <class F> double quad_finite(F f, double a, double b, double rtol, int limit) {
  struct Piece { double a, b, val, err; };
  auto rule = [&f](double lo, double hi) {
    double err = 0;
    double v = boost::math::quadrature::gauss_kronrod<double, 21>::integrate(f, lo, hi, 0, 0.0, &err);
    return Piece{lo, hi, v, err};
  };
  std::vector<Piece> ps = {rule(a, b)};
  for (int it = 1; it < limit; ++it) {
    double tot = 0, terr = 0;
    size_t worst = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
      tot += ps[i].val; terr += ps[i].err;
      if (ps[i].err > ps[worst].err) worst = i;
    }
    if (terr <= rtol * std::fabs(tot) || terr == 0) break;
    Piece w = ps[worst];
    double m = 0.5 * (w.a + w.b);
    if (!(m > w.a && m < w.b)) break;
    ps[worst] = rule(w.a, m);
    ps.push_back(rule(m, w.b));
  }
  double tot = 0;
  for (auto& p : ps) tot += p.val;
  return tot;
}
template <class F> double quad(F f, double a, double b, double rtol = 1e-13, int limit = 200) {
  if (a == b) return 0.0;
  if (std::isinf(a) && std::isinf(b)) return quad(f, a, 0.0, rtol, limit) + quad(f, 0.0, b, rtol, limit);
  if (std::isinf(b)) return quad_finite([&](double t) { return t <= 0 ? 0.0 : f(a + (1 - t) / t) / (t * t); }, 0.0, 1.0, rtol, limit);
  if (std::isinf(a)) return quad_finite([&](double t) { return t <= 0 ? 0.0 : f(b - (1 - t) / t) / (t * t); }, 0.0, 1.0, rtol, limit);
  return quad_finite(f, a, b, rtol, limit);
}

double polyval(std::initializer_list<double> c, double x) {
  double r = 0;
  for (double v : c) r = r * x + v;
  return r;
}
}  // namespace

// ---- the law of one family, standardised (loc 0, scale 1); methods mirror scipy's underscore methods --------------
struct Law {
  bool discrete = false;
  virtual ~Law() = default;
  virtual double a() const { return discrete ? 0 : -Inf; }
  virtual double b() const { return Inf; }
  virtual double pdf(double x) const { return std::exp(logpdf(x)); }
  virtual double logpdf(double x) const { return std::log(pdf(x)); }
  virtual double cdf(double x) const {
    if (discrete) {  // rv_discrete._cdf_single: sum of the mass function from the lower support end
      double k = std::floor(x), s = 0;
      for (double m = std::floor(a()); m <= k; m += 1) s += pdf(m);
      return s;
    }
    return quad([this](double t) { return pdf(t); }, a(), x);
  }
  virtual double sf(double x) const { return 1.0 - cdf(x); }
  mutable double median_cache = NaN;
  double median() const {
    if (std::isnan(median_cache)) median_cache = ppf(0.5);
    return median_cache;
  }
  virtual double logcdf(double x) const {
    if (discrete) return std::log(cdf(x));
    return x < median() ? std::log(cdf(x)) : std::log1p(-sf(x));
  }
  virtual double logsf(double x) const {
    if (discrete) return std::log(sf(x));
    return x > median() ? std::log(sf(x)) : std::log1p(-cdf(x));
  }
  virtual double ppf(double q) const { return discrete ? ppf_discrete(q) : ppf_continuous(q); }
  virtual double isf(double q) const { return ppf(1.0 - q); }
  // mean, var, skew, kurtosis; NaN entries are filled from munp by d_stats (scipy's None)
  virtual void stats(double& m, double& v, double& s, double& k) const { m = v = s = k = NaN; has_stats = false; }
  mutable bool has_stats = true;
  virtual double munp(int n) const {
    if (discrete) {
      double tot = 0;
      for (double x = std::floor(a()); x <= std::min(b(), 1e7); x += 1) {
        double d = std::pow(x, n) * pdf(x);
        tot += d;
        if (x > median() && std::fabs(d) < 1e-300) break;
      }
      return tot;
    }
    return quad([this, n](double t) { double p = pdf(t); return p == 0 ? 0.0 : std::pow(t, n) * p; }, a(), b(), 1.49e-8);
  }

  double ppf_continuous(double q) const {  // rv_continuous._ppf_single
    const double factor = 10.;
    double left = a(), right = b();
    auto f = [this, q](double x) { return cdf_public(x) - q; };
    if (std::isinf(left)) {
      left = std::min(-factor, right);
      while (f(left) > 0.) { right = left; left = left * factor; }
    }
    if (std::isinf(right)) {
      right = std::max(factor, left);
      while (f(right) < 0.) { left = right; right = right * factor; }
    }
    return brentq(f, left, right, 1e-14);
  }
  // rv_continuous.cdf restricted to the standard law (support handling around _cdf)
  double cdf_public(double x) const {
    if (std::isnan(x)) return NaN;
    if (x >= b()) return 1.0;
    if (x <= a()) return 0.0;
    return cdf(x);
  }
  double ppf_discrete(double q) const {  // _drv2_ppfsingle
    double _a = a(), _b = b(), A = _a, B = _b, qa, qb, step = 10;
    if (std::isinf(B)) {
      B = std::max(100 * q, 10.0);
      while (true) {
        if (B >= _b) { qb = 1.0; break; }
        qb = cdf(B);
        if (qb < q) { B += step; step *= 2; }
        else break;
      }
    } else {
      qb = 1.0;
    }
    qa = cdf(A);
    for (int i = 0; i < 2046; i++) {
      if (qa == q) return A;
      if (qb == q) return B;
      if (B <= A + 1) return qa > q ? A : B;
      double c = std::trunc((A + B) / 2.0);
      double qc = cdf(c);
      if (qc < q) { if (A != c) A = c; else return NaN; qa = qc; }
      else if (qc > q) { if (B != c) B = c; else return NaN; qb = qc; }
      else return c;
    }
    return NaN;
  }
};

namespace {

// ---- continuous families ----------------------------------------------------------------------------------------
struct Norm : Law {
  double pdf(double x) const override { return norm_pdf(x); }
  double logpdf(double x) const override { return norm_logpdf(x); }
  double cdf(double x) const override { return ndtr(x); }
  double logcdf(double x) const override { return log_ndtr(x); }
  double sf(double x) const override { return ndtr(-x); }
  double logsf(double x) const override { return log_ndtr(-x); }
  double ppf(double q) const override { return ndtri(q); }
  double isf(double q) const override { return -ndtri(q); }
  void stats(double& m, double& v, double& s, double& k) const override { m = 0; v = 1; s = 0; k = 0; }
};

struct StudentT : Law {
  double df;
  boost::math::students_t_distribution<double, StatPol> d;
  explicit StudentT(double df_) : df(df_), d(std::isinf(df_) ? 1.0 : df_) {}
  double pdf(double x) const override { return std::isinf(df) ? norm_pdf(x) : std::exp(logpdf(x)); }
  double logpdf(double x) const override {
    if (std::isinf(df)) return norm_logpdf(x);
    return std::log(poch(0.5 * df, 0.5)) - 0.5 * (std::log(df) + std::log(M_PI)) - (df + 1) / 2 * std::log1p(x * x / df);
  }
  double cdf(double x) const override {
    if (std::isinf(df)) return ndtr(x);
    if (std::isinf(x)) return x > 0 ? 1 : 0;
    return boost::math::cdf(d, x);
  }
  double sf(double x) const override { return cdf(-x); }
  double ppf(double q) const override { return std::isinf(df) ? ndtri(q) : boost::math::quantile(d, q); }
  double isf(double q) const override { return -ppf(q); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = df > 1 ? 0.0 : Inf;
    v = (df > 1 && df <= 2) ? Inf : (df > 2 && std::isfinite(df)) ? df / (df - 2.0) : std::isinf(df) ? 1 : NaN;
    s = df > 3 ? 0.0 : NaN;
    k = (df > 2 && df <= 4) ? Inf : (df > 4 && std::isfinite(df)) ? 6.0 / (df - 4.0) : std::isinf(df) ? 0 : NaN;
  }
};

struct Uniform : Law {
  double a() const override { return 0; }
  double b() const override { return 1; }
  double pdf(double) const override { return 1.0; }
  double cdf(double x) const override { return x; }
  double ppf(double q) const override { return q; }
  void stats(double& m, double& v, double& s, double& k) const override { m = 0.5; v = 1.0 / 12; s = 0; k = -1.2; }
};

struct Expon : Law {
  double a() const override { return 0; }
  double pdf(double x) const override { return std::exp(-x); }
  double logpdf(double x) const override { return -x; }
  double cdf(double x) const override { return -std::expm1(-x); }
  double ppf(double q) const override { return -std::log1p(-q); }
  double sf(double x) const override { return std::exp(-x); }
  double logsf(double x) const override { return -x; }
  double isf(double q) const override { return -std::log(q); }
  void stats(double& m, double& v, double& s, double& k) const override { m = 1; v = 1; s = 2; k = 6; }
};

struct Gamma : Law {
  double al;
  explicit Gamma(double a_) : al(a_) {}
  double a() const override { return 0; }
  double logpdf(double x) const override { return xlogy(al - 1.0, x) - x - gammaln(al); }
  double cdf(double x) const override { return gammainc(al, x); }
  double sf(double x) const override { return gammaincc(al, x); }
  double ppf(double q) const override { return gammaincinv(al, q); }
  double isf(double q) const override { return gammainccinv(al, q); }
  void stats(double& m, double& v, double& s, double& k) const override { m = al; v = al; s = 2.0 / std::sqrt(al); k = 6.0 / al; }
};

struct Beta : Law {
  double al, be;
  Beta(double a_, double b_) : al(a_), be(b_) {}
  double a() const override { return 0; }
  double b() const override { return 1; }
  double pdf(double x) const override {
    if (!std::isfinite(x)) return NaN;
    if (x >= 1 && be < 1) return Inf;
    if (x <= 0 && al < 1) return Inf;
    return boost::math::pdf(boost::math::beta_distribution<double, StatPol>(al, be), x);
  }
  double logpdf(double x) const override { return xlog1py(be - 1.0, -x) + xlogy(al - 1.0, x) - betaln(al, be); }
  double cdf(double x) const override { return betainc(al, be, x); }
  double sf(double x) const override { return betaincc(al, be, x); }
  double isf(double q) const override { return betainccinv(al, be, q); }
  double ppf(double q) const override { return betaincinv(al, be, q); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double apb = al + be;
    m = al / apb;
    v = al * be / (apb * apb * (apb + 1));
    s = (2 * (be - al) * std::sqrt(apb + 1)) / ((apb + 2) * std::sqrt(al * be));
    double n = 6 * ((al - be) * (al - be) * (apb + 1) - al * be * (apb + 2));
    double d = al * be * (apb + 2) * (apb + 3);
    k = n / d;
  }
};

struct Lognorm : Law {
  double s_;
  explicit Lognorm(double s) : s_(s) {}
  double a() const override { return 0; }
  double logpdf(double x) const override {
    if (x == 0) return -Inf;
    double l = std::log(x);
    return -l * l / (2 * s_ * s_) - std::log(s_ * x * std::sqrt(2 * M_PI));
  }
  double cdf(double x) const override { return ndtr(std::log(x) / s_); }
  double logcdf(double x) const override { return log_ndtr(std::log(x) / s_); }
  double ppf(double q) const override { return std::exp(s_ * ndtri(q)); }
  double sf(double x) const override { return ndtr(-(std::log(x) / s_)); }
  double logsf(double x) const override { return log_ndtr(-(std::log(x) / s_)); }
  double isf(double q) const override { return std::exp(s_ * -ndtri(q)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double p = std::exp(s_ * s_);
    m = std::sqrt(p);
    v = p * (p - 1);
    s = std::sqrt(p - 1) * (2 + p);
    k = polyval({1, 2, 3, 0, -6.0}, p);
  }
};

struct Chi2 : Law {
  double df;
  explicit Chi2(double d) : df(d) {}
  double a() const override { return 0; }
  double logpdf(double x) const override { return xlogy(df / 2. - 1, x) - x / 2. - gammaln(df / 2.) - (std::log(2) * df) / 2.; }
  double cdf(double x) const override { return chdtr(df, x); }
  double sf(double x) const override { return chdtrc(df, x); }
  double isf(double p) const override { return chdtri(df, p); }
  double ppf(double p) const override { return 2 * gammaincinv(df / 2, p); }
  void stats(double& m, double& v, double& s, double& k) const override { m = df; v = 2 * df; s = 2 * std::sqrt(2.0 / df); k = 12.0 / df; }
};

struct GenExtreme : Law {
  double c;
  explicit GenExtreme(double c_) : c(c_) {}
  double a() const override { return c < 0 ? 1.0 / std::min(c, -XMIN) : -Inf; }
  double b() const override { return c > 0 ? 1.0 / std::max(c, XMIN) : Inf; }
  double loglogcdf(double x) const { return (x == x && c != 0) ? std::log1p(-c * x) / c : -x; }
  double logpdf(double x) const override {
    double cx = (x == x && c != 0) ? c * x : 0.0;
    double logex2 = std::log1p(-cx);
    double logpex2 = loglogcdf(x);
    double pex2 = std::exp(logpex2);
    if (c == 0 && x == -Inf) logpex2 = 0.0;
    double r = !((cx == 1) || (cx == -Inf)) ? -pex2 + logpex2 - logex2 : -Inf;
    if (c == 1 && x == 1) r = 0.0;
    return r;
  }
  double logcdf(double x) const override { return -std::exp(loglogcdf(x)); }
  double cdf(double x) const override { return std::exp(logcdf(x)); }
  double sf(double x) const override { return -std::expm1(logcdf(x)); }
  double ppf(double q) const override {
    double x = -std::log(-std::log(q));
    return (x == x && c != 0) ? -std::expm1(-c * x) / c : x;
  }
  double isf(double q) const override {
    double x = -std::log(-std::log1p(-q));
    return (x == x && c != 0) ? -std::expm1(-c * x) / c : x;
  }
  void stats(double& m, double& v, double& s, double& k) const override {
    auto g = [this](double n) { return gammafn(n * c + 1); };
    double g1 = g(1), g2 = g(2), g3 = g(3), g4 = g(4);
    double g2mg12 = std::fabs(c) < 1e-7 ? std::pow(c * M_PI, 2.0) / 6.0 : g2 - std::pow(g1, 2.0);
    double gam2k = std::fabs(c) >= 1e-7 ? std::expm1(gammaln(2.0 * c + 1.0) - 2 * gammaln(c + 1.0)) / std::pow(c, 2.0) : M_PI * M_PI / 6.0;
    const double eps = 1e-14;
    double gamk = std::fabs(c) >= eps ? std::expm1(gammaln(c + 1)) / c : -EULER;
    m = c < -1.0 ? NaN : -gamk;
    v = c < -0.5 ? NaN : std::pow(g1, 2.0) * gam2k;
    if (std::fabs(c) > std::pow(eps, 0.29)) {
      double sg = c > 0 ? 1 : c < 0 ? -1 : 0;
      s = c >= -1. / 3 ? sg * (-g3 + (g2 + 2 * g2mg12) * g1) / std::pow(g2mg12, 1.5) : NaN;
    } else {
      s = 12 * std::sqrt(6) * ZETA3 / std::pow(M_PI, 3);
    }
    if (std::fabs(c) > std::pow(eps, 0.23)) k = c >= -1. / 4 ? (g4 + (-4 * g3 + 3 * (g2 + g2mg12) * g1) * g1) / (g2mg12 * g2mg12) - 3 : NaN;
    else k = 12.0 / 5.0;
  }
};

struct GenPareto : Law {
  double c;
  explicit GenPareto(double c_) : c(c_) {}
  double a() const override { return 0; }
  double b() const override { return c < 0 ? -1. / c : Inf; }
  double logpdf(double x) const override { return (x == x && c != 0) ? -xlog1py(c + 1., c * x) / c : -x; }
  double cdf(double x) const override { return -inv_boxcox1p(-x, -c); }
  double sf(double x) const override { return inv_boxcox(-x, -c); }
  double logsf(double x) const override { return (x == x && c != 0) ? -std::log1p(c * x) / c : -x; }
  double ppf(double q) const override { return -boxcox1p(-q, -c); }
  double isf(double q) const override { return -boxcox(q, -c); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double xi = c;
    m = xi < 1 ? 1 / (1 - xi) : Inf;
    v = xi < 0.5 ? 1 / std::pow(1 - xi, 2) / (1 - 2 * xi) : NaN;
    s = xi < 1. / 3 ? 2 * (1 + xi) * std::sqrt(1 - 2 * xi) / (1 - 3 * xi) : NaN;
    k = xi < 0.25 ? 3 * (1 - 2 * xi) * (2 * xi * xi + xi + 3) / (1 - 3 * xi) / (1 - 4 * xi) - 3 : NaN;
  }
};

struct SkewNorm : Law {
  double al;
  explicit SkewNorm(double a_) : al(a_) {}
  double pdf(double x) const override { return al == 0 ? norm_pdf(x) : 2. * norm_pdf(x) * ndtr(al * x); }
  double logpdf(double x) const override { return al == 0 ? norm_logpdf(x) : std::log(2) + norm_logpdf(x) + log_ndtr(al * x); }
  double cdf_of(double x, double a_) const {
    double r;
    if (std::isfinite(x)) r = boost::math::cdf(boost::math::skew_normal_distribution<double, StatPol>(0.0, 1.0, a_), x);
    else r = 1 - (std::signbit(x) ? 1 : 0);
    if (r < 1e-6 && a_ > 0) {  // scipy falls back to integrating the density
      r = boost::math::quadrature::exp_sinh<double>().integrate([x, a_](double t) {
        double u = x - t;
        return a_ == 0 ? norm_pdf(u) : 2. * norm_pdf(u) * ndtr(a_ * u); }, 0.0, Inf);
    }
    return std::min(1.0, std::max(0.0, r));
  }
  double cdf(double x) const override { return cdf_of(x, al); }
  double ppf(double q) const override { return boost::math::quantile(boost::math::skew_normal_distribution<double, StatPol>(0.0, 1.0, al), q); }
  double sf(double x) const override { return cdf_of(-x, -al); }
  double isf(double q) const override {
    return boost::math::quantile(boost::math::complement(boost::math::skew_normal_distribution<double, StatPol>(0.0, 1.0, al), q));
  }
  void stats(double& m, double& v, double& s, double& k) const override {
    double cst = std::sqrt(2 / M_PI) * al / std::sqrt(1 + al * al);
    m = cst;
    v = 1 - cst * cst;
    s = ((4 - M_PI) / 2) * std::pow(cst / std::sqrt(1 - cst * cst), 3);
    k = (2 * (M_PI - 3)) * (std::pow(cst, 4) / std::pow(1 - cst * cst, 2));
  }
};

struct JohnsonSU : Law {
  double al, be;
  JohnsonSU(double a_, double b_) : al(a_), be(b_) {}
  double pdf(double x) const override { return be * 1.0 / std::sqrt(x * x + 1.0) * norm_pdf(al + be * std::asinh(x)); }
  double cdf(double x) const override { return ndtr(al + be * std::asinh(x)); }
  double ppf(double q) const override { return std::sinh((ndtri(q) - al) / be); }
  double sf(double x) const override { return ndtr(-(al + be * std::asinh(x))); }
  double isf(double q) const override { return std::sinh((-ndtri(q) - al) / be); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double bn2 = std::pow(be, -2.), e = std::exp(bn2), ab = al / be;
    m = -std::pow(e, 0.5) * std::sinh(ab);
    v = 0.5 * std::expm1(bn2) * (e * std::cosh(2 * ab) + 1);
    double t1 = std::pow(e, .5) * std::pow(std::expm1(bn2), 0.5), t2 = 3 * std::sinh(ab), t3 = e * (e + 2) * std::sinh(3 * ab);
    s = -t1 * (t2 + t3) / (std::sqrt(2) * std::pow(1 + e * std::cosh(2 * ab), 1.5));
    double u1 = 3 + 6 * e, u2 = 4 * e * e * (e + 2) * std::cosh(2 * ab), u3 = e * e * std::cosh(4 * ab),
           u4 = -3 + 3 * e * e + 2 * e * e * e + e * e * e * e;
    k = (u1 + u2 + u3 * u4) / (2 * std::pow(1 + e * std::cosh(2 * ab), 2)) - 3;
  }
};

// ---- levy_stable: Nolan's integral representation (scipy _levy_stable, S1 parameterisation) ------------------------
struct Nolan {
  double alpha, zeta, xi, zeta_prefactor = 0, alpha_exp = 0, alpha_xi = 0, zeta_offset = 0, two_beta_div_pi = 0, pi_div_two_beta = 0,
         x0_div_term = 0, c1, c2, c3;
  Nolan(double al, double be, double x0) : alpha(al) {
    zeta = -be * std::tan(M_PI_2 * al);
    if (al != 1.) {
      xi = std::atan(-zeta) / al;
      zeta_prefactor = std::pow(std::pow(zeta, 2.) + 1., -1. / (2. * (al - 1.)));
      alpha_exp = al / (al - 1.);
      alpha_xi = std::atan(-zeta);
      zeta_offset = x0 - zeta;
      if (al < 1.) { c1 = 0.5 - xi * M_1_PI; c3 = M_1_PI; }
      else { c1 = 1.; c3 = -M_1_PI; }
      c2 = al * M_1_PI / std::fabs(al - 1.) / (x0 - zeta);
    } else {
      xi = M_PI_2;
      two_beta_div_pi = be * M_2_PI;
      pi_div_two_beta = M_PI_2 / be;
      x0_div_term = x0 / two_beta_div_pi;
      c1 = 0.;
      c2 = .5 / std::fabs(be);
      c3 = M_1_PI;
    }
  }
  double g(double theta) const {
    if (alpha != 1) {
      if (theta == -xi) return alpha < 1 ? 0 : Inf;
      if (theta == M_PI_2) return alpha < 1 ? Inf : 0;
      double ct = std::cos(theta);
      return zeta_prefactor * std::pow(ct / std::sin(alpha_xi + alpha * theta) * zeta_offset, alpha_exp) * std::cos(alpha_xi + (alpha - 1) * theta) / ct;
    }
    if (theta == -xi) return 0;
    if (theta == M_PI_2) return Inf;
    return (1 + theta * two_beta_div_pi) * std::exp((pi_div_two_beta + theta) * std::tan(theta) - x0_div_term) / std::cos(theta);
  }
};

double integrate_pieces(const std::function<double(double)>& f, std::vector<double> pts) {
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  double s = 0;
  for (size_t i = 0; i + 1 < pts.size(); ++i)
    if (pts[i + 1] > pts[i]) s += quad(f, pts[i], pts[i + 1], 1.2e-14);
  return s;
}

double bisect(const std::function<double(double)>& f, double a, double b, double xtol) {
  double fa = f(a);
  for (int i = 0; i < 200 && b - a > xtol; ++i) {
    double m = 0.5 * (a + b), fm = f(m);
    if (std::signbit(fm) == std::signbit(fa)) { a = m; fa = fm; } else b = m;
  }
  return 0.5 * (a + b);
}

const double LS_X_TOL = 0.005, LS_ALPHA_TOL = 0.005, LS_QUAD_EPS = 1.2e-14;

double round_near_zeta(double x0, double alpha, double zeta) {
  if (std::fabs(x0 - zeta) < LS_X_TOL * std::pow(alpha, 1 / alpha)) x0 = zeta;
  return x0;
}

double ls_pdf_post(double x0, double alpha, double beta) {
  Nolan n(alpha, beta, x0);
  x0 = round_near_zeta(x0, alpha, n.zeta);
  if (x0 == n.zeta) return gammafn(1 + 1 / alpha) * std::cos(n.xi) / M_PI / std::pow(1 + n.zeta * n.zeta, 1 / alpha / 2);
  if (x0 < n.zeta) return ls_pdf_post(-x0, alpha, -beta);
  if (std::fabs(-n.xi - M_PI / 2) <= 1e-14 + 1e-14 * std::fabs(M_PI / 2)) return 0.0;
  auto integrand = [&n](double th) {
    double g1 = n.g(th);
    if (!std::isfinite(g1) || g1 < 0) g1 = 0;
    return g1 * std::exp(-g1);
  };
  auto at = [&n](double h) { return [&n, h](double t) { return n.g(t) - h; }; };
  std::vector<double> pts = {-n.xi, M_PI / 2, bisect(at(1), -n.xi, M_PI / 2, LS_QUAD_EPS)};
  for (double h : {100.0, 10.0, 5.0}) pts.push_back(bisect(at(h), -n.xi, M_PI / 2, 2e-12));
  if (0 > -n.xi && 0 < M_PI / 2) pts.push_back(0);
  return n.c2 * integrate_pieces(integrand, pts);
}

double ls_cdf_post(double x0, double alpha, double beta) {
  Nolan n(alpha, beta, x0);
  x0 = round_near_zeta(x0, alpha, n.zeta);
  if ((alpha == 1 && beta < 0) || x0 < n.zeta) return 1 - ls_cdf_post(-x0, alpha, -beta);
  if (x0 == n.zeta) return 0.5 - n.xi / M_PI;
  if (std::fabs(-n.xi - M_PI / 2) <= 1e-14 + 1e-14 * std::fabs(M_PI / 2)) return n.c1;
  auto integrand = [&n](double th) { return std::exp(-n.g(th)); };
  std::vector<double> pts = {-n.xi, M_PI / 2};
  // break points where exp(-g) falls through 1e-300 ... 1 help the adaptive rule find the support
  for (double h : {1.0, 5.0, 10.0, 100.0}) pts.push_back(bisect([&n, h](double t) { return n.g(t) - h; }, -n.xi, M_PI / 2, 2e-12));
  return n.c1 + n.c3 * integrate_pieces(integrand, pts);
}

struct LevyStable : Law {
  double alpha, beta;
  LevyStable(double a_, double b_) : alpha(a_), beta(b_) {}
  void z0(double x, double& x0, double& al, double& be) const {
    double zeta = -beta * std::tan(M_PI * alpha / 2.0);
    x0 = alpha != 1 ? x + zeta : x;
    al = alpha; be = beta;
    if (std::fabs(al - 1) < LS_ALPHA_TOL) al = 1.0;
    x0 = round_near_zeta(x0, al, -be * std::tan(M_PI * al / 2.0));
  }
  double pdf(double x) const override {
    double x0, al, be;
    z0(x, x0, al, be);
    if (al == 2.0) return norm_pdf(x0 / std::sqrt(2)) / std::sqrt(2);
    if (al == 0.5 && be == 1.0) {
      double u = x0 + 1;
      if (u <= 0) return 0;
      return 1 / std::sqrt(2 * M_PI * u) / u * std::exp(-1 / (2 * u));
    }
    if (al == 0.5 && be == 0.0 && x0 != 0) {
      double z = 1 / std::sqrt(2 * M_PI * std::fabs(x0)), S = fresnel_s(z), C = fresnel_c(z), arg = 1 / (4 * std::fabs(x0));
      return (std::sin(arg) * (0.5 - S) + std::cos(arg) * (0.5 - C)) / std::sqrt(2 * M_PI * std::pow(std::fabs(x0), 3));
    }
    if (al == 1.0 && be == 0.0) return 1 / (1 + x0 * x0) / M_PI;
    return ls_pdf_post(x0, al, be);
  }
  double cdf(double x) const override {
    double x0, al, be;
    z0(x, x0, al, be);
    if (al == 2.0) return ndtr(x0 / std::sqrt(2));
    if (al == 0.5 && be == 1.0) {
      double u = x0 + 1;
      if (u <= 0) return 0;
      return erfc(std::sqrt(0.5 / u));
    }
    if (al == 1.0 && be == 0.0) return 0.5 + std::atan(x0) / M_PI;
    return ls_cdf_post(x0, al, be);
  }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = alpha > 1 ? 0 : NaN;
    v = alpha == 2 ? 2 : Inf;
    s = alpha == 2.0 ? 0.0 : NaN;
    k = alpha == 2.0 ? 0.0 : NaN;
  }
};

struct NCT : Law {
  double df, nc;
  boost::math::non_central_t_distribution<double, StatPol> d;
  boost::math::non_central_t_distribution<double, SpecPol> ds;
  NCT(double df_, double nc_) : df(df_), nc(nc_), d(df_, nc_), ds(df_, nc_) {}
  double pdf(double x) const override {
    if (!std::isfinite(x)) return NaN;
    double p = boost::math::pdf(d, x);
    if (p == 0.0) {  // pycli: recompute by quadrature where Boost underflows inside the body of the distribution
      double c = cdf(x);
      if (c > 1e-300 && c < 1 - 1e-16) p = density_by_quadrature(x);
    }
    return p;
  }
  double density_by_quadrature(double t) const {
    double half = df / 2.0, log_norm = std::log(2.0) + std::log(df) - half * std::log(2.0) - gammaln(half);
    auto integrand = [&](double s) {
      if (s <= 0) return 0.0;
      double z = t * s - nc;
      return std::exp(std::log(s) + log_norm + std::log(s) + (df - 2.0) * std::log(s * std::sqrt(df)) - s * s * df / 2.0 - 0.5 * z * z - 0.5 * std::log(2 * M_PI));
    };
    double peak = (t != 0 && nc / t > 0) ? nc / t : 0;
    double total = peak > 0 ? quad(integrand, 0.0, peak, 1e-10) + quad(integrand, peak, Inf, 1e-10) : quad(integrand, 0.0, Inf, 1e-10);
    return std::max(total, 0.0);
  }
  double cdf(double x) const override {
    if (std::isinf(x)) return x > 0 ? 1.0 : 0.0;
    double y = boost::math::cdf(ds, x);
    return (y < 0 || y > 1) ? NaN : y;
  }
  double ppf(double q) const override { return boost::math::quantile(ds, q); }
  double sf(double x) const override { return std::min(1.0, std::max(0.0, boost::math::cdf(boost::math::complement(d, x)))); }
  double isf(double q) const override { return boost::math::quantile(boost::math::complement(d, q)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = boost::math::mean(d); v = boost::math::variance(d); s = boost::math::skewness(d); k = boost::math::kurtosis_excess(d);
  }
};

struct NCF : Law {
  double dfn, dfd, nc;
  boost::math::non_central_f_distribution<double, StatPol> d;
  boost::math::non_central_f_distribution<double, SpecPol> ds;
  NCF(double a_, double b_, double c_) : dfn(a_), dfd(b_), nc(c_), d(a_, b_, c_), ds(a_, b_, c_) {}
  double a() const override { return 0; }
  double pdf(double x) const override { return std::isfinite(x) ? boost::math::pdf(d, x) : NaN; }
  double cdf(double x) const override {
    if (x < 0) return NaN;
    if (std::isinf(x)) return 1.0;
    double y = boost::math::cdf(ds, x);
    return (y < 0 || y > 1) ? NaN : y;
  }
  double ppf(double q) const override { return boost::math::quantile(ds, q); }
  double sf(double x) const override { return boost::math::cdf(boost::math::complement(d, x)); }
  double isf(double q) const override { return boost::math::quantile(boost::math::complement(d, q)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = boost::math::mean(d); v = boost::math::variance(d); s = boost::math::skewness(d);
    k = boost::math::kurtosis_excess(d) - 3;  // as scipy (ncf_gen._stats)
  }
};

// scipy/stats/_stats.pyx von_mises_cdf
double von_mises_cdf_series(double k, double x, unsigned p) {
  double s = std::sin(x), c = std::cos(x), sn = std::sin(p * x), cn = std::cos(p * x), R = 0, V = 0;
  for (unsigned n = p - 1; n > 0; n--) {
    double sn2 = sn * c - cn * s, cn2 = cn * c + sn * s;
    sn = sn2; cn = cn2;
    R = k / (2 * n + k * R);
    V = R * (sn / n + V);
  }
  return 0.5 + x / (2 * M_PI) + V / M_PI;
}
double von_mises_cdf(double k, double x) {
  double ix = std::nearbyint(x / (2 * M_PI));
  x = x - ix * (2 * M_PI);
  double r;
  if (k < 50) {
    unsigned p = unsigned(int(1 + 28. + 0.5 * k - 100. / (k + 5.)));
    r = von_mises_cdf_series(k, x, p);
    r = r < 0 ? 0 : r > 1 ? 1 : r;
  } else {
    double b = 0.79788456080286535588 / i0e(k);
    r = ndtr(b * std::sin(x / 2.));
  }
  return r + ix;
}

struct VonMises : Law {
  double kappa;
  explicit VonMises(double k) : kappa(k) {}
  double a() const override { return -M_PI; }
  double b() const override { return M_PI; }
  double pdf(double x) const override { return std::exp(kappa * cosm1(x)) / (2 * M_PI * i0e(kappa)); }
  double logpdf(double x) const override { return kappa * cosm1(x) - std::log(2 * M_PI) - std::log(i0e(kappa)); }
  double cdf(double x) const override { return von_mises_cdf(kappa, x); }
};

struct Truncnorm : Law {
  double al, be, lmass;
  static double log_diff(double lp, double lq) { return lp + std::log1p(-std::exp(lq - lp)); }
  static double log_sum(double lp, double lq) {
    double m = std::max(lp, lq);
    if (std::isinf(m)) return m;
    return m + std::log(std::exp(lp - m) + std::exp(lq - m));
  }
  static double log_gauss_mass(double a, double b) {
    if (b <= 0) return log_diff(log_ndtr(b), log_ndtr(a));
    if (a > 0) return log_diff(log_ndtr(-a), log_ndtr(-b));
    return std::log1p(-ndtr(a) - ndtr(-b));
  }
  Truncnorm(double a_, double b_) : al(a_), be(b_), lmass(log_gauss_mass(a_, b_)) {}
  double a() const override { return al; }
  double b() const override { return be; }
  double logpdf(double x) const override { return norm_logpdf(x) - lmass; }
  double logcdf_(double x) const { return log_gauss_mass(al, x) - lmass; }
  double logsf_(double x) const { return log_gauss_mass(x, be) - lmass; }
  double logcdf(double x) const override {
    double r = logcdf_(x);
    return r > -0.1 ? std::log1p(-std::exp(logsf_(x))) : r;
  }
  double logsf(double x) const override {
    double r = logsf_(x);
    return r > -0.1 ? std::log1p(-std::exp(logcdf_(x))) : r;
  }
  double cdf(double x) const override { return std::exp(logcdf(x)); }
  double sf(double x) const override { return std::exp(logsf(x)); }
  double ppf(double q) const override {
    if (al < 0) return ndtri_exp(log_sum(log_ndtr(al), std::log(q) + lmass));
    return -ndtri_exp(log_sum(log_ndtr(-be), std::log1p(-q) + lmass));
  }
  double isf(double q) const override {
    if (be < 0) return ndtri_exp(log_diff(log_ndtr(be), std::log(q) + lmass));
    return -ndtri_exp(log_diff(log_ndtr(-al), std::log1p(-q) + lmass));
  }
  void stats(double& m, double& v, double& s, double& k) const override {
    double pA = std::isfinite(al) ? std::exp(logpdf(al)) : 0, pB = std::isfinite(be) ? std::exp(logpdf(be)) : 0;
    double ab[2] = {al, be}, probs[2] = {pA, -pB};
    auto sum = [&](auto fn) { double t = 0; for (int i = 0; i < 2; ++i) if (probs[i] != 0) t += fn(probs[i], ab[i]); return t; };
    double m1 = pA - pB;
    double mu = m1;
    double m2 = 1 + sum([](double x, double y) { return x * y; });
    double mu2 = 1 + sum([mu](double x, double y) { return x * (y - mu); });
    double m3 = 2 * m1 + sum([](double x, double y) { return x * y * y; });
    double m4 = 3 * m2 + sum([](double x, double y) { return x * y * y * y; });
    double mu3 = m3 + m1 * (-3 * m2 + 2 * m1 * m1);
    double mu4 = m4 + m1 * (-4 * m3 + 3 * m1 * (2 * m2 - m1 * m1));
    m = mu; v = mu2; s = mu3 / std::pow(mu2, 1.5); k = mu4 / (mu2 * mu2) - 3;
  }
};

struct WeibullMin : Law {
  double c;
  explicit WeibullMin(double c_) : c(c_) {}
  double a() const override { return 0; }
  double pdf(double x) const override { return c * std::pow(x, c - 1) * std::exp(-std::pow(x, c)); }
  double logpdf(double x) const override { return std::log(c) + xlogy(c - 1, x) - std::pow(x, c); }
  double cdf(double x) const override { return -std::expm1(-std::pow(x, c)); }
  double ppf(double q) const override { return std::pow(-std::log1p(-q), 1.0 / c); }
  double sf(double x) const override { return std::exp(logsf(x)); }
  double logsf(double x) const override { return -std::pow(x, c); }
  double isf(double q) const override { return std::pow(-std::log(q), 1 / c); }
  void stats(double& m, double& v, double& s, double& k) const override {
    // pycli weibull_moments: mean from Gamma(1 + 1/c), central moments by quadrature (keeps digits for sharp shapes)
    double mu = gammafn(1.0 + 1.0 / c), shift = mu - 1.0;
    auto central = [&](int kk) {
      auto f = [&](double t) {
        if (t > 0) return std::pow(std::expm1(std::log(t) / c) - shift, kk) * std::exp(-t);
        return t == 0 ? std::pow(-shift, kk) : 0.0;
      };
      return quad(f, 0.0, 1.0, 1e-11) + quad(f, 1.0, Inf, 1e-11);
    };
    double m2 = central(2), m3 = central(3), m4 = central(4);
    m = mu; v = m2; s = m3 / std::pow(m2, 1.5); k = m4 / (m2 * m2) - 3.0;
  }
};

struct Cauchy : Law {
  double pdf(double x) const override { return 1.0 / M_PI / (1.0 + x * x); }
  double logpdf(double x) const override {
    double ax = std::fabs(x);
    return ax < 1 ? -LOG_PI - std::log1p(ax * ax) : -LOG_PI - (2 * std::log(ax) + std::log1p(std::pow(1 / ax, 2)));
  }
  double cdf(double x) const override { return std::atan2(1, -x) / M_PI; }
  double sf(double x) const override { return std::atan2(1, x) / M_PI; }
  double ppf(double q) const override { return boost::math::quantile(boost::math::cauchy_distribution<double, StatPol>(0, 1), q); }
  double isf(double q) const override {
    return boost::math::quantile(boost::math::complement(boost::math::cauchy_distribution<double, StatPol>(0, 1), q));
  }
  void stats(double& m, double& v, double& s, double& k) const override { m = v = s = k = NaN; }
};

struct Laplace : Law {
  double pdf(double x) const override { return 0.5 * std::exp(-std::fabs(x)); }
  double cdf(double x) const override { return x > 0 ? 1.0 - 0.5 * std::exp(-x) : 0.5 * std::exp(x); }
  double sf(double x) const override { return cdf(-x); }
  double ppf(double q) const override { return q > 0.5 ? -std::log(2 * (1 - q)) : std::log(2 * q); }
  double isf(double q) const override { return -ppf(q); }
  void stats(double& m, double& v, double& s, double& k) const override { m = 0; v = 2; s = 0; k = 3; }
};

struct Logistic : Law {
  double logpdf(double x) const override { double y = -std::fabs(x); return y - 2. * std::log1p(std::exp(y)); }
  double cdf(double x) const override { return expit(x); }
  double logcdf(double x) const override { return log_expit(x); }
  double ppf(double q) const override { return logit(q); }
  double sf(double x) const override { return expit(-x); }
  double logsf(double x) const override { return log_expit(-x); }
  double isf(double q) const override { return -logit(q); }
  void stats(double& m, double& v, double& s, double& k) const override { m = 0; v = M_PI * M_PI / 3.0; s = 0; k = 6.0 / 5.0; }
};

struct GumbelR : Law {
  double logpdf(double x) const override { return -x - std::exp(-x); }
  double cdf(double x) const override { return std::exp(-std::exp(-x)); }
  double logcdf(double x) const override { return -std::exp(-x); }
  double ppf(double q) const override { return -std::log(-std::log(q)); }
  double sf(double x) const override { return -std::expm1(-std::exp(-x)); }
  double isf(double p) const override { return -std::log(-std::log1p(-p)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = EULER; v = M_PI * M_PI / 6.0; s = 12 * std::sqrt(6) / std::pow(M_PI, 3) * ZETA3; k = 12.0 / 5;
  }
};

struct Pareto : Law {
  double bb;
  explicit Pareto(double b_) : bb(b_) {}
  double a() const override { return 1; }
  double pdf(double x) const override { return bb * std::pow(x, -bb - 1); }
  double cdf(double x) const override { return 1 - std::pow(x, -bb); }
  double ppf(double q) const override { return std::pow(1 - q, -1.0 / bb); }
  double sf(double x) const override { return std::pow(x, -bb); }
  double isf(double q) const override { return std::pow(q, -1.0 / bb); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double b = bb;
    m = b > 1 ? b / (b - 1.0) : Inf;
    v = b > 2 ? b / (b - 2.0) / std::pow(b - 1.0, 2) : Inf;
    s = b > 3 ? 2 * (b + 1.0) * std::sqrt(b - 2.0) / ((b - 3.0) * std::sqrt(b)) : NaN;
    k = b > 4 ? 6.0 * polyval({1.0, 1.0, -6, -2}, b) / polyval({1.0, -7.0, 12.0, 0.0}, b) : NaN;
  }
};

struct Rayleigh : Law {
  double a() const override { return 0; }
  double logpdf(double r) const override { return std::log(r) - 0.5 * r * r; }
  double cdf(double r) const override { return -std::expm1(-0.5 * r * r); }
  double ppf(double q) const override { return std::sqrt(-2 * std::log1p(-q)); }
  double sf(double r) const override { return std::exp(logsf(r)); }
  double logsf(double r) const override { return -0.5 * r * r; }
  double isf(double q) const override { return std::sqrt(-2 * std::log(q)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double val = 4 - M_PI;
    m = std::sqrt(M_PI / 2); v = val / 2; s = 2 * (M_PI - 3) * std::sqrt(M_PI) / std::pow(val, 1.5); k = 6 * M_PI / val - 16 / (val * val);
  }
};

struct FDist : Law {
  double dfn, dfd;
  FDist(double a_, double b_) : dfn(a_), dfd(b_) {}
  double a() const override { return 0; }
  double logpdf(double x) const override {
    double n = dfn, m = dfd;
    return m / 2 * std::log(m) + n / 2 * std::log(n) + xlogy(n / 2 - 1, x) - (((n + m) / 2) * std::log(m + n * x) + betaln(n / 2, m / 2));
  }
  double cdf(double x) const override { return fdtr(dfn, dfd, x); }
  double sf(double x) const override { return fdtrc(dfn, dfd, x); }
  double ppf(double q) const override { return fdtri(dfn, dfd, q); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double v1 = dfn, v2 = dfd, v2_2 = v2 - 2., v2_4 = v2 - 4., v2_6 = v2 - 6., v2_8 = v2 - 8.;
    m = v2 > 2 ? v2 / v2_2 : Inf;
    v = v2 > 4 ? 2 * v2 * v2 * (v1 + v2_2) / (v1 * v2_2 * v2_2 * v2_4) : Inf;
    s = v2 > 6 ? (2 * v1 + v2_2) / v2_6 * std::sqrt(v2_4 / (v1 * (v1 + v2_2))) : NaN;
    s *= std::sqrt(8.);
    k = v2 > 8 ? (8 + s * s * v2_6) / v2_8 : NaN;
    k *= 3. / 2.;
  }
};

// ---- discrete families ------------------------------------------------------------------------------------------
struct Poisson : Law {
  double mu;
  explicit Poisson(double m) : mu(m) { discrete = true; }
  double logpdf(double k) const override { return xlogy(k, mu) - gammaln(k + 1) - mu; }
  double pdf(double k) const override { return std::exp(logpdf(k)); }
  double cdf(double x) const override { return pdtr(std::floor(x), mu); }
  double sf(double x) const override { return pdtrc(std::floor(x), mu); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = mu; v = mu; s = mu > 0 ? std::sqrt(1.0 / mu) : Inf; k = mu > 0 ? 1.0 / mu : Inf;
  }
};

struct Binom : Law {
  double n, p;
  boost::math::binomial_distribution<double, StatPol> d;
  Binom(double n_, double p_) : n(n_), p(p_), d(n_, p_) { discrete = true; }
  double b() const override { return n; }
  double logpdf(double x) const override {
    double k = std::floor(x);
    double combiln = gammaln(n + 1) - (gammaln(k + 1) + gammaln(n - k + 1));
    return combiln + xlogy(k, p) + xlog1py(n - k, -p);
  }
  double pdf(double x) const override { return std::isfinite(x) ? boost::math::pdf(d, x) : NaN; }
  double cdf(double x) const override { return boost::math::cdf(d, std::floor(x)); }
  double sf(double x) const override { return boost::math::cdf(boost::math::complement(d, std::floor(x))); }
  double ppf(double q) const override { return boost::math::quantile(d, q); }
  double isf(double q) const override { return boost::math::quantile(boost::math::complement(d, q)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = n * p; v = m - n * p * p;
    double pq = p - p * p, npq_sqrt = std::sqrt(n * pq);
    s = 1.0 / npq_sqrt - (2.0 * p) / npq_sqrt;
    k = 1.0 / (n * pq) - 6.0 / n;
  }
};

const double ZIPF_MAX = 1e5;
struct Zipf : Law {
  double al, zinv;
  explicit Zipf(double a_) : al(a_) { discrete = true; zinv = 1.0 / zeta(al, 1); }
  double a() const override { return 1; }
  double pdf(double k) const override { return zinv * std::pow(k, -al); }
  void stats(double& m, double& v, double& s, double& k) const override { m = v = s = k = NaN; has_stats = false; }
  double munp(int n) const override { return al > n + 1 ? zeta(al - n, 1) / zeta(al, 1) : Inf; }
};

struct BetaBinom : Law {
  double n, al, be;
  BetaBinom(double n_, double a_, double b_) : n(n_), al(a_), be(b_) { discrete = true; }
  double b() const override { return n; }
  double logpdf(double x) const override {
    double k = std::floor(x);
    double combiln = -std::log(n + 1) - betaln(n - k + 1, k + 1);
    return combiln + betaln(k + al, n - k + be) - betaln(al, be);
  }
  double pdf(double x) const override { return std::exp(logpdf(x)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    double a = al, b = be, e_p = a / (a + b), e_q = 1 - e_p;
    m = n * e_p;
    v = n * (a + b + n) * e_p * e_q / (a + b + 1);
    s = 1.0 / std::sqrt(v);
    s *= (a + b + 2 * n) * (b - a);
    s /= (a + b + 2) * (a + b);
    double g2 = a + b;
    g2 *= (a + b - 1 + 6 * n);
    g2 += 3 * a * b * (n - 2);
    g2 += 6 * n * n;
    g2 -= 3 * e_p * b * n * (6 - n);
    g2 -= 18 * e_p * e_q * n * n;
    g2 *= (a + b) * (a + b) * (1 + a + b);
    g2 /= (n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n));
    g2 -= 3;
    k = g2;
  }
};

struct NBinom : Law {
  double n, p;
  boost::math::negative_binomial_distribution<double, StatPol> d;
  NBinom(double n_, double p_) : n(n_), p(p_), d(n_, p_) { discrete = true; }
  double pdf(double x) const override { return std::isfinite(x) ? boost::math::pdf(d, x) : NaN; }
  double logpdf(double x) const override { return gammaln(n + x) - gammaln(x + 1) - gammaln(n) + n * std::log(p) + xlog1py(x, -p); }
  double cdf(double x) const override { return boost::math::cdf(d, std::floor(x)); }
  double logcdf(double x) const override {
    double k = std::floor(x), c = cdf(k);
    return c > 0.5 ? std::log1p(-betainc(k + 1, n, 1 - p)) : std::log(c);
  }
  double sf(double x) const override { return boost::math::cdf(boost::math::complement(d, std::floor(x))); }
  double ppf(double q) const override { return boost::math::quantile(d, q); }
  double isf(double q) const override { return boost::math::quantile(boost::math::complement(d, q)); }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = boost::math::mean(d); v = boost::math::variance(d); s = boost::math::skewness(d); k = boost::math::kurtosis_excess(d);
  }
};

struct Geom : Law {
  double p;
  explicit Geom(double p_) : p(p_) { discrete = true; }
  double a() const override { return 1; }
  double pdf(double k) const override { return std::pow(1 - p, k - 1) * p; }
  double logpdf(double k) const override { return xlog1py(k - 1, -p) + std::log(p); }
  double cdf(double x) const override { return -std::expm1(std::log1p(-p) * std::floor(x)); }
  double sf(double x) const override { return std::exp(logsf(x)); }
  double logsf(double x) const override { return std::floor(x) * std::log1p(-p); }
  double ppf(double q) const override {
    double vals = std::ceil(std::log1p(-q) / std::log1p(-p));
    double temp = cdf(vals - 1);
    return (temp >= q && vals > 0) ? vals - 1 : vals;
  }
  void stats(double& m, double& v, double& s, double& k) const override {
    m = 1.0 / p; double qr = 1.0 - p; v = qr / p / p; s = (2.0 - p) / std::sqrt(qr); k = polyval({1, -6, 6}, p) / (1.0 - p);
  }
};

// ---- family table and parameter resolution ---------------------------------------------------------------------
struct Spec {
  const char* name;
  bool discrete;
  std::vector<const char*> shapes;  // required shape parameters (scipy names)
};
const std::vector<Spec>& specs() {
  static const std::vector<Spec> s = {
      {"norm", false, {}}, {"t", false, {"df"}}, {"uniform", false, {}}, {"expon", false, {}}, {"gamma", false, {"a"}},
      {"beta", false, {"a", "b"}}, {"lognorm", false, {"s"}}, {"chi2", false, {"df"}}, {"genextreme", false, {"c"}},
      {"genpareto", false, {"c"}}, {"skewnorm", false, {"a"}}, {"johnsonsu", false, {"a", "b"}}, {"levy_stable", false, {"alpha", "beta"}},
      {"nct", false, {"df", "nc"}}, {"ncf", false, {"dfn", "dfd", "nc"}}, {"vonmises", false, {"kappa"}}, {"truncnorm", false, {"a", "b"}},
      {"weibull_min", false, {"c"}}, {"cauchy", false, {}}, {"laplace", false, {}}, {"logistic", false, {}}, {"gumbel_r", false, {}},
      {"pareto", false, {"b"}}, {"rayleigh", false, {}}, {"f", false, {"dfn", "dfd"}},
      {"poisson", true, {"mu"}}, {"binom", true, {"n", "p"}}, {"zipf", true, {"a"}}, {"betabinom", true, {"n", "a", "b"}},
      {"nbinom", true, {"n", "p"}}, {"geom", true, {"p"}}};
  return s;
}

std::string join(const std::vector<std::string>& v) {
  std::string r;
  for (auto& x : v) r += (r.empty() ? "" : ", ") + x;
  return r;
}

void need(bool ok, const std::string& name, const std::string& msg) {
  if (!ok) fail("dist(\"" + name + "\"): " + msg);
}
bool is_int(double x) { return std::isfinite(x) && x == std::floor(x); }

}  // namespace

std::vector<std::string> dist_names() {
  std::vector<std::string> r;
  for (auto& s : specs()) r.push_back(s.name);
  return r;
}

std::shared_ptr<DistImpl> make_dist(const std::string& name, const Opts& o) {
  const Spec* sp = nullptr;
  for (auto& s : specs())
    if (name == s.name) sp = &s;
  if (!sp) fail("dist: unknown family '" + name + "'; known: " + join(dist_names()));
  auto d = std::make_shared<DistImpl>();
  d->name = name;
  d->discrete = sp->discrete;
  // accepted keys: the scipy names, plus pycli's original names for the first ten families
  std::map<std::string, double> got;
  std::vector<std::string> allowed;
  for (auto s : sp->shapes) allowed.push_back(s);
  allowed.push_back("loc");
  if (!sp->discrete) allowed.push_back("scale");
  std::vector<std::string> aliases;
  if (name == "uniform") aliases = {"low", "high"};
  if (name == "gamma") aliases = {"shape"};
  if (name == "lognorm") aliases = {"mu", "sigma"};
  if (name == "poisson") aliases = {"lam"};
  for (auto& kv : o.items) {
    const std::string& k = kv.first;
    bool known = std::find(allowed.begin(), allowed.end(), k) != allowed.end() || std::find(aliases.begin(), aliases.end(), k) != aliases.end();
    if (!known) {
      std::vector<std::string> all = allowed;
      all.insert(all.end(), aliases.begin(), aliases.end());
      fail("dist(\"" + name + "\"): unknown parameter '" + k + "'; parameters: " + join(all));
    }
    double v = o.num(k, NaN);
    if (!std::isfinite(v) && !(name == "t" && k == "df" && v == Inf) && !(name == "truncnorm" && (k == "a" || k == "b") && std::isinf(v)))
      fail("dist(\"" + name + "\"): parameter '" + k + "' must be finite");
    got[k] = v;
  }
  o.finish("dist");
  auto has = [&](const char* k) { return got.count(k) > 0; };
  auto take = [&](const char* k, double def) { return has(k) ? got[k] : def; };
  // aliases (pycli names) -> scipy parameterisation
  if (name == "uniform" && (has("low") || has("high"))) {
    need(has("low") && has("high"), name, "give both low and high (or loc and scale)");
    need(!has("loc") && !has("scale"), name, "use either low/high or loc/scale, not both");
    need(got["high"] > got["low"], name, "high must exceed low");
    got["loc"] = got["low"]; got["scale"] = got["high"] - got["low"];
  }
  if (name == "gamma" && has("shape")) { need(!has("a"), name, "use either a or shape"); got["a"] = got["shape"]; }
  if (name == "poisson" && has("lam")) { need(!has("mu"), name, "use either mu or lam"); got["mu"] = got["lam"]; }
  bool lognorm_mu_sigma = false;
  if (name == "lognorm" && (has("mu") || has("sigma"))) {
    need(has("mu") && has("sigma"), name, "give both mu and sigma (or s and scale)");
    need(!has("s") && !has("scale"), name, "use either mu/sigma or s/scale, not both");
    need(std::fabs(got["mu"]) <= 700, name, "mu must be within [-700, 700]");
    got["s"] = got["sigma"]; got["scale"] = std::exp(got["mu"]);
    lognorm_mu_sigma = true;
  }
  for (auto s : sp->shapes) need(has(s), name, std::string("missing parameter '") + s + "' (required: " +
                                                   join(std::vector<std::string>(sp->shapes.begin(), sp->shapes.end())) + ")");
  d->loc = take("loc", 0);
  d->scale = sp->discrete ? 1 : take("scale", 1);
  need(d->scale > 0, name, "scale must be > 0");
  for (auto s : sp->shapes) d->params.push_back({s, got[s]});
  if (lognorm_mu_sigma) { d->params.push_back({"mu", got["mu"]}); d->params.push_back({"sigma", got["sigma"]}); }
  d->params.push_back({"loc", d->loc});
  if (!sp->discrete) d->params.push_back({"scale", d->scale});
  auto P = [&](const char* k) { return got[k]; };
  // validation: scipy _argcheck plus pycli's caps
  if (name == "norm") d->law = std::make_shared<Norm>();
  else if (name == "t") { need(P("df") > 0, name, "df must be > 0"); d->law = std::make_shared<StudentT>(P("df")); }
  else if (name == "uniform") d->law = std::make_shared<Uniform>();
  else if (name == "expon") d->law = std::make_shared<Expon>();
  else if (name == "gamma") { need(P("a") > 0, name, "a (shape) must be > 0"); d->law = std::make_shared<Gamma>(P("a")); }
  else if (name == "beta") { need(P("a") > 0 && P("b") > 0, name, "a and b must be > 0"); d->law = std::make_shared<Beta>(P("a"), P("b")); }
  else if (name == "lognorm") { need(P("s") > 0, name, "s (sigma) must be > 0"); d->law = std::make_shared<Lognorm>(P("s")); }
  else if (name == "chi2") { need(P("df") > 0, name, "df must be > 0"); d->law = std::make_shared<Chi2>(P("df")); }
  else if (name == "genextreme") d->law = std::make_shared<GenExtreme>(P("c"));
  else if (name == "genpareto") d->law = std::make_shared<GenPareto>(P("c"));
  else if (name == "skewnorm") d->law = std::make_shared<SkewNorm>(P("a"));
  else if (name == "johnsonsu") { need(P("b") > 0, name, "b must be > 0"); d->law = std::make_shared<JohnsonSU>(P("a"), P("b")); }
  else if (name == "levy_stable") {
    need(P("alpha") > 0 && P("alpha") <= 2, name, "alpha must be in (0, 2]");
    need(P("beta") >= -1 && P("beta") <= 1, name, "beta must be in [-1, 1]");
    d->law = std::make_shared<LevyStable>(P("alpha"), P("beta"));
  }
  else if (name == "nct") { need(P("df") >= 1e-8, name, "df must be >= 1e-8"); d->law = std::make_shared<NCT>(P("df"), P("nc")); }
  else if (name == "ncf") {
    need(P("dfn") > 0 && P("dfd") > 0 && P("nc") >= 0, name, "dfn, dfd must be > 0 and nc >= 0");
    d->law = std::make_shared<NCF>(P("dfn"), P("dfd"), P("nc"));
  }
  else if (name == "vonmises") { need(P("kappa") > 0 && P("kappa") <= 1e5, name, "kappa must be in (0, 1e5]"); d->law = std::make_shared<VonMises>(P("kappa")); }
  else if (name == "truncnorm") { need(P("a") < P("b"), name, "a must be < b (standardised bounds)"); d->law = std::make_shared<Truncnorm>(P("a"), P("b")); }
  else if (name == "weibull_min") { need(P("c") > 0, name, "c must be > 0"); d->law = std::make_shared<WeibullMin>(P("c")); }
  else if (name == "cauchy") d->law = std::make_shared<Cauchy>();
  else if (name == "laplace") d->law = std::make_shared<Laplace>();
  else if (name == "logistic") d->law = std::make_shared<Logistic>();
  else if (name == "gumbel_r") d->law = std::make_shared<GumbelR>();
  else if (name == "pareto") { need(P("b") > 0, name, "b must be > 0"); d->law = std::make_shared<Pareto>(P("b")); }
  else if (name == "rayleigh") d->law = std::make_shared<Rayleigh>();
  else if (name == "f") { need(P("dfn") > 0 && P("dfd") > 0, name, "dfn and dfd must be > 0"); d->law = std::make_shared<FDist>(P("dfn"), P("dfd")); }
  else if (name == "poisson") { need(P("mu") >= 0, name, "mu (lam) must be >= 0"); d->law = std::make_shared<Poisson>(P("mu")); }
  else if (name == "binom") {
    need(P("n") >= 0 && is_int(P("n")), name, "n must be a non-negative integer");
    need(P("p") >= 0 && P("p") <= 1, name, "p must be in [0, 1]");
    d->law = std::make_shared<Binom>(P("n"), P("p"));
  }
  else if (name == "zipf") { need(P("a") > 1, name, "a must be > 1"); d->law = std::make_shared<Zipf>(P("a")); }
  else if (name == "betabinom") {
    need(P("n") >= 0 && P("n") <= 5000 && is_int(P("n")), name, "n must be an integer in [0, 5000]");
    need(P("a") > 0 && P("b") > 0, name, "a and b must be > 0");
    d->law = std::make_shared<BetaBinom>(P("n"), P("a"), P("b"));
  }
  else if (name == "nbinom") {
    need(P("n") > 0, name, "n must be > 0");
    need(P("p") >= 1e-15 && P("p") <= 1, name, "p must be in [1e-15, 1]");
    d->law = std::make_shared<NBinom>(P("n"), P("p"));
  }
  else if (name == "geom") { need(P("p") >= 1e-15 && P("p") <= 1, name, "p must be in [1e-15, 1]"); d->law = std::make_shared<Geom>(P("p")); }
  if (d->discrete) need(is_int(d->loc), name, "loc must be an integer for a discrete family");
  if (name != "norm" && name != "t" && name != "lognorm")
    for (auto& kv : d->params) need(std::fabs(kv.second) <= 1e12, name, "parameter '" + kv.first + "' exceeds 1e12 in magnitude");
  return d;
}

// ---- public evaluation (rv_continuous / rv_discrete wrappers) ---------------------------------------------------
namespace {
// levy_stable S1 with alpha == 1: scipy shifts the location in pdf and cdf only
double ls_loc(const DistImpl& d) {
  if (d.name == "levy_stable") {
    const auto& L = static_cast<const LevyStable&>(*d.law);
    if (L.alpha == 1.0) return d.loc + 2 * L.beta * d.scale * std::log(d.scale) / M_PI;
  }
  return d.loc;
}
// pycli: vonmises lives on loc + scale * [-pi, pi]; points outside are wrapped onto the circle for cdf/sf
double wrap_vm(const DistImpl& d, double x) {
  if (d.name != "vonmises") return x;
  double z = (x - d.loc) / d.scale;
  if (std::isfinite(z) && (z < -M_PI || z > M_PI)) z = std::fmod(std::fmod(z + M_PI, 2 * M_PI) + 2 * M_PI, 2 * M_PI) - M_PI;
  return d.loc + d.scale * z;
}
void zipf_guard(const DistImpl& d, double x) {
  if (d.name == "zipf" && std::isfinite(x) && x - d.loc > ZIPF_MAX)
    fail("zipf: cdf/sf are computed by summing the mass function and need x - loc <= 1e5 (got " + jnum(x - d.loc) + ")");
}
void need_prob(double q, const char* fn) {
  if (!std::isnan(q) && (q < 0 || q > 1)) fail(std::string("Dist.") + fn + ": probability " + jnum(q) + " is outside [0, 1]");
}
}  // namespace

double d_support_lo(const DistImpl& d) { return d.law->a() * d.scale + d.loc; }
double d_support_hi(const DistImpl& d) { return d.law->b() * d.scale + d.loc; }

double d_pdf(const DistImpl& d, double x) {
  if (std::isnan(x)) return NaN;
  const Law& L = *d.law;
  if (d.discrete) {
    double k = x - d.loc;
    if (!(k >= L.a() && k <= L.b() && std::floor(k) == k)) return 0.0;
    return std::min(1.0, std::max(0.0, L.pdf(k)));
  }
  double z = (x - ls_loc(d)) / d.scale;
  if (!(z >= L.a() && z <= L.b())) return 0.0;
  return L.pdf(z) / d.scale;
}
double d_logpdf(const DistImpl& d, double x) {
  if (std::isnan(x)) return NaN;
  const Law& L = *d.law;
  if (d.discrete) {
    double k = x - d.loc;
    if (!(k >= L.a() && k <= L.b() && std::floor(k) == k)) return -Inf;
    return L.logpdf(k);
  }
  double z = (x - d.loc) / d.scale;
  if (!(z >= L.a() && z <= L.b())) return -Inf;
  return L.logpdf(z) - std::log(d.scale);
}
double d_cdf(const DistImpl& d, double x) {
  if (std::isnan(x)) return NaN;
  zipf_guard(d, x);
  x = wrap_vm(d, x);
  const Law& L = *d.law;
  if (d.discrete) {
    double k = x - d.loc;
    if (k >= L.b()) return 1.0;
    if (!(k >= L.a())) return 0.0;
    return std::min(1.0, std::max(0.0, L.cdf(k)));
  }
  double z = (x - ls_loc(d)) / d.scale;
  if (z >= L.b()) return 1.0;
  if (!(z > L.a())) return 0.0;
  return L.cdf(z);
}
double d_logcdf(const DistImpl& d, double x) {
  if (std::isnan(x)) return NaN;
  zipf_guard(d, x);
  x = wrap_vm(d, x);
  const Law& L = *d.law;
  double z = d.discrete ? x - d.loc : (x - d.loc) / d.scale;
  if (z >= L.b()) return 0.0;
  if (d.discrete ? !(z >= L.a()) : !(z > L.a())) return -Inf;
  return L.logcdf(z);
}
double d_sf(const DistImpl& d, double x) {
  if (std::isnan(x)) return NaN;
  zipf_guard(d, x);
  x = wrap_vm(d, x);
  const Law& L = *d.law;
  if (d.discrete) {
    double k = x - d.loc;
    if (k < L.a()) return 1.0;
    if (!(k < L.b())) return 0.0;
    return std::min(1.0, std::max(0.0, L.sf(k)));
  }
  double z = (x - d.loc) / d.scale;
  if (z <= L.a()) return 1.0;
  if (!(z < L.b())) return 0.0;
  return L.sf(z);
}
double d_logsf(const DistImpl& d, double x) {
  if (std::isnan(x)) return NaN;
  zipf_guard(d, x);
  x = wrap_vm(d, x);
  const Law& L = *d.law;
  double z = d.discrete ? x - d.loc : (x - d.loc) / d.scale;
  if (d.discrete ? z < L.a() : z <= L.a()) return 0.0;
  if (!(z < L.b())) return -Inf;
  return L.logsf(z);
}
double d_ppf(const DistImpl& d, double q) {
  need_prob(q, "ppf");
  if (std::isnan(q)) return NaN;
  const Law& L = *d.law;
  if (q == 0) return L.a() * d.scale + d.loc;  // discrete: the first support point (pycli), not scipy's a - 1
  if (q == 1) return L.b() * d.scale + d.loc;
  if (d.name == "zipf" && q > d_cdf(d, d.loc + ZIPF_MAX)) return NaN;  // pycli OUT_OF_RANGE
  return L.ppf(q) * d.scale + d.loc;
}
double d_isf(const DistImpl& d, double q) {
  need_prob(q, "isf");
  if (std::isnan(q)) return NaN;
  const Law& L = *d.law;
  if (q == 1) return L.a() * d.scale + d.loc;
  if (q == 0) return L.b() * d.scale + d.loc;
  if (d.name == "zipf" && q < d_sf(d, d.loc + ZIPF_MAX)) return NaN;
  return L.isf(q) * d.scale + d.loc;
}

void d_stats(const DistImpl& d, double& mean, double& var, double& skew, double& kurt) {
  const Law& L = *d.law;
  double mu, mu2, g1, g2;
  L.has_stats = true;
  L.stats(mu, mu2, g1, g2);
  if (!L.has_stats) {  // scipy rv_generic.stats from _munp
    mu = L.munp(1);
    double mu2p = L.munp(2);
    mu2 = !std::isinf(mu) ? mu2p - mu * mu : Inf;
    double mu3p = L.munp(3);
    double mu3 = (-mu * mu - 3 * mu2) * mu + mu3p;
    g1 = mu3 / std::pow(mu2, 1.5);
    double mu4p = L.munp(4);
    double mu4 = ((-mu * mu - 6 * mu2) * mu - 4 * mu3) * mu + mu4p;
    g2 = mu4 / std::pow(mu2, 2.0) - 3.0;
  }
  mean = mu * d.scale + d.loc;
  var = mu2 * d.scale * d.scale;
  skew = g1;
  kurt = g2;
}

// ---- random variates: numpy Generator(PCG64(seed)) as pycli drew them ---------------------------------------------
std::vector<double> d_rvs(const DistImpl& d, int64_t n, uint64_t seed) {
  using namespace nprand;
  if (n < 0) fail("Dist.rvs: n must be >= 0");
  if (n > 10000000) fail("Dist.rvs: n is capped at 10,000,000");
  std::vector<double> x((size_t)n);
  Bitgen g(seed);
  const std::string& nm = d.name;
  auto par = [&](const char* k) {
    for (auto& kv : d.params) if (kv.first == k) return kv.second;
    return NaN;
  };
  double loc = d.loc, scale = d.scale;
  auto each = [&](auto f) { for (auto& v : x) v = f(); };
  if (nm == "norm") each([&] { return random_normal(&g, loc, scale); });
  else if (nm == "t") { double df = par("df"); each([&] { return loc + scale * random_standard_t(&g, df); }); }
  else if (nm == "uniform") each([&] { return random_uniform(&g, loc, scale); });
  else if (nm == "expon") each([&] { return random_exponential(&g, scale) + loc; });
  else if (nm == "gamma") { double a = par("a"); each([&] { return random_gamma(&g, a, scale) + loc; }); }
  else if (nm == "beta") { double a = par("a"), b = par("b"); each([&] { return random_beta(&g, a, b) * scale + loc; }); }
  else if (nm == "lognorm") {
    double mu = par("mu"), sigma = par("sigma"), s = par("s");
    if (!std::isnan(mu)) each([&] { return random_lognormal(&g, mu, sigma) + loc; });
    else each([&] { return std::exp(s * random_standard_normal(&g)) * scale + loc; });
  }
  else if (nm == "chi2") { double df = par("df"); each([&] { return random_chisquare(&g, df) * scale + loc; }); }
  else if (nm == "poisson") { double mu = par("mu"); each([&] { return double(random_poisson(&g, mu)) + loc; }); }
  else if (nm == "binom") { int64_t nn = int64_t(par("n")); double p = par("p"); each([&] { return double(random_binomial(&g, p, nn, &g.binomial)) + loc; }); }
  else if (nm == "nct" || nm == "skewnorm" || nm == "betabinom" || nm == "levy_stable") {
    // pycli: compound samplers from two streams, PCG64(seed) and PCG64(seed).jumped(), so a shorter n is a prefix
    Bitgen h(seed);
    h.jump();
    if (nm == "nct") {
      double df = par("df"), nc = par("nc");
      std::vector<double> z((size_t)n), v((size_t)n);
      for (auto& t : z) t = random_standard_normal(&g) + nc;
      for (auto& t : v) t = random_chisquare(&h, df);
      for (size_t i = 0; i < x.size(); ++i) x[i] = loc + scale * z[i] / std::sqrt(v[i] / df);
    } else if (nm == "skewnorm") {
      double a = par("a");
      std::vector<double> u0((size_t)n), v((size_t)n);
      for (auto& t : u0) t = random_standard_normal(&g);
      for (auto& t : v) t = random_standard_normal(&h);
      double dd = a / std::sqrt(1.0 + a * a);
      for (size_t i = 0; i < x.size(); ++i) {
        double u1 = dd * u0[i] + v[i] * std::sqrt(1.0 - dd * dd);
        x[i] = loc + scale * (u0[i] >= 0 ? u1 : -u1);
      }
    } else if (nm == "betabinom") {
      double a = par("a"), b = par("b");
      int64_t nn = int64_t(par("n"));
      std::vector<double> pr((size_t)n);
      for (auto& t : pr) t = random_beta(&g, a, b);
      for (size_t i = 0; i < x.size(); ++i) x[i] = double(random_binomial(&h, pr[i], nn, &h.binomial)) + loc;
    } else {
      double alpha = par("alpha"), beta = par("beta");
      std::vector<double> TH((size_t)n), W((size_t)n);
      for (auto& t : TH) t = -M_PI / 2 + M_PI * g.next_double();
      for (auto& t : W) t = random_standard_exponential(&h);
      for (size_t i = 0; i < x.size(); ++i) {
        double th = TH[i], w = W[i], aTH = alpha * th, bTH = beta * th, cosTH = std::cos(th), tanTH = std::tan(th), z;
        if (alpha == 1) z = 2 / M_PI * ((M_PI / 2 + bTH) * tanTH - beta * std::log((M_PI / 2 * w * cosTH) / (M_PI / 2 + bTH)));
        else if (beta == 0) z = w / (cosTH / std::tan(aTH) + std::sin(th)) * std::pow((std::cos(aTH) + std::sin(aTH) * tanTH) / w, 1.0 / alpha);
        else {
          double val0 = beta * std::tan(M_PI * alpha / 2), th0 = std::atan(val0) / alpha;
          double val3 = w / (cosTH / std::tan(alpha * (th0 + th)) + std::sin(th));
          z = val3 * std::pow((std::cos(aTH) + std::sin(aTH) * tanTH - val0 * (std::sin(aTH) - std::cos(aTH) * tanTH)) / w, 1.0 / alpha);
        }
        x[i] = loc + scale * z;
      }
    }
  }
  else if (nm == "ncf") { double a = par("dfn"), b = par("dfd"), c = par("nc"); each([&] { return random_noncentral_f(&g, a, b, c) * scale + loc; }); }
  else if (nm == "vonmises") { double k = par("kappa"); each([&] { return random_vonmises(&g, 0.0, k) * scale + loc; }); }
  else if (nm == "laplace") each([&] { return random_laplace(&g, 0, 1) * scale + loc; });
  else if (nm == "logistic") each([&] { return random_logistic(&g, 0, 1) * scale + loc; });
  else if (nm == "rayleigh") each([&] { return std::sqrt(random_chisquare(&g, 2)) * scale + loc; });
  else if (nm == "f") { double a = par("dfn"), b = par("dfd"); each([&] { return random_f(&g, a, b) * scale + loc; }); }
  else if (nm == "zipf") { double a = par("a"); each([&] { return double(random_zipf(&g, a)) + loc; }); }
  else if (nm == "nbinom") { double nn = par("n"), p = par("p"); each([&] { return double(random_negative_binomial(&g, nn, p)) + loc; }); }
  else if (nm == "geom") { double p = par("p"); each([&] { return double(random_geometric(&g, p)) + loc; }); }
  else {
    // scipy's default _rvs: the quantile function of numpy uniforms (genextreme, genpareto, johnsonsu, truncnorm,
    // weibull_min, cauchy, gumbel_r, pareto)
    const Law& L = *d.law;
    each([&] { return L.ppf(g.next_double()) * scale + loc; });
  }
  for (double v : x)
    if (!std::isfinite(v)) fail("Dist.rvs: the sampler produced non-finite values for these parameters (pycli NUMERICAL_FAILURE)");
  return x;
}

}}  // namespace lmnum::st
