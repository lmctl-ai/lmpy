// Shared declarations of the statistics family (stats.cpp, stats_special.cpp, stats_dist.cpp).
#pragma once
#include "host.hpp"
#include "stats_rng.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lmnum { namespace st {

// ---- special functions (stats_special.cpp): scipy.special semantics, NaN outside the domain, +-inf at poles ----------
double ndtr(double x);
double log_ndtr(double x);
double ndtri(double p);
double ndtri_exp(double y);
double erf(double x);
double erfc(double x);
double erfcx(double x);
double gammaln(double x);
double gammafn(double x);
double xlogy(double x, double y);
double xlog1py(double x, double y);
double expit(double x);
double logit(double p);
double log_expit(double x);
double cosm1(double x);
double i0e(double x);
double poch(double a, double m);
double betaln(double a, double b);
double betainc(double a, double b, double x);
double betaincc(double a, double b, double x);
double betaincinv(double a, double b, double p);
double betainccinv(double a, double b, double p);
double gammainc(double a, double x);
double gammaincc(double a, double x);
double gammaincinv(double a, double p);
double gammainccinv(double a, double p);
double zeta(double x, double q);
double riemann_zeta(double x);
double inv_boxcox(double x, double l);
double inv_boxcox1p(double x, double l);
double boxcox(double x, double l);
double boxcox1p(double x, double l);
double fresnel_s(double x);
double fresnel_c(double x);
double chdtr(double df, double x);
double chdtrc(double df, double x);
double chdtri(double df, double p);
double fdtr(double a, double b, double x);
double fdtrc(double a, double b, double x);
double fdtri(double a, double b, double p);
double pdtr(double k, double m);
double pdtrc(double k, double m);

// Named special functions for special(name, ...): arity 1..4.
struct SpecialFn {
  const char* name;
  int arity;
  double (*f)(const double*);
  const char* doc;
};
const std::vector<SpecialFn>& special_table();
const SpecialFn& find_special(const std::string& name, int arity);
// Airy functions Ai, Ai', Bi, Bi' (scipy.special.airy).
void airy(double x, double& ai, double& aip, double& bi, double& bip);
double logsumexp(const std::vector<double>& v);

// ---- distributions (stats_dist.cpp) -----------------------------------------------------------------------------
struct Law;
struct DistImpl {
  std::string name;
  std::vector<std::pair<std::string, double>> params;  // canonical (scipy) names in order, as resolved
  std::shared_ptr<Law> law;
  bool discrete = false;
  double loc = 0, scale = 1;
};
std::shared_ptr<DistImpl> make_dist(const std::string& name, const Opts& o);
std::vector<std::string> dist_names();

double d_pdf(const DistImpl&, double x);
double d_logpdf(const DistImpl&, double x);
double d_cdf(const DistImpl&, double x);
double d_logcdf(const DistImpl&, double x);
double d_sf(const DistImpl&, double x);
double d_logsf(const DistImpl&, double x);
double d_ppf(const DistImpl&, double q);
double d_isf(const DistImpl&, double q);
void d_stats(const DistImpl&, double& mean, double& var, double& skew, double& kurt);
double d_support_lo(const DistImpl&);
double d_support_hi(const DistImpl&);
std::vector<double> d_rvs(const DistImpl&, int64_t n, uint64_t seed);

}}  // namespace lmnum::st
