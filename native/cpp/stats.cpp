// Statistics family: distributions (pycli stats_dist / stats_dist_to), special functions (special_fn / special_fn_to),
// numpy-compatible random numbers and a few classical tests. Implementations: stats_dist.cpp, stats_special.cpp,
// stats_rng_numpy.cpp.
#include "stats_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lmnum {
namespace {

using V = std::vector<double>;
const std::string Dt = "Dist", Rt = "Rng", Vt = "Vec", Rs = "Res", Ot = "Opts";
const double NaN = std::numeric_limits<double>::quiet_NaN();

struct Dist : Obj {
  static constexpr const char* TYPE = "Dist";
  std::shared_ptr<st::DistImpl> d;
};
struct Rng : Obj {
  static constexpr const char* TYPE = "Rng";
  nprand::Bitgen g;
  explicit Rng(uint64_t seed) : g(seed) {}
};

const st::DistImpl& D(void* s) { return *static_cast<Dist*>(s)->d; }
nprand::Bitgen& G(void* s) { return static_cast<Rng*>(s)->g; }
const V& av(const Value& x) { return obj<Vec>(x).v; }

uint64_t seed_arg(int64_t s, const char* op) {
  if (s < 0) fail(std::string(op) + ": seed must be a non-negative integer (numpy SeedSequence)");
  return uint64_t(s);
}
size_t count_arg(int64_t n, const char* op) {
  if (n < 0) fail(std::string(op) + ": n must be >= 0");
  if (n > 10000000) fail(std::string(op) + ": n is capped at 10,000,000");
  return size_t(n);
}

// ---- special(): broadcasting of scalars and vectors (length-1 vectors repeat, as pycli) ------------------------------
struct Arg { bool vec; double f; const V* v; };
Value special_call(const std::string& name, const std::vector<Arg>& args) {
  const st::SpecialFn& fn = st::find_special(name, int(args.size()));
  size_t n = 1;
  bool any_vec = false;
  for (auto& a : args)
    if (a.vec) {
      any_vec = true;
      if (a.v->size() != 1) {
        if (n != 1 && a.v->size() != n) fail("special: vector arguments must have equal length (scalars and length-1 vectors broadcast)");
        n = a.v->size();
      }
    }
  for (auto& a : args) if (a.vec && a.v->empty()) n = 0;
  double buf[4];
  if (!any_vec) {
    for (size_t j = 0; j < args.size(); ++j) buf[j] = args[j].f;
    return F(fn.f(buf));
  }
  V out(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < args.size(); ++j) buf[j] = args[j].vec ? (*args[j].v)[args[j].v->size() == 1 ? 0 : i] : args[j].f;
    out[i] = fn.f(buf);
  }
  return vecv(std::move(out));
}
Arg fa(const Value& v) { return {false, v.f, nullptr}; }
Arg va(const Value& v) { return {true, 0, &av(v)}; }

// ---- Dist helpers ------------------------------------------------------------------------------------------------
using DFn = double (*)(const st::DistImpl&, double);
void reg_eval(Api& api, const char* name, DFn f, const std::string& doc) {
  api.method(Dt, name, {"float"}, "float", doc, [f](void* s, Args& a) { return F(f(D(s), a[0].f)); });
  api.method(Dt, name, {Vt}, Vt, doc + " (elementwise)", [f](void* s, Args& a) {
    const V& x = av(a[0]);
    V r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = f(D(s), x[i]);
    return vecv(std::move(r));
  });
}
void need_discrete(const st::DistImpl& d, const char* m) {
  if (!d.discrete) fail(std::string("Dist.") + m + ": " + d.name + " is continuous; use pdf/logpdf");
}

Res* moments_res(const st::DistImpl& d) {
  double m, v, s, k;
  st::d_stats(d, m, v, s, k);
  std::vector<std::string> undef;
  const char* names[4] = {"mean", "var", "skew", "kurtosis"};
  double vals[4] = {m, v, s, k};
  auto* r = make<Res>();
  for (int i = 0; i < 4; ++i) {
    r->num(names[i], vals[i]);
    if (!std::isfinite(vals[i])) undef.push_back(names[i]);
  }
  r->strs("moments_undefined", undef);
  r->str("status", "OK");
  return r;
}

// ---- tests -------------------------------------------------------------------------------------------------------
double mean_of(const V& x) { return std::accumulate(x.begin(), x.end(), 0.0) / double(x.size()); }
double var_of(const V& x, double m) {
  double s = 0;
  for (double v : x) s += (v - m) * (v - m);
  return s / double(x.size() - 1);
}
double t_two_sided(double t, double df) {
  // scipy _get_pvalue: 2 * t(df).sf(|t|)
  Opts o;
  Item it; it.kind = Item::NUM; it.f = df;
  o.put("df", it);
  auto d = st::make_dist("t", o);
  return 2 * st::d_sf(*d, std::fabs(t));
}
void need_finite(const V& x, const char* op) {
  for (double v : x) if (!std::isfinite(v)) fail(std::string(op) + ": the data contain NaN or infinity");
}
Res* ttest_res(double t, double df) {
  auto* r = make<Res>();
  r->num("statistic", t).num("pvalue", t_two_sided(t, df)).num("df", df);
  return r;
}
Res* ttest_ind(const V& a, const V& b, bool equal_var) {
  need_finite(a, "ttest_ind"); need_finite(b, "ttest_ind");
  if (a.size() < 2 || b.size() < 2) fail("ttest_ind: each sample needs at least 2 values");
  double n1 = double(a.size()), n2 = double(b.size()), m1 = mean_of(a), m2 = mean_of(b), v1 = var_of(a, m1), v2 = var_of(b, m2);
  double df, denom;
  if (equal_var) {
    df = n1 + n2 - 2.0;
    double svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df;
    denom = std::sqrt(svar * (1.0 / n1 + 1.0 / n2));
  } else {
    double vn1 = v1 / n1, vn2 = v2 / n2;
    df = (vn1 + vn2) * (vn1 + vn2) / (vn1 * vn1 / (n1 - 1) + vn2 * vn2 / (n2 - 1));
    denom = std::sqrt(vn1 + vn2);
  }
  return ttest_res((m1 - m2) / denom, df);
}

}  // namespace

void register_stats(Api& api) {
  api.section("Statistics",
              "Probability distributions (pycli `stats_dist`, `stats_dist_to`), special functions (`special_fn`, `special_fn_to`), "
              "numpy-compatible random numbers and classical tests. Distributions follow scipy.stats 1.16 parameter names and "
              "formulas (each family is a port of its scipy class; special functions are scipy's own xsf/Cephes code or Boost.Math "
              "with scipy's policies). `dist(name, opts)` takes the scipy parameters plus `loc` (and `scale` for continuous families); "
              "pycli's names are accepted too: uniform `low`/`high`, gamma `shape`, lognorm `mu`/`sigma` (of the underlying normal), "
              "poisson `lam`. Families: norm, t, uniform, expon, gamma, beta, lognorm, chi2, genextreme, genpareto, skewnorm, johnsonsu, "
              "levy_stable (S1), nct, ncf, vonmises, truncnorm, weibull_min, cauchy, laplace, logistic, gumbel_r, pareto, rayleigh, f, "
              "poisson, binom, zipf, betabinom, nbinom, geom. Probabilities outside [0, 1] are errors; NaN inputs give NaN. Discrete "
              "ppf(0) and isf(1) are the first support point (pycli). Random variates reproduce numpy.random.Generator(PCG64(seed)) "
              "exactly for the numpy-sampled families (pycli's rvs; nct, skewnorm, betabinom and levy_stable use PCG64(seed) and its "
              "jumped() stream as pycli did); the quantile-sampled families (genextreme, genpareto, johnsonsu, truncnorm, weibull_min, "
              "cauchy, gumbel_r, pareto) use numpy's uniform stream through lmnum's quantile function. Deviations from scipy: Student t "
              "cdf/ppf use Boost (scipy: CDFLIB), Bessel/Airy outside [-10, 10] use Boost (scipy: AMOS), levy_stable integrates with "
              "Gauss-Kronrod (scipy: QUADPACK), Poisson ppf uses the generic discrete search.");
  api.type(Dt, "A frozen probability distribution (`dist(\"gamma\", opts().set(\"a\", 3.0).set(\"scale\", 2.0))`).");
  api.type(Rt, "A numpy-compatible random generator: the stream of numpy.random.default_rng(seed) (PCG64 + SeedSequence). Methods advance it in place.");

  api.fn("dist", {"string"}, Dt, "distribution with default parameters (families without shape parameters: norm, uniform, expon, cauchy, laplace, logistic, gumbel_r, rayleigh)",
         [](Args& a) { auto* d = make<Dist>(); d->d = st::make_dist(a[0].s, no_opts()); return O(d); });
  api.fn("dist", {"string", Ot}, Dt, "distribution with parameters (scipy names; unknown or missing ones are errors that list the valid ones)",
         [](Args& a) { auto* d = make<Dist>(); d->d = st::make_dist(a[0].s, obj<Opts>(a[1])); return O(d); });
  api.fn("dist_names", {}, "StrVec", "the 31 distribution families", [](Args&) { return strsv(st::dist_names()); });

  reg_eval(api, "pdf", st::d_pdf, "density (the mass function for a discrete family, as pycli fn=pdf)");
  reg_eval(api, "logpdf", st::d_logpdf, "log density (log mass for a discrete family)");
  reg_eval(api, "cdf", st::d_cdf, "cumulative distribution function");
  reg_eval(api, "logcdf", st::d_logcdf, "log cdf");
  reg_eval(api, "sf", st::d_sf, "survival function 1 - cdf");
  reg_eval(api, "logsf", st::d_logsf, "log survival function");
  reg_eval(api, "ppf", st::d_ppf, "quantile function (inverse cdf); p must be in [0, 1]");
  reg_eval(api, "isf", st::d_isf, "inverse survival function; p must be in [0, 1]");
  api.method(Dt, "pmf", {"float"}, "float", "probability mass (discrete families only)", [](void* s, Args& a) { need_discrete(D(s), "pmf"); return F(st::d_pdf(D(s), a[0].f)); });
  api.method(Dt, "pmf", {Vt}, Vt, "probability mass, elementwise (discrete families only)", [](void* s, Args& a) {
    need_discrete(D(s), "pmf"); V r; for (double x : av(a[0])) r.push_back(st::d_pdf(D(s), x));
    return vecv(std::move(r)); });
  api.method(Dt, "logpmf", {"float"}, "float", "log probability mass (discrete families only)", [](void* s, Args& a) { need_discrete(D(s), "logpmf"); return F(st::d_logpdf(D(s), a[0].f)); });
  api.method(Dt, "logpmf", {Vt}, Vt, "log probability mass, elementwise (discrete families only)", [](void* s, Args& a) {
    need_discrete(D(s), "logpmf"); V r; for (double x : av(a[0])) r.push_back(st::d_logpdf(D(s), x));
    return vecv(std::move(r)); });
  api.method(Dt, "mean", {}, "float", "mean (inf or NaN where undefined, as scipy)", [](void* s, Args&) { double m, v, k, g; st::d_stats(D(s), m, v, k, g); return F(m); });
  api.method(Dt, "var", {}, "float", "variance", [](void* s, Args&) { double m, v, k, g; st::d_stats(D(s), m, v, k, g); return F(v); });
  api.method(Dt, "std", {}, "float", "standard deviation", [](void* s, Args&) { double m, v, k, g; st::d_stats(D(s), m, v, k, g); return F(std::sqrt(v)); });
  api.method(Dt, "median", {}, "float", "median ppf(0.5)", [](void* s, Args&) { return F(st::d_ppf(D(s), 0.5)); });
  api.method(Dt, "moments", {}, Rs, "pycli fn=moments: fields mean, var, skew, kurtosis (excess, as scipy), moments_undefined (names of the non-finite ones), status",
             [](void* s, Args&) { return O(moments_res(D(s))); });
  api.method(Dt, "interval", {"float"}, Vt, "central interval [ppf((1-c)/2), ppf((1+c)/2)] with confidence c in [0, 1]", [](void* s, Args& a) {
    double c = a[0].f;
    if (!(c >= 0 && c <= 1)) fail("Dist.interval: confidence must be in [0, 1]");
    return vecv(V{st::d_ppf(D(s), (1 - c) / 2), st::d_ppf(D(s), (1 + c) / 2)}); });
  api.method(Dt, "rvs", {"int", "int"}, Vt, "n random variates from seed (pycli fn=rvs; numpy Generator(PCG64(seed)); a shorter n is a prefix of a longer one)",
             [](void* s, Args& a) { return vecv(st::d_rvs(D(s), a[0].i, seed_arg(a[1].i, "Dist.rvs"))); });
  api.method(Dt, "support", {}, Vt, "[lower, upper] end of the support", [](void* s, Args&) { return vecv(V{st::d_support_lo(D(s)), st::d_support_hi(D(s))}); });
  api.method(Dt, "name", {}, "string", "family name", [](void* s, Args&) { return S(D(s).name); });
  api.method(Dt, "is_discrete", {}, "bool", "true for the discrete families", [](void* s, Args&) { return B(D(s).discrete); });
  api.method(Dt, "params", {}, Rs, "the resolved parameters (scipy names, plus loc and scale)", [](void* s, Args&) {
    auto* r = make<Res>(); for (auto& kv : D(s).params) r->num(kv.first, kv.second);
    return O(r); });

  // special functions
  const std::string sdoc = "scipy.special function by name (pycli special_fn): ";
  std::string names;
  for (auto& f : st::special_table()) names += std::string(names.empty() ? "" : ", ") + f.name + "/" + std::to_string(f.arity);
  api.fn("special", {"string", "float"}, "float", sdoc + "name/arity: " + names + ". NaN outside the domain (pycli DOMAIN_ERROR), +-inf at poles (NON_FINITE)",
         [](Args& a) { return special_call(a[0].s, {fa(a[1])}); });
  api.fn("special", {"string", Vt}, Vt, sdoc + "elementwise", [](Args& a) { return special_call(a[0].s, {va(a[1])}); });
  api.fn("special", {"string", "float", "float"}, "float", sdoc + "two arguments", [](Args& a) { return special_call(a[0].s, {fa(a[1]), fa(a[2])}); });
  api.fn("special", {"string", "float", Vt}, Vt, sdoc + "scalar first argument, vector second (e.g. jv(v, x))", [](Args& a) { return special_call(a[0].s, {fa(a[1]), va(a[2])}); });
  api.fn("special", {"string", Vt, "float"}, Vt, sdoc + "vector first argument, scalar second (e.g. boxcox(x, lmbda))", [](Args& a) { return special_call(a[0].s, {va(a[1]), fa(a[2])}); });
  api.fn("special", {"string", Vt, Vt}, Vt, sdoc + "two vectors of one length (length-1 vectors broadcast)", [](Args& a) { return special_call(a[0].s, {va(a[1]), va(a[2])}); });
  api.fn("special", {"string", "float", "float", "float"}, "float", sdoc + "three arguments (betainc family)", [](Args& a) { return special_call(a[0].s, {fa(a[1]), fa(a[2]), fa(a[3])}); });
  api.fn("special", {"string", "float", "float", Vt}, Vt, sdoc + "three arguments, vector last", [](Args& a) { return special_call(a[0].s, {fa(a[1]), fa(a[2]), va(a[3])}); });
  api.fn("special", {"string", Vt, Vt, Vt}, Vt, sdoc + "three vectors", [](Args& a) { return special_call(a[0].s, {va(a[1]), va(a[2]), va(a[3])}); });
  api.fn("special", {"string", "float", "float", "float", "float"}, "float", sdoc + "four arguments (hyp2f1)", [](Args& a) { return special_call(a[0].s, {fa(a[1]), fa(a[2]), fa(a[3]), fa(a[4])}); });
  api.fn("special", {"string", "float", "float", "float", Vt}, Vt, sdoc + "four arguments, vector last", [](Args& a) { return special_call(a[0].s, {fa(a[1]), fa(a[2]), fa(a[3]), va(a[4])}); });
  api.fn("special", {"string", Vt, Vt, Vt, Vt}, Vt, sdoc + "four vectors", [](Args& a) { return special_call(a[0].s, {va(a[1]), va(a[2]), va(a[3]), va(a[4])}); });
  api.fn("special_names", {}, "StrVec", "names accepted by special()", [](Args&) {
    std::vector<std::string> r; for (auto& f : st::special_table()) if (std::find(r.begin(), r.end(), f.name) == r.end()) r.push_back(f.name);
    return strsv(r); });
  api.fn("airy", {"float"}, Rs, "Airy functions (scipy.special.airy): fields ai, aip, bi, bip", [](Args& a) {
    double ai, aip, bi, bip; st::airy(a[0].f, ai, aip, bi, bip);
    auto* r = make<Res>(); r->num("ai", ai).num("aip", aip).num("bi", bi).num("bip", bip); return O(r); });
  api.fn("airy", {Vt}, Rs, "Airy functions elementwise: Vec fields ai, aip, bi, bip", [](Args& a) {
    V ai, aip, bi, bip;
    for (double x : av(a[0])) { double p, q, r_, t; st::airy(x, p, q, r_, t); ai.push_back(p); aip.push_back(q); bi.push_back(r_); bip.push_back(t); }
    auto* r = make<Res>(); r->vec("ai", ai).vec("aip", aip).vec("bi", bi).vec("bip", bip); return O(r); });
  api.fn("logsumexp", {Vt}, "float", "log(sum(exp(v))) without overflow (scipy.special.logsumexp)", [](Args& a) { return F(st::logsumexp(av(a[0]))); });

  // random numbers
  api.fn("rng", {"int"}, Rt, "generator seeded like numpy.random.default_rng(seed) (seed >= 0)", [](Args& a) { return O(make<Rng>(seed_arg(a[0].i, "rng"))); });
  api.method(Rt, "random", {"int"}, Vt, "n uniforms in [0, 1) (Generator.random)", [](void* s, Args& a) {
    V r(count_arg(a[0].i, "Rng.random")); for (auto& x : r) x = G(s).next_double();
    return vecv(std::move(r)); });
  api.method(Rt, "uniform", {"int", "float", "float"}, Vt, "n uniforms in [lo, hi) (Generator.uniform)", [](void* s, Args& a) {
    V r(count_arg(a[0].i, "Rng.uniform")); double lo = a[1].f, rg = a[2].f - a[1].f;
    if (!std::isfinite(rg)) fail("Rng.uniform: hi - lo must be finite");
    for (auto& x : r) x = nprand::random_uniform(&G(s), lo, rg);
    return vecv(std::move(r)); });
  api.method(Rt, "normal", {"int", "float", "float"}, Vt, "n normal variates with mean mu and standard deviation sd (Generator.normal)", [](void* s, Args& a) {
    if (!(a[2].f >= 0)) fail("Rng.normal: sd must be >= 0");
    V r(count_arg(a[0].i, "Rng.normal")); for (auto& x : r) x = nprand::random_normal(&G(s), a[1].f, a[2].f);
    return vecv(std::move(r)); });
  api.method(Rt, "standard_normal", {"int"}, Vt, "n standard normal variates (Generator.standard_normal)", [](void* s, Args& a) {
    V r(count_arg(a[0].i, "Rng.standard_normal")); for (auto& x : r) x = nprand::random_standard_normal(&G(s));
    return vecv(std::move(r)); });
  api.method(Rt, "exponential", {"int", "float"}, Vt, "n exponential variates with the given scale (Generator.exponential)", [](void* s, Args& a) {
    if (!(a[1].f >= 0)) fail("Rng.exponential: scale must be >= 0");
    V r(count_arg(a[0].i, "Rng.exponential")); for (auto& x : r) x = nprand::random_exponential(&G(s), a[1].f);
    return vecv(std::move(r)); });
  api.method(Rt, "gamma", {"int", "float", "float"}, Vt, "n gamma variates (shape, scale) (Generator.gamma)", [](void* s, Args& a) {
    if (!(a[1].f >= 0 && a[2].f >= 0)) fail("Rng.gamma: shape and scale must be >= 0");
    V r(count_arg(a[0].i, "Rng.gamma")); for (auto& x : r) x = nprand::random_gamma(&G(s), a[1].f, a[2].f);
    return vecv(std::move(r)); });
  api.method(Rt, "beta", {"int", "float", "float"}, Vt, "n beta variates (Generator.beta)", [](void* s, Args& a) {
    if (!(a[1].f > 0 && a[2].f > 0)) fail("Rng.beta: a and b must be > 0");
    V r(count_arg(a[0].i, "Rng.beta")); for (auto& x : r) x = nprand::random_beta(&G(s), a[1].f, a[2].f);
    return vecv(std::move(r)); });
  api.method(Rt, "poisson", {"int", "float"}, Vt, "n Poisson counts (Generator.poisson)", [](void* s, Args& a) {
    if (!(a[1].f >= 0 && a[1].f < 1e10)) fail("Rng.poisson: lam must be in [0, 1e10)");
    V r(count_arg(a[0].i, "Rng.poisson")); for (auto& x : r) x = double(nprand::random_poisson(&G(s), a[1].f));
    return vecv(std::move(r)); });
  api.method(Rt, "binomial", {"int", "int", "float"}, Vt, "n binomial counts with `trials` and p (Generator.binomial)", [](void* s, Args& a) {
    if (a[1].i < 0 || !(a[2].f >= 0 && a[2].f <= 1)) fail("Rng.binomial: trials must be >= 0 and p in [0, 1]");
    V r(count_arg(a[0].i, "Rng.binomial")); for (auto& x : r) x = double(nprand::random_binomial(&G(s), a[2].f, a[1].i, &G(s).binomial));
    return vecv(std::move(r)); });
  api.method(Rt, "integers", {"int", "int", "int"}, Vt, "n integers in [lo, hi) (Generator.integers, int64, Lemire's method)", [](void* s, Args& a) {
    int64_t lo = a[1].i, hi = a[2].i;
    if (hi <= lo) fail("Rng.integers: hi must exceed lo");
    uint64_t rng = uint64_t(hi) - uint64_t(lo) - 1;
    V r(count_arg(a[0].i, "Rng.integers"));
    for (auto& x : r) x = double(int64_t(nprand::random_bounded_uint64(&G(s), uint64_t(lo), rng, 0, false)));
    return vecv(std::move(r)); });
  api.method(Rt, "choice", {Vt, "int", "bool"}, Vt, "n elements of a vector, with or without replacement (Generator.choice, shuffle=True)", [](void* s, Args& a) {
    const V& pop = av(a[0]);
    size_t n = count_arg(a[1].i, "Rng.choice");
    int64_t ps = int64_t(pop.size());
    if (ps == 0 && n) fail("Rng.choice: the population is empty");
    std::vector<int64_t> idx(n);
    auto& g = G(s);
    if (a[2].b) {
      if (ps > 0) for (auto& i : idx) i = int64_t(nprand::random_bounded_uint64(&g, 0, uint64_t(ps - 1), 0, false));
    } else {
      int64_t sz = int64_t(n);
      if (sz > ps) fail("Rng.choice: cannot take a larger sample than the population without replacement");
      auto shuffle_int = [&g](int64_t m, int64_t first, int64_t* data) {
        for (int64_t i = m - 1; i >= first; --i) {
          int64_t j = int64_t(nprand::random_bounded_uint64(&g, 0, uint64_t(i), 0, false));
          std::swap(data[i], data[j]);
        }
      };
      if (ps > 10000 && sz > ps / 50) {
        std::vector<int64_t> all((size_t)ps);
        std::iota(all.begin(), all.end(), 0);
        shuffle_int(ps, std::max<int64_t>(ps - sz, 1), all.data());
        std::copy(all.end() - sz, all.end(), idx.begin());
      } else if (sz > 0) {  // Floyd's algorithm with numpy's open-addressing set, then a shuffle
        uint64_t set_size = uint64_t(1.2 * double(sz)), mask = set_size;
        for (int sh : {1, 2, 4, 8, 16, 32}) mask |= mask >> sh;
        std::vector<uint64_t> hs(size_t(mask + 1), ~uint64_t(0));
        for (int64_t j = ps - sz; j < ps; ++j) {
          uint64_t val = nprand::random_bounded_uint64(&g, 0, uint64_t(j), 0, false), loc = val & mask;
          while (hs[loc] != ~uint64_t(0) && hs[loc] != val) loc = (loc + 1) & mask;
          if (hs[loc] == ~uint64_t(0)) { hs[loc] = val; idx[size_t(j - ps + sz)] = int64_t(val); }
          else {
            loc = uint64_t(j) & mask;
            while (hs[loc] != ~uint64_t(0)) loc = (loc + 1) & mask;
            hs[loc] = uint64_t(j);
            idx[size_t(j - ps + sz)] = j;
          }
        }
        shuffle_int(sz, 1, idx.data());
      }
    }
    V r(n);
    for (size_t i = 0; i < n; ++i) r[i] = pop[size_t(idx[i])];
    return vecv(std::move(r)); });
  auto shuffle = [](nprand::Bitgen& g, V& x) {  // Generator.shuffle, 1-D fast path
    for (size_t i = x.size(); i-- > 1;) std::swap(x[i], x[nprand::random_interval(&g, i)]);
  };
  api.method(Rt, "permutation", {"int"}, Vt, "a random permutation of 0..n-1 (Generator.permutation)", [shuffle](void* s, Args& a) {
    V r(count_arg(a[0].i, "Rng.permutation")); std::iota(r.begin(), r.end(), 0.0); shuffle(G(s), r); return vecv(std::move(r)); });
  api.method(Rt, "shuffle", {Vt}, Vt, "a shuffled copy of a vector (Generator.permutation of an array)", [shuffle](void* s, Args& a) {
    V r = av(a[0]); shuffle(G(s), r); return vecv(std::move(r)); });
  api.method(Rt, "jumped", {}, Rt, "an independent generator: this state advanced by (phi - 1) * 2^128 draws (PCG64.jumped)", [](void* s, Args&) {
    auto* r = make<Rng>(0); r->g = G(s); r->g.jump(); return O(r); });

  // tests
  api.fn("ttest_1samp", {Vt, "float"}, Rs, "one-sample t test against popmean (scipy.stats.ttest_1samp, two-sided): statistic, pvalue, df", [](Args& a) {
    const V& x = av(a[0]); need_finite(x, "ttest_1samp");
    if (x.size() < 2) fail("ttest_1samp: needs at least 2 values");
    double m = mean_of(x), n = double(x.size());
    return O(ttest_res((m - a[1].f) / std::sqrt(var_of(x, m) / n), n - 1)); });
  api.fn("ttest_ind", {Vt, Vt}, Rs, "two-sample t test with pooled variance (scipy.stats.ttest_ind, two-sided)", [](Args& a) { return O(ttest_ind(av(a[0]), av(a[1]), true)); });
  api.fn("ttest_ind", {Vt, Vt, Ot}, Rs, "two-sample t test; option equal_var (default true; false = Welch)", [](Args& a) {
    const Opts& o = obj<Opts>(a[2]); bool ev = o.flag("equal_var", true); o.finish("ttest_ind"); return O(ttest_ind(av(a[0]), av(a[1]), ev)); });
  api.fn("pearsonr", {Vt, Vt}, Rs, "Pearson correlation with the two-sided p-value of the exact beta null (scipy.stats.pearsonr): statistic, pvalue, df", [](Args& a) {
    const V& x = av(a[0]); const V& y = av(a[1]);
    if (x.size() != y.size()) fail("pearsonr: vectors differ in length");
    if (x.size() < 2) fail("pearsonr: needs at least 2 pairs");
    need_finite(x, "pearsonr"); need_finite(y, "pearsonr");
    double mx = mean_of(x), my = mean_of(y), nx = 0, ny = 0;
    V xm(x.size()), ym(y.size());
    for (size_t i = 0; i < x.size(); ++i) { xm[i] = x[i] - mx; ym[i] = y[i] - my; nx += xm[i] * xm[i]; ny += ym[i] * ym[i]; }
    nx = std::sqrt(nx); ny = std::sqrt(ny);
    if (nx == 0 || ny == 0) fail("pearsonr: an input is constant; the correlation is undefined");
    double r = 0;
    for (size_t i = 0; i < x.size(); ++i) r += (xm[i] / nx) * (ym[i] / ny);
    r = std::max(-1.0, std::min(1.0, r));
    double n = double(x.size()), ab = n / 2 - 1, p;
    if (n == 2) p = 1.0;
    else p = 2 * st::betaincc(ab, ab, (std::fabs(r) + 1) / 2);
    auto* res = make<Res>(); res->num("statistic", r).num("pvalue", p).num("df", n - 2); return O(res); });
}

}  // namespace lmnum
