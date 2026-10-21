// numpy-compatible random numbers: the PCG64 bit generator seeded through numpy's SeedSequence, and the samplers of
// numpy.random.Generator (stats_rng_numpy.cpp). Bitgen(seed) produces the same stream as
// numpy.random.Generator(numpy.random.PCG64(seed)) (equivalently numpy.random.default_rng(seed)).
// PCG64 is M.E. O'Neill's PCG XSL RR 128/64 (pcg-random.org, MIT/Apache-2.0); the seeding and jump follow numpy's
// bit_generator.pyx / _pcg64.pyx (BSD-3-Clause).
#pragma once
#include <cstdint>

namespace lmpy { namespace nprand {

// numpy's cached BTPE set-up (values only speed up repeated draws; they never change the outcome).
struct binomial_t {
  int has_binomial = 0;
  double psave = 0; int64_t nsave = 0; double r = 0, q = 0, fm = 0; int64_t m = 0;
  double p1 = 0, xm = 0, xl = 0, xr = 0, c = 0, laml = 0, lamr = 0, p2 = 0, p3 = 0, p4 = 0;
};

struct Bitgen {
  using u128 = unsigned __int128;
  u128 state = 0, inc = 0;
  int has_uint32 = 0;
  uint32_t uinteger = 0;
  binomial_t binomial;

  static constexpr u128 mult() { return (u128(2549297995355413924ULL) << 64) | 4865540595714422341ULL; }
  void step() { state = state * mult() + inc; }
  uint64_t next64() {
    step();
    uint64_t hi = uint64_t(state >> 64), lo = uint64_t(state);
    unsigned rot = unsigned(state >> 122);
    uint64_t v = hi ^ lo;
    return (v >> rot) | (v << ((-rot) & 63));
  }
  uint32_t next32() {
    if (has_uint32) { has_uint32 = 0; return uinteger; }
    uint64_t n = next64();
    has_uint32 = 1;
    uinteger = uint32_t(n >> 32);
    return uint32_t(n & 0xffffffffULL);
  }
  double next_double() { return double(next64() >> 11) * (1.0 / 9007199254740992.0); }

  // PCG64(seed): SeedSequence(seed).generate_state(4, uint64) -> initstate (words 0, 1), initseq (words 2, 3).
  explicit Bitgen(uint64_t seed);
  // numpy PCG64.advance(delta) (resets the cached 32-bit half) and jumped(): advance by jumps * (phi - 1) * 2^128.
  void advance(u128 delta);
  void jump(uint64_t jumps = 1);
};

double random_standard_exponential(Bitgen*);
double random_standard_normal(Bitgen*);
double random_standard_gamma(Bitgen*, double shape);
double random_loggam(double x);
double random_normal(Bitgen*, double loc, double scale);
double random_exponential(Bitgen*, double scale);
double random_uniform(Bitgen*, double lower, double range);
double random_gamma(Bitgen*, double shape, double scale);
double random_beta(Bitgen*, double a, double b);
double random_chisquare(Bitgen*, double df);
double random_f(Bitgen*, double dfnum, double dfden);
double random_standard_cauchy(Bitgen*);
double random_pareto(Bitgen*, double a);
double random_weibull(Bitgen*, double a);
double random_laplace(Bitgen*, double loc, double scale);
double random_gumbel(Bitgen*, double loc, double scale);
double random_logistic(Bitgen*, double loc, double scale);
double random_lognormal(Bitgen*, double mean, double sigma);
double random_rayleigh(Bitgen*, double mode);
double random_standard_t(Bitgen*, double df);
int64_t random_poisson(Bitgen*, double lam);
int64_t random_negative_binomial(Bitgen*, double n, double p);
int64_t random_binomial(Bitgen*, double p, int64_t n, binomial_t* binomial);
double random_noncentral_chisquare(Bitgen*, double df, double nonc);
double random_noncentral_f(Bitgen*, double dfnum, double dfden, double nonc);
double random_vonmises(Bitgen*, double mu, double kappa);
int64_t random_geometric(Bitgen*, double p);
int64_t random_zipf(Bitgen*, double a);
uint64_t random_interval(Bitgen*, uint64_t max);
uint64_t random_bounded_uint64(Bitgen*, uint64_t off, uint64_t rng, uint64_t mask, bool use_masked);

}}  // namespace lmpy::nprand
