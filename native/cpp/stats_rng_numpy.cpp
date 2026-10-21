// numpy.random.Generator samplers, ported from numpy 2.3.3 (numpy/random/src/distributions/distributions.c) with the
// function bodies unchanged, so that lmpy reproduces the exact streams of numpy.random.Generator(PCG64(seed)).
// Only the double-precision samplers lmpy needs are kept; the bit generator interface is lmpy's Bitgen (stats_rng.hpp).
// numpy is BSD-3-Clause, Copyright (c) 2005-2025, NumPy Developers (https://github.com/numpy/numpy/blob/main/LICENSE.txt).
// Parts of distributions.c are themselves derived from: Hoermann (PTRS Poisson, 1993), Kachitvichyanukul & Schmeiser
// (BTPE binomial, 1988), Marsaglia & Tsang (gamma, 2000; ziggurat, 2000), Lemire (bounded integers, 2019).
#include "stats_rng.hpp"
#include "stats_ziggurat.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>

// numpy keeps unused buffer arguments for templating symmetry; keep the bodies unmodified.
#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace lmpy { namespace nprand {

#define RAND_INT_TYPE int64_t
#define RAND_INT_MAX INT64_MAX
#define MIN(x, y) (((x) < (y)) ? x : y)
#define MAX(x, y) (((x) > (y)) ? x : y)
#define npy_log1p log1p
#define npy_isnan std::isnan
#define NPY_NAN NAN
using std::log; using std::exp; using std::pow; using std::sqrt; using std::floor; using std::ceil; using std::fabs;
using std::log1p; using std::expm1; using std::cos; using std::acos; using std::fmod;

typedef Bitgen bitgen_t;

static inline uint32_t next_uint32(bitgen_t *bitgen_state) { return bitgen_state->next32(); }
static inline uint64_t next_uint64(bitgen_t *bitgen_state) { return bitgen_state->next64(); }
static inline double next_double(bitgen_t *bitgen_state) { return bitgen_state->next_double(); }

static double standard_exponential_unlikely(bitgen_t *bitgen_state,
                                                uint8_t idx, double x) {
  if (idx == 0) {
    /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
    return ziggurat_exp_r - npy_log1p(-next_double(bitgen_state));
  } else if ((fe_double[idx - 1] - fe_double[idx]) * next_double(bitgen_state) +
                 fe_double[idx] <
             exp(-x)) {
    return x;
  } else {
    return random_standard_exponential(bitgen_state);
  }
}

double random_standard_exponential(bitgen_t *bitgen_state) {
  uint64_t ri;
  uint8_t idx;
  double x;
  ri = next_uint64(bitgen_state);
  ri >>= 3;
  idx = ri & 0xFF;
  ri >>= 8;
  x = ri * we_double[idx];
  if (ri < ke_double[idx]) {
    return x; /* 98.9% of the time we return here 1st try */
  }
  return standard_exponential_unlikely(bitgen_state, idx, x);
}


double random_standard_normal(bitgen_t *bitgen_state) {
  uint64_t r;
  int sign;
  uint64_t rabs;
  int idx;
  double x, xx, yy;
  for (;;) {
    /* r = e3n52sb8 */
    r = next_uint64(bitgen_state);
    idx = r & 0xff;
    r >>= 8;
    sign = r & 0x1;
    rabs = (r >> 1) & 0x000fffffffffffff;
    x = rabs * wi_double[idx];
    if (sign & 0x1)
      x = -x;
    if (rabs < ki_double[idx])
      return x; /* 99.3% of the time return here */
    if (idx == 0) {
      for (;;) {
        /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
        xx = -ziggurat_nor_inv_r * npy_log1p(-next_double(bitgen_state));
        yy = -npy_log1p(-next_double(bitgen_state));
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                                     : ziggurat_nor_r + xx;
      }
    } else {
      if (((fi_double[idx - 1] - fi_double[idx]) * next_double(bitgen_state) +
           fi_double[idx]) < exp(-0.5 * x * x))
        return x;
    }
  }
}


double random_standard_gamma(bitgen_t *bitgen_state,
                                            double shape) {
  double b, c;
  double U, V, X, Y;

  if (shape == 1.0) {
    return random_standard_exponential(bitgen_state);
  } else if (shape == 0.0) {
    return 0.0;
  } else if (shape < 1.0) {
    for (;;) {
      U = next_double(bitgen_state);
      V = random_standard_exponential(bitgen_state);
      if (U <= 1.0 - shape) {
        X = pow(U, 1. / shape);
        if (X <= V) {
          return X;
        }
      } else {
        Y = -log((1 - U) / shape);
        X = pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  } else {
    b = shape - 1. / 3.;
    c = 1. / sqrt(9 * b);
    for (;;) {
      do {
        X = random_standard_normal(bitgen_state);
        V = 1.0 + c * X;
      } while (V <= 0.0);

      V = V * V * V;
      U = next_double(bitgen_state);
      if (U < 1.0 - 0.0331 * (X * X) * (X * X))
        return (b * V);
      /* log(0.0) ok here */
      if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
        return (b * V);
    }
  }
}


double random_loggam(double x) {
  double x0, x2, lg2pi, gl, gl0;
  RAND_INT_TYPE k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};

  if ((x == 1.0) || (x == 2.0)) {
    return 0.0;
  } else if (x < 7.0) {
    n = (RAND_INT_TYPE)(7 - x);
  } else {
    n = 0;
  }
  x0 = x + n;
  x2 = (1.0 / x0) * (1.0 / x0);
  /* log(2 * M_PI) */
  lg2pi = 1.8378770664093453e+00;
  gl0 = a[9];
  for (k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
  if (x < 7.0) {
    for (k = 1; k <= n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

double random_normal(bitgen_t *bitgen_state, double loc, double scale) {
  return loc + scale * random_standard_normal(bitgen_state);
}

double random_exponential(bitgen_t *bitgen_state, double scale) {
  return scale * random_standard_exponential(bitgen_state);
}

double random_uniform(bitgen_t *bitgen_state, double lower, double range) {
  return lower + range * next_double(bitgen_state);
}

double random_gamma(bitgen_t *bitgen_state, double shape, double scale) {
  return scale * random_standard_gamma(bitgen_state, shape);
}

#define BETA_TINY_THRESHOLD 3e-103

/*
 *  Note: random_beta assumes that a != 0 and b != 0.
 */
double random_beta(bitgen_t *bitgen_state, double a, double b) {
  double Ga, Gb;

  if ((a <= 1.0) && (b <= 1.0)) {
    double U, V, X, Y, XpY;

    if (a < BETA_TINY_THRESHOLD && b < BETA_TINY_THRESHOLD) {
      /*
       * When a and b are this small, the probability that the
       * sample would be a double precision float that is not
       * 0 or 1 is less than approx. 1e-100.  So we use the
       * proportion a/(a + b) and a single uniform sample to
       * generate the result.
       */
      U = next_double(bitgen_state);
      return (a + b)*U < a;
    }

    /* Use Johnk's algorithm */

    while (1) {
      U = next_double(bitgen_state);
      V = next_double(bitgen_state);
      X = pow(U, 1.0 / a);
      Y = pow(V, 1.0 / b);
      XpY = X + Y;
      /* Reject if both U and V are 0.0, which is approx 1 in 10^106 */
      if ((XpY <= 1.0) && (U + V > 0.0)) {
        if ((X > 0) && (Y > 0)) {
          return X / XpY;
        } else {
          /*
           * Either X or Y underflowed to 0, so we lost information in
           * U**(1/a) or V**(1/b). We still compute X/(X+Y) here, but we
           * work with logarithms as much as we can to avoid the underflow.
           */
          double logX = log(U)/a;
          double logY = log(V)/b;
          double delta = logX - logY;
          if (delta > 0) {
            return exp(-log1p(exp(-delta)));
          }
          else {
            return exp(delta - log1p(exp(delta)));
          }
        }
      }
    }
  } else {
    Ga = random_standard_gamma(bitgen_state, a);
    Gb = random_standard_gamma(bitgen_state, b);
    return Ga / (Ga + Gb);
  }
}

double random_chisquare(bitgen_t *bitgen_state, double df) {
  return 2.0 * random_standard_gamma(bitgen_state, df / 2.0);
}

double random_f(bitgen_t *bitgen_state, double dfnum, double dfden) {
  return ((random_chisquare(bitgen_state, dfnum) * dfden) /
          (random_chisquare(bitgen_state, dfden) * dfnum));
}

double random_standard_cauchy(bitgen_t *bitgen_state) {
  return random_standard_normal(bitgen_state) / random_standard_normal(bitgen_state);
}

double random_pareto(bitgen_t *bitgen_state, double a) {
  return expm1(random_standard_exponential(bitgen_state) / a);
}

double random_weibull(bitgen_t *bitgen_state, double a) {
  if (a == 0.0) {
    return 0.0;
  }
  return pow(random_standard_exponential(bitgen_state), 1. / a);
}

double random_power(bitgen_t *bitgen_state, double a) {
  return pow(-expm1(-random_standard_exponential(bitgen_state)), 1. / a);
}

double random_laplace(bitgen_t *bitgen_state, double loc, double scale) {
  double U;

  U = next_double(bitgen_state);
  if (U >= 0.5) {
    U = loc - scale * log(2.0 - U - U);
  } else if (U > 0.0) {
    U = loc + scale * log(U + U);
  } else {
    /* Reject U == 0.0 and call again to get next value */
    U = random_laplace(bitgen_state, loc, scale);
  }
  return U;
}

double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) {
  double U;

  U = 1.0 - next_double(bitgen_state);
  if (U < 1.0) {
    return loc - scale * log(-log(U));
  }
  /* Reject U == 1.0 and call again to get next value */
  return random_gumbel(bitgen_state, loc, scale);
}

double random_logistic(bitgen_t *bitgen_state, double loc, double scale) {
  double U;

  U = next_double(bitgen_state);
  if (U > 0.0) {
    return loc + scale * log(U / (1.0 - U));
  }
  /* Reject U == 0.0 and call again to get next value */
  return random_logistic(bitgen_state, loc, scale);
}

double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma) {
  return exp(random_normal(bitgen_state, mean, sigma));
}

double random_rayleigh(bitgen_t *bitgen_state, double mode) {
  return mode * sqrt(2.0 * random_standard_exponential(bitgen_state));
}

double random_standard_t(bitgen_t *bitgen_state, double df) {
  double num, denom;

  num = random_standard_normal(bitgen_state);
  denom = random_standard_gamma(bitgen_state, df / 2);
  return sqrt(df / 2) * num / sqrt(denom);
}

static RAND_INT_TYPE random_poisson_mult(bitgen_t *bitgen_state, double lam) {
  RAND_INT_TYPE X;
  double prod, U, enlam;

  enlam = exp(-lam);
  X = 0;
  prod = 1.0;
  while (1) {
    U = next_double(bitgen_state);
    prod *= U;
    if (prod > enlam) {
      X += 1;
    } else {
      return X;
    }
  }
}

/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
static RAND_INT_TYPE random_poisson_ptrs(bitgen_t *bitgen_state, double lam) {
  RAND_INT_TYPE k;
  double U, V, slam, loglam, a, b, invalpha, vr, us;

  slam = sqrt(lam);
  loglam = log(lam);
  b = 0.931 + 2.53 * slam;
  a = -0.059 + 0.02483 * b;
  invalpha = 1.1239 + 1.1328 / (b - 3.4);
  vr = 0.9277 - 3.6224 / (b - 2);

  while (1) {
    U = next_double(bitgen_state) - 0.5;
    V = next_double(bitgen_state);
    us = 0.5 - fabs(U);
    k = (RAND_INT_TYPE)floor((2 * a / us + b) * U + lam + 0.43);
    if ((us >= 0.07) && (V <= vr)) {
      return k;
    }
    if ((k < 0) || ((us < 0.013) && (V > us))) {
      continue;
    }
    /* log(V) == log(0.0) ok here */
    /* if U==0.0 so that us==0.0, log is ok since always returns */
    if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
        (-lam + k * loglam - random_loggam(k + 1))) {
      return k;
    }
  }
}

RAND_INT_TYPE random_poisson(bitgen_t *bitgen_state, double lam) {
  if (lam >= 10) {
    return random_poisson_ptrs(bitgen_state, lam);
  } else if (lam == 0) {
    return 0;
  } else {
    return random_poisson_mult(bitgen_state, lam);
  }
}

RAND_INT_TYPE random_negative_binomial(bitgen_t *bitgen_state, double n,
                                       double p) {
  double Y = random_gamma(bitgen_state, n, (1 - p) / p);
  return random_poisson(bitgen_state, Y);
}

RAND_INT_TYPE random_binomial_btpe(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                                   double p, binomial_t *binomial) {
  double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
  double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
  RAND_INT_TYPE m, y, k, i;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    /* initialize */
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->r = r = MIN(p, 1.0 - p);
    binomial->q = q = 1.0 - r;
    binomial->fm = fm = n * r + r;
    binomial->m = m = (RAND_INT_TYPE)floor(binomial->fm);
    binomial->p1 = p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
    binomial->xm = xm = m + 0.5;
    binomial->xl = xl = xm - p1;
    binomial->xr = xr = xm + p1;
    binomial->c = c = 0.134 + 20.5 / (15.3 + m);
    a = (fm - xl) / (fm - xl * r);
    binomial->laml = laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    binomial->lamr = lamr = a * (1.0 + a / 2.0);
    binomial->p2 = p2 = p1 * (1.0 + 2.0 * c);
    binomial->p3 = p3 = p2 + c / laml;
    binomial->p4 = p4 = p3 + c / lamr;
  } else {
    r = binomial->r;
    q = binomial->q;
    fm = binomial->fm;
    m = binomial->m;
    p1 = binomial->p1;
    xm = binomial->xm;
    xl = binomial->xl;
    xr = binomial->xr;
    c = binomial->c;
    laml = binomial->laml;
    lamr = binomial->lamr;
    p2 = binomial->p2;
    p3 = binomial->p3;
    p4 = binomial->p4;
  }

/* sigh ... */
Step10:
  nrq = n * r * q;
  u = next_double(bitgen_state) * p4;
  v = next_double(bitgen_state);
  if (u > p1)
    goto Step20;
  y = (RAND_INT_TYPE)floor(xm - p1 * v + u);
  goto Step60;

Step20:
  if (u > p2)
    goto Step30;
  x = xl + (u - p1) / c;
  v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
  if (v > 1.0)
    goto Step10;
  y = (RAND_INT_TYPE)floor(x);
  goto Step50;

Step30:
  if (u > p3)
    goto Step40;
  y = (RAND_INT_TYPE)floor(xl + log(v) / laml);
  /* Reject if v==0.0 since previous cast is undefined */
  if ((y < 0) || (v == 0.0))
    goto Step10;
  v = v * (u - p2) * laml;
  goto Step50;

Step40:
  y = (RAND_INT_TYPE)floor(xr - log(v) / lamr);
  /* Reject if v==0.0 since previous cast is undefined */
  if ((y > n) || (v == 0.0))
    goto Step10;
  v = v * (u - p3) * lamr;

Step50:
  k = llabs(y - m);
  if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
    goto Step52;

  s = r / q;
  a = s * (n + 1);
  F = 1.0;
  if (m < y) {
    for (i = m + 1; i <= y; i++) {
      F *= (a / i - s);
    }
  } else if (m > y) {
    for (i = y + 1; i <= m; i++) {
      F /= (a / i - s);
    }
  }
  if (v > F)
    goto Step10;
  goto Step60;

Step52:
  rho =
      (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
  t = -k * k / (2 * nrq);
  /* log(0.0) ok here */
  A = log(v);
  if (A < (t - rho))
    goto Step60;
  if (A > (t + rho))
    goto Step10;

  x1 = y + 1;
  f1 = m + 1;
  z = n + 1 - m;
  w = n - y + 1;
  x2 = x1 * x1;
  f2 = f1 * f1;
  z2 = z * z;
  w2 = w * w;
  if (A > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
           (y - m) * log(w * r / (x1 * q)) +
           (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w /
               166320.)) {
    goto Step10;
  }

Step60:
  if (p > 0.5) {
    y = n - y;
  }

  return y;
}

RAND_INT_TYPE random_binomial_inversion(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                                        double p, binomial_t *binomial) {
  double q, qn, np, px, U;
  RAND_INT_TYPE X, bound;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->q = q = 1.0 - p;
    binomial->r = qn = exp(n * log(q));
    binomial->c = np = n * p;
    binomial->m = bound = (RAND_INT_TYPE)MIN(n, np + 10.0 * sqrt(np * q + 1));
  } else {
    q = binomial->q;
    qn = binomial->r;
    np = binomial->c;
    bound = binomial->m;
  }
  X = 0;
  px = qn;
  U = next_double(bitgen_state);
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = next_double(bitgen_state);
    } else {
      U -= px;
      px = ((n - X + 1) * p * px) / (X * q);
    }
  }
  return X;
}

int64_t random_binomial(bitgen_t *bitgen_state, double p, int64_t n,
                        binomial_t *binomial) {
  double q;

  if ((n == 0LL) || (p == 0.0f))
    return 0;

  if (p <= 0.5) {
    if (p * n <= 30.0) {
      return random_binomial_inversion(bitgen_state, n, p, binomial);
    } else {
      return random_binomial_btpe(bitgen_state, n, p, binomial);
    }
  } else {
    q = 1.0 - p;
    if (q * n <= 30.0) {
      return n - random_binomial_inversion(bitgen_state, n, q, binomial);
    } else {
      return n - random_binomial_btpe(bitgen_state, n, q, binomial);
    }
  }
}

double random_noncentral_chisquare(bitgen_t *bitgen_state, double df,
                                   double nonc) {
  if (npy_isnan(nonc)) {
    return NPY_NAN;
  }
  if (nonc == 0) {
    return random_chisquare(bitgen_state, df);
  }
  if (1 < df) {
    const double Chi2 = random_chisquare(bitgen_state, df - 1);
    const double n = random_standard_normal(bitgen_state) + sqrt(nonc);
    return Chi2 + n * n;
  } else {
    const RAND_INT_TYPE i = random_poisson(bitgen_state, nonc / 2.0);
    return random_chisquare(bitgen_state, df + 2 * i);
  }
}

double random_noncentral_f(bitgen_t *bitgen_state, double dfnum, double dfden,
                           double nonc) {
  double t = random_noncentral_chisquare(bitgen_state, dfnum, nonc) * dfden;
  return t / (random_chisquare(bitgen_state, dfden) * dfnum);
}

double random_wald(bitgen_t *bitgen_state, double mean, double scale) {
  double U, X, Y;
  double mu_2l;

  mu_2l = mean / (2 * scale);
  Y = random_standard_normal(bitgen_state);
  Y = mean * Y * Y;
  X = mean + mu_2l * (Y - sqrt(4 * scale * Y + Y * Y));
  U = next_double(bitgen_state);
  if (U <= mean / (mean + X)) {
    return X;
  } else {
    return mean * mean / X;
  }
}

double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa) {
  double s;
  double U, V, W, Y, Z;
  double result, mod;
  int neg;
  if (npy_isnan(kappa)) {
    return NPY_NAN;
  }
  if (kappa < 1e-8) {
    /* Use a uniform for very small values of kappa */
    return M_PI * (2 * next_double(bitgen_state) - 1);
  } else {
    /* with double precision rho is zero until 1.4e-8 */
    if (kappa < 1e-5) {
      /*
       * second order taylor expansion around kappa = 0
       * precise until relatively large kappas as second order is 0
       */
      s = (1. / kappa + kappa);
    } else {
      if (kappa <= 1e6) {
        /* Path for 1e-5 <= kappa <= 1e6 */
        double r = 1 + sqrt(1 + 4 * kappa * kappa);
        double rho = (r - sqrt(2 * r)) / (2 * kappa);
        s = (1 + rho * rho) / (2 * rho);
      } else {
        /* Fallback to wrapped normal distribution for kappa > 1e6 */
        result = mu + sqrt(1. / kappa) * random_standard_normal(bitgen_state);
        /* Ensure result is within bounds */
        if (result < -M_PI) {
          result += 2*M_PI;
        }
        if (result > M_PI) {
          result -= 2*M_PI;
        }
        return result;
      }
    }

    while (1) {
      U = next_double(bitgen_state);
      Z = cos(M_PI * U);
      W = (1 + s * Z) / (s + Z);
      Y = kappa * (s - W);
      V = next_double(bitgen_state);
      /*
       * V==0.0 is ok here since Y >= 0 always leads
       * to accept, while Y < 0 always rejects
       */
      if ((Y * (2 - Y) - V >= 0) || (log(Y / V) + 1 - Y >= 0)) {
        break;
      }
    }

    U = next_double(bitgen_state);

    result = acos(W);
    if (U < 0.5) {
      result = -result;
    }
    result += mu;
    neg = (result < 0);
    mod = fabs(result);
    mod = (fmod(mod + M_PI, 2 * M_PI) - M_PI);
    if (neg) {
      mod *= -1;
    }

    return mod;
  }
}

int64_t random_logseries(bitgen_t *bitgen_state, double p) {
  double q, r, U, V;
  int64_t result;

  r = npy_log1p(-p);

  while (1) {
    V = next_double(bitgen_state);
    if (V >= p) {
      return 1;
    }
    U = next_double(bitgen_state);
    q = -expm1(r * U);
    if (V <= q * q) {
      result = (int64_t)floor(1 + log(V) / log(q));
      if ((result < 1) || (V == 0.0)) {
        continue;
      } else {
        return result;
      }
    }
    if (V >= q) {
      return 1;
    }
    return 2;
  }
}

/*
 * RAND_INT_TYPE is used to share integer generators with RandomState which
 * used long in place of int64_t. If changing a distribution that uses
 * RAND_INT_TYPE, then the original unmodified copy must be retained for
 * use in RandomState by copying to the legacy distributions source file.
 */

/* Still used but both generator and mtrand via legacy_random_geometric */
RAND_INT_TYPE random_geometric_search(bitgen_t *bitgen_state, double p) {
  double U;
  RAND_INT_TYPE X;
  double sum, prod, q;

  X = 1;
  sum = prod = p;
  q = 1.0 - p;
  U = next_double(bitgen_state);
  while (U > sum) {
    prod *= q;
    sum += prod;
    X++;
  }
  return X;
}

int64_t random_geometric_inversion(bitgen_t *bitgen_state, double p) {
  double z = ceil(-random_standard_exponential(bitgen_state) / npy_log1p(-p));
  /*
   * The constant 9.223372036854776e+18 is the smallest double that is
   * larger than INT64_MAX.
   */
  if (z >= 9.223372036854776e+18) {
    return INT64_MAX;
  }
  return (int64_t) z;
}

int64_t random_geometric(bitgen_t *bitgen_state, double p) {
  if (p >= 0.333333333333333333333333) {
    return random_geometric_search(bitgen_state, p);
  } else {
    return random_geometric_inversion(bitgen_state, p);
  }
}

RAND_INT_TYPE random_zipf(bitgen_t *bitgen_state, double a) {
  double am1, b, Umin;

  if (a >= 1025) {
    /*
     * If a exceeds 1025, the calculation of b will overflow and the loop
     * will not terminate.  It is safe to simply return 1 here, because the
     * probability of generating a value greater than 1 in this case is
     * less than 3e-309.
     */
    return (RAND_INT_TYPE) 1;
  }
  am1 = a - 1.0;
  b = pow(2.0, am1);
  /*
   * In the while loop, X is generated from the uniform distribution (Umin, 1].
   * Values below Umin would result in X being rejected because it is too
   * large, so there is no point in including them in the distribution of U.
   */
  Umin = pow((double) RAND_INT_MAX, -am1);
  while (1) {
    double U01, T, U, V, X;

    /*
     * U is sampled from (Umin, 1]. Note that Umin might be 0, and we don't
     * want U to be 0.
     */
    U01 = next_double(bitgen_state);
    U = U01*Umin + (1 - U01);
    V = next_double(bitgen_state);
    X = floor(pow(U, -1.0 / am1));
    /*
     * The real result may be above what can be represented in a signed
     * long. Since this is a straightforward rejection algorithm, we can
     * just reject this value. This function then models a Zipf
     * distribution truncated to sys.maxint.
     */
    if (X > (double)RAND_INT_MAX || X < 1.0) {
      continue;
    }

    T = pow(1.0 + 1.0 / X, am1);
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) {
      return (RAND_INT_TYPE)X;
    }
  }
}

uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) {
  uint64_t mask, value;
  if (max == 0) {
    return 0;
  }

  mask = max;

  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;

  /* Search a random value in [0..mask] <= max */
  if (max <= 0xffffffffUL) {
    while ((value = (next_uint32(bitgen_state) & mask)) > max)
      ;
  } else {
    while ((value = (next_uint64(bitgen_state) & mask)) > max)
      ;
  }
  return value;
}

/* Bounded generators */
static inline uint64_t gen_mask(uint64_t max) {
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  return mask;
}


/* Static `masked rejection` function called by random_bounded_uint64(...) */
static inline uint64_t bounded_masked_uint64(bitgen_t *bitgen_state,
                                             uint64_t rng, uint64_t mask) {
  uint64_t val;

  while ((val = (next_uint64(bitgen_state) & mask)) > rng)
    ;

  return val;
}

/* Static `masked rejection` function called by
 * random_buffered_bounded_uint32(...) */
static inline uint32_t
buffered_bounded_masked_uint32(bitgen_t *bitgen_state, uint32_t rng,
                               uint32_t mask, int *bcnt, uint32_t *buf) {
  /*
   * The buffer and buffer count are not used here but are included to allow
   * this function to be templated with the similar uint8 and uint16
   * functions
   */

  uint32_t val;

  while ((val = (next_uint32(bitgen_state) & mask)) > rng)
    ;

  return val;
}

/* Static `masked rejection` function called by
 * random_buffered_bounded_uint16(...) */

static inline uint64_t bounded_lemire_uint64(bitgen_t *bitgen_state,
                                             uint64_t rng) {
  /*
   * Uses Lemire's algorithm - https://arxiv.org/abs/1805.10941
   *
   * Note: `rng` should not be 0xFFFFFFFFFFFFFFFF. When this happens `rng_excl`
   * becomes zero.
   */
  const uint64_t rng_excl = rng + 1;

  assert(rng != 0xFFFFFFFFFFFFFFFFULL);

#if __SIZEOF_INT128__
  /* 128-bit uint available (e.g. GCC/clang). `m` is the __uint128_t scaled
   * integer. */
  __uint128_t m;
  uint64_t leftover;

  /* Generate a scaled random number. */
  m = ((__uint128_t)next_uint64(bitgen_state)) * rng_excl;

  /* Rejection sampling to remove any bias. */
  leftover = m & 0xFFFFFFFFFFFFFFFFULL;

  if (leftover < rng_excl) {
    /* `rng_excl` is a simple upper bound for `threshold`. */
    const uint64_t threshold = (UINT64_MAX - rng) % rng_excl;

    while (leftover < threshold) {
      m = ((__uint128_t)next_uint64(bitgen_state)) * rng_excl;
      leftover = m & 0xFFFFFFFFFFFFFFFFULL;
    }
  }

  return (m >> 64);
#else
  /* 128-bit uint NOT available (e.g. MSVS). `m1` is the upper 64-bits of the
   * scaled integer. */
  uint64_t m1;
  uint64_t x;
  uint64_t leftover;

  x = next_uint64(bitgen_state);

  /* Rejection sampling to remove any bias. */
  leftover = x * rng_excl; /* The lower 64-bits of the mult. */

  if (leftover < rng_excl) {
    /* `rng_excl` is a simple upper bound for `threshold`. */
    const uint64_t threshold = (UINT64_MAX - rng) % rng_excl;

    while (leftover < threshold) {
      x = next_uint64(bitgen_state);
      leftover = x * rng_excl;
    }
  }

#if defined(_MSC_VER) && defined(_WIN64)
  /* _WIN64 architecture. Use the __umulh intrinsic to calc `m1`. */
  m1 = __umulh(x, rng_excl);
#else
  /* 32-bit architecture. Emulate __umulh to calc `m1`. */
  {
    uint64_t x0, x1, rng_excl0, rng_excl1;
    uint64_t w0, w1, w2, t;

    x0 = x & 0xFFFFFFFFULL;
    x1 = x >> 32;
    rng_excl0 = rng_excl & 0xFFFFFFFFULL;
    rng_excl1 = rng_excl >> 32;
    w0 = x0 * rng_excl0;
    t = x1 * rng_excl0 + (w0 >> 32);
    w1 = t & 0xFFFFFFFFULL;
    w2 = t >> 32;
    w1 += x0 * rng_excl1;
    m1 = x1 * rng_excl1 + w2 + (w1 >> 32);
  }
#endif

  return m1;
#endif
}

/* Static `Lemire rejection` function called by
 * random_buffered_bounded_uint32(...) */
static inline uint32_t buffered_bounded_lemire_uint32(
    bitgen_t *bitgen_state, uint32_t rng, int *bcnt, uint32_t *buf) {
  /*
   * Uses Lemire's algorithm - https://arxiv.org/abs/1805.10941
   *
   * The buffer and buffer count are not used here but are included to allow
   * this function to be templated with the similar uint8 and uint16
   * functions
   *
   * Note: `rng` should not be 0xFFFFFFFF. When this happens `rng_excl` becomes
   * zero.
   */
  const uint32_t rng_excl = rng + 1;

  uint64_t m;
  uint32_t leftover;

  assert(rng != 0xFFFFFFFFUL);

  /* Generate a scaled random number. */
  m = ((uint64_t)next_uint32(bitgen_state)) * rng_excl;

  /* Rejection sampling to remove any bias */
  leftover = m & 0xFFFFFFFFUL;

  if (leftover < rng_excl) {
    /* `rng_excl` is a simple upper bound for `threshold`. */
    const uint32_t threshold = (UINT32_MAX - rng) % rng_excl;

    while (leftover < threshold) {
      m = ((uint64_t)next_uint32(bitgen_state)) * rng_excl;
      leftover = m & 0xFFFFFFFFUL;
    }
  }

  return (m >> 32);
}

uint64_t random_bounded_uint64(bitgen_t *bitgen_state, uint64_t off,
                               uint64_t rng, uint64_t mask, bool use_masked) {
  if (rng == 0) {
    return off;
  } else if (rng <= 0xFFFFFFFFUL) {
    /* Call 32-bit generator if range in 32-bit. */
    if (rng == 0xFFFFFFFFUL) {
      /*
       * The 32-bit Lemire method does not handle rng=0xFFFFFFFF, so we'll
       * call next_uint32 directly.  This also works when use_masked is True,
       * so we handle both cases here.
       */
      return off + (uint64_t) next_uint32(bitgen_state);
    }
    if (use_masked) {
      return off + buffered_bounded_masked_uint32(bitgen_state, rng, mask, NULL,
                                                  NULL);
    } else {
      return off +
             buffered_bounded_lemire_uint32(bitgen_state, rng, NULL, NULL);
    }
  } else if (rng == 0xFFFFFFFFFFFFFFFFULL) {
    /* Lemire64 doesn't support inclusive rng = 0xFFFFFFFFFFFFFFFF. */
    return off + next_uint64(bitgen_state);
  } else {
    if (use_masked) {
      return off + bounded_masked_uint64(bitgen_state, rng, mask);
    } else {
      return off + bounded_lemire_uint64(bitgen_state, rng);
    }
  }
}

// ---- SeedSequence and PCG64 seeding (numpy bit_generator.pyx, _pcg64.pyx, pcg64.c) ------------------------------
namespace {
const uint32_t INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
const uint32_t MIX_MULT_L = 0xca01f9dd, MIX_MULT_R = 0x4973f715, XSHIFT = 16;
uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= MULT_A;
  value *= hash_const;
  value ^= value >> XSHIFT;
  return value;
}
uint32_t mix(uint32_t x, uint32_t y) {
  uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
  result ^= result >> XSHIFT;
  return result;
}
}  // namespace

Bitgen::Bitgen(uint64_t seed) {
  // SeedSequence(seed): entropy words are the 32-bit limbs of the integer, lowest first ([0] for 0); pool size 4.
  uint32_t entropy[2];
  int n_entropy = 0;
  if (seed == 0) entropy[n_entropy++] = 0;
  for (uint64_t s = seed; s > 0; s >>= 32) entropy[n_entropy++] = uint32_t(s & 0xffffffffULL);
  uint32_t pool[4];
  uint32_t hash_const = INIT_A;
  for (int i = 0; i < 4; ++i) pool[i] = hashmix(i < n_entropy ? entropy[i] : 0u, hash_const);
  for (int i_src = 0; i_src < 4; ++i_src)
    for (int i_dst = 0; i_dst < 4; ++i_dst)
      if (i_src != i_dst) pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], hash_const));
  // (an entropy longer than the pool would be mixed in here; a 64-bit seed has at most two words)
  // generate_state(4, uint64): 8 uint32 words, little-endian pairs.
  uint32_t words[8];
  uint32_t hc = INIT_B;
  for (int i = 0; i < 8; ++i) {
    uint32_t v = pool[i % 4];
    v ^= hc;
    hc *= MULT_B;
    v *= hc;
    v ^= v >> XSHIFT;
    words[i] = v;
  }
  uint64_t w[4];
  for (int i = 0; i < 4; ++i) w[i] = uint64_t(words[2 * i]) | (uint64_t(words[2 * i + 1]) << 32);
  u128 initstate = (u128(w[0]) << 64) | w[1];
  u128 initseq = (u128(w[2]) << 64) | w[3];
  // pcg_setseq_128_srandom_r
  state = 0;
  inc = (initseq << 1u) | 1u;
  step();
  state += initstate;
  step();
  has_uint32 = 0;
  uinteger = 0;
}

void Bitgen::advance(u128 delta) {
  // pcg_advance_lcg_128 (Brown, "Random number generation with arbitrary strides", 1994)
  u128 cur_mult = mult(), cur_plus = inc, acc_mult = 1u, acc_plus = 0u;
  while (delta > 0) {
    if (delta & 1) {
      acc_mult *= cur_mult;
      acc_plus = acc_plus * cur_mult + cur_plus;
    }
    cur_plus = (cur_mult + 1) * cur_plus;
    cur_mult *= cur_mult;
    delta /= 2;
  }
  state = acc_mult * state + acc_plus;
  has_uint32 = 0;
  uinteger = 0;
}

void Bitgen::jump(uint64_t jumps) {
  const u128 step_size = (u128(0x9e3779b97f4a7c15ULL) << 64) | 0xf39cc0605cedc835ULL;
  advance(step_size * u128(jumps));
}

#undef RAND_INT_TYPE
#undef RAND_INT_MAX
#undef MIN
#undef MAX
#undef npy_log1p
#undef npy_isnan
#undef NPY_NAN
}}  // namespace lmpy::nprand
