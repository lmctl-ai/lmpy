// IIR filter design and filtering. Ported from scipy.signal 1.16 _filter_design.py (buttap, cheb1ap, cheb2ap,
// ellipap with _ellipdeg/_arc_jac_sn, besselap with _campos_zeros/_aberth/_norm_factor, lp2*_zpk, bilinear_zpk,
// _cplxreal, zpk2sos, zpk2tf, freqz, sosfreqz) and _signaltools.py (lfilter_zi, sosfilt_zi, filtfilt, sosfiltfilt,
// odd/even/const extension) plus the sosfilt/lfilter inner loops (_sosfilt.pyx, lfilter.c.src). scipy is
// BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. 2003-2024 SciPy Developers.
// Differences: Bessel polynomial zeros are refined by Aberth iteration on the polynomial in 50-digit arithmetic
// (scipy iterates on scipy.special.kve); complete elliptic integrals use the arithmetic-geometric mean.
#include "signal_impl.hpp"
#include "host.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <algorithm>
#include <numeric>

namespace lmnum {
namespace sig {
namespace {

using mp = boost::multiprecision::cpp_bin_float_50;

// numpy complex division (Smith's algorithm, as in numpy's loops)
cd cdiv(cd a, cd b) {
  double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
  double abr = std::fabs(br), abi = std::fabs(bi);
  if (abr >= abi) {
    if (abr == 0 && abi == 0) return cd(ar / abr, ai / abi);
    double rat = bi / br, scl = 1.0 / (br + bi * rat);
    return cd((ar + ai * rat) * scl, (ai - ar * rat) * scl);
  }
  double rat = br / bi, scl = 1.0 / (bi + br * rat);
  return cd((ar * rat + ai) * scl, (ai * rat - ar) * scl);
}
cd cprod(const CV& v) {
  if (v.empty()) return cd(1, 0);
  cd r = v[0];
  for (size_t i = 1; i < v.size(); ++i) r = cmul(r, v[i]);
  return r;
}
CV cneg(const CV& v) { CV r(v.size()); for (size_t i = 0; i < v.size(); ++i) r[i] = -v[i]; return r; }
bool isreal(cd z) { return z.imag() == 0; }

// ---- elliptic helpers ----
double agm(double a, double b) {
  for (int i = 0; i < 100; ++i) {
    double an = (a + b) / 2, bn = std::sqrt(a * b);
    if (std::fabs(an - bn) <= 1e-16 * an) return (an + bn) / 2;
    a = an; b = bn;
  }
  return (a + b) / 2;
}
// K(m) = pi / (2 agm(1, sqrt(1 - m))); ellipkm1(p) = K(1 - p) without forming 1 - p.
double ellipk(double m) { if (m == 1) return INFINITY; return M_PI / (2 * agm(1.0, std::sqrt(1 - m))); }
double ellipkm1(double p) { if (p == 0) return INFINITY; return M_PI / (2 * agm(1.0, std::sqrt(p))); }
// Jacobi elliptic functions sn, cn, dn of parameter m by the descending AGM (Abramowitz & Stegun 16.4), with the
// series of A&S 16.13 (m near 0) and 16.15 (m near 1), as in the Cephes ellpj algorithm scipy uses.
void ellipj(double u, double m, double& sn, double& cn, double& dn) {
  if (m < 0 || m > 1 || std::isnan(m)) { sn = cn = dn = NAN; return; }
  if (m < 1.0e-9) {
    double t = std::sin(u), b = std::cos(u), ai = 0.25 * m * (u - t * b);
    sn = t - ai * b; cn = b + ai * t; dn = 1.0 - 0.5 * m * t * t;
    return;
  }
  if (m >= 0.9999999999) {
    double ai = 0.25 * (1.0 - m), b = std::cosh(u), t = std::tanh(u), phi = 1.0 / b, twon = b * std::sinh(u);
    sn = t + ai * (twon - u) / (b * b);
    ai *= t * phi;
    cn = phi - ai * (twon - u);
    dn = phi + ai * (twon + u);
    return;
  }
  double a[9], c[9];
  a[0] = 1.0;
  double b = std::sqrt(1.0 - m), twon = 1.0;
  c[0] = std::sqrt(m);
  int i = 0;
  while (std::fabs(c[i] / a[i]) > 1.11022302462515654042e-16) {
    if (i > 7) break;
    double ai = a[i];
    ++i;
    c[i] = (ai - b) / 2.0;
    double t = std::sqrt(ai * b);
    a[i] = (ai + b) / 2.0;
    b = t;
    twon *= 2.0;
  }
  double phi = twon * a[i] * u, bb = 0;
  do {
    double t = c[i] * std::sin(phi) / a[i];
    bb = phi;
    phi = (std::asin(t) + phi) / 2.0;
  } while (--i);
  sn = std::sin(phi);
  double t = std::cos(phi);
  cn = t;
  dn = t / std::cos(phi - bb);
}
double pow10m1(double x) { return std::expm1(std::log(10.0) * x); }
double ellipdeg(int n, double m1) {
  double K1 = ellipk(m1), K1p = ellipkm1(m1);
  double q1 = std::exp(-M_PI * K1p / K1), q = std::pow(q1, 1.0 / n);
  double num = 0, den = 0;
  for (int m = 0; m <= 7; ++m) num += std::pow(q, double(m * (m + 1)));
  for (int m = 1; m <= 8; ++m) den += std::pow(q, double(m * m));
  den = 1 + 2 * den;
  return 16 * q * std::pow(num / den, 4.0);
}
cd complement(cd kx) { return std::pow((1.0 - kx) * (1.0 + kx), 0.5); }
cd arc_jac_sn(cd w, double m) {
  double k = std::sqrt(m);
  if (k > 1) return cd(NAN, 0);
  if (k == 1) return std::atanh(w);
  std::vector<double> ks = {k};
  int niter = 0;
  while (ks.back() != 0) {
    double kp = std::sqrt((1 - ks.back()) * (1 + ks.back()));
    ks.push_back((1 - kp) / (1 + kp));
    if (++niter > 10) fail("iir: Landen transformation not converging (ellip)");
  }
  double K = 1;
  for (size_t i = 1; i < ks.size(); ++i) K *= 1 + ks[i];
  K *= M_PI / 2;
  cd wn = w;
  for (size_t i = 0; i + 1 < ks.size(); ++i) wn = 2.0 * wn / ((1 + ks[i + 1]) * (1.0 + complement(ks[i] * wn)));
  cd u = 2 / M_PI * std::asin(wn);
  return K * u;
}
double arc_jac_sc1(double w, double m) {
  cd z = arc_jac_sn(cd(0, w), m);
  if (std::fabs(z.real()) > 1e-14) fail("iir: elliptic design failed (arc_jac_sc1)");
  return z.imag();
}

// ---- Bessel helpers ----
struct mpc { mp re, im; };
mpc operator+(const mpc& a, const mpc& b) { return {a.re + b.re, a.im + b.im}; }
mpc operator-(const mpc& a, const mpc& b) { return {a.re - b.re, a.im - b.im}; }
mpc operator*(const mpc& a, const mpc& b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
mpc operator/(const mpc& a, const mpc& b) {
  mp d = b.re * b.re + b.im * b.im;
  return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
}
mp mabs(const mpc& a) { return boost::multiprecision::sqrt(a.re * a.re + a.im * a.im); }

double npp_polyval(double x, std::vector<double> c) {
  double r = c.back();
  for (size_t i = c.size() - 1; i-- > 0;) r = c[i] + r * x;
  return r;
}
CV campos_zeros(int n) {
  if (n == 1) return {cd(-1, 0)};
  double N = n;
  double s = npp_polyval(N, {0, 0, 2, 0, -3, 1});
  double b3 = npp_polyval(N, {16, -8}) / s, b2 = npp_polyval(N, {-24, -12, 12}) / s;
  double b1 = npp_polyval(N, {8, 24, -12, -2}) / s, b0 = npp_polyval(N, {0, -6, 0, 5, -1}) / s;
  double r = npp_polyval(N, {0, 0, 2, 1});
  double a1 = npp_polyval(N, {-6, -6}) / r, a2 = 6 / r;
  CV out;
  for (int k = 1; k <= n; ++k) out.push_back(cd(npp_polyval(k, {0, a1, a2}), npp_polyval(k, {b0, b1, b2, b3})));
  return out;
}
// Zeros of the ordinary Bessel polynomial y_N(x) = sum_k (N+k)! / ((N-k)! k!) (x/2)^k.
CV bessel_zeros(int N) {
  std::vector<mp> c(size_t(N) + 1);
  for (int k = 0; k <= N; ++k) {
    mp v = 1;
    for (int j = N - k + 1; j <= N + k; ++j) v *= j;  // (N+k)!/(N-k)!
    for (int j = 2; j <= k; ++j) v /= j;
    for (int j = 0; j < k; ++j) v /= 2;
    c[size_t(k)] = v;
  }
  CV x0 = campos_zeros(N);
  std::vector<mpc> x(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) x[size_t(i)] = {mp(x0[size_t(i)].real()), mp(x0[size_t(i)].imag())};
  const mp tol("1e-32");  // far below binary64 resolution; the 50-digit evaluation noise floor is higher for N near 20
  bool done = false;
  for (int it = 0; it < 500 && !done; ++it) {
    std::vector<mpc> alpha(static_cast<size_t>(N));
    done = true;
    for (int k = 0; k < N; ++k) {
      mpc f = {c[size_t(N)], 0}, fp = {0, 0};
      for (int j = N - 1; j >= 0; --j) { fp = fp * x[size_t(k)] + f; f = f * x[size_t(k)] + mpc{c[size_t(j)], 0}; }
      alpha[size_t(k)] = mpc{0, 0} - f / fp;
    }
    for (int k = 0; k < N; ++k) {
      mpc beta = {0, 0};
      for (int j = 0; j < N; ++j) if (j != k) beta = beta + mpc{1, 0} / (x[size_t(k)] - x[size_t(j)]);
      mpc step = alpha[size_t(k)] / (mpc{1, 0} + alpha[size_t(k)] * beta);
      x[size_t(k)] = x[size_t(k)] + step;
      if (mabs(step) > tol * (1 + mabs(x[size_t(k)]))) done = false;
    }
  }
  if (!done) fail("iir: Bessel polynomial zeros failed to converge");
  CV xd(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) xd[size_t(i)] = cd(double(x[size_t(i)].re), double(x[size_t(i)].im));
  CV r(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) r[size_t(i)] = (xd[size_t(i)] + std::conj(xd[size_t(N - 1 - i)])) / 2.0;
  return r;
}
// scipy optimize.newton secant iteration as used by _norm_factor (x0 = 1.5, tol 1.48e-8, maxiter 50)
template <class F> double secant(F func, double x0) {
  double p0 = x0, eps = 1e-4, p1 = x0 * (1 + eps);
  p1 += p1 >= 0 ? eps : -eps;
  double q0 = func(p0), q1 = func(p1);
  if (std::fabs(q1) < std::fabs(q0)) { std::swap(p0, p1); std::swap(q0, q1); }
  for (int itr = 0; itr < 50; ++itr) {
    double p;
    if (q1 == q0) return (p1 + p0) / 2;
    if (std::fabs(q1) > std::fabs(q0)) p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1);
    else p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
    if (std::fabs(p - p1) <= 1.48e-8) return p;
    p0 = p1; q0 = q1; p1 = p; q1 = func(p1);
  }
  fail("iir: Bessel magnitude normalisation did not converge");
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- prototypes
Zpk buttap(int N) {
  Zpk f;
  for (int m = -N + 1; m < N; m += 2) {
    double th = M_PI * double(m) / (2 * double(N));
    f.p.push_back(-std::exp(cd(0, th)));
  }
  f.k = 1;
  return f;
}
Zpk cheb1ap(int N, double rp) {
  Zpk f;
  if (N == 0) { f.k = std::pow(10, -rp / 20); return f; }
  double eps = std::sqrt(std::pow(10, 0.1 * rp) - 1.0), mu = 1.0 / N * std::asinh(1 / eps);
  for (int m = -N + 1; m < N; m += 2) {
    double th = M_PI * double(m) / (2 * double(N));
    f.p.push_back(-std::sinh(cd(mu, th)));
  }
  f.k = cprod(cneg(f.p)).real();
  if (N % 2 == 0) f.k = f.k / std::sqrt(1 + eps * eps);
  return f;
}
Zpk cheb2ap(int N, double rs) {
  Zpk f;
  if (N == 0) { f.k = 1; return f; }
  double de = 1.0 / std::sqrt(std::pow(10, 0.1 * rs) - 1), mu = std::asinh(1.0 / de) / N;
  std::vector<int> m;
  if (N % 2) { for (int i = -N + 1; i < 0; i += 2) m.push_back(i); for (int i = 2; i < N; i += 2) m.push_back(i); }
  else for (int i = -N + 1; i < N; i += 2) m.push_back(i);
  for (int mi : m) f.z.push_back(cd(-0.0, 1.0 / std::sin(double(mi) * M_PI / (2.0 * N))));
  for (int i = -N + 1; i < N; i += 2) {
    cd q = -std::exp(cd(0, M_PI * double(i) / (2 * double(N))));
    q = cd(std::sinh(mu) * q.real(), std::cosh(mu) * q.imag());
    f.p.push_back(cdiv(cd(1, 0), q));
  }
  f.k = cdiv(cprod(cneg(f.p)), cprod(cneg(f.z))).real();
  return f;
}
Zpk ellipap(int N, double rp, double rs) {
  Zpk f;
  const double EPS = 2e-16;
  if (N == 0) { f.k = std::pow(10, -rp / 20); return f; }
  if (N == 1) {
    double p = -std::sqrt(1.0 / pow10m1(0.1 * rp));
    f.p = {cd(p, 0)};
    f.k = -p;
    f.real = true;
    return f;
  }
  double eps_sq = pow10m1(0.1 * rp), eps = std::sqrt(eps_sq), ck1_sq = eps_sq / pow10m1(0.1 * rs);
  if (ck1_sq == 0) fail("iir: cannot design an elliptic filter with these rp and rs");
  double val0 = ellipk(ck1_sq);
  double m = ellipdeg(N, ck1_sq), capk = ellipk(m);
  std::vector<double> s, c, d;
  for (int j = 1 - N % 2; j < N; j += 2) {
    double sn, cn, dn;
    ellipj(double(j) * capk / N, m, sn, cn, dn);
    s.push_back(sn); c.push_back(cn); d.push_back(dn);
  }
  CV z;
  for (double sn : s) if (std::fabs(sn) > EPS) z.push_back(cd(0, 1.0 / (std::sqrt(m) * sn)));
  size_t nz = z.size();
  for (size_t i = 0; i < nz; ++i) z.push_back(std::conj(z[i]));
  double r = arc_jac_sc1(1. / eps, ck1_sq), v0 = capk * r / (N * val0);
  double sv, cv, dv;
  ellipj(v0, 1 - m, sv, cv, dv);
  CV p;
  for (size_t i = 0; i < s.size(); ++i) {
    double den = 1 - std::pow(d[i] * sv, 2.0);
    p.push_back(cd(-(c[i] * d[i] * sv * cv) / den, -(s[i] * dv) / den));
  }
  size_t np0 = p.size();
  if (N % 2) {
    double ss = 0;
    for (auto& q : p) ss += (cmul(q, std::conj(q))).real();
    double thr = EPS * std::sqrt(ss);
    for (size_t i = 0; i < np0; ++i) if (std::fabs(p[i].imag()) > thr) p.push_back(std::conj(p[i]));
  } else {
    for (size_t i = 0; i < np0; ++i) p.push_back(std::conj(p[i]));
  }
  f.z = z; f.p = p;
  f.k = cdiv(cprod(cneg(p)), cprod(cneg(z))).real();
  if (N % 2 == 0) f.k = f.k / std::sqrt(1 + eps_sq);
  return f;
}
Zpk besselap(int N, const std::string& norm) {
  Zpk f;
  if (N == 0) { f.k = 1; return f; }
  CV x = bessel_zeros(N);
  for (auto& v : x) f.p.push_back(cdiv(cd(1, 0), v));
  // a_last = (2N)! / (N! 2^N), exact below 2^53 for N <= 15 and correctly rounded above (as Python's int -> float)
  mp al = 1;
  for (int j = N + 1; j <= 2 * N; ++j) al *= j;
  for (int j = 0; j < N; ++j) al /= 2;
  double a_last = double(al);
  if (norm == "delay" || norm == "mag") {
    f.k = a_last;
    if (norm == "mag") {
      CV p = f.p;
      double k = f.k;
      auto G = [&](double w) {
        CV t(p.size());
        for (size_t i = 0; i < p.size(); ++i) t[i] = cd(0, w) - p[i];
        return std::abs(cdiv(cd(k, 0), cprod(t))) - 1 / std::sqrt(2.0);
      };
      double nf = secant(G, 1.5);
      for (auto& q : f.p) q = q / nf;
      f.k = std::pow(nf, -N) * a_last;
    }
  } else if (norm == "phase") {
    double fac = std::pow(10, -std::log10(double(al)) / N);
    for (auto& q : f.p) q = q * fac;
    f.k = 1;
  } else {
    fail("iir: bessel norm must be phase, delay or mag; got '" + norm + "'");
  }
  return f;
}

// ---------------------------------------------------------------------------------------------------- transforms
static int rel_degree(const Zpk& f) {
  if (f.p.size() < f.z.size()) fail("iir: improper transfer function (more zeros than poles)");
  return int(f.p.size() - f.z.size());
}
Zpk lp2lp_zpk(const Zpk& f, double wo) {
  int deg = rel_degree(f);
  Zpk r;
  for (auto& z : f.z) r.z.push_back(cmul(cd(wo, 0), z));
  for (auto& p : f.p) r.p.push_back(f.real ? cd(wo * p.real(), 0.0) : cmul(cd(wo, 0), p));
  r.real = f.real;
  r.k = f.k * std::pow(wo, deg);
  return r;
}
Zpk lp2hp_zpk(const Zpk& f, double wo) {
  int deg = rel_degree(f);
  Zpk r;
  for (auto& z : f.z) r.z.push_back(cdiv(cd(wo, 0), z));
  for (auto& p : f.p) r.p.push_back(f.real ? cd(wo / p.real(), 0.0) : cdiv(cd(wo, 0), p));
  for (int i = 0; i < deg; ++i) r.z.push_back(cd(0, 0));
  r.real = f.real;
  r.k = f.k * cdiv(cprod(cneg(f.z)), cprod(cneg(f.p))).real();
  return r;
}
static void shift_pairs(const CV& in, double wo, CV& out) {
  CV plus, minus;
  for (auto& v : in) {
    cd s = std::sqrt(cmul(v, v) - wo * wo);
    plus.push_back(v + s);
    minus.push_back(v - s);
  }
  out.insert(out.end(), plus.begin(), plus.end());
  out.insert(out.end(), minus.begin(), minus.end());
}
Zpk lp2bp_zpk(const Zpk& f, double wo, double bw) {
  int deg = rel_degree(f);
  CV zl, pl;
  for (auto& z : f.z) zl.push_back(cdiv(cmul(z, cd(bw, 0)), cd(2, 0)));
  for (auto& p : f.p) pl.push_back(f.real ? cd(p.real() * bw / 2, 0.0) : cdiv(cmul(p, cd(bw, 0)), cd(2, 0)));
  Zpk r;
  shift_pairs(zl, wo, r.z);
  shift_pairs(pl, wo, r.p);
  for (int i = 0; i < deg; ++i) r.z.push_back(cd(0, 0));
  r.k = f.k * std::pow(bw, deg);
  return r;
}
Zpk lp2bs_zpk(const Zpk& f, double wo, double bw) {
  int deg = rel_degree(f);
  CV zh, ph;
  for (auto& z : f.z) zh.push_back(cdiv(cd(bw / 2, 0), z));
  for (auto& p : f.p) ph.push_back(f.real ? cd((bw / 2) / p.real(), 0.0) : cdiv(cd(bw / 2, 0), p));
  Zpk r;
  shift_pairs(zh, wo, r.z);
  shift_pairs(ph, wo, r.p);
  for (int i = 0; i < deg; ++i) r.z.push_back(cd(0, wo));
  for (int i = 0; i < deg; ++i) r.z.push_back(cd(0, -wo));
  r.k = f.k * cdiv(cprod(cneg(f.z)), cprod(cneg(f.p))).real();
  return r;
}
Zpk bilinear_zpk(const Zpk& f, double fs) {
  int deg = rel_degree(f);
  double fs2 = 2.0 * fs;
  Zpk r;
  CV a, b;
  // numpy promotes fs2 to complex: (fs2 + 0j) - z keeps the sign rules of a complex subtraction
  auto rsub = [fs2](cd v) { return cd(fs2 - v.real(), 0.0 - v.imag()); };
  auto radd = [fs2](cd v) { return cd(fs2 + v.real(), 0.0 + v.imag()); };
  for (auto& z : f.z) { r.z.push_back(cdiv(radd(z), rsub(z))); a.push_back(rsub(z)); }
  for (auto& p : f.p) {
    if (f.real) { r.p.push_back(cd((fs2 + p.real()) / (fs2 - p.real()), 0.0)); b.push_back(cd(fs2 - p.real(), 0.0)); }
    else { r.p.push_back(cdiv(radd(p), rsub(p))); b.push_back(rsub(p)); }
  }
  r.real = f.real;
  for (int i = 0; i < deg; ++i) r.z.push_back(cd(-1, 0));
  r.k = f.k * cdiv(cprod(a), cprod(b)).real();
  return r;
}

Zpk iirfilter_zpk(const std::string& ftype, int N, const V& wn, const std::string& btype, double rp, double rs, const std::string& bnorm) {
  Zpk f;
  if (ftype == "butter") f = buttap(N);
  else if (ftype == "bessel") f = besselap(N, bnorm);
  else if (ftype == "cheby1") { if (std::isnan(rp)) fail("iir: passband ripple (rp) must be provided to design a Chebyshev I filter"); f = cheb1ap(N, rp); }
  else if (ftype == "cheby2") { if (std::isnan(rs)) fail("iir: stopband attenuation (rs) must be provided to design a Chebyshev II filter"); f = cheb2ap(N, rs); }
  else if (ftype == "ellip") {
    if (std::isnan(rp) || std::isnan(rs)) fail("iir: both rp and rs must be provided to design an elliptic filter");
    f = ellipap(N, rp, rs);
  } else fail("iir: kind must be butter, cheby1, cheby2, ellip or bessel; got '" + ftype + "'");
  const double fs = 2.0;
  V warped;
  for (double w : wn) warped.push_back(2 * fs * std::tan(M_PI * w / fs));
  if (btype == "lowpass") f = lp2lp_zpk(f, warped[0]);
  else if (btype == "highpass") f = lp2hp_zpk(f, warped[0]);
  else {
    double bw = warped[1] - warped[0], wo = std::sqrt(warped[0] * warped[1]);
    f = btype == "bandpass" ? lp2bp_zpk(f, wo, bw) : lp2bs_zpk(f, wo, bw);
  }
  return bilinear_zpk(f, fs);
}

// ---------------------------------------------------------------------------------------------------- zpk -> sos / tf
namespace {
CV npoly(const CV& roots) {
  CV a = {cd(1, 0)};
  for (auto& r : roots) {
    CV n(a.size() + 1, cd(0, 0));
    for (size_t i = 0; i < a.size(); ++i) { n[i] += a[i]; n[i + 1] += cmul(a[i], -r); }
    a = n;
  }
  return a;
}
// scipy _cplxreal: conjugate-pair representatives (positive imaginary part) and the real values.
void cplxreal(const CV& z0, CV& zc, CV& zr) {
  zc.clear(); zr.clear();
  if (z0.empty()) return;
  const double tol = 100 * std::numeric_limits<double>::epsilon();
  CV z = z0;
  std::stable_sort(z.begin(), z.end(), [](cd a, cd b) {
    if (a.real() != b.real()) return a.real() < b.real();
    return std::fabs(a.imag()) < std::fabs(b.imag());
  });
  CV rest;
  for (auto& v : z) { if (std::fabs(v.imag()) <= tol * std::abs(v)) zr.push_back(cd(v.real(), 0)); else rest.push_back(v); }
  if (rest.empty()) return;
  CV zp, zn;
  for (auto& v : rest) { if (v.imag() > 0) zp.push_back(v); else if (v.imag() < 0) zn.push_back(v); }
  if (zp.size() != zn.size()) fail("zpk2sos: a complex value has no matching conjugate");
  // runs of (approximately) equal real part, each sorted by |imag|
  size_t n = zp.size(), i = 0;
  while (i < n) {
    size_t j = i;
    while (j + 1 < n && zp[j + 1].real() - zp[j].real() <= tol * std::abs(zp[j])) ++j;
    if (j > i) {
      auto byim = [](cd a, cd b) { return std::fabs(a.imag()) < std::fabs(b.imag()); };
      std::stable_sort(zp.begin() + long(i), zp.begin() + long(j + 1), byim);
      std::stable_sort(zn.begin() + long(i), zn.begin() + long(j + 1), byim);
    }
    i = j + 1;
  }
  for (size_t k = 0; k < n; ++k) {
    if (std::abs(zp[k] - std::conj(zn[k])) > tol * std::abs(zn[k])) fail("zpk2sos: a complex value has no matching conjugate");
    zc.push_back((zp[k] + std::conj(zn[k])) / 2.0);
  }
}
size_t nearest_idx(const CV& fro, cd to, const char* which) {
  std::vector<size_t> order(fro.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return std::abs(fro[a] - to) < std::abs(fro[b] - to); });
  std::string w = which;
  for (size_t o : order) {
    if (w == "any") return o;
    bool r = isreal(fro[o]);
    if ((w == "real") == r) return o;
  }
  fail("zpk2sos: no matching zero for pairing");
}
void single_sos(const CV& z, const CV& p, double* row) {
  CV b = npoly(z), a = npoly(p);
  for (int i = 0; i < 6; ++i) row[i] = 0;
  for (size_t i = 0; i < b.size(); ++i) row[3 - b.size() + i] = b[i].real();
  for (size_t i = 0; i < a.size(); ++i) row[6 - a.size() + i] = a[i].real();
}
size_t idx_worst(const CV& p) {
  size_t best = 0;
  for (size_t i = 1; i < p.size(); ++i) if (std::fabs(1 - std::abs(p[i])) < std::fabs(1 - std::abs(p[best]))) best = i;
  return best;
}
size_t count_real(const CV& v) { size_t c = 0; for (auto& x : v) c += isreal(x); return c; }
}  // namespace

Eigen::MatrixXd zpk2sos(const Zpk& f) {
  CV z = f.z, p = f.p;
  if (z.empty() && p.empty()) {
    Eigen::MatrixXd s(1, 6);
    s << f.k, 0, 0, 1, 0, 0;
    return s;
  }
  while (p.size() < z.size()) p.push_back(cd(0, 0));
  while (z.size() < p.size()) z.push_back(cd(0, 0));
  size_t n_sections = (std::max(p.size(), z.size()) + 1) / 2;
  if (p.size() % 2 == 1) { p.push_back(cd(0, 0)); z.push_back(cd(0, 0)); }
  {
    CV c, r;
    cplxreal(z, c, r); z = c; z.insert(z.end(), r.begin(), r.end());
    cplxreal(p, c, r); p = c; p.insert(p.end(), r.begin(), r.end());
  }
  Eigen::MatrixXd sos = Eigen::MatrixXd::Zero(long(n_sections), 6);
  double row[6];
  for (long si = long(n_sections) - 1; si >= 0; --si) {
    size_t i1 = idx_worst(p);
    cd p1 = p[i1];
    p.erase(p.begin() + long(i1));
    if (isreal(p1) && count_real(p) == 0) {
      size_t zi = nearest_idx(z, p1, "real");
      cd z1 = z[zi];
      z.erase(z.begin() + long(zi));
      single_sos({z1, cd(0, 0)}, {p1, cd(0, 0)}, row);
    } else if (p.size() + 1 == z.size() && !isreal(p1) && count_real(p) == 1 && count_real(z) == 1) {
      size_t zi = nearest_idx(z, p1, "complex");
      cd z1 = z[zi];
      z.erase(z.begin() + long(zi));
      single_sos({z1, std::conj(z1)}, {p1, std::conj(p1)}, row);
    } else {
      cd p2;
      if (isreal(p1)) {
        CV reals;
        std::vector<size_t> idx;
        for (size_t i = 0; i < p.size(); ++i) if (isreal(p[i])) { reals.push_back(p[i]); idx.push_back(i); }
        size_t i2 = idx[idx_worst(reals)];
        p2 = p[i2];
        p.erase(p.begin() + long(i2));
      } else {
        p2 = std::conj(p1);
      }
      if (!z.empty()) {
        size_t zi = nearest_idx(z, p1, "any");
        cd z1 = z[zi];
        z.erase(z.begin() + long(zi));
        if (!isreal(z1)) single_sos({z1, std::conj(z1)}, {p1, p2}, row);
        else if (!z.empty()) {
          size_t zj = nearest_idx(z, p1, "real");
          cd z2 = z[zj];
          z.erase(z.begin() + long(zj));
          single_sos({z1, z2}, {p1, p2}, row);
        } else single_sos({z1}, {p1, p2}, row);
      } else single_sos({}, {p1, p2}, row);
    }
    for (int j = 0; j < 6; ++j) sos(si, j) = row[j];
  }
  for (int j = 0; j < 3; ++j) sos(0, j) *= f.k;
  return sos;
}

void zpk2tf(const Zpk& f, V& b, V& a) {
  CV bz = npoly(f.z), ap = npoly(f.p);
  b.clear(); a.clear();
  for (auto& v : bz) b.push_back(f.k * v.real());
  for (auto& v : ap) a.push_back(v.real());
}

double max_pole_radius(const Eigen::MatrixXd& sos) {
  mp radius = 0;
  for (long r = 0; r < sos.rows(); ++r) {
    double bd = sos(r, 4);
    mp a = sos(r, 3), b = bd, c = sos(r, 5);
    mp disc = b * b - 4 * a * c, section;
    if (disc < 0) section = boost::multiprecision::sqrt(boost::multiprecision::abs(c / a));
    else {
      mp root = boost::multiprecision::sqrt(disc);
      mp q = -(b + (std::signbit(bd) ? mp(-root) : root)) / 2;
      section = q != 0 ? std::max(mp(boost::multiprecision::abs(q / a)), mp(boost::multiprecision::abs(c / q))) : mp(0);
    }
    if (section > radius) radius = section;
  }
  return double(radius);
}

// ---------------------------------------------------------------------------------------------------- filtering
V sosfilt(const Eigen::MatrixXd& sos, const V& x, Eigen::MatrixXd* zi) {
  long ns = sos.rows();
  Eigen::MatrixXd z = zi ? *zi : Eigen::MatrixXd::Zero(ns, 2);
  V y(x.size());
  for (size_t n = 0; n < x.size(); ++n) {
    double xc = x[n];
    for (long s = 0; s < ns; ++s) {
      double xn = sos(s, 0) * xc + z(s, 0);
      z(s, 0) = sos(s, 1) * xc - sos(s, 4) * xn + z(s, 1);
      z(s, 1) = sos(s, 2) * xc - sos(s, 5) * xn;
      xc = xn;
    }
    y[n] = xc;
  }
  if (zi) *zi = z;
  return y;
}

V lfilter_zi(const V& b0, const V& a0) {
  V a = a0, b = b0;
  while (a.size() > 1 && a[0] == 0.0) a.erase(a.begin());
  if (a.empty() || a[0] == 0) fail("lfilter_zi: there must be at least one nonzero a coefficient");
  if (a[0] != 1.0) { double d = a[0]; for (auto& v : b) v /= d; for (auto& v : a) v /= d; }
  size_t n = std::max(a.size(), b.size());
  a.resize(n, 0.0); b.resize(n, 0.0);
  if (n == 1) return {};
  long m = long(n - 1);
  Eigen::MatrixXd IminusA = Eigen::MatrixXd::Identity(m, m);
  for (long j = 0; j < m; ++j) IminusA(j, 0) -= -a[size_t(j + 1)] / a[0];
  for (long i = 1; i < m; ++i) IminusA(i - 1, i) -= 1.0;
  Eigen::VectorXd B(m);
  for (long i = 0; i < m; ++i) B(i) = b[size_t(i + 1)] - a[size_t(i + 1)] * b[0];
  Eigen::VectorXd zi = IminusA.partialPivLu().solve(B);
  return V(zi.data(), zi.data() + zi.size());
}

Eigen::MatrixXd sosfilt_zi(const Eigen::MatrixXd& sos) {
  Eigen::MatrixXd zi(sos.rows(), 2);
  double scale = 1.0;
  for (long s = 0; s < sos.rows(); ++s) {
    V b = {sos(s, 0), sos(s, 1), sos(s, 2)}, a = {sos(s, 3), sos(s, 4), sos(s, 5)};
    V z = lfilter_zi(b, a);
    z.resize(2, 0.0);
    zi(s, 0) = scale * z[0];
    zi(s, 1) = scale * z[1];
    scale *= (b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2]);
  }
  return zi;
}

namespace {
V extend(const V& x, long edge, const std::string& padtype) {
  if (padtype == "none" || edge == 0) return x;
  long n = long(x.size());
  V out;
  out.reserve(size_t(n + 2 * edge));
  for (long i = edge; i >= 1; --i) {
    double v = x[size_t(i)];
    out.push_back(padtype == "odd" ? 2 * x[0] - v : padtype == "even" ? v : x[0]);
  }
  out.insert(out.end(), x.begin(), x.end());
  for (long i = 1; i <= edge; ++i) {
    double v = x[size_t(n - 1 - i)];
    out.push_back(padtype == "odd" ? 2 * x[size_t(n - 1)] - v : padtype == "even" ? v : x[size_t(n - 1)]);
  }
  return out;
}
long validate_pad(const std::string& padtype, long padlen, long ntaps, size_t n, const char* op) {
  if (padtype != "odd" && padtype != "even" && padtype != "constant" && padtype != "none")
    fail(std::string(op) + ": padtype must be odd, even, constant or none; got '" + padtype + "'");
  long edge = padtype == "none" ? 0 : (padlen < 0 ? 3 * ntaps : padlen);
  if (long(n) <= edge) fail(std::string(op) + ": the length of the input vector x must be greater than padlen, which is " + std::to_string(edge));
  return edge;
}
}  // namespace

V sosfiltfilt(const Eigen::MatrixXd& sos, const V& x, const std::string& padtype, long padlen) {
  long ntaps = 2 * sos.rows() + 1;
  long z2 = 0, z5 = 0;
  for (long s = 0; s < sos.rows(); ++s) { z2 += sos(s, 2) == 0; z5 += sos(s, 5) == 0; }
  ntaps -= std::min(z2, z5);
  long edge = validate_pad(padtype, padlen, ntaps, x.size(), "filtfilt");
  V ext = extend(x, edge, padtype);
  Eigen::MatrixXd zi = sosfilt_zi(sos);
  Eigen::MatrixXd z = zi * ext[0];
  V y = sosfilt(sos, ext, &z);
  double y0 = y.back();
  std::reverse(y.begin(), y.end());
  z = zi * y0;
  y = sosfilt(sos, y, &z);
  std::reverse(y.begin(), y.end());
  if (edge > 0) y = V(y.begin() + edge, y.end() - edge);
  return y;
}

V lfilter(const V& b0, const V& a0, const V& x, V* zi) {
  if (a0.empty() || b0.empty()) fail("lfilter: b and a must be non-empty");
  if (a0[0] == 0) fail("lfilter: a[0] must be nonzero");
  size_t n = std::max(a0.size(), b0.size());
  V b = b0, a = a0;
  b.resize(n, 0.0); a.resize(n, 0.0);
  double d = a[0];
  for (size_t i = 0; i < n; ++i) { b[i] /= d; a[i] /= d; }
  V Z = zi ? *zi : V(n - 1, 0.0);
  if (Z.size() != n - 1) fail("lfilter: the initial state has the wrong length");
  V y(x.size());
  for (size_t k = 0; k < x.size(); ++k) {
    double xn = x[k], yn;
    if (n > 1) {
      yn = Z[0] + b[0] * xn;
      for (size_t j = 0; j + 2 < n; ++j) Z[j] = Z[j + 1] + xn * b[j + 1] - yn * a[j + 1];
      Z[n - 2] = xn * b[n - 1] - yn * a[n - 1];
    } else {
      yn = xn * b[0];
    }
    y[k] = yn;
  }
  if (zi) *zi = Z;
  return y;
}

V filtfilt(const V& b, const V& a, const V& x, const std::string& padtype, long padlen) {
  long edge = validate_pad(padtype, padlen, long(std::max(a.size(), b.size())), x.size(), "filtfilt");
  V ext = extend(x, edge, padtype);
  V zi = lfilter_zi(b, a), z(zi.size());
  for (size_t i = 0; i < zi.size(); ++i) z[i] = zi[i] * ext[0];
  V y = lfilter(b, a, ext, &z);
  double y0 = y.back();
  std::reverse(y.begin(), y.end());
  for (size_t i = 0; i < zi.size(); ++i) z[i] = zi[i] * y0;
  y = lfilter(b, a, y, &z);
  std::reverse(y.begin(), y.end());
  if (edge > 0) y = V(y.begin() + edge, y.end() - edge);
  return y;
}

CV freqz(const V& b, const V& a, const V& w) {
  CV h(w.size());
  for (size_t i = 0; i < w.size(); ++i) {
    cd zm1(std::cos(w[i]), -std::sin(w[i]));
    auto pv = [&](const V& c) {
      cd r(c.back(), 0);
      for (size_t k = c.size() - 1; k-- > 0;) r = cd(c[k], 0) + cmul(r, zm1);
      return r;
    };
    h[i] = cdiv(pv(b), pv(a));
  }
  return h;
}
CV sosfreqz(const Eigen::MatrixXd& sos, const V& w) {
  CV h(w.size(), cd(1, 0));
  for (long s = 0; s < sos.rows(); ++s) {
    CV r = freqz({sos(s, 0), sos(s, 1), sos(s, 2)}, {sos(s, 3), sos(s, 4), sos(s, 5)}, w);
    for (size_t i = 0; i < w.size(); ++i) h[i] = cmul(h[i], r[i]);
  }
  return h;
}

}  // namespace sig
}  // namespace lmnum
