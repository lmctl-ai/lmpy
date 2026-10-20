// Interpolation: pycli `interp_eval` / `interp_eval_to` (scipy.interpolate). Every 1-D kind is held as a piecewise polynomial
// in scipy's PPoly form (local power basis, right-continuous at the breakpoints, the end pieces extended for
// extrapolation) and evaluated, differentiated and integrated the way scipy's _ppoly does it. The constructions are ports of
// scipy 1.16.2 (BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. 2003-2024 SciPy Developers):
//   linear     make_interp_spline(k=1)                cubic    CubicSpline (bc_type not-a-knot/natural/clamped/periodic)
//   pchip      PchipInterpolator (Fritsch-Carlson)    akima    Akima1DInterpolator(method="akima"/"makima")
//   smoothing  make_smoothing_spline (lam or GCV)     rbf      RBFInterpolator (n-D; LU solve like dgesv)
//   nearest    interp1d(kind="nearest") (an lmnum extra: ties at a midpoint take the lower node, as numpy searchsorted left)
#include "host.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <limits>

namespace lmnum {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
const std::string It = "Interp", Vt = "Vec", Mt = "Mat", Rt = "Res";

// ------------------------------------------------------------------------------------------------- PPoly
struct PPoly {
  std::vector<double> x;  // breakpoints, ascending
  MatrixXd c;             // (k + 1) x (x.size() - 1); row 0 multiplies the highest power

  // scipy _ppoly.find_interval (ascending, extrapolate): right-continuous, the last interval closed on the right
  long interval(double xv) const {
    long nx = long(x.size());
    double a = x[0], b = x[size_t(nx - 1)];
    if (!(a <= xv && xv <= b)) {
      if (xv < a) return 0;
      if (xv > b) return nx - 2;
      return -1;  // NaN
    }
    if (xv == b) return nx - 2;
    long i = long(std::upper_bound(x.begin(), x.end(), xv) - x.begin()) - 1;
    return std::min(std::max(i, 0L), nx - 2);
  }
  // _ppoly.evaluate_poly1: dx > 0 derivative, dx < 0 antiderivative, terms summed from the lowest power
  double poly1(double s, long ci, int dx) const {
    long K = c.rows();
    double res = 0, z = 1;
    if (dx < 0) for (int kp = 0; kp < -dx; ++kp) z *= s;
    for (long kp = 0; kp < K; ++kp) {
      double prefactor = 1;
      if (dx > 0) {
        if (kp < dx) continue;
        for (long k = kp; k > kp - dx; --k) prefactor *= double(k);
      } else if (dx < 0) {
        for (long k = kp; k < kp - dx; ++k) prefactor /= double(k + 1);
      }
      res = res + c(K - kp - 1, ci) * z * prefactor;
      if (kp < K - 1 && kp >= dx) z = z * s;
    }
    return res;
  }
  double eval(double xv, int nu) const {
    long i = interval(xv);
    if (i < 0) return NAN;
    return poly1(xv - x[size_t(i)], i, nu);
  }
  double integrate(double a, double b) const {  // _ppoly.integrate with extrapolation
    double sign = 1;
    if (b < a) { std::swap(a, b); sign = -1; }
    long s = interval(a), e = interval(b);
    if (s < 0 || e < 0) return NAN;
    double out;
    if (s < e) {
      double vtot = poly1(x[size_t(s + 1)] - x[size_t(s)], s, -1);
      double va = poly1(a - x[size_t(s)], s, -1);
      out = vtot - va;
      for (long i = s + 1; i < e; ++i) out += poly1(x[size_t(i + 1)] - x[size_t(i)], i, -1);
      out += poly1(b - x[size_t(e)], e, -1);
    } else {
      out = poly1(b - x[size_t(s)], s, -1) - poly1(a - x[size_t(s)], s, -1);
    }
    return sign * out;
  }
};

// CubicHermiteSpline coefficients from values and slopes
PPoly hermite(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& dydx) {
  size_t m = x.size() - 1;
  PPoly p; p.x = x; p.c.resize(4, Eigen::Index(m));
  for (size_t i = 0; i < m; ++i) {
    double dx = x[i + 1] - x[i];
    double slope = (y[i + 1] - y[i]) / dx;
    double t = (dydx[i] + dydx[i + 1] - 2 * slope) / dx;
    p.c(0, Eigen::Index(i)) = t / dx;
    p.c(1, Eigen::Index(i)) = (slope - dydx[i]) / dx - t;
    p.c(2, Eigen::Index(i)) = dydx[i];
    p.c(3, Eigen::Index(i)) = y[i];
  }
  return p;
}

double sgn(double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }

// PchipInterpolator._find_derivatives
std::vector<double> pchip_slopes(const std::vector<double>& x, const std::vector<double>& y) {
  size_t n = x.size();
  std::vector<double> hk(n - 1), mk(n - 1), dk(n, 0.0);
  for (size_t i = 0; i + 1 < n; ++i) { hk[i] = x[i + 1] - x[i]; mk[i] = (y[i + 1] - y[i]) / hk[i]; }
  if (n == 2) { dk[0] = dk[1] = mk[0]; return dk; }
  for (size_t i = 0; i + 2 < n; ++i) {
    bool cond = sgn(mk[i + 1]) != sgn(mk[i]) || mk[i + 1] == 0 || mk[i] == 0;
    if (cond) { dk[i + 1] = 0.0; continue; }
    double w1 = 2 * hk[i + 1] + hk[i], w2 = hk[i + 1] + 2 * hk[i];
    double whmean = (w1 / mk[i] + w2 / mk[i + 1]) / (w1 + w2);
    dk[i + 1] = 1.0 / whmean;
  }
  auto edge = [](double h0, double h1, double m0, double m1) {
    double d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    bool mask = sgn(d) != sgn(m0);
    bool mask2 = sgn(m0) != sgn(m1) && std::fabs(d) > 3.0 * std::fabs(m0);
    if (mask) return 0.0;
    if (mask2) return 3.0 * m0;
    return d;
  };
  dk[0] = edge(hk[0], hk[1], mk[0], mk[1]);
  dk[n - 1] = edge(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
  return dk;
}

// Akima1DInterpolator slopes (akima / makima)
std::vector<double> akima_slopes(const std::vector<double>& x, const std::vector<double>& y, bool makima) {
  size_t n = x.size();
  std::vector<double> t(n);
  if (n == 2) { double mk = (y[1] - y[0]) / (x[1] - x[0]); t[0] = t[1] = mk; return t; }
  std::vector<double> m(n + 3);
  for (size_t i = 0; i + 1 < n; ++i) m[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
  m[1] = 2.0 * m[2] - m[3];
  m[0] = 2.0 * m[1] - m[2];
  m[n + 1] = 2.0 * m[n] - m[n - 1];
  m[n + 2] = 2.0 * m[n + 1] - m[n];
  for (size_t i = 0; i < n; ++i) t[i] = 0.5 * (m[i + 3] + m[i]);
  std::vector<double> dm(n + 2), f1(n), f2(n), f12(n);
  for (size_t i = 0; i + 1 < m.size(); ++i) dm[i] = std::fabs(m[i + 1] - m[i]);
  for (size_t i = 0; i < n; ++i) {
    if (makima) {
      double pm_hi = std::fabs(m[i + 3] + m[i + 2]), pm_lo = std::fabs(m[i + 1] + m[i]);
      f1[i] = dm[i + 2] + 0.5 * pm_hi;
      f2[i] = dm[i] + 0.5 * pm_lo;
    } else { f1[i] = dm[i + 2]; f2[i] = dm[i]; }
    f12[i] = f1[i] + f2[i];
  }
  double mx = -INFINITY;
  for (double v : f12) mx = std::max(mx, v);
  for (size_t i = 0; i < n; ++i)
    if (f12[i] > 1e-9 * mx) t[i] = (f1[i] * m[i + 1] + f2[i] * m[i + 2]) / f12[i];
  return t;
}

// Tridiagonal solve with partial pivoting (the algorithm of LAPACK dgtsv). dl: sub-diagonal, d: diagonal, du: super-diagonal.
std::vector<double> tridiag(std::vector<double> dl, std::vector<double> d, std::vector<double> du, std::vector<double> b, const char* op) {
  size_t n = d.size();
  if (n == 1) { if (d[0] == 0) fail(std::string(op) + ": singular spline system"); return {b[0] / d[0]}; }
  std::vector<double> du2(n, 0.0);
  for (size_t i = 0; i + 1 < n; ++i) {
    if (std::fabs(d[i]) >= std::fabs(dl[i])) {
      if (d[i] == 0) fail(std::string(op) + ": singular spline system");
      double fact = dl[i] / d[i];
      d[i + 1] -= fact * du[i];
      b[i + 1] -= fact * b[i];
    } else {
      double fact = d[i] / dl[i];
      d[i] = dl[i];
      double temp = d[i + 1];
      d[i + 1] = du[i] - fact * temp;
      if (i + 2 < n) { du2[i] = du[i + 1]; du[i + 1] = -fact * du2[i]; }
      du[i] = temp;
      temp = b[i];
      b[i] = b[i + 1];
      b[i + 1] = temp - fact * b[i + 1];
    }
  }
  if (d[n - 1] == 0) fail(std::string(op) + ": singular spline system");
  std::vector<double> xs(n);
  xs[n - 1] = b[n - 1] / d[n - 1];
  xs[n - 2] = (b[n - 2] - du[n - 2] * xs[n - 1]) / d[n - 2];
  for (size_t k = n - 2; k-- > 0;) xs[k] = (b[k] - du[k] * xs[k + 1] - du2[k] * xs[k + 2]) / d[k];
  return xs;
}

// CubicSpline slopes. bc: "not-a-knot", "natural" (2nd derivative 0), "clamped" (1st derivative 0), "periodic".
std::vector<double> spline_slopes(const std::vector<double>& x, const std::vector<double>& y, const std::string& bc_type) {
  size_t n = x.size();
  std::vector<double> dx(n - 1), slope(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) { dx[i] = x[i + 1] - x[i]; slope[i] = (y[i + 1] - y[i]) / dx[i]; }
  struct Bc { int kind; double value; };  // 0 not-a-knot, 3 periodic, 1/2 derivative order with value
  auto conv = [](const std::string& s) -> Bc {
    if (s == "not-a-knot") return {0, 0};
    if (s == "periodic") return {3, 0};
    if (s == "clamped") return {1, 0};
    return {2, 0};  // natural
  };
  Bc bs = conv(bc_type), be = conv(bc_type);
  if (n == 2) {
    if (bs.kind == 0 || bs.kind == 3) bs = {1, slope[0]};
    if (be.kind == 0 || be.kind == 3) be = {1, slope[0]};
  }
  std::vector<double> s(n);
  if (n == 3 && bs.kind == 0 && be.kind == 0) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b;
    A(0, 0) = 1; A(0, 1) = 1;
    A(1, 0) = dx[1]; A(1, 1) = 2 * (dx[0] + dx[1]); A(1, 2) = dx[0];
    A(2, 1) = 1; A(2, 2) = 1;
    b(0) = 2 * slope[0];
    b(1) = 3 * (dx[0] * slope[1] + dx[1] * slope[0]);
    b(2) = 2 * slope[1];
    Eigen::Vector3d r = A.partialPivLu().solve(b);
    return {r(0), r(1), r(2)};
  }
  if (n == 3 && bs.kind == 3) {
    double num = 0, den = 0;
    for (size_t i = 0; i < 2; ++i) { num += slope[i] / dx[i]; den += 1.0 / dx[i]; }
    double t = num / den;
    return {t, t, t};
  }
  // banded system: diag A[1], super A[0] (row i, col i+1), sub A[2] (row i+1, col i)
  std::vector<double> diag(n, 0.0), sup(n, 0.0), sub(n, 0.0), b(n, 0.0);  // sup[j] = A[0, j], sub[j] = A[2, j]
  for (size_t i = 1; i + 1 < n; ++i) diag[i] = 2 * (dx[i - 1] + dx[i]);
  for (size_t j = 2; j < n; ++j) sup[j] = dx[j - 2];
  for (size_t j = 0; j + 2 < n; ++j) sub[j] = dx[j + 1];
  for (size_t i = 1; i + 1 < n; ++i) b[i] = 3 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
  if (bs.kind == 3) {
    // periodic: the reduced (n-1) system with a cyclic correction (scipy's two banded solves)
    if (n < 4) fail("interp(cubic): bc_type periodic needs at least 4 nodes here");
    size_t m = n - 1;
    diag[0] = 2 * (dx[n - 2] + dx[0]);
    sup[1] = dx[n - 2];
    double a_m1_0 = dx[n - 3], a_m1_m2 = dx[n - 2], a_m1_m1 = 2 * (dx[n - 2] + dx[n - 3]), a_m2_m1 = dx[n - 4], a_0_m1 = dx[0];
    b[0] = 3 * (dx[0] * slope[n - 2] + dx[n - 2] * slope[0]);
    b[m - 1] = 3 * (dx[m - 1] * slope[m - 2] + dx[m - 2] * slope[m - 1]);
    // Ac = A[:, :m-1] (size m-1)
    size_t k = m - 1;
    std::vector<double> dl(k - 1), dd(k), du(k - 1), b1(k), b2(k, 0.0);
    for (size_t i = 0; i < k; ++i) dd[i] = diag[i];
    for (size_t i = 0; i + 1 < k; ++i) { du[i] = sup[i + 1]; dl[i] = sub[i]; }
    for (size_t i = 0; i < k; ++i) b1[i] = b[i];
    b2[0] = -a_0_m1;
    b2[k - 1] = -a_m2_m1;
    std::vector<double> s1 = tridiag(dl, dd, du, b1, "interp"), s2 = tridiag(dl, dd, du, b2, "interp");
    double s_m1 = (b[m - 1] - a_m1_0 * s1[0] - a_m1_m2 * s1[k - 1]) / (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[k - 1]);
    for (size_t i = 0; i < k; ++i) s[i] = s1[i] + s_m1 * s2[i];
    s[n - 2] = s_m1;
    s[n - 1] = s[0];
    return s;
  }
  if (bs.kind == 0) {
    diag[0] = dx[1];
    sup[1] = x[2] - x[0];
    double d = x[2] - x[0];
    b[0] = ((dx[0] + 2 * d) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / d;
  } else if (bs.kind == 1) {
    diag[0] = 1; sup[1] = 0; b[0] = bs.value;
  } else {
    diag[0] = 2 * dx[0]; sup[1] = dx[0]; b[0] = -0.5 * bs.value * dx[0] * dx[0] + 3 * (y[1] - y[0]);
  }
  if (be.kind == 0) {
    diag[n - 1] = dx[n - 3];
    sub[n - 2] = x[n - 1] - x[n - 3];
    double d = x[n - 1] - x[n - 3];
    b[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3] + (2 * d + dx[n - 2]) * dx[n - 3] * slope[n - 2]) / d;
  } else if (be.kind == 1) {
    diag[n - 1] = 1; sub[n - 2] = 0; b[n - 1] = be.value;
  } else {
    diag[n - 1] = 2 * dx[n - 2]; sub[n - 2] = dx[n - 2]; b[n - 1] = 0.5 * be.value * dx[n - 2] * dx[n - 2] + 3 * (y[n - 1] - y[n - 2]);
  }
  std::vector<double> dl(n - 1), du(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) { du[i] = sup[i + 1]; dl[i] = sub[i]; }
  return tridiag(dl, diag, du, b, "interp");
}

// ------------------------------------------------------------------------------------------------- smoothing spline
// B-spline basis values (nu = 0) or derivatives at xv in interval ell (scipy _bspl _deBoor_D).
void deboor(const std::vector<double>& t, double xv, int k, long ell, int m, double* out) {
  std::vector<double> h(size_t(k + 1), 0.0), hh(size_t(k + 1), 0.0);
  h[0] = 1;
  for (int j = 1; j <= k - m; ++j) {
    for (int i = 0; i < j; ++i) hh[size_t(i)] = h[size_t(i)];
    h[0] = 0;
    for (int n = 1; n <= j; ++n) {
      long ind = ell + n;
      double xb = t[size_t(ind)], xa = t[size_t(ind - j)];
      if (xb == xa) { h[size_t(n)] = 0; continue; }
      double w = hh[size_t(n - 1)] / (xb - xa);
      h[size_t(n - 1)] += w * (xb - xv);
      h[size_t(n)] = w * (xv - xa);
    }
  }
  for (int j = k - m + 1; j <= k; ++j) {
    for (int i = 0; i < j; ++i) hh[size_t(i)] = h[size_t(i)];
    h[0] = 0;
    for (int n = 1; n <= j; ++n) {
      long ind = ell + n;
      double xb = t[size_t(ind)], xa = t[size_t(ind - j)];
      if (xb == xa) { h[size_t(n)] = 0; continue; }
      double w = j * hh[size_t(n - 1)] / (xb - xa);
      h[size_t(n - 1)] -= w;
      h[size_t(n)] = w;
    }
  }
  for (int i = 0; i <= k; ++i) out[i] = h[size_t(i)];
}

std::vector<double> coeff_of_divided_diff(const double* x, size_t n) {
  std::vector<double> r(n);
  for (size_t i = 0; i < n; ++i) {
    double pp = 1.0;
    for (size_t k = 0; k < n; ++k) if (k != i) pp *= x[i] - x[k];
    r[i] = 1.0 / pp;
  }
  return r;
}

using Band = std::vector<std::vector<double>>;  // rows of band storage, each of length n

// general banded solve (scipy solve_banded((l, u), ab, b)): ab[u + i - j][j] = A(i, j); sparse LU underneath
std::vector<double> solve_band(int l, int u, const Band& ab, const std::vector<double>& b) {
  Eigen::Index n = Eigen::Index(b.size());
  std::vector<Eigen::Triplet<double>> tr;
  for (Eigen::Index j = 0; j < n; ++j)
    for (int r = 0; r < l + u + 1; ++r) {
      Eigen::Index i = r - u + j;
      if (i < 0 || i >= n || ab[size_t(r)][size_t(j)] == 0) continue;
      tr.emplace_back(i, j, ab[size_t(r)][size_t(j)]);
    }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(tr.begin(), tr.end());
  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  lu.compute(A);
  if (lu.info() != Eigen::Success) fail("interp(smoothing): the penalised system is singular");
  VectorXd x = lu.solve(to_eigen(b));
  return from_eigen(x);
}

struct SmoothFit { PPoly pp; double lam; };

SmoothFit smoothing_spline(const std::vector<double>& x, const std::vector<double>& y, double lam_in, bool have_lam) {
  size_t n = x.size();
  if (n <= 4) fail("interp(smoothing): needs at least 5 nodes");
  std::vector<double> t;
  for (int i = 0; i < 3; ++i) t.push_back(x[0]);
  t.insert(t.end(), x.begin(), x.end());
  for (int i = 0; i < 3; ++i) t.push_back(x[n - 1]);
  long nb = long(n) + 2;  // number of basis functions
  // design matrix rows: 4 nonzeros starting at column ell - 3
  std::vector<long> ell(n);
  std::vector<std::array<double, 4>> vals(n);
  for (size_t i = 0; i < n; ++i) {
    long l = (i + 1 < n) ? long(i) + 3 : nb - 1;
    ell[i] = l;
    deboor(t, x[i], 3, l, 0, vals[i].data());
  }
  auto Xb = [&](long i, long col) {
    long first = ell[size_t(i)] - 3;
    if (col < first || col > first + 3) return 0.0;
    return vals[size_t(i)][size_t(col - first)];
  };
  long N = long(n);
  Band X(5, std::vector<double>(n, 0.0));
  for (int i = 1; i < 4; ++i)
    for (long k = 0; k < N - 4; ++k) X[size_t(i)][size_t(2 + k)] = Xb(i + k, 3 + k);
  X[1][1] = Xb(0, 0);
  X[2][0] = (x[2] + x[1] - 2 * x[0]) * Xb(0, 0);
  X[2][1] = Xb(1, 1) + Xb(1, 2);
  X[3][0] = (x[2] - x[0]) * Xb(1, 1);
  X[3][1] = Xb(2, 2);
  X[1][n - 2] = Xb(N - 3, N - 1);
  X[1][n - 1] = (x[n - 1] - x[n - 3]) * Xb(N - 2, N);
  X[2][n - 2] = Xb(N - 2, N - 1) + Xb(N - 2, N);
  X[2][n - 1] = (2 * x[n - 1] - x[n - 2] - x[n - 3]) * Xb(N - 1, N + 1);
  X[3][n - 2] = Xb(N - 1, N + 1);
  Band wE(5, std::vector<double>(n, 0.0));
  { auto c = coeff_of_divided_diff(&x[0], 3); for (int r = 0; r < 3; ++r) wE[size_t(2 + r)][0] = c[size_t(r)]; }
  { auto c = coeff_of_divided_diff(&x[0], 4); for (int r = 0; r < 4; ++r) wE[size_t(1 + r)][1] = c[size_t(r)]; }
  for (size_t j = 2; j + 2 < n; ++j) {
    auto c = coeff_of_divided_diff(&x[j - 2], 5);
    for (int r = 0; r < 5; ++r) wE[size_t(r)][j] = (x[j + 2] - x[j - 2]) * c[size_t(r)];
  }
  { auto c = coeff_of_divided_diff(&x[n - 4], 4); for (int r = 0; r < 4; ++r) wE[size_t(r)][n - 2] = -c[size_t(r)]; }
  { auto c = coeff_of_divided_diff(&x[n - 3], 3); for (int r = 0; r < 3; ++r) wE[size_t(r)][n - 1] = c[size_t(r)]; }
  for (auto& row : wE) for (auto& v : row) v *= 6;

  auto combine = [&](double lam) {
    Band A = X;
    for (size_t r = 0; r < 5; ++r) for (size_t j = 0; j < n; ++j) A[r][j] += lam * wE[r][j];
    return A;
  };
  double lam = lam_in;
  if (!have_lam) {
    // _compute_optimal_gcv_parameter with unit weights
    auto xtwy = [&](const Band& Y) {
      Band res(4, std::vector<double>(n, 0.0));
      for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < std::min<size_t>(n - i, 4); ++j) {
          double s = 0;
          for (size_t r = j; r < 5; ++r) s += X[r][i] * Y[r - j][i + j];
          res[3 - j][i + j] = s;
        }
      return res;
    };
    Band XtWX = xtwy(X), XtE = xtwy(wE);
    auto b_inv = [&](const Band& A) {
      // cholesky_banded (upper, 3 super-diagonals): U[3 + r - c][c] = U(r, c), A = U^T U
      Band U(4, std::vector<double>(n, 0.0));
      auto a = [&](size_t r, size_t c) { return A[3 + r - c][c]; };
      auto u = [&](size_t r, size_t c) -> double& { return U[3 + r - c][c]; };
      for (size_t j = 0; j < n; ++j) {
        double s = a(j, j);
        for (size_t k = (j >= 3 ? j - 3 : 0); k < j; ++k) s -= u(k, j) * u(k, j);
        if (!(s > 0)) fail("interp(smoothing): the GCV problem seems ill-posed (matrix not positive definite)");
        u(j, j) = std::sqrt(s);
        for (size_t i = j + 1; i <= std::min(j + 3, n - 1); ++i) {
          double v = a(j, i);
          for (size_t k = (i >= 3 ? i - 3 : 0); k < j; ++k) v -= u(k, j) * u(k, i);
          u(j, i) = v / u(j, j);
        }
      }
      for (size_t i = 2; i < 5; ++i)
        for (size_t c = i - 1; c < n; ++c) U[4 - i][c] /= U[3][c - (i - 1)];
      std::vector<double> D(n);
      for (size_t c = 0; c < n; ++c) { D[c] = 1.0 / (U[3][c] * U[3][c]); U[3][c] = 1.0; }
      Band B(4, std::vector<double>(n, 0.0));
      for (size_t i = n; i-- > 0;) {
        size_t rng = std::min<size_t>(3, n - i - 1);
        for (size_t jj = rng + 1; jj-- > 0;) {
          double s = 0;
          if (jj == 0) {
            for (size_t k = 1; k <= rng; ++k) s -= U[3 - k][i + k] * B[3 - k][i + k];
            s += D[i];
            B[3][i] = s;
          } else {
            for (size_t k = 1; k <= rng; ++k) {
              size_t diag = k > jj ? k - jj : jj - k;
              size_t ind = i + std::min(k, jj);
              s -= U[3 - k][i + k] * B[3 - diag][ind + diag];
            }
            B[3 - jj][i + jj] = s;
          }
        }
      }
      std::fill(B[0].begin(), B[0].end(), 0.0);
      return B;
    };
    auto gcv = [&](double lm) {
      std::vector<double> c = solve_band(2, 2, combine(lm), y);
      std::vector<double> res(n, 0.0);
      for (size_t i = 0; i < n; ++i)
        for (size_t j = (i + 3 > n ? i + 3 - n : 0); j < std::min<size_t>(5, i + 3); ++j) res[i] += wE[j][i + 2 - j] * c[i + 2 - j];
      double nn = 0;
      for (double r : res) nn += (lm * r) * (lm * r);
      double numer = nn / double(n);
      Band lhs = XtWX;
      for (size_t r = 0; r < 4; ++r) for (size_t j = 0; j < n; ++j) lhs[r][j] += lm * XtE[r][j];
      Band bb = b_inv(lhs);
      double tr = 0;
      for (size_t r = 0; r < 4; ++r) for (size_t j = 0; j < n; ++j) tr += bb[r][j] * XtWX[r][j] * (r < 3 ? 2.0 : 1.0);
      double denom = (1 - tr / double(n)) * (1 - tr / double(n));
      return numer / denom;
    };
    // scipy.optimize.minimize_scalar(method="bounded") on [0, n], xatol 1e-5, maxiter 500
    const double sqrt_eps = std::sqrt(2.2e-16), golden_mean = 0.5 * (3.0 - std::sqrt(5.0)), xatol = 1e-5;
    double a = 0, b = double(n);
    double fulc = a + golden_mean * (b - a), nfc = fulc, xf = fulc, rat = 0, e = 0, xx = xf;
    double fx = gcv(xx), ffulc = fx, fnfc = fx, fu = INFINITY;
    int num = 1;
    double xm = 0.5 * (a + b), tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0, tol2 = 2.0 * tol1;
    while (std::fabs(xf - xm) > tol2 - 0.5 * (b - a)) {
      bool golden = true;
      if (std::fabs(e) > tol1) {
        golden = false;
        double r = (xf - nfc) * (fx - ffulc), q = (xf - fulc) * (fx - fnfc);
        double p = (xf - fulc) * q - (xf - nfc) * r;
        q = 2.0 * (q - r);
        if (q > 0.0) p = -p;
        q = std::fabs(q);
        r = e;
        e = rat;
        if (std::fabs(p) < std::fabs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)) {
          rat = (p + 0.0) / q;
          xx = xf + rat;
          if (xx - a < tol2 || b - xx < tol2) { double si = sgn(xm - xf) + ((xm - xf) == 0 ? 1 : 0); rat = tol1 * si; }
        } else golden = true;
      }
      if (golden) { e = xf >= xm ? a - xf : b - xf; rat = golden_mean * e; }
      double si = sgn(rat) + (rat == 0 ? 1 : 0);
      xx = xf + si * std::max(std::fabs(rat), tol1);
      fu = gcv(xx);
      ++num;
      if (fu <= fx) {
        if (xx >= xf) a = xf; else b = xf;
        fulc = nfc; ffulc = fnfc; nfc = xf; fnfc = fx; xf = xx; fx = fu;
      } else {
        if (xx < xf) a = xx; else b = xx;
        if (fu <= fnfc || nfc == xf) { fulc = nfc; ffulc = fnfc; nfc = xx; fnfc = fu; }
        else if (fu <= ffulc || fulc == xf || fulc == nfc) { fulc = xx; ffulc = fu; }
      }
      xm = 0.5 * (a + b);
      tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
      tol2 = 2.0 * tol1;
      if (num >= 500) break;
    }
    if (std::isnan(xf) || std::isnan(fx)) fail("interp(smoothing): unable to find the minimum of the GCV function");
    lam = xf;
  }
  std::vector<double> c = solve_band(2, 2, combine(lam), y);
  std::vector<double> cc;
  double c0 = c[0], c1 = c[1], cm0 = c[n - 1], cm1 = c[n - 2];
  size_t T = t.size();
  cc.push_back(c0 * (t[5] + t[4] - 2 * t[3]) + c1);
  cc.push_back(c0 * (t[5] - t[3]) + c1);
  for (size_t i = 1; i + 1 < n; ++i) cc.push_back(c[i]);
  cc.push_back(cm0 * (t[T - 4] - t[T - 6]) + cm1);
  cc.push_back(cm0 * (2 * t[T - 4] - t[T - 5] - t[T - 6]) + cm1);
  // BSpline -> PPoly on the nodes (PPoly.from_spline: derivatives at the left end of each interval)
  SmoothFit f;
  f.lam = lam;
  f.pp.x = x;
  f.pp.c.resize(4, Eigen::Index(n - 1));
  double fact[4] = {1, 1, 2, 6};
  for (size_t i = 0; i + 1 < n; ++i) {
    long l = long(i) + 3;
    for (int m = 0; m <= 3; ++m) {
      double h[4];
      deboor(t, x[i], 3, l, m, h);
      double v = 0;
      for (int a = 0; a < 4; ++a) v += h[a] * cc[size_t(l - 3 + a)];
      f.pp.c(3 - m, Eigen::Index(i)) = v / fact[m];
    }
  }
  return f;
}

// ------------------------------------------------------------------------------------------------- RBF
double rbf_kernel(const std::string& k, double r) {
  if (k == "linear") return -r;
  if (k == "thin_plate_spline") return r == 0 ? 0.0 : r * r * std::log(r);
  if (k == "cubic") return r * r * r;
  if (k == "quintic") return -(r * r * r * r * r);
  if (k == "multiquadric") return -std::sqrt(r * r + 1);
  if (k == "inverse_multiquadric") return 1 / std::sqrt(r * r + 1);
  if (k == "inverse_quadratic") return 1 / (r * r + 1);
  return std::exp(-(r * r));  // gaussian
}

struct Rbf {
  MatrixXd y;  // nodes, one row each
  std::string kernel;
  double epsilon = 1;
  std::vector<std::vector<int>> powers;
  VectorXd shift, scale, coeffs;
  void poly_row(const Eigen::RowVectorXd& xhat, double* out) const {
    for (size_t j = 0; j < powers.size(); ++j) {
      double p = 1;
      for (Eigen::Index d = 0; d < xhat.size(); ++d) p *= std::pow(xhat(d), double(powers[j][size_t(d)]));
      out[j] = p;
    }
  }
  double eval(const Eigen::RowVectorXd& x) const {
    Eigen::Index p = y.rows();
    double s = 0;
    Eigen::RowVectorXd xeps = x * epsilon;
    for (Eigen::Index i = 0; i < p; ++i) s += rbf_kernel(kernel, (xeps - y.row(i) * epsilon).norm()) * coeffs(i);
    Eigen::RowVectorXd xhat = (x - shift.transpose()).cwiseQuotient(scale.transpose());
    std::vector<double> pv(powers.size());
    poly_row(xhat, pv.data());
    for (size_t j = 0; j < powers.size(); ++j) s += pv[j] * coeffs(p + Eigen::Index(j));
    return s;
  }
};

std::vector<std::vector<int>> monomial_powers(int ndim, int degree) {
  std::vector<std::vector<int>> out;
  for (int deg = 0; deg <= degree; ++deg) {
    // combinations_with_replacement(range(ndim), deg) in lexicographic order
    std::vector<int> idx(size_t(deg), 0);
    while (true) {
      std::vector<int> p(size_t(ndim), 0);
      for (int v : idx) p[size_t(v)] += 1;
      out.push_back(p);
      int i = deg - 1;
      while (i >= 0 && idx[size_t(i)] == ndim - 1) --i;
      if (i < 0) break;
      int v = idx[size_t(i)] + 1;
      for (int k = i; k < deg; ++k) idx[size_t(k)] = v;
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------------- Interp object
struct Interp : Obj {
  static constexpr const char* TYPE = "Interp";
  std::string kind, bc_type, extrapolate = "error", status = "OK", message;
  PPoly pp;
  std::vector<double> nx, ny;  // nodes (nearest)
  Rbf rbf;
  bool is_rbf = false;
  double lam = NAN, interpolation_error = NAN;
  int dim = 1;

  void need_ok(const char* op) const {
    if (status != "OK") fail("Interp." + std::string(op) + ": the " + kind + " interpolant has status " + status + ": " + message);
  }
  double lo() const { return nx.front(); }
  double hi() const { return nx.back(); }
  // returns false when the value is NaN by the extrapolate policy
  bool check_range(double v, const char* op) const {
    if (std::isnan(v)) fail("Interp." + std::string(op) + ": x is NaN");
    if (v >= lo() && v <= hi()) return true;
    if (extrapolate == "extend") return true;
    if (extrapolate == "nan") return false;
    fail("Interp." + std::string(op) + ": x = " + jnum(v) + " is outside the node range [" + jnum(lo()) + ", " + jnum(hi()) +
         "] (extrapolate is \"error\"; set extrapolate to \"nan\" or \"extend\")");
  }
  double value(double v, int nu, const char* op) const {
    need_ok(op);
    if (is_rbf) {
      if (dim != 1) fail("Interp." + std::string(op) + ": this rbf interpolant has " + std::to_string(dim) + " dimensions; pass a Mat with one row per point");
      Eigen::RowVectorXd r(1); r(0) = v;
      return rbf_point(r, nu, op);
    }
    if (nu < 0) fail("Interp." + std::string(op) + ": derivative order must be >= 0");
    if (!check_range(v, op)) return NAN;
    if (kind == "nearest") {
      if (nu > 0) return 0.0;
      size_t n = nx.size();
      // interp1d nearest: searchsorted(midpoints, v, side='left'), clipped
      size_t i = 0;
      while (i + 1 < n && (nx[i] + nx[i + 1]) / 2 < v) ++i;
      return ny[i];
    }
    return pp.eval(v, nu);
  }
  double rbf_point(const Eigen::RowVectorXd& r, int nu, const char* op) const {
    if (nu != 0) fail("Interp." + std::string(op) + ": the rbf interpolant has no derivatives (order 0 only)");
    bool inside = true;
    for (Eigen::Index d = 0; d < r.size(); ++d) {
      if (std::isnan(r(d))) fail("Interp." + std::string(op) + ": point contains NaN");
      if (r(d) < rbf.y.col(d).minCoeff() || r(d) > rbf.y.col(d).maxCoeff()) inside = false;
    }
    if (!inside) {
      if (extrapolate == "nan") return NAN;
      if (extrapolate == "error") fail("Interp." + std::string(op) + ": a point lies outside the bounding box of the nodes (extrapolate is \"error\")");
    }
    return rbf.eval(r);
  }
  double integral(double a, double b) const {
    need_ok("integrate");
    if (is_rbf) fail("Interp.integrate: not available for rbf");
    if (!std::isfinite(a) || !std::isfinite(b)) fail("Interp.integrate: bounds must be finite");
    if (extrapolate != "extend" && (std::min(a, b) < lo() || std::max(a, b) > hi()))
      fail("Interp.integrate: bounds [" + jnum(a) + ", " + jnum(b) + "] leave the node range [" + jnum(lo()) + ", " + jnum(hi()) + "] (allowed with extrapolate \"extend\")");
    if (kind == "nearest") {
      double sign = 1;
      if (b < a) { std::swap(a, b); sign = -1; }
      size_t n = nx.size();
      std::vector<double> edges = {-INFINITY};
      for (size_t i = 0; i + 1 < n; ++i) edges.push_back((nx[i] + nx[i + 1]) / 2);
      edges.push_back(INFINITY);
      double s = 0;
      for (size_t i = 0; i < n; ++i) {
        double l = std::max(a, edges[i]), h = std::min(b, edges[i + 1]);
        if (h > l) s += ny[i] * (h - l);
      }
      return sign * s;
    }
    return pp.integrate(a, b);
  }
};

void check_nodes(const std::vector<double>& x, const std::vector<double>& y, size_t min_n, const std::string& kind) {
  if (x.size() != y.size()) fail("interp(" + kind + "): x has " + std::to_string(x.size()) + " values and y " + std::to_string(y.size()));
  if (x.size() < min_n) fail("interp(" + kind + "): needs at least " + std::to_string(min_n) + " nodes; got " + std::to_string(x.size()));
  for (size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) fail("interp(" + kind + "): x and y must be finite (node " + std::to_string(i) + ")");
    if (i && !(x[i] > x[i - 1])) fail("interp(" + kind + "): x must be strictly increasing (" + (x[i] == x[i - 1] ? "duplicate x " : "x decreases at ") + jnum(x[i]) + ")");
  }
}

std::string read_extrapolate(const Opts& o) {
  std::string e = o.str("extrapolate", "error");
  if (e != "error" && e != "nan" && e != "extend") fail("interp: extrapolate must be error, nan or extend; got '" + e + "'");
  return e;
}

Interp* build_1d(const std::string& kind, const std::vector<double>& x, const std::vector<double>& y, const Opts& o) {
  auto* I = make<Interp>();
  I->kind = kind;
  I->extrapolate = read_extrapolate(o);
  static const std::vector<std::string> kinds = {"linear", "nearest", "cubic", "pchip", "akima", "makima", "smoothing", "rbf"};
  if (std::find(kinds.begin(), kinds.end(), kind) == kinds.end())
    fail("interp: kind must be linear, nearest, cubic, pchip, akima, makima, smoothing or rbf; got '" + kind + "'");
  if (kind == "rbf") return nullptr;
  std::string bc = o.str("bc_type", "not-a-knot");
  bool has_lam = o.has("lam");
  double lam = o.num("lam", 0);
  if (kind != "cubic" && o.has("bc_type")) fail("interp: bc_type applies to kind cubic only");
  if (kind != "smoothing" && has_lam) fail("interp: lam applies to kind smoothing only");
  o.finish("interp");
  check_nodes(x, y, kind == "smoothing" ? 5 : 2, kind);
  I->nx = x; I->ny = y;
  if (kind == "linear") {
    PPoly p; p.x = x; p.c.resize(2, Eigen::Index(x.size() - 1));
    for (size_t i = 0; i + 1 < x.size(); ++i) { p.c(0, Eigen::Index(i)) = (y[i + 1] - y[i]) / (x[i + 1] - x[i]); p.c(1, Eigen::Index(i)) = y[i]; }
    I->pp = p;
  } else if (kind == "cubic") {
    if (bc != "not-a-knot" && bc != "natural" && bc != "clamped" && bc != "periodic") fail("interp(cubic): bc_type must be not-a-knot, natural, clamped or periodic; got '" + bc + "'");
    if (bc == "periodic" && std::fabs(y.front() - y.back()) > 1e-15 + 1e-15 * std::fabs(y.back()))
      fail("interp(cubic): bc_type periodic needs y[0] == y[-1]");
    I->bc_type = bc;
    I->pp = hermite(x, y, spline_slopes(x, y, bc));
  } else if (kind == "pchip") {
    I->pp = hermite(x, y, pchip_slopes(x, y));
  } else if (kind == "akima" || kind == "makima") {
    I->pp = hermite(x, y, akima_slopes(x, y, kind == "makima"));
  } else if (kind == "smoothing") {
    if (has_lam && !(lam >= 0 && std::isfinite(lam))) fail("interp(smoothing): lam must be finite and >= 0");
    double h = (x.back() - x.front()) / double(x.size() - 1);
    if (has_lam && lam / (h * h * h) > 1e11) fail("interp(smoothing): lam / h^3 = " + jnum(lam / (h * h * h)) + " exceeds 1e11 (h = mean node spacing): scipy's solver returns a wrong curve there; use a smaller lam");
    SmoothFit f = smoothing_spline(x, y, lam, has_lam);
    I->pp = f.pp;
    I->lam = f.lam;
  }
  return I;
}

Interp* build_rbf(const MatrixXd& pts, const std::vector<double>& y, const Opts& o) {
  auto* I = make<Interp>();
  I->kind = "rbf";
  I->is_rbf = true;
  I->extrapolate = read_extrapolate(o);
  std::string kernel = o.str("kernel", "thin_plate_spline");
  static const std::vector<std::string> kernels = {"linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"};
  if (std::find(kernels.begin(), kernels.end(), kernel) == kernels.end())
    fail("interp(rbf): kernel must be one of linear, thin_plate_spline, cubic, quintic, multiquadric, inverse_multiquadric, inverse_quadratic, gaussian; got '" + kernel + "'");
  double epsilon = o.num("epsilon", 1.0);
  double smoothing = o.num("smoothing", 0.0);
  int min_degree = kernel == "multiquadric" || kernel == "linear" ? 0 : kernel == "thin_plate_spline" || kernel == "cubic" ? 1 : kernel == "quintic" ? 2 : -1;
  int64_t degree = o.integer("degree", std::max(min_degree, 0));
  o.finish("interp");
  if (!(epsilon > 0) || !std::isfinite(epsilon)) fail("interp(rbf): epsilon must be > 0");
  if (!(smoothing >= 0) || !std::isfinite(smoothing)) fail("interp(rbf): smoothing must be >= 0");
  if (degree < -1 || (degree > -1 && degree < min_degree)) fail("interp(rbf): degree must be -1 or at least " + std::to_string(min_degree) + " for kernel " + kernel);
  if (degree > 3) fail("interp(rbf): degree must be at most 3");
  Eigen::Index p = pts.rows(), nd = pts.cols();
  if (p != Eigen::Index(y.size())) fail("interp(rbf): x has " + std::to_string(p) + " rows and y " + std::to_string(y.size()) + " values");
  if (p == 0 || nd == 0) fail("interp(rbf): no nodes");
  if (!pts.allFinite()) fail("interp(rbf): x must be finite");
  for (double v : y) if (!std::isfinite(v)) fail("interp(rbf): y must be finite");
  Rbf& R = I->rbf;
  R.y = pts; R.kernel = kernel; R.epsilon = epsilon;
  R.powers = monomial_powers(int(nd), int(degree));
  Eigen::Index r = Eigen::Index(R.powers.size());
  if (r > p) fail("interp(rbf): at least " + std::to_string(r) + " nodes are required for degree " + std::to_string(degree) + " in " + std::to_string(nd) + " dimensions");
  I->dim = int(nd);
  VectorXd mins = pts.colwise().minCoeff().transpose(), maxs = pts.colwise().maxCoeff().transpose();
  R.shift = (maxs + mins) / 2;
  R.scale = (maxs - mins) / 2;
  for (Eigen::Index d = 0; d < nd; ++d) if (R.scale(d) == 0.0) R.scale(d) = 1.0;
  MatrixXd lhs = MatrixXd::Zero(p + r, p + r);
  MatrixXd yeps = pts * epsilon;
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j <= i; ++j) { lhs(i, j) = rbf_kernel(kernel, (yeps.row(i) - yeps.row(j)).norm()); lhs(j, i) = lhs(i, j); }
  std::vector<double> pv(static_cast<size_t>(r));
  for (Eigen::Index i = 0; i < p; ++i) {
    Eigen::RowVectorXd yhat = (pts.row(i) - R.shift.transpose()).cwiseQuotient(R.scale.transpose());
    R.poly_row(yhat, pv.data());
    for (Eigen::Index j = 0; j < r; ++j) { lhs(i, p + j) = pv[size_t(j)]; lhs(p + j, i) = pv[size_t(j)]; }
  }
  for (Eigen::Index i = 0; i < p; ++i) lhs(i, i) += smoothing;
  VectorXd rhs = VectorXd::Zero(p + r);
  for (Eigen::Index i = 0; i < p; ++i) rhs(i) = y[size_t(i)];
  Eigen::PartialPivLU<MatrixXd> lu(lhs);
  bool singular = false;
  for (Eigen::Index i = 0; i < lhs.rows(); ++i) if (lu.matrixLU()(i, i) == 0) singular = true;
  if (!singular) { R.coeffs = lu.solve(rhs); singular = !R.coeffs.allFinite(); }
  if (singular) {
    I->status = "SINGULAR";
    I->message = "the RBF system is singular (duplicate or degenerate points for this kernel and degree)";
    return I;
  }
  if (smoothing == 0) {
    double fit = 0, sc = 1.0;
    for (Eigen::Index i = 0; i < p; ++i) { fit = std::max(fit, std::fabs(R.eval(pts.row(i)) - y[size_t(i)])); sc = std::max(sc, std::fabs(y[size_t(i)])); }
    I->interpolation_error = fit;
    if (!(fit <= 1e-6 * sc)) {
      I->status = "ILL_CONDITIONED";
      I->message = "the RBF system is too ill-conditioned: the fit misses its own nodes by " + jnum(fit);
    }
  }
  return I;
}

Interp& self(void* s) { return *static_cast<Interp*>(s); }

}  // namespace

void register_interp(Api& api) {
  api.section("Interpolation",
              "1-D interpolants and n-D radial basis functions (pycli `interp_eval`, `interp_eval_to`; scipy.interpolate). "
              "`interp(kind, x, y[, opts])` builds an `Interp` once; evaluate, differentiate and integrate it as often as needed. "
              "Kinds: linear (make_interp_spline k=1), nearest (interp1d nearest), cubic (CubicSpline, option `bc_type` "
              "not-a-knot (default) / natural / clamped / periodic), pchip (monotone Fritsch-Carlson, PchipInterpolator), akima, "
              "makima (Akima1DInterpolator), smoothing (make_smoothing_spline; `lam` >= 0, omitted = generalised cross-validation; "
              "lam / h^3 is capped at 1e11 as in pycli), rbf (RBFInterpolator; `kernel` thin_plate_spline (default), linear, cubic, "
              "quintic, multiquadric, inverse_multiquadric, inverse_quadratic, gaussian; `epsilon` (1), `smoothing` (0), `degree`). "
              "x must be finite and strictly increasing (rbf: one row per node). Option `extrapolate`: error (default: a point "
              "outside the nodes is a run-time error), nan, or extend (the end polynomial; rbf: the nodes' bounding box is the "
              "range). At a node a piecewise interpolant takes its right-hand piece. An rbf whose system is singular or that misses "
              "its own nodes by more than 1e-6 of the data scale has status SINGULAR / ILL_CONDITIONED and cannot be evaluated.");
  api.type(It, "Interpolant built by `interp`: piecewise polynomial (PPoly form) or radial basis function.");
  api.fn("interp", {"string", Vt, Vt}, It, "build a 1-D interpolant of kind linear, nearest, cubic, pchip, akima, makima, smoothing or rbf (default options)",
         [](Args& a) {
           if (a[0].s == "rbf") { MatrixXd m = to_eigen(obj<Vec>(a[1]).v); return O(build_rbf(m, obj<Vec>(a[2]).v, no_opts())); }
           return O(build_1d(a[0].s, obj<Vec>(a[1]).v, obj<Vec>(a[2]).v, no_opts())); });
  api.fn("interp", {"string", Vt, Vt, "Opts"}, It, "build a 1-D interpolant with options (bc_type, lam, kernel, epsilon, smoothing, degree, extrapolate)",
         [](Args& a) {
           if (a[0].s == "rbf") { MatrixXd m = to_eigen(obj<Vec>(a[1]).v); return O(build_rbf(m, obj<Vec>(a[2]).v, obj<Opts>(a[3]))); }
           return O(build_1d(a[0].s, obj<Vec>(a[1]).v, obj<Vec>(a[2]).v, obj<Opts>(a[3]))); });
  api.fn("interp", {"string", Mt, Vt}, It, "build an n-D radial basis function interpolant (kind must be rbf; one row of x per node)", [](Args& a) {
    if (a[0].s != "rbf") fail("interp: a Mat of nodes needs kind rbf; got '" + a[0].s + "'");
    return O(build_rbf(obj<Mat>(a[1]).m, obj<Vec>(a[2]).v, no_opts())); });
  api.fn("interp", {"string", Mt, Vt, "Opts"}, It, "build an n-D radial basis function interpolant with options (kernel, epsilon, smoothing, degree, extrapolate)", [](Args& a) {
    if (a[0].s != "rbf") fail("interp: a Mat of nodes needs kind rbf; got '" + a[0].s + "'");
    return O(build_rbf(obj<Mat>(a[1]).m, obj<Vec>(a[2]).v, obj<Opts>(a[3]))); });

  api.method(It, "eval", {"float"}, "float", "value at x", [](void* s, Args& a) { return F(self(s).value(a[0].f, 0, "eval")); });
  api.method(It, "eval", {Vt}, Vt, "values at each x", [](void* s, Args& a) {
    std::vector<double> r; for (double v : obj<Vec>(a[0]).v) r.push_back(self(s).value(v, 0, "eval")); return vecv(std::move(r)); });
  api.method(It, "eval", {Mt}, Vt, "rbf values at the points (one row per point)", [](void* s, Args& a) {
    Interp& I = self(s);
    I.need_ok("eval");
    if (!I.is_rbf) fail("Interp.eval: a Mat of points needs an rbf interpolant; pass a Vec for kind " + I.kind);
    const MatrixXd& m = obj<Mat>(a[0]).m;
    if (m.cols() != I.dim) fail("Interp.eval: points have " + std::to_string(m.cols()) + " columns; the nodes have " + std::to_string(I.dim));
    std::vector<double> r; for (Eigen::Index i = 0; i < m.rows(); ++i) r.push_back(I.rbf_point(m.row(i), 0, "eval")); return vecv(std::move(r)); });
  api.method(It, "deriv", {"float", "int"}, "float", "derivative of order nu at x (0 = value; above the degree it is 0; right-hand value at a node)", [](void* s, Args& a) {
    return F(self(s).value(a[0].f, int(a[1].i), "deriv")); });
  api.method(It, "deriv", {Vt, "int"}, Vt, "derivative of order nu at each x", [](void* s, Args& a) {
    std::vector<double> r; for (double v : obj<Vec>(a[0]).v) r.push_back(self(s).value(v, int(a[1].i), "deriv")); return vecv(std::move(r)); });
  api.method(It, "integrate", {"float", "float"}, "float", "definite integral from a to b (negative when b < a; bounds in the node range unless extrapolate is extend)",
             [](void* s, Args& a) { return F(self(s).integral(a[0].f, a[1].f)); });
  api.method(It, "knots", {}, Vt, "breakpoints of the piecewise polynomial (the nodes; rbf: empty)", [](void* s, Args&) {
    return vecv(self(s).is_rbf || self(s).kind == "nearest" ? (self(s).is_rbf ? std::vector<double>{} : self(s).nx) : self(s).pp.x); });
  api.method(It, "coeffs", {}, Mt, "PPoly coefficients: one row per power (highest first), one column per interval in the local variable x - knot (nearest/rbf: the rbf weights as one column, nearest: the node values)", [](void* s, Args&) {
    Interp& I = self(s);
    if (I.is_rbf) return matv(MatrixXd(I.rbf.coeffs));
    if (I.kind == "nearest") return matv(MatrixXd(to_eigen(I.ny)));
    return matv(I.pp.c); });
  api.method(It, "status", {}, "string", "OK, or SINGULAR / ILL_CONDITIONED for an rbf that could not be built", [](void* s, Args&) { return S(self(s).status); });
  api.method(It, "info", {}, Rt, "fields: status, message, kind, n_nodes, extrapolate, bc_type (cubic), lam (smoothing: the given or GCV value), interpolation_error (rbf)", [](void* s, Args&) {
    Interp& I = self(s);
    auto* r = make<Res>();
    r->str("status", I.status).str("message", I.message).str("kind", I.kind);
    r->integer("n_nodes", I.is_rbf ? int64_t(I.rbf.y.rows()) : int64_t(I.nx.size())).str("extrapolate", I.extrapolate);
    if (I.kind == "cubic") r->str("bc_type", I.bc_type);
    if (I.kind == "smoothing") r->num("lam", I.lam);
    if (I.is_rbf) r->num("interpolation_error", I.interpolation_error).integer("dim", I.dim);
    return O(r); });
}

}  // namespace lmnum
