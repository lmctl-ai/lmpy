// Windows, FFT, detrending, spectral estimates, convolution and Savitzky-Golay smoothing.
// Ported from scipy.signal 1.16 (_windows.py, _spectral_py.py, _short_time_fft.py, _signaltools.py, _savitzky_golay.py)
// and numpy 2.3 (fft helpers, convolve, pairwise summation); both BSD-3-Clause.
#include "signal_impl.hpp"
#include "host.hpp"
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"
#include <boost/math/special_functions/bessel.hpp>
#include <algorithm>

namespace lmnum {
namespace sig {

// numpy loops_utils.h.src pairwise_sum (8 accumulators, blocks of 128).
double pairwise_sum(const double* a, size_t n) {
  if (n < 8) {
    double res = -0.0;
    for (size_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

cd cmul(cd a, cd b) { return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()); }

// ---------------------------------------------------------------------------------------------------- windows
namespace {
V np_linspace(double start, double stop, long num) {
  V y(size_t(std::max(num, 0L)));
  if (num <= 0) return y;
  if (num == 1) { y[0] = start; return y; }
  double step = (stop - start) / double(num - 1);
  for (long i = 0; i < num; ++i) y[size_t(i)] = double(i) * step + start;
  y.back() = stop;
  return y;
}
V general_cosine(long M, const std::vector<double>& a) {
  V fac = np_linspace(-M_PI, M_PI, M), w(size_t(M), 0.0);
  for (size_t k = 0; k < a.size(); ++k)
    for (long i = 0; i < M; ++i) w[size_t(i)] += a[k] * std::cos(double(k) * fac[size_t(i)]);
  return w;
}
}  // namespace

const std::vector<std::string>& window_names() {
  static const std::vector<std::string> n = {"hann", "hamming", "blackman", "boxcar", "bartlett", "flattop", "parzen", "bohman", "blackmanharris",
                                             "nuttall", "barthann", "cosine", "triang", "tukey", "kaiser", "gaussian"};
  return n;
}

V window(const std::string& name0, long M, bool sym, double param) {
  std::string name = name0;
  // scipy get_window aliases
  if (name == "hanning" || name == "han") name = "hann";
  else if (name == "hamm" || name == "ham") name = "hamming";
  else if (name == "black" || name == "blk") name = "blackman";
  else if (name == "box" || name == "ones" || name == "rect" || name == "rectangular") name = "boxcar";
  else if (name == "bart" || name == "brt") name = "bartlett";
  else if (name == "flat" || name == "flt") name = "flattop";
  else if (name == "parz" || name == "par") name = "parzen";
  else if (name == "bman" || name == "bmn") name = "bohman";
  else if (name == "blackharr") name = "blackmanharris";
  else if (name == "nutl" || name == "nut") name = "nuttall";
  else if (name == "brthan" || name == "bth") name = "barthann";
  else if (name == "halfcosine") name = "cosine";
  else if (name == "traing" || name == "tri") name = "triang";
  else if (name == "tuk") name = "tukey";
  else if (name == "ksr") name = "kaiser";
  else if (name == "gauss" || name == "gss") name = "gaussian";
  if (std::find(window_names().begin(), window_names().end(), name) == window_names().end()) {
    std::string all;
    for (auto& s : window_names()) all += (all.empty() ? "" : ", ") + s;
    fail("window: unknown window '" + name0 + "' (one of " + all + ")");
  }
  if ((name == "kaiser" || name == "gaussian") && std::isnan(param))
    fail(std::string("window: ") + name + " needs its shape parameter (" + (name == "kaiser" ? "beta" : "std") + ")");
  if (M < 0) fail("window: the length must be non-negative");
  if (M <= 1) return V(size_t(M), 1.0);
  if (name == "tukey") {
    double alpha = std::isnan(param) ? 0.5 : param;
    if (alpha <= 0) return V(size_t(M), 1.0);
    if (alpha >= 1.0) name = "hann";
  }
  bool trunc = !sym;
  if (trunc) M += 1;
  V w;
  if (name == "hann") w = general_cosine(M, {0.5, 1. - 0.5});
  else if (name == "hamming") w = general_cosine(M, {0.54, 1. - 0.54});
  else if (name == "blackman") w = general_cosine(M, {0.42, 0.50, 0.08});
  else if (name == "nuttall") w = general_cosine(M, {0.3635819, 0.4891775, 0.1365995, 0.0106411});
  else if (name == "blackmanharris") w = general_cosine(M, {0.35875, 0.48829, 0.14128, 0.01168});
  else if (name == "flattop") w = general_cosine(M, {0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368});
  else if (name == "boxcar") w = V(size_t(M), 1.0);
  else if (name == "triang") {
    long h = (M + 1) / 2;
    V a;
    for (long n = 1; n <= h; ++n) a.push_back(M % 2 == 0 ? (2.0 * double(n) - 1.0) / double(M) : 2.0 * double(n) / (double(M) + 1.0));
    w = a;
    if (M % 2 == 0) w.insert(w.end(), a.rbegin(), a.rend());
    else w.insert(w.end(), a.rbegin() + 1, a.rend());
  } else if (name == "parzen") {
    double start = -(double(M) - 1) / 2.0;
    for (long i = 0; i < M; ++i) {
      double n = start + double(i), an = std::fabs(n);
      w.push_back(an <= (double(M) - 1) / 4.0 ? (1 - 6 * std::pow(an / (double(M) / 2.0), 2.0) + 6 * std::pow(an / (double(M) / 2.0), 3.0))
                                              : 2 * std::pow(1 - an / (double(M) / 2.0), 3.0));
    }
  } else if (name == "bohman") {
    V l = np_linspace(-1, 1, M);
    w.push_back(0.0);
    for (long i = 1; i + 1 < M; ++i) {
      double fac = std::fabs(l[size_t(i)]);
      w.push_back((1 - fac) * std::cos(M_PI * fac) + 1.0 / M_PI * std::sin(M_PI * fac));
    }
    w.push_back(0.0);
  } else if (name == "bartlett") {
    for (long i = 0; i < M; ++i) {
      double n = double(i);
      w.push_back(n <= (double(M) - 1) / 2.0 ? 2.0 * n / double(M - 1) : 2.0 - 2.0 * n / double(M - 1));
    }
  } else if (name == "barthann") {
    for (long i = 0; i < M; ++i) {
      double fac = std::fabs(double(i) / (double(M) - 1.0) - 0.5);
      w.push_back(0.62 - 0.48 * fac + 0.38 * std::cos(2 * M_PI * fac));
    }
  } else if (name == "cosine") {
    for (long i = 0; i < M; ++i) w.push_back(std::sin(M_PI / double(M) * (double(i) + .5)));
  } else if (name == "tukey") {
    double alpha = std::isnan(param) ? 0.5 : param;
    long width = long(std::floor(alpha * double(M - 1) / 2.0));
    for (long i = 0; i < M; ++i) {
      double n = double(i);
      if (i <= width) w.push_back(0.5 * (1 + std::cos(M_PI * (-1 + 2.0 * n / alpha / double(M - 1)))));
      else if (i < M - width - 1) w.push_back(1.0);
      else w.push_back(0.5 * (1 + std::cos(M_PI * (-2.0 / alpha + 1 + 2.0 * n / alpha / double(M - 1)))));
    }
  } else if (name == "kaiser") {
    double alpha = (double(M) - 1) / 2.0, den = boost::math::cyl_bessel_i(0, param);
    for (long i = 0; i < M; ++i) {
      double r = (double(i) - alpha) / alpha;
      w.push_back(boost::math::cyl_bessel_i(0, param * std::sqrt(1 - std::pow(r, 2.0))) / den);
    }
  } else if (name == "gaussian") {
    double sig2 = 2 * param * param;
    for (long i = 0; i < M; ++i) {
      double n = double(i) - (double(M) - 1.0) / 2.0;
      w.push_back(std::exp(-(n * n) / sig2));
    }
  }
  if (trunc) w.pop_back();
  return w;
}

// ---------------------------------------------------------------------------------------------------- FFT
CV rfft(const V& x, size_t n) {
  if (n == 0) fail("rfft: the transform length must be positive");
  V in(n, 0.0);
  std::copy(x.begin(), x.begin() + long(std::min(n, x.size())), in.begin());
  CV out(n / 2 + 1);
  pocketfft::r2c<double>({n}, {sizeof(double)}, {sizeof(cd)}, 0, pocketfft::FORWARD, in.data(), out.data(), 1.0);
  return out;
}
CV fft(const CV& x, size_t n, bool inverse) {
  if (n == 0) fail("fft: the transform length must be positive");
  CV in(n, cd(0, 0));
  std::copy(x.begin(), x.begin() + long(std::min(n, x.size())), in.begin());
  CV out(n);
  pocketfft::c2c<double>({n}, {sizeof(cd)}, {sizeof(cd)}, {0}, inverse ? pocketfft::BACKWARD : pocketfft::FORWARD, in.data(), out.data(),
                         inverse ? 1.0 / double(n) : 1.0);
  return out;
}
V irfft(const CV& X, size_t n) {
  if (n == 0) fail("irfft: the output length must be positive");
  CV in(n / 2 + 1, cd(0, 0));
  std::copy(X.begin(), X.begin() + long(std::min(in.size(), X.size())), in.begin());
  V out(n);
  pocketfft::c2r<double>({n}, {sizeof(cd)}, {sizeof(double)}, 0, pocketfft::BACKWARD, in.data(), out.data(), 1.0 / double(n));
  return out;
}
V rfftfreq(size_t n, double d) {
  double val = 1.0 / (double(n) * d);
  V r(n / 2 + 1);
  for (size_t i = 0; i < r.size(); ++i) r[i] = double(i) * val;
  return r;
}
V fftfreq(size_t n, double d) {
  double val = 1.0 / (double(n) * d);
  V r(n);
  size_t N = (n - 1) / 2 + 1;
  for (size_t i = 0; i < N; ++i) r[i] = double(i) * val;
  long neg = -long(n / 2);
  for (size_t i = N; i < n; ++i) r[i] = double(neg++) * val;
  return r;
}

// ---------------------------------------------------------------------------------------------------- detrend
V detrend(const V& x, const std::string& type) {
  if (type == "none") return x;
  V r = x;
  if (type == "constant" || type == "c") {
    double m = np_mean(x);
    for (auto& v : r) v -= m;
    return r;
  }
  if (type != "linear" && type != "l") fail("detrend: type must be constant, linear or none; got '" + type + "'");
  // scipy fits A = [t, 1] with t = (1..N)/N by least squares; the centred normal equations give the same line.
  size_t n = x.size();
  if (n == 0) return r;
  V t(n);
  for (size_t i = 0; i < n; ++i) t[i] = double(i + 1) / double(n);
  double tb = np_mean(t), yb = np_mean(x), sxy = 0, sxx = 0;
  for (size_t i = 0; i < n; ++i) { sxy += (t[i] - tb) * (x[i] - yb); sxx += (t[i] - tb) * (t[i] - tb); }
  double slope = sxx > 0 ? sxy / sxx : 0.0, icpt = yb - slope * tb;
  for (size_t i = 0; i < n; ++i) r[i] = x[i] - (slope * t[i] + icpt);
  return r;
}

// ---------------------------------------------------------------------------------------------------- spectra
namespace {
double median_of(V v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}
// scipy _median_bias
double median_bias(size_t n) {
  double s = 0;
  for (size_t i = 1; i <= (n - 1) / 2; ++i) { double ii2 = 2.0 * double(i); s += 1.0 / (ii2 + 1) - 1.0 / ii2; }
  return 1 + s;
}
// One windowed, detrended segment's spectrum (rfft for one-sided, fft otherwise).
CV segment_fft(const V& x, size_t start, const SpecParams& p) {
  V seg(x.begin() + long(start), x.begin() + long(start + p.nperseg));
  seg = detrend(seg, p.detrend);
  for (size_t i = 0; i < seg.size(); ++i) seg[i] *= p.win[i];
  if (p.onesided) return rfft(seg, p.nfft);
  CV c(seg.size());
  for (size_t i = 0; i < seg.size(); ++i) c[i] = cd(seg[i], 0.0);
  return fft(c, p.nfft, false);
}
double win_scale(const SpecParams& p) {
  V w2(p.win.size());
  for (size_t i = 0; i < w2.size(); ++i) w2[i] = p.win[i] * p.win[i];
  if (p.scaling == "density") return 1.0 / (p.fs * pairwise_sum(w2.data(), w2.size()));
  double s = pairwise_sum(p.win.data(), p.win.size());
  return 1.0 / (s * s);
}
}  // namespace

CsdOut csd(const V& x0, const V& y0, bool same, const SpecParams& p) {
  V x = x0, y = same ? x0 : y0;
  size_t n = std::max(x.size(), y.size());
  x.resize(n, 0.0);
  y.resize(n, 0.0);
  size_t hop = p.nperseg - p.noverlap, nseg = (n - p.noverlap) / hop;
  double scale = win_scale(p);
  size_t nf = p.onesided ? p.nfft / 2 + 1 : p.nfft;
  std::vector<CV> per(nseg);
  for (size_t s = 0; s < nseg; ++s) {
    CV X = segment_fft(x, s * hop, p), Y = same ? X : segment_fft(y, s * hop, p);
    CV r(nf);
    for (size_t k = 0; k < nf; ++k) {
      r[k] = cmul(std::conj(X[k]), Y[k]) * scale;
      if (p.onesided && k >= 1 && (p.nfft % 2 ? true : k + 1 < nf)) r[k] *= 2.0;
    }
    per[s] = std::move(r);
  }
  CsdOut out;
  out.n_segments = nseg;
  out.pxy.resize(nf);
  for (size_t k = 0; k < nf; ++k) {
    V re(nseg), im(nseg);
    for (size_t s = 0; s < nseg; ++s) { re[s] = per[s][k].real(); im[s] = per[s][k].imag(); }
    if (nseg == 1) out.pxy[k] = per[0][k];
    else if (p.average == "median") {
      double b = median_bias(nseg);
      out.pxy[k] = cd(median_of(re) / b, same ? 0.0 : median_of(im) / b);
    } else {
      out.pxy[k] = cd(pairwise_sum(re.data(), nseg) / double(nseg), pairwise_sum(im.data(), nseg) / double(nseg));
    }
    if (same) out.pxy[k] = cd(out.pxy[k].real(), 0.0);
  }
  out.f = p.onesided ? rfftfreq(p.nfft, 1 / p.fs) : fftfreq(p.nfft, 1 / p.fs);
  return out;
}

SpectrogramOut spectrogram(const V& x, const SpecParams& p) {
  size_t n = x.size(), hop = p.nperseg - p.noverlap, nseg = (n - p.noverlap) / hop;
  double scale = win_scale(p);
  size_t nf = p.onesided ? p.nfft / 2 + 1 : p.nfft;
  SpectrogramOut out;
  out.sxx = Eigen::MatrixXd::Zero(long(nf), long(nseg));
  for (size_t s = 0; s < nseg; ++s) {
    CV X = segment_fft(x, s * hop, p);
    for (size_t k = 0; k < nf; ++k) {
      double v = (cmul(std::conj(X[k]), X[k]) * scale).real();
      if (p.onesided && k >= 1 && (p.nfft % 2 ? true : k + 1 < nf)) v *= 2.0;
      out.sxx(long(k), long(s)) = v;
    }
  }
  out.f = p.onesided ? rfftfreq(p.nfft, 1 / p.fs) : fftfreq(p.nfft, 1 / p.fs);
  // numpy.arange(nperseg/2, n - nperseg/2 + 1, hop) / fs
  double start = double(p.nperseg) / 2, stop = double(n) - double(p.nperseg) / 2 + 1;
  size_t cnt = size_t(std::ceil((stop - start) / double(hop)));
  for (size_t i = 0; i < cnt; ++i) out.t.push_back((start + double(i) * double(hop)) / p.fs);
  return out;
}

// ---------------------------------------------------------------------------------------------------- convolution, Savitzky-Golay
V convolve(const V& a0, const V& v0, const std::string& mode) {
  const V& a = a0.size() >= v0.size() ? a0 : v0;
  const V& v = a0.size() >= v0.size() ? v0 : a0;
  if (v.empty()) fail("convolve: an input is empty");
  size_t n = a.size(), m = v.size();
  V full(n + m - 1, 0.0);
  for (size_t i = 0; i < full.size(); ++i) {
    size_t jlo = i >= n - 1 ? i - (n - 1) : 0, jhi = std::min(i, m - 1);
    double s = 0;
    for (size_t j = jlo; j <= jhi; ++j) s += v[j] * a[i - j];
    full[i] = s;
  }
  if (mode == "full") return full;
  if (mode == "same") { size_t off = (m - 1) / 2; return V(full.begin() + long(off), full.begin() + long(off + n)); }
  if (mode == "valid") return V(full.begin() + long(m - 1), full.begin() + long(n));
  fail("convolve: mode must be full, same or valid; got '" + mode + "'");
}

V savgol_coeffs(int window_length, int polyorder, int deriv, double delta) {
  if (window_length < 1) fail("savgol: window_length must be positive");
  if (polyorder < 0) fail("savgol: polyorder must be non-negative");
  if (polyorder >= window_length) fail("savgol: polyorder must be less than window_length");
  if (deriv < 0) fail("savgol: deriv must be non-negative");
  int halflen = window_length / 2, rem = window_length % 2;
  double pos = rem == 0 ? halflen - 0.5 : halflen;
  if (deriv > polyorder) return V(size_t(window_length), 0.0);
  Eigen::MatrixXd A(polyorder + 1, window_length);
  for (int j = 0; j < window_length; ++j) {
    double x = -pos + double(window_length - 1 - j);  // reversed for convolution
    for (int o = 0; o <= polyorder; ++o) A(o, j) = std::pow(x, double(o));
  }
  Eigen::VectorXd y = Eigen::VectorXd::Zero(polyorder + 1);
  double fact = 1;
  for (int i = 2; i <= deriv; ++i) fact *= i;
  y(deriv) = fact / std::pow(delta, double(deriv));
  // minimum-norm least squares (scipy.linalg.lstsq) of the underdetermined system A c = y
  Eigen::VectorXd c = A.completeOrthogonalDecomposition().solve(y);
  return V(c.data(), c.data() + c.size());
}

namespace {
// scipy.ndimage boundary extension for convolve1d; index i may lie outside [0, n).
double ext_at(const V& x, long i, const std::string& mode, double cval) {
  long n = long(x.size());
  if (i >= 0 && i < n) return x[size_t(i)];
  if (mode == "constant") return cval;
  if (mode == "nearest") return x[i < 0 ? 0 : size_t(n - 1)];
  if (mode == "wrap") { long k = ((i % n) + n) % n; return x[size_t(k)]; }
  if (mode == "mirror") {
    if (n == 1) return x[0];
    long period = 2 * (n - 1), k = ((i % period) + period) % period;
    return x[size_t(k < n ? k : period - k)];
  }
  // reflect (d c b a | a b c d | d c b a)
  long period = 2 * n, k = ((i % period) + period) % period;
  return x[size_t(k < n ? k : period - 1 - k)];
}
// numpy.polyfit of degree `deg` on t = 0..len-1, evaluated (with derivative) at the given points.
void fit_edge(const V& x, long wstart, long wstop, long istart, long istop, int polyorder, int deriv, double delta, V& y) {
  long len = wstop - wstart;
  Eigen::MatrixXd A(len, polyorder + 1);
  Eigen::VectorXd b(len);
  for (long i = 0; i < len; ++i) {
    for (int o = 0; o <= polyorder; ++o) A(i, o) = std::pow(double(i), double(polyorder - o));
    b(i) = x[size_t(wstart + i)];
  }
  Eigen::VectorXd scl = A.colwise().norm().transpose();
  for (int o = 0; o <= polyorder; ++o) if (scl(o) == 0) scl(o) = 1;
  Eigen::MatrixXd As = A * scl.cwiseInverse().asDiagonal();
  Eigen::VectorXd c = As.completeOrthogonalDecomposition().solve(b).cwiseQuotient(scl);
  // coefficients high power first; differentiate `deriv` times
  std::vector<double> pc(c.data(), c.data() + c.size());
  for (int d = 0; d < deriv; ++d) {
    if (pc.size() <= 1) { pc = {0.0}; break; }
    std::vector<double> q(pc.size() - 1);
    size_t deg = pc.size() - 1;
    for (size_t k = 0; k < q.size(); ++k) q[k] = pc[k] * double(deg - k);
    pc = q;
  }
  for (long i = istart; i < istop; ++i) {
    double t = double(i - wstart), v = 0;
    for (double coef : pc) v = v * t + coef;
    y[size_t(i)] = v / std::pow(delta, double(deriv));
  }
}
}  // namespace

V savgol_filter(const V& x, int window_length, int polyorder, int deriv, double delta, const std::string& mode, double cval) {
  if (mode != "interp" && mode != "mirror" && mode != "constant" && mode != "nearest" && mode != "wrap")
    fail("savgol: mode must be interp, mirror, constant, nearest or wrap; got '" + mode + "'");
  V w = savgol_coeffs(window_length, polyorder, deriv, delta);
  long n = long(x.size()), W = window_length;
  if (mode == "interp" && W > n) fail("savgol: with mode interp, window_length must not exceed the signal length");
  // ndimage.convolve1d: reversed weights, origin -1 for even lengths, then correlate1d
  V wr(w.rbegin(), w.rend());
  long origin = W % 2 == 0 ? -1 : 0, size1 = W / 2;
  std::string m = mode == "interp" ? "constant" : mode;
  double cv = mode == "interp" ? 0.0 : cval;
  V y(static_cast<size_t>(n));
  for (long i = 0; i < n; ++i) {
    double s = 0;
    for (long j = 0; j < W; ++j) s += wr[size_t(j)] * ext_at(x, i + j - size1 - origin, m, cv);
    y[size_t(i)] = s;
  }
  if (mode == "interp") {
    long half = W / 2;
    fit_edge(x, 0, W, 0, half, polyorder, deriv, delta, y);
    fit_edge(x, n - W, n, n - half, n, polyorder, deriv, delta, y);
  }
  return y;
}

}  // namespace sig
}  // namespace lmnum
