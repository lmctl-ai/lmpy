// Signal processing: IIR design and filtering (Filter), spectral estimates, peak finding, FFT, windows, detrending,
// convolution and Savitzky-Golay smoothing. pycli ops covered: sig_filter(_to), sig_spectrum(_to), sig_peaks(_to).
// The numerics live in signal_design.cpp, signal_spectral.cpp and signal_peaks.cpp (ports of scipy.signal).
#include "host.hpp"
#include "signal_impl.hpp"
#include <algorithm>

namespace lmnum {
namespace {

using namespace sig;
using Eigen::MatrixXd;
const std::string Vt = "Vec", Rt = "Res", Ot = "Opts", Mt = "Mat", Ft = "Filter";

// A designed (or user-supplied) digital IIR filter in second-order sections.
struct Filter : Obj {
  static constexpr const char* TYPE = "Filter";
  std::string kind = "sos", btype = "";
  int order = 0;
  V cutoff;
  double fs = NAN;  // NaN: frequencies are fractions of Nyquist (scipy without fs)
  bool has_zpk = false;
  Zpk zpk;
  MatrixXd sos;
  double radius = 0;
  bool stable = true, marginal = false;
  double fs_eff() const { return std::isnan(fs) ? 2.0 : fs; }
};
Filter& flt(void* s) { return *static_cast<Filter*>(s); }
const V& av(const Value& v) { return obj<Vec>(v).v; }
V lvec(const std::vector<long>& v) { return V(v.begin(), v.end()); }

void need_finite(const V& x, const std::string& op, const char* what = "x") {
  for (double v : x) if (!std::isfinite(v)) fail(op + ": " + what + " contains NaN or infinity");
}

std::string norm_btype(const std::string& b) {
  if (b == "low" || b == "lowpass" || b == "l" || b == "lp") return "lowpass";
  if (b == "high" || b == "highpass" || b == "h" || b == "hp") return "highpass";
  if (b == "band" || b == "bandpass" || b == "pass" || b == "bp") return "bandpass";
  if (b == "bandstop" || b == "stop" || b == "bands" || b == "bs") return "bandstop";
  fail("iir: btype must be lowpass, highpass, bandpass or bandstop; got '" + b + "'");
}
std::string norm_kind(const std::string& k, std::string& bnorm) {
  if (k == "butter" || k == "butterworth" || k == "butterw") return "butter";
  if (k == "cheby1" || k == "chebyshev1" || k == "chebyshevi") return "cheby1";
  if (k == "cheby2" || k == "chebyshev2" || k == "chebyshevii") return "cheby2";
  if (k == "ellip" || k == "elliptic" || k == "cauer") return "ellip";
  if (k == "bessel") return "bessel";
  if (k == "bessel_phase") { bnorm = "phase"; return "bessel"; }
  if (k == "bessel_delay") { bnorm = "delay"; return "bessel"; }
  if (k == "bessel_mag") { bnorm = "mag"; return "bessel"; }
  fail("iir: kind must be butter, cheby1, cheby2, ellip or bessel; got '" + k + "'");
}

void finalize(Filter* f) {
  if (!f->sos.allFinite()) fail("iir: the design produced non-finite second-order sections");
  f->radius = max_pole_radius(f->sos);
  f->stable = f->radius < 1.0;
  f->marginal = f->radius >= 1.0 - 1e-6;
}

Filter* design(const std::string& kind0, int64_t order, const V& cutoff, const Opts& o) {
  std::string bnorm = "phase";
  std::string kind = norm_kind(kind0, bnorm);
  std::string btype = o.has("btype") ? norm_btype(o.str("btype", "")) : (cutoff.size() == 2 ? "bandpass" : "lowpass");
  double fs = o.num("fs", NAN), rp = o.num("rp", NAN), rs = o.num("rs", NAN);
  if (kind == "bessel") bnorm = o.str("norm", bnorm);
  o.finish("iir");
  if (order < 1 || order > 20) fail("iir: order must be 1..20; got " + std::to_string(order));
  if (!std::isnan(fs) && !(fs > 0 && std::isfinite(fs))) fail("iir: fs must be a positive finite number");
  if (!std::isnan(rp) && rp < 0) fail("iir: passband ripple (rp) must be positive");
  if (!std::isnan(rs) && rs < 0) fail("iir: stopband attenuation (rs) must be positive");
  bool band = btype == "bandpass" || btype == "bandstop";
  if (band && cutoff.size() != 2) fail("iir: " + btype + " needs two cutoff frequencies (low, high); got " + std::to_string(cutoff.size()));
  if (!band && cutoff.size() != 1) fail("iir: " + btype + " needs one cutoff frequency; got " + std::to_string(cutoff.size()));
  V wn;
  for (double c : cutoff) wn.push_back(std::isnan(fs) ? c : c / (fs / 2));
  for (double w : wn)
    if (!(w > 0 && w < 1)) {
      if (!std::isnan(fs)) fail("iir: digital filter critical frequencies must be 0 < cutoff < fs/2 (fs=" + jnum(fs) + " -> fs/2=" + jnum(fs / 2) + ")");
      fail("iir: digital filter critical frequencies must be 0 < cutoff < 1 (fractions of Nyquist; set fs for Hz)");
    }
  if (band && !(wn[0] < wn[1])) fail("iir: the first cutoff must be less than the second");
  auto* f = make<Filter>();
  f->kind = kind; f->btype = btype; f->order = int(order); f->cutoff = cutoff; f->fs = fs;
  f->zpk = iirfilter_zpk(kind, int(order), wn, btype, rp, rs, bnorm);
  f->has_zpk = true;
  f->sos = zpk2sos(f->zpk);
  finalize(f);
  return f;
}

Filter* from_sos(const MatrixXd& sos, double fs) {
  if (sos.cols() != 6 || sos.rows() < 1) fail("sos_filter: sos must have shape (n_sections, 6); got " + std::to_string(sos.rows()) + "x" + std::to_string(sos.cols()));
  for (long r = 0; r < sos.rows(); ++r) if (sos(r, 3) != 1.0) fail("sos_filter: the first denominator coefficient of each section (column 3) must be 1");
  if (!(std::isnan(fs) || (fs > 0 && std::isfinite(fs)))) fail("sos_filter: fs must be a positive finite number");
  auto* f = make<Filter>();
  f->sos = sos; f->fs = fs;
  f->order = 0;
  for (long r = 0; r < sos.rows(); ++r) f->order += sos(r, 5) != 0 ? 2 : (sos(r, 4) != 0 ? 1 : 0);
  finalize(f);
  return f;
}

void need_stable(const Filter& f, const char* op) {
  if (!f.stable) fail(std::string("Filter.") + op + ": the filter is unstable (max pole radius " + jnum(f.radius) + "): lower the order or widen the band");
}

Res* response_res(const CV& h, const V& freqs) {
  V mag, ph;
  for (auto& v : h) { mag.push_back(std::hypot(v.real(), v.imag())); ph.push_back(std::atan2(v.imag(), v.real())); }
  auto* r = make<Res>();
  r->vec("freqs", freqs).vec("magnitude", mag).vec("phase", ph);
  return r;
}
V to_omega(const V& f, double fs) { V w; for (double v : f) w.push_back(2 * M_PI * v / fs); return w; }

// ---- spectral options ----
struct WinChoice { std::string name; V custom; double param = NAN; bool is_custom = false; };
WinChoice read_window(const Opts& o, const std::string& def, double def_param = NAN) {
  WinChoice w;
  w.name = def; w.param = def_param;
  const Item* it = o.find("window");
  if (!it) return w;
  if (it->kind == Item::STR) { w.name = it->s; w.param = NAN; }
  else if (it->kind == Item::VEC) { w.is_custom = true; w.custom = it->vec; w.name = "custom"; }
  else fail("option 'window' must be a window name (string) or the window values (Vec)");
  return w;
}
std::string read_detrend(const Opts& o, const std::string& op) {
  const Item* it = o.find("detrend");
  if (!it) return "constant";
  if (it->kind == Item::BOOL) { if (it->b) fail(op + ": detrend=true is ambiguous; use \"constant\" or \"linear\""); return "none"; }
  if (it->kind != Item::STR) fail(op + ": option 'detrend' must be constant, linear, none (or false)");
  std::string d = it->s;
  if (d == "false" || d == "none") return "none";
  if (d == "c") d = "constant";
  if (d == "l") d = "linear";
  if (d != "constant" && d != "linear") fail(op + ": detrend must be constant, linear or none; got '" + it->s + "'");
  return d;
}
const int64_t NPERSEG_CAP = 65536, NFFT_CAP = 131072;
V make_window(const WinChoice& w, size_t n, const std::string& op) {
  if (w.is_custom) {
    if (w.custom.size() != n) fail(op + ": the window has " + std::to_string(w.custom.size()) + " values; nperseg is " + std::to_string(n));
    return w.custom;
  }
  return window(w.name, long(n), false, w.param);
}

// Builds the welch/csd/spectrogram parameters (scipy defaults; `def_overlap_div` 2 for welch, 8 for spectrogram).
SpecParams spec_params(const Opts& o, size_t n, const std::string& op, const std::string& def_window, double def_param, int def_overlap_div,
                       bool has_average, WinChoice& wc) {
  SpecParams p;
  wc = read_window(o, def_window, def_param);
  int64_t nperseg = o.integer("nperseg", -1), noverlap = o.integer("noverlap", -1), nfft = o.integer("nfft", -1);
  p.detrend = read_detrend(o, op);
  p.onesided = o.flag("return_onesided", true);
  p.scaling = o.str("scaling", "density");
  if (has_average) p.average = o.str("average", "mean");
  o.finish(op);
  if (p.scaling != "density" && p.scaling != "spectrum") fail(op + ": scaling must be density or spectrum; got '" + p.scaling + "'");
  if (p.average != "mean" && p.average != "median") fail(op + ": average must be mean or median; got '" + p.average + "'");
  if (n == 0) fail(op + ": x is empty");
  if (nperseg == -1) nperseg = wc.is_custom ? int64_t(wc.custom.size()) : 256;
  else if (nperseg < 1) fail(op + ": nperseg must be a positive integer");
  if (nperseg > NPERSEG_CAP) fail(op + ": nperseg is capped at " + std::to_string(NPERSEG_CAP));
  if (!wc.is_custom && size_t(nperseg) > n) nperseg = int64_t(n);  // scipy warns and uses the signal length
  if (wc.is_custom && size_t(nperseg) > n) fail(op + ": the window is longer than the signal");
  p.nperseg = size_t(nperseg);
  p.win = make_window(wc, p.nperseg, op);
  if (nfft == -1) nfft = nperseg;
  else if (nfft < nperseg) fail(op + ": nfft must be greater than or equal to nperseg");
  if (nfft > NFFT_CAP) fail(op + ": nfft is capped at " + std::to_string(NFFT_CAP));
  p.nfft = size_t(nfft);
  if (noverlap == -1) noverlap = nperseg / def_overlap_div;
  else if (noverlap < 0) fail(op + ": noverlap must be non-negative");
  if (noverlap >= nperseg) fail(op + ": noverlap must be less than nperseg");
  p.noverlap = size_t(noverlap);
  return p;
}

void put_peak(Res* r, const V& f, const V& mag) {
  size_t k = size_t(std::max_element(mag.begin(), mag.end()) - mag.begin());
  // numpy argmax: the first maximum; NaN cannot occur (inputs are finite)
  r->integer("peak_index", int64_t(k)).num("peak_freq", f[k]);
}

Res* spectrum_res(const std::string& method, const WinChoice& wc, size_t n, const CsdOut& c, const SpecParams& p) {
  auto* r = make<Res>();
  V pxx;
  for (auto& v : c.pxy) pxx.push_back(v.real());
  for (double v : pxx) if (!std::isfinite(v)) fail(method + ": the spectral estimate is not finite");
  double df = p.fs / double(p.nfft);
  V w2;
  for (double v : p.win) w2.push_back(v * v);
  double sw = pairwise_sum(p.win.data(), p.win.size());
  double enbw = p.fs * pairwise_sum(w2.data(), w2.size()) / std::pow(std::fabs(sw), 2.0);
  double total = pairwise_sum(pxx.data(), pxx.size()) * df / (p.scaling == "spectrum" ? enbw : 1.0);
  r->str("status", "OK").str("method", method).str("window", wc.name).integer("n_samples", int64_t(n)).integer("n_segments", int64_t(c.n_segments));
  r->integer("n_bins", int64_t(c.f.size())).num("df", df).vec("f", c.f).vec("pxx", pxx).num("total_power", total).num("enbw", enbw);
  put_peak(r, c.f, pxx);
  return r;
}

Res* periodogram_op(const V& x0, double fs, const Opts& o) {
  const std::string op = "periodogram";
  need_finite(x0, op);
  if (!(fs > 0 && std::isfinite(fs))) fail(op + ": fs must be a positive finite number");
  V x = x0;
  WinChoice wc = read_window(o, "boxcar");
  int64_t nfft = o.integer("nfft", -1);
  SpecParams p;
  p.fs = fs;
  p.detrend = read_detrend(o, op);
  p.onesided = o.flag("return_onesided", true);
  p.scaling = o.str("scaling", "density");
  o.finish(op);
  if (p.scaling != "density" && p.scaling != "spectrum") fail(op + ": scaling must be density or spectrum; got '" + p.scaling + "'");
  if (x.empty()) fail(op + ": x is empty");
  size_t n = x.size(), nperseg;
  if (nfft != -1 && nfft < 1) fail(op + ": nfft must be a positive integer");
  if (nfft > NFFT_CAP) fail(op + ": nfft is capped at " + std::to_string(NFFT_CAP));
  if (nfft == -1 || size_t(nfft) >= n) nperseg = n;
  else { x.resize(size_t(nfft)); nperseg = size_t(nfft); }
  p.nperseg = nperseg;
  p.nfft = nfft == -1 ? nperseg : std::max(size_t(nfft), nperseg);
  p.noverlap = 0;
  p.win = make_window(wc, nperseg, op);
  CsdOut c = csd(x, x, true, p);
  return spectrum_res(op, wc, x0.size(), c, p);
}

Res* welch_op(const V& x, double fs, const Opts& o) {
  need_finite(x, "welch");
  if (!(fs > 0 && std::isfinite(fs))) fail("welch: fs must be a positive finite number");
  WinChoice wc;
  SpecParams p = spec_params(o, x.size(), "welch", "hann", NAN, 2, true, wc);
  p.fs = fs;
  CsdOut c = csd(x, x, true, p);
  return spectrum_res("welch", wc, x.size(), c, p);
}

Res* csd_op(const V& x, const V& y, double fs, const Opts& o, bool coherence) {
  std::string op = coherence ? "coherence" : "csd";
  need_finite(x, op); need_finite(y, op, "y");
  if (!(fs > 0 && std::isfinite(fs))) fail(op + ": fs must be a positive finite number");
  if (x.empty() || y.empty()) fail(op + ": x and y must be non-empty");
  WinChoice wc;
  size_t n = std::max(x.size(), y.size());
  SpecParams p = spec_params(o, n, op, "hann", NAN, 2, !coherence, wc);
  p.fs = fs;
  auto* r = make<Res>();
  r->str("status", "OK").str("method", op).str("window", wc.name).integer("n_samples", int64_t(n));
  CsdOut c = csd(x, y, false, p);
  r->integer("n_segments", int64_t(c.n_segments)).integer("n_bins", int64_t(c.f.size())).num("df", fs / double(p.nfft)).vec("f", c.f);
  V mag;
  if (coherence) {
    SpecParams pc = p;
    pc.scaling = "density";
    pc.average = "mean";
    CsdOut cx = csd(x, x, true, pc), cy = csd(y, y, true, pc), cxy = csd(x, y, false, pc);
    for (size_t k = 0; k < cxy.pxy.size(); ++k) {
      double a = std::abs(cxy.pxy[k]);
      mag.push_back(std::pow(a, 2.0) / cx.pxy[k].real() / cy.pxy[k].real());
    }
    r->vec("cxy", mag);
  } else {
    V re, im;
    for (auto& v : c.pxy) { re.push_back(v.real()); im.push_back(v.imag()); mag.push_back(std::abs(v)); }
    r->vec("pxy_re", re).vec("pxy_im", im).vec("magnitude", mag);
  }
  for (double v : mag) if (!std::isfinite(v)) fail(op + ": the spectral estimate is not finite");
  put_peak(r, c.f, mag);
  return r;
}

Res* spectrogram_op(const V& x, double fs, const Opts& o) {
  need_finite(x, "spectrogram");
  if (!(fs > 0 && std::isfinite(fs))) fail("spectrogram: fs must be a positive finite number");
  WinChoice wc;
  SpecParams p = spec_params(o, x.size(), "spectrogram", "tukey", 0.25, 8, false, wc);
  p.fs = fs;
  SpectrogramOut s = spectrogram(x, p);
  auto* r = make<Res>();
  r->str("status", "OK").str("window", wc.name);
  r->integer("n_segments", int64_t(s.t.size())).integer("n_bins", int64_t(s.f.size())).vec("f", s.f).vec("t", s.t).mat("sxx", s.sxx);
  return r;
}

// ---- peaks ----
struct Cond { bool given = false; double lo = NAN, hi = NAN; };
Cond read_cond(const Opts& o, const std::string& key) {
  Cond c;
  const Item* it = o.find(key);
  if (!it) return c;
  c.given = true;
  if (it->kind == Item::NUM) c.lo = it->f;
  else if (it->kind == Item::INT) c.lo = double(it->i);
  else if (it->kind == Item::VEC && (it->vec.size() == 1 || it->vec.size() == 2)) { c.lo = it->vec[0]; if (it->vec.size() == 2) c.hi = it->vec[1]; }
  else fail("find_peaks: option '" + key + "' must be a number (minimum) or a Vec (min, max) with nan for an open end");
  return c;
}
std::vector<char> select_prop(const V& v, const Cond& c) {
  std::vector<char> keep(v.size(), 1);
  for (size_t i = 0; i < v.size(); ++i) {
    if (!std::isnan(c.lo) && !(c.lo <= v[i])) keep[i] = 0;
    if (!std::isnan(c.hi) && !(v[i] <= c.hi)) keep[i] = 0;
  }
  return keep;
}
template <class T> std::vector<T> take(const std::vector<T>& v, const std::vector<char>& keep) {
  std::vector<T> r;
  for (size_t i = 0; i < v.size(); ++i) if (keep[i]) r.push_back(v[i]);
  return r;
}
long read_wlen(const Opts& o, const std::string& op) {
  const Item* it = o.find("wlen");
  if (!it) return -1;
  double w;
  if (it->kind == Item::NUM) w = it->f;
  else if (it->kind == Item::INT) w = double(it->i);
  else fail(op + ": option 'wlen' must be a number");
  if (!(w > 1)) fail(op + ": wlen must be larger than 1; got " + jnum(w));
  return long(std::ceil(w));
}
std::vector<long> peak_indices(const V& p, size_t n, const std::string& op) {
  std::vector<long> r;
  for (double v : p) {
    if (v != std::floor(v) || !std::isfinite(v)) fail(op + ": peaks must be integer indexes; got " + jnum(v));
    if (v < 0 || v >= double(n)) fail(op + ": peak " + jnum(v) + " is not a valid index for x (length " + std::to_string(n) + ")");
    r.push_back(long(v));
  }
  return r;
}

Res* find_peaks_op(const V& x, const Opts& o) {
  const std::string op = "find_peaks";
  Cond height = read_cond(o, "height"), threshold = read_cond(o, "threshold"), prominence = read_cond(o, "prominence");
  Cond width = read_cond(o, "width"), plateau = read_cond(o, "plateau_size");
  double distance = o.num("distance", NAN);
  long wlen = read_wlen(o, op);
  double rel_height = o.num("rel_height", 0.5);
  o.finish(op);
  if (!std::isnan(distance) && distance < 1) fail("find_peaks: distance must be greater or equal to 1");
  std::vector<long> peaks, le, re;
  local_maxima_1d(x, peaks, le, re);
  std::vector<std::pair<std::string, V>> props;
  auto filter_all = [&](const std::vector<char>& keep) {
    peaks = take(peaks, keep);
    for (auto& kv : props) kv.second = take(kv.second, keep);
  };
  if (plateau.given) {
    V sizes;
    for (size_t i = 0; i < peaks.size(); ++i) sizes.push_back(double(re[i] - le[i] + 1));
    props.push_back({"plateau_sizes", sizes});
    props.push_back({"left_edges", lvec(le)});
    props.push_back({"right_edges", lvec(re)});
    filter_all(select_prop(sizes, plateau));
  }
  if (height.given) {
    V h;
    for (long p : peaks) h.push_back(x[size_t(p)]);
    props.push_back({"peak_heights", h});
    filter_all(select_prop(h, height));
  }
  if (threshold.given) {
    V lt, rt;
    std::vector<char> keep(peaks.size(), 1);
    for (size_t i = 0; i < peaks.size(); ++i) {
      size_t p = size_t(peaks[i]);
      lt.push_back(x[p] - x[p - 1]);
      rt.push_back(x[p] - x[p + 1]);
      if (!std::isnan(threshold.lo) && !(threshold.lo <= std::min(lt[i], rt[i]))) keep[i] = 0;
      if (!std::isnan(threshold.hi) && !(std::max(lt[i], rt[i]) <= threshold.hi)) keep[i] = 0;
    }
    props.push_back({"left_thresholds", lt});
    props.push_back({"right_thresholds", rt});
    filter_all(keep);
  }
  if (!std::isnan(distance)) {
    V pri;
    for (long p : peaks) pri.push_back(x[size_t(p)]);
    filter_all(select_by_peak_distance(peaks, pri, distance));
  }
  if (prominence.given || width.given) {
    V prom;
    std::vector<long> lb, rb;
    peak_prominences(x, peaks, wlen, prom, lb, rb);
    props.push_back({"prominences", prom});
    props.push_back({"left_bases", lvec(lb)});
    props.push_back({"right_bases", lvec(rb)});
  }
  auto prop = [&](const std::string& k) -> V& { for (auto& kv : props) if (kv.first == k) return kv.second; fail("find_peaks: internal property " + k); };
  if (prominence.given) filter_all(select_prop(prop("prominences"), prominence));
  if (width.given) {
    V w, wh, li, ri;
    std::vector<long> lb, rb;
    for (double v : prop("left_bases")) lb.push_back(long(v));
    for (double v : prop("right_bases")) rb.push_back(long(v));
    peak_widths(x, peaks, rel_height, prop("prominences"), lb, rb, w, wh, li, ri);
    props.push_back({"widths", w});
    props.push_back({"width_heights", wh});
    props.push_back({"left_ips", li});
    props.push_back({"right_ips", ri});
    filter_all(select_prop(w, width));
  }
  auto* r = make<Res>();
  r->str("status", peaks.empty() ? "NO_PEAKS" : "OK").vec("indices", lvec(peaks)).integer("n_peaks", int64_t(peaks.size()));
  for (auto& kv : props) r->vec(kv.first, kv.second);
  return r;
}

Res* prominences_op(const V& x, const V& pk, const Opts& o) {
  long wlen = read_wlen(o, "peak_prominences");
  o.finish("peak_prominences");
  std::vector<long> peaks = peak_indices(pk, x.size(), "peak_prominences"), lb, rb;
  V prom;
  peak_prominences(x, peaks, wlen, prom, lb, rb);
  auto* r = make<Res>();
  r->vec("prominences", prom).vec("left_bases", lvec(lb)).vec("right_bases", lvec(rb));
  return r;
}
Res* widths_op(const V& x, const V& pk, const Opts& o) {
  double rel = o.num("rel_height", 0.5);
  long wlen = read_wlen(o, "peak_widths");
  o.finish("peak_widths");
  std::vector<long> peaks = peak_indices(pk, x.size(), "peak_widths"), lb, rb;
  V prom, w, wh, li, ri;
  peak_prominences(x, peaks, wlen, prom, lb, rb);
  peak_widths(x, peaks, rel, prom, lb, rb, w, wh, li, ri);
  auto* r = make<Res>();
  r->vec("widths", w).vec("width_heights", wh).vec("left_ips", li).vec("right_ips", ri);
  return r;
}

Res* complex_res(const CV& c) {
  V re, im;
  for (auto& v : c) { re.push_back(v.real()); im.push_back(v.imag()); }
  auto* r = make<Res>();
  r->vec("re", re).vec("im", im);
  return r;
}
CV to_complex(const V& re, const V& im, const char* op) {
  if (re.size() != im.size()) fail(std::string(op) + ": re and im lengths differ (" + std::to_string(re.size()) + " and " + std::to_string(im.size()) + ")");
  CV c(re.size());
  for (size_t i = 0; i < re.size(); ++i) c[i] = cd(re[i], im[i]);
  return c;
}

V window_op(const std::string& name, int64_t n, const Opts& o) {
  bool sym = o.flag("sym", false);
  double param = NAN;
  if (o.has("alpha")) param = o.num("alpha", NAN);
  if (o.has("beta")) param = o.num("beta", NAN);
  if (o.has("std")) param = o.num("std", NAN);
  o.finish("window");
  if (n < 0) fail("window: the length must be non-negative");
  return window(name, long(n), sym, param);
}

V savgol_op(const V& x, int64_t wl, int64_t po, const Opts& o) {
  int64_t deriv = o.integer("deriv", 0);
  double delta = o.num("delta", 1.0), cval = o.num("cval", 0.0);
  std::string mode = o.str("mode", "interp");
  o.finish("savgol");
  if (wl < 1 || wl > 100001) fail("savgol: window_length must be 1..100001");
  return savgol_filter(x, int(wl), int(po), int(deriv), delta, mode, cval);
}

}  // namespace

void register_signal(Api& api) {
  api.section("Signal processing",
              "IIR filter design and filtering (`iir` -> `Filter`, pycli `sig_filter`), spectral estimates (`periodogram`, `welch`, `csd`, "
              "`coherence`, `spectrogram`; pycli `sig_spectrum`), peak finding (`find_peaks`, `peak_prominences`, `peak_widths`; pycli "
              "`sig_peaks`), FFT (pocketfft), windows, detrending, convolution and Savitzky-Golay smoothing. Ports of scipy.signal 1.16: "
              "the same zpk prototypes, bilinear transform and pole-zero pairing (`zpk2sos`, pairing nearest), the same padding in "
              "`filtfilt` (odd extension, scipy's default padlen) and the same find_peaks selection order. Frequencies are in Hz when `fs` "
              "is given, otherwise fractions of the Nyquist frequency (scipy's convention).");
  api.type(Ft, "Digital IIR filter in second-order sections (scipy `iirfilter(..., output='sos')`).");

  auto iir_doc = std::string("design a digital IIR filter (pycli sig_filter): kind butter, cheby1, cheby2, ellip, bessel; order 1..20; cutoff in Hz with ") +
                 "`fs`, else a fraction of Nyquist; one cutoff = lowpass, two = bandpass by default";
  api.fn("iir", {"string", "int", "float"}, Ft, iir_doc, [](Args& a) { return O(design(a[0].s, a[1].i, {a[2].f}, no_opts())); });
  api.fn("iir", {"string", "int", Vt}, Ft, "iir with one (low/high) or two (band) cutoff frequencies", [](Args& a) { return O(design(a[0].s, a[1].i, av(a[2]), no_opts())); });
  api.fn("iir", {"string", "int", "float", Ot}, Ft,
         "iir with options: `btype` lowpass/highpass/bandpass/bandstop, `fs` (Hz), `rp` (dB, cheby1/ellip), `rs` (dB, cheby2/ellip), "
         "`norm` (bessel: phase (default), delay, mag)",
         [](Args& a) { return O(design(a[0].s, a[1].i, {a[2].f}, obj<Opts>(a[3]))); });
  api.fn("iir", {"string", "int", Vt, Ot}, Ft, "iir with cutoff frequencies and options (see above)", [](Args& a) { return O(design(a[0].s, a[1].i, av(a[2]), obj<Opts>(a[3]))); });
  api.fn("sos_filter", {Mt}, Ft, "a Filter from given second-order sections (n x 6, a0 = 1 in column 3); frequencies as fractions of Nyquist",
         [](Args& a) { return O(from_sos(obj<Mat>(a[0]).m, NAN)); });
  api.fn("sos_filter", {Mt, "float"}, Ft, "a Filter from second-order sections with a sampling frequency fs (Hz)", [](Args& a) { return O(from_sos(obj<Mat>(a[0]).m, a[1].f)); });

  api.method(Ft, "sos", {}, Mt, "second-order sections (n_sections x 6: b0 b1 b2 a0 a1 a2)", [](void* s, Args&) { return matv(flt(s).sos); });
  api.method(Ft, "n_sections", {}, "int", "number of second-order sections", [](void* s, Args&) { return I(flt(s).sos.rows()); });
  api.method(Ft, "order", {}, "int", "filter order as designed (for sos_filter: the number of nonzero denominator terms)", [](void* s, Args&) { return I(flt(s).order); });
  api.method(Ft, "fs", {}, "float", "sampling frequency (NaN when frequencies are fractions of Nyquist)", [](void* s, Args&) { return F(flt(s).fs); });
  api.method(Ft, "zpk", {}, Rt, "zeros, poles and gain of the digital design: fields z_re, z_im, p_re, p_im, k (designed filters only)", [](void* s, Args&) {
    const Filter& f = flt(s);
    if (!f.has_zpk) fail("Filter.zpk: only designed filters (iir) carry zeros and poles");
    V zr, zi, pr, pi;
    for (auto& v : f.zpk.z) { zr.push_back(v.real()); zi.push_back(v.imag()); }
    for (auto& v : f.zpk.p) { pr.push_back(v.real()); pi.push_back(v.imag()); }
    auto* r = make<Res>();
    r->vec("z_re", zr).vec("z_im", zi).vec("p_re", pr).vec("p_im", pi).num("k", f.zpk.k);
    return O(r); });
  api.method(Ft, "ba", {}, Rt, "transfer-function coefficients b, a (scipy zpk2tf; poorly conditioned for high orders: prefer sos)", [](void* s, Args&) {
    const Filter& f = flt(s);
    V b, a;
    if (f.has_zpk) zpk2tf(f.zpk, b, a);
    else {
      b = {1.0}; a = {1.0};
      for (long r = 0; r < f.sos.rows(); ++r) {
        V nb(b.size() + 2, 0.0), na(a.size() + 2, 0.0);
        for (size_t i = 0; i < b.size(); ++i) for (int j = 0; j < 3; ++j) nb[i + size_t(j)] += b[i] * f.sos(r, j);
        for (size_t i = 0; i < a.size(); ++i) for (int j = 0; j < 3; ++j) na[i + size_t(j)] += a[i] * f.sos(r, 3 + j);
        b = nb; a = na;
      }
    }
    auto* r = make<Res>();
    r->vec("b", b).vec("a", a);
    return O(r); });
  api.method(Ft, "response", {Vt}, Rt, "frequency response at the given frequencies (Hz with fs, else fractions of Nyquist): fields freqs, magnitude, phase (radians)",
             [](void* s, Args& a) { const Filter& f = flt(s); return O(response_res(sosfreqz(f.sos, to_omega(av(a[0]), f.fs_eff())), av(a[0]))); });
  api.method(Ft, "gain", {"float"}, "float", "|H| at one frequency (Hz with fs, else a fraction of Nyquist)", [](void* s, Args& a) {
    const Filter& f = flt(s);
    cd h = sosfreqz(f.sos, to_omega({a[0].f}, f.fs_eff()))[0];
    return F(std::hypot(h.real(), h.imag())); });
  api.method(Ft, "filter", {Vt}, Vt, "causal filtering with zero initial state (scipy sosfilt)", [](void* s, Args& a) {
    const Filter& f = flt(s); need_stable(f, "filter"); need_finite(av(a[0]), "Filter.filter");
    V y = sosfilt(f.sos, av(a[0]), nullptr);
    need_finite(y, "Filter.filter", "the filtered signal");
    return vecv(std::move(y)); });
  api.method(Ft, "filtfilt", {Vt}, Vt, "zero-phase forward-backward filtering (scipy sosfiltfilt: odd extension, padlen 3*(2*n_sections+1-...))", [](void* s, Args& a) {
    const Filter& f = flt(s); need_stable(f, "filtfilt"); need_finite(av(a[0]), "Filter.filtfilt");
    V y = sosfiltfilt(f.sos, av(a[0]), "odd", -1);
    need_finite(y, "Filter.filtfilt", "the filtered signal");
    return vecv(std::move(y)); });
  api.method(Ft, "filtfilt", {Vt, Ot}, Vt, "sosfiltfilt with options `padtype` (odd, even, constant, none) and `padlen` (int)", [](void* s, Args& a) {
    const Filter& f = flt(s); const Opts& o = obj<Opts>(a[1]);
    std::string pt = o.str("padtype", "odd"); int64_t pl = o.integer("padlen", -1); o.finish("filtfilt");
    if (o.has("padlen") && pl < 0) fail("Filter.filtfilt: padlen must be non-negative");
    need_stable(f, "filtfilt"); need_finite(av(a[0]), "Filter.filtfilt");
    V y = sosfiltfilt(f.sos, av(a[0]), pt, pl);
    need_finite(y, "Filter.filtfilt", "the filtered signal");
    return vecv(std::move(y)); });
  api.method(Ft, "stable", {}, "bool", "all poles strictly inside the unit circle (max_pole_radius < 1)", [](void* s, Args&) { return B(flt(s).stable); });
  api.method(Ft, "marginal", {}, "bool", "a pole within 1e-6 of the unit circle (rings for a very long time)", [](void* s, Args&) { return B(flt(s).marginal); });
  api.method(Ft, "max_pole_radius", {}, "float", "largest pole modulus of the rounded sections (50-digit arithmetic, as pycli)", [](void* s, Args&) { return F(flt(s).radius); });
  api.method(Ft, "status", {}, "string", "OK, or UNSTABLE (filter and filtfilt refuse an unstable filter)", [](void* s, Args&) { return S(flt(s).stable ? "OK" : "UNSTABLE"); });
  api.method(Ft, "info", {}, Rt, "pycli sig_filter fields: status, family, order, btype, n_sections, sos, max_pole_radius, stable, marginal", [](void* s, Args&) {
    const Filter& f = flt(s);
    auto* r = make<Res>();
    r->str("status", f.stable ? "OK" : "UNSTABLE").str("family", f.kind).integer("order", f.order).str("btype", f.btype);
    r->integer("n_sections", f.sos.rows()).mat("sos", f.sos).num("max_pole_radius", f.radius).flag("stable", f.stable).flag("marginal", f.marginal);
    return O(r); });

  // ---- transfer-function filtering ----
  api.fn("lfilter", {Vt, Vt, Vt}, Vt, "filter x with transfer function b/a, zero initial state (scipy lfilter, direct form II transposed)", [](Args& a) {
    need_finite(av(a[2]), "lfilter"); return vecv(lfilter(av(a[0]), av(a[1]), av(a[2]), nullptr)); });
  api.fn("filtfilt", {Vt, Vt, Vt}, Vt, "zero-phase filtering with b/a (scipy filtfilt: odd extension, padlen 3*max(len(a), len(b)))", [](Args& a) {
    need_finite(av(a[2]), "filtfilt"); return vecv(filtfilt(av(a[0]), av(a[1]), av(a[2]), "odd", -1)); });
  api.fn("filtfilt", {Vt, Vt, Vt, Ot}, Vt, "filtfilt with `padtype` (odd, even, constant, none) and `padlen`", [](Args& a) {
    const Opts& o = obj<Opts>(a[3]); std::string pt = o.str("padtype", "odd"); int64_t pl = o.integer("padlen", -1); o.finish("filtfilt");
    if (o.has("padlen") && pl < 0) fail("filtfilt: padlen must be non-negative");
    need_finite(av(a[2]), "filtfilt"); return vecv(filtfilt(av(a[0]), av(a[1]), av(a[2]), pt, pl)); });
  api.fn("freqz", {Vt, Vt, Vt}, Rt, "frequency response of b/a at angular frequencies w (rad/sample): fields freqs, magnitude, phase", [](Args& a) {
    return O(response_res(freqz(av(a[0]), av(a[1]), av(a[2])), av(a[2]))); });
  api.fn("freqz", {Vt, Vt, Vt, "float"}, Rt, "frequency response of b/a at frequencies in Hz for sampling frequency fs", [](Args& a) {
    if (!(a[3].f > 0)) fail("freqz: fs must be positive");
    return O(response_res(freqz(av(a[0]), av(a[1]), to_omega(av(a[2]), a[3].f)), av(a[2]))); });

  // ---- spectra ----
  api.fn("periodogram", {Vt, "float"}, Rt,
         "periodogram (pycli sig_spectrum method periodogram; boxcar window, constant detrend, density): fields status, method, window, n_samples, "
         "n_segments, n_bins, df, f, pxx, total_power, enbw, peak_index, peak_freq",
         [](Args& a) { return O(periodogram_op(av(a[0]), a[1].f, no_opts())); });
  api.fn("periodogram", {Vt, "float", Ot}, Rt, "periodogram with `window` (name or Vec), `nfft`, `detrend` (constant, linear, none), `return_onesided`, `scaling` (density, spectrum)",
         [](Args& a) { return O(periodogram_op(av(a[0]), a[1].f, obj<Opts>(a[2]))); });
  api.fn("welch", {Vt, "float"}, Rt, "Welch PSD (hann, nperseg 256 or the signal length, 50% overlap): same fields as periodogram",
         [](Args& a) { return O(welch_op(av(a[0]), a[1].f, no_opts())); });
  api.fn("welch", {Vt, "float", Ot}, Rt,
         "Welch PSD with `window`, `nperseg`, `noverlap`, `nfft`, `detrend`, `return_onesided`, `scaling`, `average` (mean, median)",
         [](Args& a) { return O(welch_op(av(a[0]), a[1].f, obj<Opts>(a[2]))); });
  api.fn("csd", {Vt, Vt, "float"}, Rt, "cross spectral density (Welch): fields f, pxy_re, pxy_im, magnitude, df, n_segments, n_bins, peak_index, peak_freq",
         [](Args& a) { return O(csd_op(av(a[0]), av(a[1]), a[2].f, no_opts(), false)); });
  api.fn("csd", {Vt, Vt, "float", Ot}, Rt, "csd with the welch options", [](Args& a) { return O(csd_op(av(a[0]), av(a[1]), a[2].f, obj<Opts>(a[3]), false)); });
  api.fn("coherence", {Vt, Vt, "float"}, Rt, "magnitude-squared coherence |Pxy|^2 / (Pxx Pyy): fields f, cxy, df, n_segments, n_bins, peak_index, peak_freq",
         [](Args& a) { return O(csd_op(av(a[0]), av(a[1]), a[2].f, no_opts(), true)); });
  api.fn("coherence", {Vt, Vt, "float", Ot}, Rt, "coherence with `window`, `nperseg`, `noverlap`, `nfft`, `detrend`", [](Args& a) {
    return O(csd_op(av(a[0]), av(a[1]), a[2].f, obj<Opts>(a[3]), true)); });
  api.fn("spectrogram", {Vt, "float"}, Rt, "spectrogram (scipy defaults: tukey 0.25 window, nperseg 256, noverlap nperseg/8, density): fields f, t, sxx (Mat, frequencies x times)",
         [](Args& a) { return O(spectrogram_op(av(a[0]), a[1].f, no_opts())); });
  api.fn("spectrogram", {Vt, "float", Ot}, Rt, "spectrogram with `window`, `nperseg`, `noverlap`, `nfft`, `detrend`, `return_onesided`, `scaling`",
         [](Args& a) { return O(spectrogram_op(av(a[0]), a[1].f, obj<Opts>(a[2]))); });
  api.fn("window", {"string", "int"}, Vt,
         "periodic window of length n as scipy get_window: hann, hamming, blackman, boxcar, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, "
         "barthann, cosine, triang, tukey (alpha 0.5)",
         [](Args& a) { return vecv(window_op(a[0].s, a[1].i, no_opts())); });
  api.fn("window", {"string", "int", Ot}, Vt, "window with `sym` (bool, symmetric instead of periodic) and the shape parameter `alpha` (tukey), `beta` (kaiser) or `std` (gaussian)",
         [](Args& a) { return vecv(window_op(a[0].s, a[1].i, obj<Opts>(a[2]))); });
  api.fn("detrend", {Vt, "string"}, Vt, "remove the mean (constant) or the least-squares line (linear) (scipy.signal.detrend)", [](Args& a) {
    std::string t = a[1].s;
    if (t != "constant" && t != "linear" && t != "c" && t != "l") fail("detrend: type must be constant or linear; got '" + t + "'");
    return vecv(detrend(av(a[0]), t)); });

  // ---- peaks ----
  api.fn("find_peaks", {Vt}, Rt, "all local maxima (scipy find_peaks, pycli sig_peaks): fields status (OK or NO_PEAKS), indices, n_peaks",
         [](Args& a) { return O(find_peaks_op(av(a[0]), no_opts())); });
  api.fn("find_peaks", {Vt, Ot}, Rt,
         "find_peaks with `height`, `threshold`, `prominence`, `width`, `plateau_size` (a minimum, or Vec (min, max), nan = open), `distance`, `wlen`, "
         "`rel_height`; adds scipy's properties for the options given: peak_heights, left/right_thresholds, prominences, left/right_bases, widths, "
         "width_heights, left/right_ips, plateau_sizes, left/right_edges",
         [](Args& a) { return O(find_peaks_op(av(a[0]), obj<Opts>(a[1]))); });
  api.fn("peak_prominences", {Vt, Vt}, Rt, "prominence of each peak index (scipy peak_prominences): fields prominences, left_bases, right_bases",
         [](Args& a) { return O(prominences_op(av(a[0]), av(a[1]), no_opts())); });
  api.fn("peak_prominences", {Vt, Vt, Ot}, Rt, "peak_prominences with `wlen` (window length in samples)", [](Args& a) { return O(prominences_op(av(a[0]), av(a[1]), obj<Opts>(a[2]))); });
  api.fn("peak_widths", {Vt, Vt}, Rt, "width of each peak at half prominence (scipy peak_widths): fields widths, width_heights, left_ips, right_ips",
         [](Args& a) { return O(widths_op(av(a[0]), av(a[1]), no_opts())); });
  api.fn("peak_widths", {Vt, Vt, Ot}, Rt, "peak_widths with `rel_height` (default 0.5) and `wlen`", [](Args& a) { return O(widths_op(av(a[0]), av(a[1]), obj<Opts>(a[2]))); });

  // ---- FFT ----
  api.fn("rfft", {Vt}, Rt, "real-input FFT (numpy.fft.rfft): fields re, im (n/2+1 values)", [](Args& a) {
    if (av(a[0]).empty()) fail("rfft: x is empty"); return O(complex_res(rfft(av(a[0]), av(a[0]).size()))); });
  api.fn("rfft", {Vt, "int"}, Rt, "rfft of x truncated or zero-padded to n", [](Args& a) {
    if (a[1].i < 1) fail("rfft: n must be positive"); return O(complex_res(rfft(av(a[0]), size_t(a[1].i)))); });
  api.fn("rfft_abs", {Vt}, Vt, "magnitudes |rfft(x)|", [](Args& a) {
    if (av(a[0]).empty()) fail("rfft_abs: x is empty");
    V m; for (auto& v : rfft(av(a[0]), av(a[0]).size())) m.push_back(std::hypot(v.real(), v.imag())); return vecv(std::move(m)); });
  api.fn("irfft", {Vt, Vt, "int"}, Vt, "inverse of rfft with output length n (numpy.fft.irfft)", [](Args& a) {
    if (a[2].i < 1) fail("irfft: n must be positive"); return vecv(irfft(to_complex(av(a[0]), av(a[1]), "irfft"), size_t(a[2].i))); });
  api.fn("fft", {Vt, Vt}, Rt, "complex FFT of re + i im (numpy.fft.fft): fields re, im", [](Args& a) {
    CV c = to_complex(av(a[0]), av(a[1]), "fft"); if (c.empty()) fail("fft: x is empty"); return O(complex_res(fft(c, c.size(), false))); });
  api.fn("ifft", {Vt, Vt}, Rt, "inverse complex FFT with 1/n scaling (numpy.fft.ifft): fields re, im", [](Args& a) {
    CV c = to_complex(av(a[0]), av(a[1]), "ifft"); if (c.empty()) fail("ifft: x is empty"); return O(complex_res(fft(c, c.size(), true))); });
  api.fn("rfftfreq", {"int", "float"}, Vt, "sample frequencies of rfft for n samples spaced d apart (numpy.fft.rfftfreq)", [](Args& a) {
    if (a[0].i < 1) fail("rfftfreq: n must be positive"); return vecv(rfftfreq(size_t(a[0].i), a[1].f)); });
  api.fn("fftfreq", {"int", "float"}, Vt, "sample frequencies of fft (numpy.fft.fftfreq)", [](Args& a) {
    if (a[0].i < 1) fail("fftfreq: n must be positive"); return vecv(fftfreq(size_t(a[0].i), a[1].f)); });
  api.fn("fftshift", {Vt}, Vt, "move the zero-frequency term to the centre (numpy.fft.fftshift)", [](Args& a) {
    const V& x = av(a[0]); size_t n = x.size(), s = n / 2; V r(n);
    for (size_t i = 0; i < n; ++i) r[(i + s) % n] = x[i];
    return vecv(std::move(r)); });

  // ---- misc ----
  api.fn("convolve", {Vt, Vt, "string"}, Vt, "discrete linear convolution (numpy.convolve), mode full, same or valid", [](Args& a) {
    return vecv(convolve(av(a[0]), av(a[1]), a[2].s)); });
  api.fn("savgol", {Vt, "int", "int"}, Vt, "Savitzky-Golay smoothing (scipy savgol_filter, mode interp): window_length, polyorder", [](Args& a) {
    need_finite(av(a[0]), "savgol"); return vecv(savgol_op(av(a[0]), a[1].i, a[2].i, no_opts())); });
  api.fn("savgol", {Vt, "int", "int", Ot}, Vt, "savgol with `deriv`, `delta`, `mode` (interp, mirror, nearest, constant, wrap), `cval`", [](Args& a) {
    need_finite(av(a[0]), "savgol"); return vecv(savgol_op(av(a[0]), a[1].i, a[2].i, obj<Opts>(a[3]))); });
  api.fn("savgol_coeffs", {"int", "int"}, Vt, "Savitzky-Golay convolution coefficients (scipy savgol_coeffs)", [](Args& a) {
    if (a[0].i < 1 || a[0].i > 100001) fail("savgol_coeffs: window_length must be 1..100001");
    return vecv(savgol_coeffs(int(a[0].i), int(a[1].i), 0, 1.0)); });
}

}  // namespace lmnum
