// Signal-processing numerics behind signal.cpp (no DSL types here): windows, FFT, detrending, spectral estimates,
// IIR design (zpk prototypes, frequency transforms, bilinear transform, zpk -> second-order sections), filtering,
// peak finding and Savitzky-Golay smoothing. Ported from scipy.signal 1.16 (BSD-3-Clause, Copyright (c) 2001-2002
// Enthought, Inc. 2003-2024 SciPy Developers) and numpy (BSD-3-Clause); the comments name the scipy function each
// routine follows.
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

namespace lmnum {
namespace sig {

using cd = std::complex<double>;
using V = std::vector<double>;
using CV = std::vector<cd>;

// ---- small numerics shared by the signal files ----
double pairwise_sum(const double* a, size_t n);  // numpy's pairwise summation (add.reduce on a contiguous axis)
inline double np_mean(const V& x) { return x.empty() ? NAN : pairwise_sum(x.data(), x.size()) / double(x.size()); }
cd cmul(cd a, cd b);  // (ar br - ai bi) + (ar bi + ai br) i, the numpy formula (no special inf/nan handling)

// ---- windows (scipy.signal.windows / get_window) ----
// sym=false is the periodic window get_window returns (fftbins=True). `param` is the window's shape parameter
// (tukey alpha, kaiser beta, gaussian std); NaN means the window's default (tukey 0.5) or "missing" (kaiser, gaussian).
V window(const std::string& name, long n, bool sym, double param);
const std::vector<std::string>& window_names();

// ---- FFT (pocketfft, the code numpy/scipy use) ----
CV rfft(const V& x, size_t n);                  // input truncated or zero-padded to n; n/2+1 outputs
CV fft(const CV& x, size_t n, bool inverse);    // inverse includes the 1/n factor (numpy norm="backward")
V irfft(const CV& X, size_t n);                 // numpy.fft.irfft(X, n)
V rfftfreq(size_t n, double d);
V fftfreq(size_t n, double d);

// ---- detrending (scipy.signal.detrend, one segment) ----
// type: "constant"/"c", "linear"/"l", or "none" (identity).
V detrend(const V& x, const std::string& type);

// ---- spectral estimates (scipy.signal.csd / welch / periodogram / coherence / spectrogram) ----
struct SpecParams {
  double fs = 1.0;
  V win;                       // window values (length nperseg)
  size_t nperseg = 0, noverlap = 0, nfft = 0;
  std::string detrend = "constant";
  bool onesided = true;
  std::string scaling = "density";
  std::string average = "mean";
};
struct CsdOut { V f; CV pxy; size_t n_segments = 0; };
// Cross spectral density of x and y (same=true: power spectral density, imaginary part dropped).
CsdOut csd(const V& x, const V& y, bool same, const SpecParams& p);
struct SpectrogramOut { V f, t; Eigen::MatrixXd sxx; };
SpectrogramOut spectrogram(const V& x, const SpecParams& p);

// ---- IIR design (scipy.signal.iirfilter, digital) ----
// real: scipy holds the zeros and poles as a real array (ellipap of order 1), so the transforms use real arithmetic and
// every imaginary part is +0 (this decides the square-root branch in lp2bp/lp2bs).
struct Zpk { CV z, p; double k = 1.0; bool real = false; };
Zpk buttap(int n);
Zpk cheb1ap(int n, double rp);
Zpk cheb2ap(int n, double rs);
Zpk ellipap(int n, double rp, double rs);
Zpk besselap(int n, const std::string& norm);
Zpk lp2lp_zpk(const Zpk& f, double wo);
Zpk lp2hp_zpk(const Zpk& f, double wo);
Zpk lp2bp_zpk(const Zpk& f, double wo, double bw);
Zpk lp2bs_zpk(const Zpk& f, double wo, double bw);
Zpk bilinear_zpk(const Zpk& f, double fs);
// Full digital design: wn normalised to Nyquist = 1 (0 < wn < 1), btype lowpass/highpass/bandpass/bandstop.
Zpk iirfilter_zpk(const std::string& ftype, int order, const V& wn, const std::string& btype, double rp, double rs, const std::string& bessel_norm);
Eigen::MatrixXd zpk2sos(const Zpk& f);              // pairing "nearest" (scipy's digital default)
void zpk2tf(const Zpk& f, V& b, V& a);               // numpy.poly of zeros and poles (real parts)
double max_pole_radius(const Eigen::MatrixXd& sos);  // from the rounded sections, in 50-digit arithmetic (as pycli)

// ---- filtering ----
V sosfilt(const Eigen::MatrixXd& sos, const V& x, Eigen::MatrixXd* zi);  // zi (n_sections x 2) in/out, may be null
Eigen::MatrixXd sosfilt_zi(const Eigen::MatrixXd& sos);
// padtype "odd" / "even" / "constant" / "none"; padlen < 0 means scipy's default.
V sosfiltfilt(const Eigen::MatrixXd& sos, const V& x, const std::string& padtype, long padlen);
V lfilter(const V& b, const V& a, const V& x, V* zi);
V lfilter_zi(const V& b, const V& a);
V filtfilt(const V& b, const V& a, const V& x, const std::string& padtype, long padlen);
// Frequency response of b/a at angular frequencies w (rad/sample): scipy freqz with an array worN.
CV freqz(const V& b, const V& a, const V& w);
CV sosfreqz(const Eigen::MatrixXd& sos, const V& w);

// ---- peaks (scipy.signal.find_peaks and _peak_finding_utils) ----
void local_maxima_1d(const V& x, std::vector<long>& mid, std::vector<long>& left, std::vector<long>& right);
std::vector<char> select_by_peak_distance(const std::vector<long>& peaks, const V& priority, double distance);
// wlen < 0: whole signal (scipy wlen=None); otherwise the integer window length (>= 2 applies a window).
void peak_prominences(const V& x, const std::vector<long>& peaks, long wlen, V& prom, std::vector<long>& lb, std::vector<long>& rb);
void peak_widths(const V& x, const std::vector<long>& peaks, double rel_height, const V& prom, const std::vector<long>& lb,
                 const std::vector<long>& rb, V& widths, V& heights, V& lips, V& rips);

// ---- misc ----
V convolve(const V& a, const V& v, const std::string& mode);  // numpy.convolve
V savgol_coeffs(int window_length, int polyorder, int deriv, double delta);
V savgol_filter(const V& x, int window_length, int polyorder, int deriv, double delta, const std::string& mode, double cval);

}  // namespace sig
}  // namespace lmnum
