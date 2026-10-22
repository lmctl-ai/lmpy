# Source: the SciPy Fourier transforms tutorial (scipy.fft: the sum of a 50 Hz and an 80 Hz sine, N = 600 samples at 800 Hz,
# with and without a Blackman window) and the SciPy reference examples for scipy.signal.periodogram and scipy.signal.welch
# (a 2 Vrms sine at 1234 Hz in 0.001 V**2/Hz of white noise sampled at 10 kHz, density and flattop spectrum scalings, and the
# mean versus median average after a 10-sample burst).
#   https://docs.scipy.org/doc/scipy/tutorial/fft.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
# The periodogram/welch examples use an unseeded np.random.default_rng(); here the seed is 1234 so lmpy's rng(1234)
# reproduces the noise. Prints the answers as one JSON object; examples/python/scipy-fft-welch.lms pins them.
import json
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman

# scipy.fft tutorial
N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
amp = 2.0/N * np.abs(yf[0:N//2])
w = blackman(N)
ywf = fft(y*w)
amp_w = 2.0/N * np.abs(ywf[0:N//2])
top2, _ = signal.find_peaks(amp, height=0.2)  # the two tone lines (50 Hz leaks onto 49.3 and 50.7 Hz)

# scipy.signal.periodogram / welch examples
rng = np.random.default_rng(1234)
fs = 10e3
NN = 1e5
a = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(NN) / fs
s = a*np.sin(2*np.pi*freq*time)
s += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

f_p, P_den = signal.periodogram(s, fs)
f_ps, P_spec = signal.periodogram(s, fs, 'flattop', scaling='spectrum')
f_w, W_den = signal.welch(s, fs, nperseg=1024)
f_ws, W_spec = signal.welch(s, fs, 'flattop', 1024, scaling='spectrum')
s[int(NN//2):int(NN//2)+10] *= 50.
f_m, W_mean = signal.welch(s, fs, nperseg=1024)
f_med, W_med = signal.welch(s, fs, nperseg=1024, average='median')

print(json.dumps({
    "fft_freqs": xf.tolist(),
    "fft_amplitude": amp.tolist(),
    "fft_peak_freqs": xf[top2].tolist(),
    "fft_peak_amplitudes": amp[top2].tolist(),
    "fft_blackman_amplitude": amp_w.tolist(),
    "periodogram_noise_floor": float(np.mean(P_den[25000:])),
    "periodogram_peak_freq": float(f_p[np.argmax(P_den)]),
    "periodogram_flattop_rms": float(np.sqrt(P_spec.max())),
    "welch_freqs": f_w.tolist(),
    "welch_density": W_den.tolist(),
    "welch_noise_floor": float(np.mean(W_den[256:])),
    "welch_flattop_rms": float(np.sqrt(W_spec.max())),
    "welch_burst_mean_floor": float(np.mean(W_mean[256:])),
    "welch_burst_median_floor": float(np.mean(W_med[256:])),
    "welch_burst_median": W_med.tolist(),
}))
