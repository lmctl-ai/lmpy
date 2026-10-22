# Source: the SciPy reference examples for scipy.signal.butter (10 Hz + 20 Hz two-tone signal sampled at 1 kHz, 10th-order
# 15 Hz high-pass in second-order sections, sosfilt), scipy.signal.sosfiltfilt (step signal with noise, 4th-order low-pass at
# 0.125 x Nyquist, zero-phase) and scipy.signal.freqz_sos (15th-order elliptic band-pass, response at 1500 points).
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_sos.html
# The sosfiltfilt example draws its noise from an unseeded np.random.default_rng(); here the seed is 2024 so lmpy's rng(2024)
# reproduces it. The analog butter/freqs example and the sosfilt_zi initial-state comparison are omitted (lmpy has no analog
# design or zi). Prints the answers as one JSON object; examples/python/scipy-butter-sosfilt.lms pins them.
import json
import numpy as np
from scipy import signal

# scipy.signal.butter example: remove the 10 Hz tone with a 15 Hz high-pass
t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, sig)

# scipy.signal.sosfiltfilt example
rng = np.random.default_rng(2024)
n = 201
t2 = np.linspace(0, 1, n)
x = 1 + (t2 < 0.5) - 0.25*t2**2 + 0.05*rng.standard_normal(n)
sos4 = signal.butter(4, 0.125, output='sos')
y = signal.sosfiltfilt(sos4, x)

# scipy.signal.freqz_sos example
sos_bp = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass', output='sos')
w, h = signal.freqz_sos(sos_bp, worN=1500)

print(json.dumps({
    "hp_sos": sos.tolist(),
    "hp_filtered": filtered.tolist(),
    "hp_tail_rms": float(np.sqrt(np.mean(filtered[500:]**2))),
    "lp_sos": sos4.tolist(),
    "lp_zero_phase": y.tolist(),
    "bp_n_sections": len(sos_bp),
    "bp_w": w.tolist(),
    "bp_magnitude": np.abs(h).tolist(),
    "bp_peak_gain_db": float(20*np.log10(np.abs(h).max())),
}))
