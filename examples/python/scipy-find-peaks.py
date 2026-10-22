# Source: the SciPy reference examples for scipy.signal.peak_prominences and scipy.signal.peak_widths (a test signal with two
# overlaid harmonics, x = sin(t) + 0.6 sin(2.6 t) on 1000 points of [0, 6 pi], plus the 9-sample edge case with wlen=3.1),
# and the scipy.signal.find_peaks example conditions (height, distance, prominence upper bound, prominence + width).
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
# The find_peaks page runs its conditions on scipy.datasets.electrocardiogram(), a downloaded dataset; here they run on the
# two-harmonic signal plus seeded noise (np.random.default_rng(7), scale 0.05) instead.
# Prints the answers as one JSON object; examples/python/scipy-find-peaks.lms pins them.
import json
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

x = np.linspace(0, 6 * np.pi, 1000)
x = np.sin(x) + 0.6 * np.sin(2.6 * x)
peaks, _ = find_peaks(x)
prominences = peak_prominences(x, peaks)[0]
contour_heights = x[peaks] - prominences
results_half = peak_widths(x, peaks, rel_height=0.5)
results_full = peak_widths(x, peaks, rel_height=1)

e = np.array([0, 1, 0, 3, 1, 3, 0, 4, 0])
edge = peak_prominences(e, np.array([5]))
edge_w = peak_prominences(e, np.array([5]), wlen=3.1)

rng = np.random.default_rng(7)
noisy = x + 0.05 * rng.standard_normal(x.size)
p_height, _ = find_peaks(noisy, height=0)
p_dist, _ = find_peaks(noisy, distance=150)
p_low, props_low = find_peaks(noisy, prominence=(None, 0.6))
p_big, props_big = find_peaks(noisy, prominence=1, width=20)

print(json.dumps({
    "peaks": peaks.tolist(),
    "prominences": prominences.tolist(),
    "contour_heights": contour_heights.tolist(),
    "widths_half": results_half[0].tolist(),
    "width_heights_half": results_half[1].tolist(),
    "left_ips_half": results_half[2].tolist(),
    "right_ips_half": results_half[3].tolist(),
    "widths_full": results_full[0].tolist(),
    "edge_prominence": edge[0].tolist(), "edge_left_base": edge[1].tolist(), "edge_right_base": edge[2].tolist(),
    "edge_wlen_prominence": edge_w[0].tolist(), "edge_wlen_left_base": edge_w[1].tolist(),
    "edge_wlen_right_base": edge_w[2].tolist(),
    "noisy_height0_count": len(p_height),
    "noisy_distance150": p_dist.tolist(),
    "noisy_distance150_diff": np.diff(p_dist).tolist(),
    "noisy_low_prominence_count": len(p_low),
    "noisy_low_prominence_max": float(props_low["prominences"].max()),
    "noisy_big_peaks": p_big.tolist(),
    "noisy_big_prominences": props_big["prominences"].tolist(),
    "noisy_big_widths": props_big["widths"].tolist(),
}))
