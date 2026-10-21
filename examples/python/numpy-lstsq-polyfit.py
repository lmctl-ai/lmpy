# Source: the NumPy reference examples for numpy.linalg.lstsq ("fit a line, y = mx + c, through some noisy data-points")
# and numpy.polyfit (cubic fit, evaluated with numpy.poly1d). https://numpy.org/doc/stable/reference/
# Prints the answers as one JSON object; examples/python/numpy-lstsq-polyfit.lms pins them.
import json
import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x, np.ones(len(x))]).T
(m, c), residuals, rank, sv = np.linalg.lstsq(A, y)

xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
yp = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
z = np.polyfit(xp, yp, 3)
p = np.poly1d(z)

print(json.dumps({
    "slope": m, "intercept": c, "residual_ss": residuals[0], "rank": int(rank), "singular_values": sv.tolist(),
    "cubic": z.tolist(), "p_at_0_5": p(0.5), "p_at_3_5": p(3.5), "p_at_10": p(10),
}))
