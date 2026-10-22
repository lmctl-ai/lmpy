# Source: the SciPy reference example for scipy.optimize.curve_fit (exponential decay a*exp(-b*x) + c with Gaussian noise,
# then the same fit constrained to 0 <= a <= 3, 0 <= b <= 1, 0 <= c <= 0.5, and the condition number of pcov).
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# The docs draw the noise from an unseeded np.random.default_rng(); here it is seeded (2024) so the answers are reproducible.
# Prints the answers as one JSON object; examples/python/scipy-curve-fit.lms pins them.
import json
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
rng = np.random.default_rng(2024)
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise

popt, pcov = curve_fit(func, xdata, ydata)
popt_b, pcov_b = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))

print(json.dumps({
    "ydata_head": ydata[:5].tolist(),
    "popt": popt.tolist(), "perr": np.sqrt(np.diag(pcov)).tolist(), "pcov": pcov.tolist(),
    "cond_pcov": np.linalg.cond(pcov),
    "popt_bounded": popt_b.tolist(),
}))
