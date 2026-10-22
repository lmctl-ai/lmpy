# Source: the SciPy reference examples for scipy.interpolate.CubicSpline (sin sampled at 0..9 evaluated with its first three
# derivatives on arange(-0.5, 9.6, 0.1), and the periodic unit circle), Akima1DInterpolator (akima versus makima on the step
# data [-1, -1, -1, 0, 1, 1, 1]; the 1-D interpolation tutorial adds PchipInterpolator to the same comparison),
# make_smoothing_spline (x**3 + x**2 sin(4x) plus noise, lam chosen by generalised cross-validation) and RBFInterpolator
# (y = (x1 + x2) exp(-6 |x|^2) at 100 scattered points of [-1, 1]^2, thin-plate spline, evaluated on a grid).
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html
#   https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
# Deviations, so that lmpy can draw the same data: make_smoothing_spline's np.random.seed(1234) legacy stream becomes
# np.random.default_rng(1234), and RBFInterpolator's scrambled Halton points become 2*rng.random((100, 2)) - 1 with
# default_rng(42); the RBF grid is 20 x 20 instead of 50 x 50. The CubicSpline bc_type=((1, 0), (1, 3)) example is omitted
# (lmpy's clamped condition fixes both end slopes at 0). Prints one JSON object; scipy-interpolate.lms pins it.
import json
import numpy as np
from scipy.interpolate import (Akima1DInterpolator, CubicSpline, PchipInterpolator, RBFInterpolator,
                               make_smoothing_spline)

# CubicSpline, not-a-knot, extrapolating to -0.5 and 9.5 as the example does
x = np.arange(10)
y = np.sin(x)
cs = CubicSpline(x, y)
xs = np.arange(-0.5, 9.6, 0.1)

# CubicSpline, periodic unit circle (one spline per coordinate)
theta = 2 * np.pi * np.linspace(0, 1, 5)
yc = np.c_[np.cos(theta), np.sin(theta)]
cc = CubicSpline(theta, yc, bc_type='periodic')
xt = 2 * np.pi * np.linspace(0, 1, 100)

# Akima / makima / pchip
xa = np.linspace(1, 7, 7)
ya = np.array([-1, -1, -1, 0, 1, 1, 1])
xsa = np.linspace(min(xa), max(xa), num=100)
y_akima = Akima1DInterpolator(xa, ya, method="akima")(xsa)
y_makima = Akima1DInterpolator(xa, ya, method="makima")(xsa)
y_pchip = PchipInterpolator(xa, ya)(xsa)

# make_smoothing_spline
rng = np.random.default_rng(1234)
n = 200
def func(x):
    return x**3 + x**2 * np.sin(4 * x)
xn = np.sort(rng.random(n) * 4 - 2)
yn = func(xn) + rng.normal(scale=1.5, size=n)
spl = make_smoothing_spline(xn, yn)
grid = np.linspace(xn[0], xn[-1], 400)
spl_fixed = make_smoothing_spline(xn, yn, lam=0.01)

# RBFInterpolator
rng2 = np.random.default_rng(42)
xobs = 2 * rng2.random((100, 2)) - 1
yobs = np.sum(xobs, axis=1) * np.exp(-6 * np.sum(xobs**2, axis=1))
xgrid = np.mgrid[-1:1:20j, -1:1:20j]
xflat = xgrid.reshape(2, -1).T
yflat = RBFInterpolator(xobs, yobs)(xflat)

print(json.dumps({
    "cs_s": cs(xs).tolist(),
    "cs_s1": cs(xs, 1).tolist(),
    "cs_s2": cs(xs, 2).tolist(),
    "cs_s3": cs(xs, 3).tolist(),
    "circle_slope_at_0": cc(0, 1).tolist(),
    "circle_x": cc(xt)[:, 0].tolist(),
    "circle_y": cc(xt)[:, 1].tolist(),
    "akima": y_akima.tolist(),
    "makima": y_makima.tolist(),
    "pchip": y_pchip.tolist(),
    "smoothing_gcv_rss": float(np.sum((spl(xn) - yn)**2)),
    "smoothing_gcv_norm": float(np.linalg.norm(spl(grid))),
    "smoothing_lam_0_01": spl_fixed(grid).tolist(),
    "rbf_grid": yflat.tolist(),
}))
