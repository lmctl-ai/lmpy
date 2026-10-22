# Source: the SciPy integration tutorial, "General integration (quad)" and "General multiple integration"
# (https://docs.scipy.org/doc/scipy/tutorial/integrate.html), and the reference examples of scipy.integrate.quad and
# scipy.integrate.dblquad (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html, .dblquad.html):
#   - the Bessel function J_2.5 on [0, 4.5] and its closed form through the Fresnel integral S;
#   - a x^2 + b on [0, 1] with args (2, 1); x^2 on [0, 4]; a x on [0, 1] for a = 1, 3;
#   - infinite ranges: exp(-x) on [0, inf), the Gaussian on (-inf, inf), and the tutorial's vectorised exponential
#     integral E_3(x) = int_1^inf exp(-x t) / t^3 dt at x = 1, 1.5, ..., 3.5 (compared with special.expn);
#   - dblquad over a box (x y^2, [0, 2] x [0, 1]) and over a triangle (x y, 0 <= x <= 0.5, 0 <= y <= 1 - 2x).
# Not reproduced: the tutorial's quad of expint over [0, inf) and dblquad over infinite ranges (nested adaptive quadrature).
# Prints the answers as one JSON object; examples/python/scipy-quad.lms pins them.
import json
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from numpy import sqrt, sin, cos, pi
from scipy.integrate import quad, dblquad

result = integrate.quad(lambda x: special.jv(2.5, x), 0, 4.5)
I = sqrt(2/pi)*(18.0/27*sqrt(2)*cos(4.5) - 4.0/27*sqrt(2)*sin(4.5) +
                sqrt(2*pi) * special.fresnel(3/sqrt(pi))[0])


def integrand(x, a, b):
    return a*x**2 + b


I_ab = quad(integrand, 0, 1, args=(2, 1))
x2 = quad(lambda x: x**2, 0, 4)
invexp = quad(lambda x: np.exp(-x), 0, np.inf)
f = lambda x, a: a*x
ax1 = quad(f, 0, 1, args=(1,))
ax3 = quad(f, 0, 1, args=(3,))


def integrand_e(t, n, x):
    return np.exp(-x*t) / t**n


def expint(n, x):
    return quad(integrand_e, 1, np.inf, args=(n, x))[0]


vec_expint = np.vectorize(expint)
gauss = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)

box = dblquad(lambda y, x: x*y**2, 0, 2, 0, 1)
area = dblquad(lambda x, y: x*y, 0, 0.5, lambda x: 0, lambda x: 1-2*x)

print(json.dumps({
    "jv25": result[0], "jv25_exact": I,
    "ax2_plus_b": I_ab[0], "x_squared": x2[0], "invexp": invexp[0], "ax_1": ax1[0], "ax_3": ax3[0],
    "expint3": vec_expint(3, np.arange(1.0, 4.0, 0.5)).tolist(),
    "expn3": special.expn(3, np.arange(1.0, 4.0, 0.5)).tolist(),
    "gaussian": gauss[0],
    "dblquad_box": box[0], "dblquad_triangle": area[0],
}))
