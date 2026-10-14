"""Independent SciPy oracle plus a hand-computed integer inverse."""
import numpy as np
from scipy.linalg import solve
A = [[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]]
B = np.eye(3).tolist()
INVERSE = [[-24., 18., 5.], [20., -15., -4.], [-5., 4., 1.]]
def expected():
    result = solve(A, B)
    np.testing.assert_allclose(result, INVERSE, rtol=1e-13, atol=1e-13)
    return result
