# Source: SciPy sparse linear algebra documentation:
#   - scipy.sparse.linalg.LaplacianNd reference example: the 2-D Laplacian with Dirichlet boundary conditions (here on a
#     12 x 10 grid instead of 2 x 3) and its exact eigenvalues lap.eigenvalues();
#   - the Poisson system -L x = 1 on that grid solved directly (spsolve) and by conjugate gradients (cg, rtol 1e-10);
#   - the scipy.sparse.linalg.cg reference example (4 x 4 SPD matrix, atol 1e-5);
#   - the ARPACK tutorial (https://docs.scipy.org/doc/scipy/tutorial/arpack.html): X = rng.random((100, 100)) - 0.5,
#     X X^T, the three largest-magnitude eigenvalues by eigsh(which='LM') against eigh, and the three smallest by
#     eigsh(which='SM', maxiter=5000). The tutorial uses an unseeded default_rng(); here it is seeded (7).
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
# Prints the answers as one JSON object; examples/python/scipy-sparse-laplacian.lms pins them.
import json
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_array
from scipy.sparse.linalg import LaplacianNd, spsolve, cg, eigsh

grid_shape = (12, 10)
n = int(np.prod(grid_shape))
lap = LaplacianNd(grid_shape, boundary_conditions='dirichlet')
A = -lap.tosparse().astype(np.float64).tocsc()   # symmetric positive definite
b = np.ones(n)
x_direct = spsolve(A, b)
x_cg, info = cg(A, b, rtol=1e-10)
exact = lap.eigenvalues()                        # ascending, all negative
w_small = eigsh(A, 4, which='SA')[0]
w_large = eigsh(A, 4, which='LA')[0]

P = np.array([[4, 0, 1, 0],
              [0, 5, 0, 0],
              [1, 0, 3, 2],
              [0, 0, 2, 4]])
Pa = csc_array(P)
pb = np.array([-1, -0.5, -1, 2])
px, exit_code = cg(Pa, pb, atol=1e-5)

rng = np.random.default_rng(7)
X = rng.random((100, 100)) - 0.5
X = np.dot(X, X.T)  # create a symmetric matrix
evals_all, evecs_all = eigh(X)
evals_large, evecs_large = eigsh(X, 3, which='LM')
evals_small, evecs_small = eigsh(X, 3, which='SM', maxiter=5000)

print(json.dumps({
    "nnz": int(A.nnz),
    "x_direct_sum": float(x_direct.sum()), "x_direct_max": float(x_direct.max()), "x_direct_head": x_direct[:5].tolist(),
    "cg_converged": bool(info == 0), "cg_x_head": x_cg[:5].tolist(),
    "exact_smallest": (-exact[::-1][:4]).tolist(), "exact_largest": (-exact[:4][::-1]).tolist(),
    "eigsh_smallest": w_small.tolist(), "eigsh_largest": w_large.tolist(),
    "cg_doc_converged": bool(exit_code == 0), "cg_doc_allclose": bool(np.allclose(Pa.dot(px), pb)),
    "evals_all_top3": evals_all[-3:].tolist(), "evals_large": evals_large.tolist(),
    "evals_all_bottom3": evals_all[:3].tolist(), "evals_small": evals_small.tolist(),
}))
