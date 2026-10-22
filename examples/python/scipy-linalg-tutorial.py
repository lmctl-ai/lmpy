# Source: the SciPy linear algebra tutorial (https://docs.scipy.org/doc/scipy/tutorial/linalg.html): inverse, solving a
# linear system, determinant, norms, eigenvalues/eigenvectors and the SVD of its example matrices; plus the reference
# examples of scipy.linalg.solve, scipy.linalg.eigh (full spectrum, and the eigenvalues below 10 of the 5x5 matrix),
# scipy.linalg.lu (the 4x4 matrix with its permutation) and scipy.linalg.expm (Euler's identity expm(1j a) for
# a = [[1, 2], [-1, 3]]), https://docs.scipy.org/doc/scipy/reference/linalg.html
# Eigenvectors and singular vectors are defined up to sign; they are reported with their largest-magnitude component made
# positive (a singular pair u_i, v_i is flipped together), so any correct implementation prints the same numbers.
# Prints the answers as one JSON object; examples/python/scipy-linalg-tutorial.lms pins them.
import json
import numpy as np
from scipy import linalg
from scipy.linalg import eigh, lu, expm


def fix_sign(v):
    return v * np.sign(v[np.argmax(np.abs(v))])


# inverse
A = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
inv_A = linalg.inv(A)
# solving a linear system
A2 = np.array([[1, 2], [3, 4]])
b2 = np.array([[5], [6]])
x2 = np.linalg.solve(A2, b2)
xs = linalg.solve(np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]]), np.array([2, 4, -1]))
# determinant and norms
det_A2 = linalg.det(A2)
det_b = linalg.det(np.array([[0, 2, 3], [4, 5, 6], [7, 8, 9]]))
norms = [linalg.norm(A2), linalg.norm(A2, 'fro'), linalg.norm(A2, 1), linalg.norm(A2, -1), linalg.norm(A2, np.inf)]
# eigenvalues and eigenvectors
la, v = linalg.eig(A2)
l1, l2 = la
v1 = np.array(v[:, 0]).T
eig_residual = linalg.norm(A2.dot(v1) - l1*v1)
order = np.argsort(la.real)
# SVD
A3 = np.array([[1, 2, 3], [4, 5, 6]])
U, s, Vh = linalg.svd(A3)
U_fixed, Vh_fixed = U.copy(), Vh[:2].copy()
for i in range(2):
    sg = np.sign(U[np.argmax(np.abs(U[:, i])), i])
    U_fixed[:, i] *= sg
    Vh_fixed[i] *= sg
# eigh
Ah = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
w, vh = eigh(Ah)
B = np.array([[34, -4, -10, -7, 2],
              [-4, 7, 2, 12, 0],
              [-10, 2, 44, 2, -19],
              [-7, 12, 2, 79, -34],
              [2, 0, -19, -34, 29]])
w_below_10 = eigh(B, eigvals_only=True, subset_by_value=[-np.inf, 10])
# LU
Al = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
p, l, u = lu(Al)
p_idx, _, _ = lu(Al, p_indices=True)
# expm: Euler's identity for a matrix
a = np.array([[1.0, 2.0], [-1.0, 3.0]])
e = expm(1j*a)

print(json.dumps({
    "inv": inv_A.tolist(), "solve_2x2": x2.ravel().tolist(), "solve_3x3": xs.tolist(),
    "det": det_A2, "det_b": det_b, "norms": norms,
    "eigenvalues": la.real[order].tolist(), "eigenvalues_imag": la.imag[order].tolist(),
    "eigenvectors": [fix_sign(v[:, k].real).tolist() for k in order],
    "eig_residual_small": bool(eig_residual < 1e-12),
    "singular_values": s.tolist(), "U": U_fixed.tolist(), "Vh": Vh_fixed.tolist(),
    "eigh_w": w.tolist(), "eigh_v": [fix_sign(vh[:, k]).tolist() for k in range(4)],
    "eigh_below_10": w_below_10.tolist(),
    "lu_p": p.tolist(), "lu_p_indices": p_idx.tolist(), "lu_l": l.tolist(), "lu_u": u.tolist(),
    "expm_ia_real": e.real.tolist(), "expm_ia_imag": e.imag.tolist(),
}))
