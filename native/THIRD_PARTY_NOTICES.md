# lmpy third-party notices

lmpy is MIT (see `licenses/lmpy-LICENSE`, the pycli licence). The release executable statically links the components below.
Every one is permissively licensed. Their full licence texts are in `licenses/`. The exact versions and source archive
hashes are pinned in `deps.lock`.

| Component | Version | Licence | Text | Use |
|---|---|---|---|---|
| lmcode-cpp (`cpp/dslvm.hpp`, `cpp/stdlib/`, `cpp/std/`) | commit in `sdk.lock` | MIT | `licenses/lmcode-cpp-LICENSE` | bytecode loader, verifier, VM and the built-in std catalog |
| Eigen | 3.4.0 | MPL-2.0 (built with `EIGEN_MPL2_ONLY`; files unmodified, source at the pinned URL) | `licenses/eigen-COPYING.MPL2` | dense and sparse linear algebra |
| Boost (header-only libraries) | 1.86.0 | BSL-1.0 | `licenses/boost-LICENSE_1_0.txt` | special functions, distributions, quadrature, odeint, graph |
| pocketfft (C++ header) | commit 4d865cba | BSD-3-Clause | `licenses/pocketfft-LICENSE.md` | FFT |
| nanoflann | 1.6.1 | BSD-2-Clause | `licenses/nanoflann-COPYING` | kd-tree nearest neighbours |
| LBFGSpp | 0.3.0 | MIT | `licenses/LBFGSpp-LICENSE.md` | L-BFGS(-B) |
| Spectra | 1.0.1 | MPL-2.0 (files unmodified, source at the pinned URL) | `licenses/spectra-LICENSE` | symmetric sparse eigenvalues |
| xsf (scipy special functions) | commit 4fff9b2 (pinned by scipy 1.16.2) | BSD-3-Clause; bundles Cephes (BSD-3-Clause) and Faddeeva (MIT) | `licenses/xsf-LICENSE`, `licenses/xsf-LICENSES_bundled.txt` | special functions and distribution kernels |
| HiGHS | 1.7.2 | MIT | `licenses/HiGHS-LICENSE.txt` | linear and mixed-integer programming |

Runtime libraries:
- libstdc++ and libgcc are linked statically under the GCC Runtime Library Exception.
- glibc (LGPL-2.1-or-later) is linked **dynamically**, as the system C library: it is the executable's only shared
  dependency (enforced by `test.sh`). `-DLMPY_FULL_STATIC=ON` links it statically too; do not publish such a binary
  without relinking materials.

Ported algorithms (credited in the file headers that contain them):
- scipy (BSD-3-Clause): solve_ivp solvers and tableaux, filter design and zpk2sos, peak finding, spectral estimators,
  savgol, BFGS/Nelder-Mead/Powell/SLSQP drivers, least_squares (TRF/dogbox/LM), root finders, linprog result conventions.
- numpy (BSD-3-Clause): PCG64, SeedSequence, Generator samplers (ziggurat normal/exponential, gamma, binomial, ...),
  MT19937 RandomState (used for scikit-learn-compatible k-means and fold shuffles).
- statsmodels (BSD-3-Clause): SARIMAX state-space likelihood, MacKinnon ADF response surfaces, KPSS tables, GLM IRLS.
- scikit-learn (BSD-3-Clause): coordinate descent, k-means++ seeding, fold assignment, metrics, svd_flip.
- networkx (BSD-3-Clause): Dinitz max flow and minimum cut partition, Yen k-shortest paths.
- QUADPACK (public domain): adaptive Gauss-Kronrod quadrature. MINPACK (MINPACK licence, via Eigen's
  `unsupported/NonLinearOptimization`, MPL-2.0): Powell hybrid and Levenberg-Marquardt root finding.
