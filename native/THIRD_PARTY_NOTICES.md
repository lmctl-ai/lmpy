# lmnum third-party notices

lmnum is MIT (see `licenses/lmnum-LICENSE`, the pycli licence). The release executable statically links the components below.
Every one is permissively licensed. Their full licence texts are in `licenses/`. The exact versions and source archive
hashes are pinned in `deps.lock`.

| Component | Version | Licence | Text | Use |
|---|---|---|---|---|
| cpp-dsl-sdk (`cpp/dslvm.hpp`) | commit in `sdk.lock` | MIT | `licenses/cpp-dsl-sdk-LICENSE` | bytecode loader, verifier and VM |
| Eigen | 3.4.0 | MPL-2.0 (built with `EIGEN_MPL2_ONLY`; files unmodified, source at the pinned URL) | `licenses/eigen-COPYING.MPL2` | dense and sparse linear algebra |
| Boost (header-only libraries) | 1.86.0 | BSL-1.0 | `licenses/boost-LICENSE_1_0.txt` | special functions, distributions, quadrature, odeint, graph |
| pocketfft (C++ header) | commit 4d865cba | BSD-3-Clause | `licenses/pocketfft-LICENSE.md` | FFT |
| nanoflann | 1.6.1 | BSD-2-Clause | `licenses/nanoflann-COPYING` | kd-tree nearest neighbours |
| LBFGSpp | 0.3.0 | MIT | `licenses/LBFGSpp-LICENSE.md` | L-BFGS(-B) |
| Spectra | 1.0.1 | MPL-2.0 (files unmodified, source at the pinned URL) | `licenses/spectra-LICENSE` | symmetric sparse eigenvalues |
| HiGHS | 1.7.2 | MIT | `licenses/HiGHS-LICENSE.txt` | linear and mixed-integer programming |

Runtime libraries:
- libstdc++ and libgcc are linked statically under the GCC Runtime Library Exception.
- The Linux build also links **glibc statically**. glibc is LGPL-2.1-or-later. Before a public binary release, either
  build against musl (MIT), which is the plan in `../durable-memory/native-plan.md` §0, or ship the object files needed to
  relink (as lmcad does). See the open items in `../durable-memory/native-dsl.md`.

Ported algorithms and their sources (BSD-licensed scipy, statsmodels, scikit-learn and numpy code; public-domain QUADPACK,
MINPACK and netlib routines) are credited in comments in the files that contain them.
