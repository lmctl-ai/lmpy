# S8b: matrix functions, sparse algebra, special functions, root systems, time-series tests (first half)

**Scope decision.** The brief for S8b is too large for one coherent turn, so this commit series finishes a self-contained half and leaves the rest explicit:

| Done | Not done (still to implement in S8b-2) |
|---|---|
| `la_matrix_fn(_to)`, `la_sparse(_to)`, `special_fn(_to)`, `opt_root`, `sm_tsa_tools`; mpmath in the test image; recipes `oracle-tsa-tools`, `sparse-poisson`, `oracle-root` | `graph_paths(_to)` and `graph_flow(_to)` with networkx (hash-locked for both architectures, notices, edges as a frame, all-pairs n <= 5000 via artifact, negative-cycle status, k_shortest <= 10, min_cost_flow; recipe `graph-route-plan.lms`); the `stats_dist` long tail (per-family parameter tables, pdf/logpdf/cdf/logcdf/sf/logsf/ppf/isf/moments, levy_stable warning, R cross-oracle at 1e-12 on norm/t/gamma/beta); the optional `special-vs-mpmath` note |

The deferred items need a new dependency with notices (networkx) and a large per-family table; each is independent of what is here, and nothing in this half references them.

## Shape
* `src/s8b.hpp` + `py/s8b.py` hold `la_matrix_fn`, `la_sparse`, `special_fn` and their twins inside the S8a plumbing (`S8Extra`, `tensor_write`, the shared `s7_emit`/`s7_guard`/`s7_cap` helpers, the one
  `artifacts[name]` descriptor, `time_limit`). `opt_root` is a member of the S3 numeric family (`Context`, `Guard`, the expression AST, the shared nonfinite policy). `sm_tsa_tools` is an `sm_*`
  operation: `acf`/`pacf` call `sm_tsa_acf` itself (one implementation), `adfuller`/`kpss`/`ljung_box` are new; `statsmodels.stats.diagnostic` joined the trusted import list and patsy still never loads.
* Lazy imports per group (`scipy.linalg`, `scipy.sparse.linalg`, `scipy.special`; statsmodels only for `sm_tsa_tools`). The first call of several scipy routines imports helper modules the import guard
  blocks (pickle, urllib), so the worker runs each routine once on a tiny input in the trusted import phase (`la_matrix_fn`: expm/logm/sqrtm/inv/pinv; `la_sparse`: all six solvers; `special_fn`: a few).

## Caps (C++ before spawn, repeated in the worker where a corrupted request could break them)
| Quantity | Cap |
|---|---|
| dense matrix | 200000 numbers (n = 447 square), inline <= ~900 numbers (logm/sqrtm: twice the count, they can be complex) |
| sparse | rows/cols <= 100000, entries <= 200000, spsolve / sigma <= 2000 unknowns and n*stored_entries <= 20M, k <= 50 and below the dimension, maxiter <= 1e6, eigen ncv k+2..dim, svds k<ncv<min(shape) |
| special functions | 200000 values per argument, 34 functions, integer arguments checked (`n >= 0`, `k` in -1000..) |
| opt_root | 16 equations/unknowns, max_evals <= 2e6 (S3 budgets) |
| sm_tsa_tools | lags <= 100 entries in 1..100, maxlag <= 1000 |

## Honest statuses
Singular and ill-conditioned inputs, non-convergence of ARPACK/CG/GMRES/root solvers, special-function domain errors/poles and unattainable tolerances are all statuses with the evidence
(`condition_number`, `relative_residual`, `n_converged`, `first_index`, `residual_max`); a non-OK status never writes an artifact, and the solver's own success flags are verified against recomputed residuals.

## Oracles (independent of the code under test)
Closed forms: expm of a rotation generator (cos/sin), logm of the rotation (the generator), sqrtm of a constructed SPD matrix `Q diag Q^T` and of negative diagonals (complex split), adjugate/cofactor
inverses with exact rational arithmetic, rank-1 pseudo-inverse `v u^T / (|u|^2 |v|^2)`, the 1-D Poisson matrix (hand Thomas solver, analytic spectrum, analytic eigenvectors), triangular matrices
(eigenvalues = diagonal), block rotations (eigenvalues +-i r), diagonal SVD, an exact Gauss-Jordan solution of a 3x3 system, the circle/line system. Special functions: mpmath at 50 digits (round trips through
the mpmath forward function for the inverse functions). Time series: a hand OLS Dickey-Fuller regression with the published MacKinnon coefficients, a hand KPSS statistic with the published critical-value table and
its interpolation, the Ljung-Box formula with a closed-form even-degree chi-square tail, Durbin-Levinson on the dot-product ACF. Cross-tool results: `oracle-findings.md`.

## S8b-1 review corrections
Numerical certification, parameter conventions, all review-to-test mappings and resource evidence are in [s8b1-review-notes.md](s8b1-review-notes.md).
`opt_root` uses `start` for initial unknowns and returns `x`; S3 `opt_minimize`/`opt_least_squares` use `x0` and return `par`. Fixed coefficients are expression literals (there is no `params` argument in opt_root). Complex S8b outputs use `_re`/`_im` names plus `complex:true` consistently within this family; older S1 operations retain their published format.
