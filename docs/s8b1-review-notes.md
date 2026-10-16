# S8b-1 review fixes

Reviews: Opus, Fable, Astra, Argon and Kimi, against ae17438. No ranked finding is deferred or declined.
The test-first commit df07798 ran against the original test-image binary: 23 methods, exit 1,
26 failing subtests and one error. Coverage-only assertions for Kimi's already-correct paths passed;
the scratch mutation audit below proves those assertions detect the requested incorrect variants.

| Findings | Correction / regression in tests/s8b_review.py (plus existing oracle tests) |
|---|---|
| Argon B1 | Normalise ADF/KPSS/Ljung-Box by max absolute data value; statistics/p-values invariant at 1e160 and 1e-165. ADF IC restored by `2*n_common*log(scale)` for AIC/BIC, with n_common=n-1-maxlag, not merely 2*log(scale). Narrow ValueError to NUMERICAL_FAILURE; validate every statistic, p, critical value and IC finite. |
| Argon B2 | logm uses the inverse's conditioning threshold (cond >= 1/eps => SINGULAR), plus exp-roundtrip certification for retained matrices. Rank-deficient 2x2 and 3x3 fixtures, nonsingular Jordan control. |
| Fable 1 | pinv verifies YAY=Y and symmetric projectors; discarded mass AYA-A is `truncation`, not failure for intentional rank reduction. Exact diagonal cutoff oracle. |
| Kimi 1; Fable 3 | Inverse backward residual `||AY-I||F/(||A||F||Y||F+sqrt(n))`, separate identity_residual and condition warning, scaled-matrix and Hilbert(10) fixtures. Dividing only by ||A|| would be dimensionally wrong and scale dependent; this implements the requested relative check with the standard backward-error denominator. |
| Opus 1 | Both SuperLU direct and sigma paths limited before spawn to n<=2000 AND n*stored_entries<=20M. Repeated worker check. Original n=20000/40000 heavy reproducers refused; random symmetric fill at the new boundary measured by scripts/measure-s8b1-review.py. Structured Poisson success alone did not justify the old cap. |
| Astra 1 | Normalise matrix/RHS, detect zero from entries, stable scaled norm, restore x and recompute residual from restored binary64 x. Tiny/ordinary/large RHS, all three linear tasks, read/write twins. |
| Opus 2 | Requested 10*rtol remains binding; no attainable-floor waiver. If solver info=0 but only the floor is met: TOLERANCE_UNATTAINABLE, tolerance_met:false. Hilbert(12) and strict Poisson(2000) tests; write twin creates no artifact. |
| Opus 3; Argon M2 | Check svds both defining equations AND vector orthogonality; reject bad triples. Matrix-norm scaling for eigen/SVD residuals, large-norm zero-mode oracle. |
| Fable 2; Astra 4; Argon M2 | Exact singular shift maps to SINGULAR in read twin and COMMAND_FAILED/SINGULAR in write twin; no artifact. ARPACK exceptions remain domain statuses. |
| Astra 3 | svds explicit k<ncv<min(shape), checked before spawn; reject ncv=dimension, accept k+1; separate eigen ncv rules retained. |
| Argon M3 | Sparse frame values use S8 exact-int64 conversion (INT64_NOT_EXACT beyond representable precision), and linear tasks omit inapplicable k. |
| Argon M4 | Integer/default Ljung-Box lags expand model_df+1..m; explicit lists each exceed model_df. Closed-form chi-square tail and exact lag-list assertions. |
| Argon M5 | Length-1 arrays/NPY/safetensors broadcast with scalars; read and write paths, artifact composition and length-1 reducer. |
| Argon m1 | Output-sized nonfinite masks, NaN-prioritised first_index with matching first_args/message, separate n_nan/n_inf, reduction n_bad<=1. |
| Argon m2; Kimi 2 | Nullable alpha default; acf/pacf resolve .05, all other tests reject explicitly supplied alpha before spawn. Strictly increasing unique Ljung-Box lags. |
| Argon m3 | Explicit unused spsolve rtol/maxiter rejected before spawn. |
| Argon M1; Astra 2 | Bracket path never evaluates start. Nonfinite endpoint => validation NONFINITE_BRACKET. Endpoint root iterations=0. Conflicting explicit method and system maxiter rejected before spawn; systems use max_evals. |
| Astra 5 | ADF reports/passes effective ceil Schwert maximum clipped by sample/trend terms; fixtures n=12 (4), n=160 (14). |
| Kimi 3 | sm_tsa_tools short ACF/PACF => TOO_SHORT. Legacy sm_tsa_acf statuses retained. |
| Kimi 4 | Independent inverse residual formula, captured PCG64 start vector, signed complex eigenpairs and Av=wv, SVD orthogonality/equations, hand adjusted ACF. All five scratch variants yield expected test exit 1. |
| Fable 4 | ADF CDF discrepancy traced independently with mpmath65; corrected docs/oracle-findings.md and recipe comment. README documents R tseries/urca conventions; xtol schema states residual_tol remains binding. |

The existing direct Ljung-Box oracle additionally caught an alias in the newly added finiteness collection:
copy the statistic list before appending other diagnostics, preserving the public result's length.
The original Poisson artifact oracle accepted a measured ~6e-11 residual at rtol=1e-12. It now asserts
TOLERANCE_UNATTAINABLE/no artifact at that tolerance, then verifies the artifact against the hand Thomas solution at attainable rtol=1e-10.

## Cross-slice conventions
opt_root keeps S3 named `start` unknowns (no separate `params`; constants are expression literals).
S8b complex results use `*_re`/`*_im` with `complex:true`; older S1 operation formats are unchanged.
No callable/evaluation escape hatch was added. No dependencies or wheel locks changed.
The inverse condition warning communicates forward accuracy limitations; a small backward residual does not promise accurate forward digits.

## Evidence
- Red reproducer command: `docker run --rm --pull=never --network=none -v $PWD:/src/pycli -w /src/pycli pycli:test python tests/s8b_review.py /build-pycli/bin/pycli_dev` (original image): exit 1, 26 failures + 1 error.
- Scratch mutation runner: `scripts/check-s8b-review-mutations.py`; raw results: `measurement-s8b1-mutations.json` (runner exit 0, five expected mutant test exits 1). No mutant enters production.
- Fresh production-container resource probe: `scripts/measure-s8b1-review.py`; final results and acceptance are recorded in docs/timings.md and acceptance-s8b1-review.json.
- Runtime timing comparisons use sequential fresh shim calls on an idle host; original S8b cap figures below are historical and do not establish the new arbitrary-fill limits.
