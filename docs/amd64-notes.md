# amd64 notes: the first cross-architecture evidence and the tight-tolerance audit

Development and every measurement in this repository happen on aarch64 (docs/decisions.md). The first GitHub Actions run
(`38901596246`, 2026-10-16) was the first time the tree ran on amd64 (`ubuntu-24.04`, the same hash-locked wheels from
`docker/requirements-amd64.lock`): arm64 passed everything; amd64 built both images and passed 36 of 38 CTest groups. The two
failures, their diagnosis and the audit they triggered are recorded here. Determinism tiers (docs/decisions.md): bit-exact only within
one image on one CPU family; tolerance tiers across architectures, default <= 1e-8 unless a test justifies tighter.

## The two amd64 failures

| Group / test | Observation on amd64 | Diagnosis | Class | Disposition |
|---|---|---|---|---|
| `s3_ops` `LeastSquares.test_sigma_weights_and_absolute_sigma` | fitted intercept 5.8e-10 from the normal-equations solution (relative 1.01e-8 > rtol 1e-8); aarch64 2.6e-11 | `opt_least_squares` (scipy `trf`, `tol` 1e-15) stops where the weighted cost no longer decreases in binary64. The cost is known to about eps * chi2; along the weakest curvature direction of X'WX (lambda_min = 87.55, chi2 = 6.04) that is a parameter error of sqrt(eps * chi2 / lambda_min) = 3.9e-9 absolute. Both architectures stopped inside that floor; a relative 1e-8 on an intercept of 0.057 asked for 5.7e-10 absolute, below what the curvature can guarantee. amd64's different OpenBLAS kernels / FMA contraction change the iterate path, not the answer's quality (chi2 excess 2.5e-20 relative). | (i) test tighter than the problem's conditioning | tolerance replaced by `atol = 4 * sqrt(eps * chi2 / lambda_min)` computed in the test from the fixture (derivation in the test comment); a scratch worker that ignores `sigma` lands 1e-2 away and still fails the test (`~/pycli-ci/mutant-run.py` harness, see below) |
| `s7_review` `Review.test_near_unit_poles_use_actual_quadratics` | `max_pole_radius` 1.000000000039027 vs the 80-digit Decimal reference 1.0 (3.9e-11); aarch64 agrees to 1e-14 | The worker computed the pole moduli of the rounded second-order sections in `numpy.longdouble`: 113-bit quad precision on aarch64 (software), **64-bit x87 extended on amd64**. The discriminant b^2 - 4ac of a near-unit section (b ~ -2, 4ac ~ 4) cancels about ten decimal digits, which a 64-bit mantissa cannot absorb; the 14-place test was only valid on the platform whose longdouble is quad. The stability verdict itself was unaffected (3.9e-11 against the 1e-6 `marginal` band), but "extended precision" was a platform-dependent promise. | (ii) platform-dependent numerics in the worker (not a wrong verdict, a non-portable precision claim) | fixed: `s7_pole_radius` evaluates the stable quadratic form in Python `decimal` at 50 digits (platform-independent; at most 10 sections per design, cost negligible) and rounds once at the end; the 14-place test stays; a scratch mutant at 16 digits fails it by 2.5e-11, proving the test still catches the original class of error |

Harness used for the mutation checks (not shipped; the pattern is `scripts/check-s8b2-review-mutations.py`): assemble the ten `py/*.py`
files into one worker with one string replaced, run the selected test against `pycli_dev` with `PYCLI_PYTHON` pointing at a wrapper that
executes that worker. Results: fixed worker OK; `ctx.prec = 16` mutant FAIL (2.46e-11); sigma-ignoring mutant FAIL.

## Audit of assertions tighter than 1e-10 on floating-point results

Method: grep for `places >= 11`, `delta` / `rtol` / `atol` below 1e-10, exact `==` on float lists and exact float literals in every
`tests/*.py` (199 lines), classified by what the compared quantity depends on. Evidence beyond reasoning: all groups except the two above
passed on amd64 in run 38901596246, so every other tight assertion already has one amd64 data point.

| Class | What it depends on | Verdict | Examples |
|---|---|---|---|
| A. exact integers, rationals and closed forms reached by a fixed sequence of IEEE operations | no BLAS / LAPACK; sums in a fixed order, divisions, `sqrt` of exact values | legitimately exact (bit-identical on every IEEE-754 platform) | `s8_ops` metrics `places=15` (ratios of small integers, hand AUC), `s8b2_review` discrete quantiles `==`, `s1_arrays` norm `sqrt(14)`, `s4_review` final state of `y' = 1`, `s1_la` LU of a 2x2 integer matrix (`1/3`, `2/3`), `s7_ops` zero cross-spectra `atol 1e-14`, `s8b2_graph` lengths and `tolerance` (products of decimal literals) |
| B. glue oracles: pycli compared with numpy / scipy computed **in the same image, in the test process** | the same OpenBLAS / LAPACK / libm build as the worker | legitimately bit-exact within one image (the decisions.md tier); on amd64 both sides run on amd64 | `s0` `la_solve` vs the in-process oracle (`rtol 1e-13`), `s1_la` `lstsq` / `svdvals` / `cholesky` vs `scipy.linalg` (`1e-12`), `s6_ops` ridge / NB / LDA / PCA parameters vs numpy (`1e-10..1e-12`), `s6_review` artifact halves `1e-13`, `s8_ops` `frame_to_array` fit `1e-12`, `s0` `stats_dist` fixtures `==` (in-process scipy) |
| C. closed forms reached **through LAPACK** (SVD / eigen / solve of tiny well-conditioned matrices) | LAPACK is backward stable: error a few eps times the norm; FMA and kernel choice move the last bit or two | tight tolerances of 1e-15 / 1e-14 were not justified by conditioning; relaxed to 1e-13 (still 100x tighter than the 1e-8 tier and far below any wrong-formula error) | `s1_la`: SVD `s = [3, 2]` 1e-15 -> 1e-13; eigh `w = [1, 3]`, eigenvector magnitudes, rotation eigenvalues `+-i` 1e-14 -> 1e-13; `w_re` of the rotation `atol` 1e-15 -> 1e-14. Kept: solve `[2, 3, -1]` 1e-12, det `1/2160` 1e-15 absolute on a 3x3 (error ~ eps * |det| * n), `cond / 1e8` 1e-12, norms (fixed-order sums) |
| D. special functions and distribution functions against mpmath or closed forms at 1e-12 .. 1e-14 | scipy's C / C++ code (cephes, Boost, xsf): no BLAS; FMA contraction can change the last bit (1e-16) | legitimate at 1e-12 .. 1e-13 (passed on amd64); the three 1e-14 / 1e-15 relative checks are kept because they passed on amd64 and bound well-conditioned elementary evaluations (`t` cdf df = 1e300 = normal limit, `norm` cdf at 1.96, special functions at ordinary arguments): an amd64 failure there would be a one-ulp libm difference and would be relaxed to 1e-14 with that note, not hidden | `review_numeric` Cauchy cdf `5e-16` (atan of small integers, glibc libm, identical on both arches in the run), `s8b_review` special functions `rtol 1e-14`, `s8b2_dist` moments `places 12` |
| E. exact float output equality | the printed shortest round-trip decimal of a library value | one case was not legitimate: `review_numeric` compared `norm.ppf` output with three literal doubles (cephes `ndtri`); changed to `delta 1e-15` (a few ulps) so a last-bit FMA difference is not a failure while any real change still is. The other exact comparisons are class A or B | `review_numeric` ndtri literals |
| F. iterative solvers (optimisation, least squares, ODE, roots, iterative sparse) | the stopping point depends on the rounding path | tolerances derived from conditioning or the solver's own tolerance: `s3_ops` least squares (this note), roots `1e-12` on bracketing methods (interval width, platform-independent), quadrature `1e-12 .. 1e-15` on smooth integrands (QUADPACK error estimate is far below), ODE `2e-8 .. 2e-9` (solver tolerances), sparse `TOLERANCE_UNATTAINABLE` honesty; `s3_ops` `fun` of a BFGS run `delta 1e-12` is against the in-process scipy optimum on the same image (class B) | |

Nothing else was changed: the arm64 and amd64 runs both exercise the relaxed assertions, and every relaxation is tighter than the
1e-8 default tier by at least four orders of magnitude.

## Time-limit fixture (s4_review)

`opt_linprog` with `time_limit: 0.001` on a 60-variable binary knapsack answers `TIME_LIMIT` by one of two documented paths: HiGHS
stops with model status 13, or - under host load - the 1 ms budget is already spent before HiGHS starts and the worker answers
`'wall budget exhausted before HiGHS'` without HiGHS fields (py/s4.py). The test accepted only the first path and failed once on a
loaded aarch64 host (docs/timings.md); it now asserts `TIME_LIMIT`, no primal, and that the answer came by one of the two paths, which
proves the limit reached the solver layer on any machine.

## CI reporting

`.github/workflows/ci.yml` runs every gate after the test-image build with `if: ${{ !cancelled() && steps.<dependency>.outcome == 'success' }}`,
so a failing CTest no longer hides the notices, runtime-build, schema / provenance and recipe results of the same architecture; a gate
summary table is appended to the job summary, and the job still fails when any gate failed.

## Status

See README "What pycli is not" for the current amd64 statement; it is updated from the CI result, never from local reasoning.
