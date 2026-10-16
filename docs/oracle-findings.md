# Cross-tool oracle findings (S5, statsmodels 0.14.6, aarch64, 2026-10-14)

Recipes: `examples/oracle-sm-regression.lms`, `oracle-sm-glm.lms`, `oracle-sm-arima.lms`, run for real through the host shim with
`PATH=pycli/bin:rcli/bin:/tmp/oracle-bin`. Legs: pycli vs R 4.3.3 golden constants (`scripts/s5-r-golden.py`), vs rcli, vs lmstats `statsrust`.

| Recipe | Legs | Largest absolute difference | Tolerance |
|---|---|---|---|
| OLS (40 rows, factor + interaction) | R, rcli fit_lm, lmstats regression | < 5e-13 (prints 0.000000000000 at 12 digits) | 1e-9 |
| Poisson GLM (log link) | R, rcli fit_glm, lmstats glm | 4.1e-8 (std_err[0], vs R and rcli); estimates and lmstats identical | 1e-6 |
| ARMA(1,1) with forecasts | R, rcli ts_arima, lmstats arima | params 1.5e-5 vs R, 6.2e-6 vs lmstats (was 3.9e-5) | 1e-3 |

All recipes exit 0 in real, `--check` and `--dry-run` modes.

## Discrepancies and conventions (reported, not hidden)

1. **GLM standard errors vs R/rcli differ by up to 4e-8.** Estimates agree to the printed 12 digits. R stops IRLS when the relative deviance
   change is below 1e-8 (`glm.control(epsilon)`); statsmodels iterates to its own `tol`, so the final working weights differ slightly.
   lmstats agrees with pycli exactly (same library). Not a defect; the tolerance of 1e-6 documents it.
2. **ARIMA parameters: the former 3e-5 gap to lmstats/R was optimizer termination, now traced and reduced.** On the 120-point ARMA(1,1):

   | Point | ar.L1 | ma.L1 | sigma2 | log-likelihood |
   |---|---|---|---|---|
   | pycli before (lbfgs only) | 0.5745595 | 0.0544121 | 1.0650405 | -174.286540834 |
   | R `arima` (fixture) | 0.5745259 | 0.0544513 | 1.0650504 | -174.286540823 |
   | lmstats `statsrust` | 0.5745361 | 0.0544429 | 1.0650461 | -174.286540820 |
   | pycli now (lbfgs + Nelder-Mead polish) | 0.5745377 | 0.0544368 | 1.0650504 | -174.286540819 |

   pycli evaluates lmstats' point to the same likelihood (7e-11 apart, via `start_params` with `maxiter: 1`), so the models are identical; the
   optimizers stop at different points of a ridge where 1.4e-8 in llf spans 3e-5 in the MA coefficient. Cause: statsmodels' default score is a
   finite-difference gradient, so L-BFGS stalls (tightening `pgtol`/`factr` does not help: llf stays -174.2865408337). The analytic `harvey`
   score converges but is 7-40x slower and fails with missing data. pycli therefore polishes the converged L-BFGS point with a short
   Nelder-Mead run (models with at most 8 parameters, kept only if llf rises; `polished` in the result). Measured now: pycli vs lmstats
   1.6e-6 / 6.2e-6 / 4.3e-6 (ar, ma, sigma2), vs R 1.2e-5 / 1.5e-5 / 2e-8; R's own point has the lowest llf of the three, so the remaining
   gap to R is R's termination, not pycli's.

3. **AIC/BIC.** statsmodels counts `k_params` = fitted coefficients (columns); R's `AIC(lm)` counts one more parameter (sigma). pycli reports the
   statsmodels value; for OLS, `aic_R = aic_sm + 2` and `bic_R = bic_sm + ln(n)`. **Gamma GLM (measured, R fixture):** pycli `llf` -26.606969
   is statsmodels' log-likelihood at the Pearson scale, R's `logLik` -26.596440 uses deviance/n (recomputing the gamma log-density at
   deviance/n reproduces R to 6 digits); pycli `aic` 57.2139 (k = 2) vs R 59.1929 = -2 logLik + 2*3 (R counts the dispersion). Gaussian GLM
   `llf`/`aic` equal `sm_ols`.
8. **GLM inference follows `summary.glm`** (found in review): t with `df_resid` for gaussian/gamma, z for poisson/binomial. Before the fix
   gamma p-values were z-based (9.2e-5 vs R's 3.7e-4 for the first coefficient). Poisson `n_iter` is 4 where R reports `iter` 5 (R counts the
   final evaluation).
4. **SARIMAX `trend="c"` is an intercept** in the state-space mean equation; R `arima` reports the process mean. Compare with `trend="n"` on centred data.
5. **ARIMA covariance.** pycli defaults to `opg` (statsmodels' default); R-like standard errors correspond to `cov_type="approx"` (observed
   information); the two differ by about 10% on the 120-point test series.
6. **rcli `fit_lm` limits.** It accepts numeric columns and plain additive formulas only: `x1:x2`, `x1*x2`, `I(...)` give
   `unsupported formula syntax` and string data give `data values must be numbers`. The recipe therefore passes dummy columns and the product
   column; the numbers then agree exactly. This is an rcli feature gap, not a numerical discrepancy.
7. **rcli must be started from a directory below `$HOME`** (otherwise INVALID_REQUEST); recipes should be run from there.

## S6 ML (aarch64)

`oracle-ml-linear.lms`: hand optimum intercept=1, slope=2; pycli `[1,2]`, rcli `fit_lm` `[1,2]`, largest absolute difference 0 on the printed doubles (tolerance 1e-9).
`oracle-kmeans.lms`: separated six-row blobs; pycli vs the hand partition **ARI=1**, pycli vs rcli Lloyd **ARI=1** after relabeling. sklearn labels are zero-based and R labels one-based; comparing pair co-membership removes that representational difference. Same integer seed does not imply identical RNG algorithms or initial centers across R/sklearn; the fixture has a unique separated partition.
`ml-classify-cv.lms`: four-fold stratified logistic accuracy **0.975**, mean log_loss **0.20075185634341353**. Saved/reloaded probabilities for x=-2,0,2: positive class `[0.08541919955327777,0.4706034083858738,0.8943013481682253]`.
The optional lmstats leg prints its setup hint (no prebuilt `statsrust` on PATH); the requested lmda checkout/prebuilt binary is absent, so no lmda build or numerical parity claim is made. All 13 recipes, including the older S0-S5 recipes, were run through the host shim; exact exit codes and stdout are in `measurement-s6-recipes.json`.

## S7 (interp / signal / spatial; scipy 1.16.2, aarch64, 2026-10-15)

Recipes: `examples/oracle-interp.lms` (rcli `interp_approx`, `interp_spline`; R 4.3.3) and `examples/oracle-spectrum.lms` (lmda `series_spectrum`, built from
lmda commit 36e5285, `cargo build --release --offline`). Both exit 0 in real, `--check` and `--dry-run` modes.

| Comparison | Largest difference | Tolerance |
|---|---|---|
| not-a-knot cubic vs the cubic polynomial it samples (closed form) | 1.1e-13 | 1e-10 |
| linear vs R `approx(method = "linear")` | 4.4e-16 | 1e-9 |
| cubic natural vs R `splinefun(method = "natural")` | 6.7e-16 | 1e-9 |
| cubic periodic vs R `splinefun(method = "periodic")` | 4.4e-16 | 1e-9 |
| periodogram: Parseval, tone powers vs closed forms (N = 512) | 1.4e-15 relative | 1e-9 |
| periodogram (boxcar, fs = 1, no detrend) vs lmda `series_spectrum`, 257 bins | < 1e-18 relative to the largest bin | 1e-9 |

1. **R `fmm` is not not-a-knot, and R `monoH.FC` is not PCHIP.** On the recipe's 9 nodes R's default `fmm` spline differs from pycli's not-a-knot cubic by up to
   0.0596, and `monoH.FC` from `pchip` by up to 0.344. These are different interpolants (different end conditions; Fritsch-Carlson vs Fritsch-Butland slope
   formulas), both correct for their definitions; the recipe prints the differences as information and asserts only the identical methods. Neither tool is wrong.
2. **lmda `series_spectrum` conventions.** Frequencies are cycles per sample, power is the one-sided density with the DC bin undoubled, which is exactly
   `scipy.signal.periodogram(x, fs=1, window="boxcar", detrend=False)`; lmda's default `detrend: mean` corresponds to `detrend: constant`. The recipe uses `detrend: none`.
3. **scipy `make_smoothing_spline` silently loses accuracy for a huge penalty** (found while writing the least-squares-line oracle; not a pycli-vs-other-tool
   disagreement). On 15 unit-scale nodes the maximum deviation of the fit from the exact large-penalty limit (the least-squares line) is 1.9e-7 at lam = 1e6,
   3.2e-6 at 1e8, 5.1e-4 at 1e10, **0.18 at 1e12 and 65.6 at 1e14**, with no warning. pycli therefore refuses `lam / h^3 > 1e11` (h = mean node spacing) with
   `BAD_PARAM` before any work. The fit stays correct below the cap to ~1e-5 of the data scale.
4. **No unstable SOS design was found.** Seven extreme designs (butter order 20 bandpass with a 2e-4 wide band, cheby1 order 20 at wn = 5e-4, an elliptic
   bandstop 1e-5 wide, butter at wn = 1e-6, cheby2 band 1e-5 wide, ...) all produced finite sections with largest pole radius <= 1 - 2.4e-9: second-order
   sections are more robust than `ba`, but finite-precision designs can round a pole to one: `UNSTABLE` is reachable. Stability now uses extended-precision quadratics of the rounded SOS, avoiding companion-matrix root error. `marginal` flags radius >= 1 - 1e-6.
5. **QHull degenerate and worst-case inputs.** Collinear/coplanar/duplicate/too-few points raise `QhullError` (or are pre-checked) and become `DEGENERATE`. Cyclic
   polytopes (moment-curve points) are the worst case: d=4 with n=5000 and d=6 with n=300 ran past the 50 s kill and d=5 with n=1000 crashed the output
   limit, which is why the point caps shrink with the dimension (README, docs/timings.md).

## S8a (table/array bridge, joins, rolling windows, metrics, seeded sampling; lmda commit 36e5285, R 4.3.3, 2026-10-15)

Recipes: `oracle-metrics.lms`, `oracle-frame.lms` (lmda), `oracle-rvs.lms` (rcli `dist_quantile`). All exit 0 in real, `--check` and `--dry-run` modes. lmda was built once with
`cargo build --release --offline` (15 minutes on the shared host).

| Comparison | Result | Tolerance |
|---|---|---|
| accuracy, balanced accuracy, precision, recall, F1, ROC AUC, Brier vs the hand values (TP 3, FP 2, FN 1, TN 2) and vs lmda `eval_classify` / `eval_roc` | difference < 1e-15 (all ratios of small integers) | 1e-12 |
| tied scores: ROC AUC 0.75 and Brier 0.1975 vs lmda | identical | 1e-12 |
| outer join with duplicate keys on both sides vs lmda `table_join` | the same 6 rows in the same order | exact |
| 3-row rolling mean, sum, std, min, max, median over 9 rows vs lmda `series_rolling` | largest difference 1e-15; the same two leading missing values | 1e-12 |
| seeded samples (norm, gamma, beta, chi2, t; 1000 draws each) vs R's quantile function at p = 0.1 ... 0.9 | largest deviation 2.85 binomial standard errors (limit 4.5) | 4.5 SE |

1. **No discrepancy was found between pycli, lmda and the hand values.** Conventions that coincide by design and were checked: NA keys never match in either join
   (a left row with an NA key stays unmatched; an unmatched right row with an NA key is appended), duplicate-key matches come in right-row order, the outer join appends
   the unmatched right rows last, rolling std uses ddof = 1, balanced accuracy is the mean recall of the classes present in the truth, and tied scores count 1/2.
2. **pandas differs on NA keys, pycli deliberately does not follow it.** `pandas.merge` matches NaN keys with each other; pycli (like lmda and SQL) never matches a missing key.
   The test oracle is a nested-loop join, not pandas, so the choice is explicit.
3. **lmda `series_rolling` skips blank CSV rows** instead of treating them as missing observations; the recipe uses complete data, so the comparison is unaffected. pycli keeps a
   masked missing value as a row and skips it inside the window.
4. **PCG64 and R's Mersenne-Twister streams differ**, so samples are never compared value by value; the recipe compares each family's empirical CDF with R's quantile
   function instead. rcli's `rng_draw` has no PCG64 option.

## S8b (matrix functions, sparse algebra, special functions, root systems, time-series tests; scipy 1.16.2 / statsmodels 0.14.6, 2026-10-16)

Recipes: `oracle-tsa-tools.lms` (rcli `ts_acf`, lmstats `acf_pacf`/`adf`/`kpss`/`ljung_box`), `oracle-root.lms` (pycli `num_root`, rcli `num_root`), `sparse-poisson.lms` (analytic spectrum, exact solution). All
exit 0 in real, `--check` and `--dry-run` modes. Special functions are checked against mpmath at 50 digits in the test image (`tests/s8b_ops.py`), not in a recipe.

| Comparison | Result | Tolerance |
|---|---|---|
| ACF lags 1-4 vs the dot-product formula, R `acf()` and lmstats `acf_pacf` | differences <= 1e-15 (printed) | 1e-9 |
| PACF (ywadjusted) vs lmstats | <= 1e-15 | 1e-9 |
| Ljung-Box Q(2), Q(4) and p-values vs the formula and a closed-form chi-square tail | relative 1e-15 | 1e-9 |
| lmstats `ljung_box` Q and p for lags 1-4, and all other statistics and correlations of the recipe | largest difference 4.4e-14 | 1e-9 |
| ADF statistic, used lag (11), nobs (108) vs lmstats | statistic equal to 12 decimals; same lag and nobs | 1e-9 |
| ADF MacKinnon p-value vs lmstats (0.014945880 vs 0.014945879) | **1.4e-9** | 1e-8 |
| KPSS statistic, p-value, lags (4) vs lmstats | statistic equal to 12 decimals, p and lags identical | 1e-9 |
| root of x^3 - 2x - 5: pycli brentq, `num_root`, R `uniroot` vs 2.0945514815423265 | <= 5e-16 | 1e-12 |
| circle/line system (hybr and lm) vs sqrt 2 | <= 1e-12 | 1e-8 |
| 1-D Poisson (n = 120): spsolve / cg / gmres vs x(1-x); eigsh (SA and LA) vs 4 sin^2(j pi / (2(n+1))) | below the 12 printed decimals | 1e-9 |

1. **ADF p-value: 1.4e-9 between pycli (statsmodels) and lmstats.** Same statistic, same lag, same critical values, but the MacKinnon polynomial is evaluated with a different normal CDF; the recipe tolerance for this one
   quantity is 1e-8. The approximation error is on **lmstats' normal-CDF side**. The polynomial argument is z = -2.171521681210366; independent mpmath at 65 digits gives Phi(z) = 0.01494588002672663, matching pycli/statsmodels 0.014945880026727; lmstats gives 0.014945878644889. The absolute error is 1.38184e-9. It is small compared with the response-surface uncertainty, but is not an equally accurate CDF. The review regression recomputes the polynomial and mpmath tail independently.
2. **lmstats `acf_pacf` defaults to the `ywadjusted` PACF**, pycli to `ywm` (R's convention); the recipe sets the method on both sides. lmstats rejects `ywm`.
3. **Convergence verdicts the solvers do not give by themselves.** `scipy.optimize.root` (hybr) with its default step tolerance 1.49e-8 stops with max|f| = 1.5e-8 on x_i^2 = i (16 unknowns) and reports success; pycli
   recomputes the residual and, with the default `residual_tol` 1e-8, would have called that NO_CONVERGENCE, so the default `xtol` is 1e-12 (the same system then reaches 2e-13). `x^2 + 1 = 0` makes hybr and lm stop
   at the minimum of the residual norm; both are reported as NO_CONVERGENCE with the residual 1.0. For cg, the residual a double-precision iteration reaches is bounded by about eps ||A|| ||x|| / ||b||: on the
   2000-unknown Poisson matrix `rtol=1e-12` produced a true relative residual of 6e-11 while the iteration reported convergence, this is now TOLERANCE_UNATTAINABLE, never OK. The artifact oracle requests attainable rtol=1e-10; a separate regression asserts honesty for unattainable tolerances.
4. **ARPACK is slow, not wrong, on the clustered end of a Poisson spectrum**: eigsh SA with k = 5 on n = 66000 (and cg/gmres at n = 66000) exhaust the default 10 s budget (TIME_LIMIT); shift-invert with sigma = 0 returns the same
   eigenvalues in 0.9 s in the original structured probe. S8b-1 review caps now reject that large shift-invert factorisation before spawn; use shift-invert only within the new 2000/20M factorisation limits.

## S8b-2: graph operations and the stats_dist long tail

Oracles are independent of the implementation: tests/oracle/s8b2.py has pure-Python Dijkstra, Bellman-Ford, BFS, Floyd-Warshall, brute-force k shortest simple paths, Edmonds-Karp with the residual min cut and successive shortest paths
(no networkx or scipy), run on textbook graphs (CLRS shortest paths and the CLRS flow network, max flow 23) and seeded fuzz; distribution functions use hand formulas and mpmath (40 digits, test image only). Recipes `graph-route-plan.lms`
(hand values), `oracle-dist.lms` (R through rcli) and `distributions-tour.lms` (closed forms) exit 0 in real, `--check` and `--dry-run` modes.

| Comparison | Result | Tolerance |
|---|---|---|
| pdf, log pdf, cdf, sf of norm, t, chi2, gamma, beta, binom, poisson vs R (`dist_density`, `dist_probability`; 217 values per run, recipe) | largest relative difference 6e-15 | 1e-12 |
| continuous ppf (norm, t, chi2, gamma, beta) vs R `q*` | largest 3.4e-12 (t, df 5, p = 0.001 or 0.999) | 1e-10 |
| discrete ppf (binom, poisson) vs R | identical integers | exact |
| genextreme, genpareto, skewnorm, johnsonsu, vonmises, truncnorm pdf/cdf vs hand formulas / mpmath quadrature | relative 1e-11 (vonmises cdf tail: absolute 1e-12) | 1e-9..1e-12 |
| nct cdf/pdf vs mpmath quadrature of the chi-square mixture; ncf cdf vs the Poisson mixture of incomplete betas | relative 1e-8 / 1e-7 | 1e-8 / 1e-7 |
| zipf, betabinom mass vs mpmath (zeta, beta function) | 1e-12 | 1e-12 |
| levy_stable vs Cauchy (alpha 1), normal (alpha 2) and Levy (alpha 1/2, beta 1) closed forms | 1e-6 .. 1e-5 | 1e-6 .. 1e-5 |
| rvs KS statistics (n = 20000, bound 1.95/sqrt(n)) and 5-sigma means / 5 % variances | all pass | see tests |
| graph oracles: seeded fuzz loops of 12 to 40 random graphs per property (paths with 3 methods, all pairs incl. zero-weight edges, max-flow = min-cut, min-cost flow, k shortest vs brute force), textbook graphs | all equal | exact (integers) |

1. **scipy and R parameterise several families differently.** gamma: scipy `scale` = 1 / R `rate`; norm: `loc`/`scale` vs `mean`/`sd`; genextreme: scipy `c` = -xi of the usual GEV shape xi (R `evd` uses xi); truncnorm: scipy's `a`, `b` are
   **standardised** bounds ((bound - loc) / scale), not bounds on the variable; nbinom counts failures before the `n`th success (R `size`, `prob`) with support starting at 0; geom starts at 1 (R `rgeom` starts at 0); lognorm in pycli is
   `mu`, `sigma` of the underlying normal, not scipy's `s`, `scale`; levy_stable uses scipy's default S1 parameterisation (other packages default to different stable parameterisations). Each is documented in the `dist` parameter text.
2. **vonmises `loc` wraps instead of re-centring the support.** scipy returns variates in [-pi, pi] whatever `loc` is: with `loc = 0.3`, `kappa = 2` the linear sample mean is 0.28, not 0.3 (the circular mean direction is 0.3). The tests compare the
   circular mean, the schema says so, and the zero-loc case is exact.
3. **scipy's discrete families without a closed-form cdf sum the mass function.** zipf and betabinom `cdf`/`ppf` are O(x) / O(n) per point; with `x` up to 1e9 or `n` up to 1e9 the worker ran out of memory or reached the 50 s hard kill in the first measurement
   (betabinom n = 1e9: crash after 17 s; zipf a = 1.0001, x to 1e9: killed at 51.7 s). Pre-spawn caps now: betabinom n <= 5000, zipf cdf/sf |x| <= 1e5, zipf quantile range 1e5 (`OUT_OF_RANGE` beyond it).
4. **scipy's generic discrete ppf can raise** (`zipf.ppf` with a < 2: "updating stopped, endless loop"); pycli maps this to `NUMERICAL_FAILURE` instead of an internal error. Heavy-tailed zipf quantiles beyond the cap are `OUT_OF_RANGE`.
5. **t ppf differs from R by 3.4e-12 relative at the tails**; both are iterative. cdf and pdf agree to 1e-14. The recipe bound for continuous ppf is therefore 1e-10, the pdf/cdf/sf bound 1e-12.
6. **levy_stable pdf/cdf are numerical integrations**: slow (about 1.5 ms per point for pdf/cdf and 17 ms for ppf in the measured runs) and good to roughly 1e-6 relative at best; every call carries a warning and at most 200 points are accepted. vonmises cdf/ppf cost about 40 ms per point (scipy integrates), so it shares the 200-point limit.
7. **Moments**: scipy reports `inf` or `nan` for undefined moments (t with df <= 1, genpareto with c >= 1/2 for the variance); pycli reports them as `NA` and lists them in `moments_undefined` rather than as numbers. Moments of genpareto c = 0.25 have
   no finite excess kurtosis (needs c < 1/4), which the test originally missed.
8. **networkx**: parallel edges and self loops are normalised before the algorithms see them (lightest edge kept for paths, capacities summed for flows, self loops dropped); `network_simplex` does not accept parallel edges, so min-cost flow rejects them as
   `DUPLICATE_EDGE` instead of silently merging costs. networkx's `shortest_simple_paths` raises on negative weights; `k_shortest` therefore requires non-negative weights (`BAD_PARAM`). `scipy.sparse.csgraph` keeps stored zeros of a sparse matrix as zero-length edges (a dense zero would mean no edge); the all-pairs code builds a sparse
   matrix, and a fuzz test with weights in 0..4 checks zero-weight ties against Floyd-Warshall.

