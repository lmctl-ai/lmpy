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
   sections do not go unstable the way `ba` coefficients do. `UNSTABLE` is a defensive status; `marginal` (radius >= 1 - 1e-6) is the reachable warning.
5. **QHull degenerate and worst-case inputs.** Collinear/coplanar/duplicate/too-few points raise `QhullError` (or are pre-checked) and become `DEGENERATE`. Cyclic
   polytopes (moment-curve points) are the worst case: d=4 with n=5000 and d=6 with n=300 ran past the 50 s kill and d=5 with n=1000 crashed the output
   limit, which is why the point caps shrink with the dimension (README, docs/timings.md).
