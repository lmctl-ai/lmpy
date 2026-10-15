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
