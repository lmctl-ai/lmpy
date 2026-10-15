# Cross-tool oracle findings (S5, statsmodels 0.14.6, aarch64, 2026-10-14)

Recipes: `examples/oracle-sm-regression.lms`, `oracle-sm-glm.lms`, `oracle-sm-arima.lms`, run for real through the host shim with
`PATH=pycli/bin:rcli/bin:/tmp/oracle-bin`. Legs: pycli vs R 4.3.3 golden constants (`scripts/s5-r-golden.py`), vs rcli, vs lmstats `statsrust`.

| Recipe | Legs | Largest absolute difference | Tolerance |
|---|---|---|---|
| OLS (40 rows, factor + interaction) | R, rcli fit_lm, lmstats regression | < 5e-13 (prints 0.000000000000 at 12 digits) | 1e-9 |
| Poisson GLM (log link) | R, rcli fit_glm, lmstats glm | 4.1e-8 (std_err[0], vs R and rcli); estimates and lmstats identical | 1e-6 |
| ARMA(1,1) with forecasts | R, rcli ts_arima, lmstats arima | params 3.9e-5, forecasts 3.6e-6 | 1e-3 |

All recipes exit 0 in real, `--check` and `--dry-run` modes.

## Discrepancies and conventions (reported, not hidden)

1. **GLM standard errors vs R/rcli differ by up to 4e-8.** Estimates agree to the printed 12 digits. R stops IRLS when the relative deviance
   change is below 1e-8 (`glm.control(epsilon)`); statsmodels iterates to its own `tol`, so the final working weights differ slightly.
   lmstats agrees with pycli exactly (same library). Not a defect; the tolerance of 1e-6 documents it.
2. **ARIMA parameters differ by 1e-5 from R and lmstats.** Different optimisers and likelihood evaluation (R: `arima` CSS-ML with BFGS;
   statsmodels: L-BFGS on the state-space likelihood). pycli and lmstats both use statsmodels SARIMAX yet still differ by 3e-5 in an
   ARMA coefficient, so lmstats evidently uses different starting values or optimiser settings; the cause was not investigated further.
3. **AIC/BIC.** statsmodels counts `k_params` = fitted coefficients (columns); R's `AIC(lm)` counts one more parameter (sigma). pycli reports the
   statsmodels value; for OLS, `aic_R = aic_sm + 2` and `bic_R = bic_sm + ln(n)`. Gamma GLM AIC also differs in how the dispersion is
   counted; this comparison was not measured.
4. **SARIMAX `trend="c"` is an intercept** in the state-space mean equation; R `arima` reports the process mean. Compare with `trend="n"` on centred data.
5. **ARIMA covariance.** pycli defaults to `opg` (statsmodels' default); R-like standard errors correspond to `cov_type="approx"` (observed
   information); the two differ by about 10% on the 120-point test series.
6. **rcli `fit_lm` limits.** It accepts numeric columns and plain additive formulas only: `x1:x2`, `x1*x2`, `I(...)` give
   `unsupported formula syntax` and string data give `data values must be numbers`. The recipe therefore passes dummy columns and the product
   column; the numbers then agree exactly. This is an rcli feature gap, not a numerical discrepancy.
7. **rcli must be started from a directory below `$HOME`** (otherwise INVALID_REQUEST); recipes should be run from there.
