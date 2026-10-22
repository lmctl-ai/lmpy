# Python recipes and their lmpy versions

Each recipe is a pair:

- `NAME.py` is a published Python recipe: an official documentation example or a tutorial from numpy, scipy, pandas,
  statsmodels, scikit-learn or networkx. Its header cites the source and lists any deviation, such as a fixed seed where
  the docs use an unseeded generator, or a bundled dataset where the docs download one. It prints its answers as one
  JSON object.
- `NAME.lms` computes the same answers with lmpy in one `_dsl_` program. It records them with `out.put` under the same
  key names and pins the Python answers with `out.close` / `out.check`. It therefore passes on its own, with no Python
  installed, and `native/test.sh` runs it on all five CI platforms.

**Compatibility is measured, not assumed.** `check-parity.sh` runs both sides and compares every value the `.lms`
records with Python's value under the same key.

- **Python versions:** Python runs with pycli's pinned stack: numpy 2.3.3, scipy 1.16.2, pandas 2.3.3, statsmodels
  0.14.6, scikit-learn 1.7.2, networkx.
- **Tolerance:** the default is relative 1e-9 (absolute below 1e-12). A looser tolerance has to be declared in the
  `.lms` header (`// parity-rtol: X` or `// parity-rtol KEY: X`), and every one is justified next to it.

```sh
examples/python/check-parity.sh            # all recipes; needs native/.deps/oracle-venv (scripts in native/) and lmpy
examples/python/check-parity.sh scipy-quad # one recipe
LMPY=native/build/lmpy lmscript examples/python/scipy-quad.lms   # one recipe standalone, checked against its pins
```

## Recipes

| Recipe | Original | Values compared | Tolerance |
|---|---|---|---|
| numpy-lstsq-polyfit | numpy `linalg.lstsq` and `polyfit` reference examples | 13 | 1e-12 |
| scipy-curve-fit | `curve_fit` docs (decay fit, bounds) | 24 | 1e-8; covariance 5e-5 (1) |
| scipy-minimize-rosenbrock | optimize tutorial: Nelder-Mead, BFGS with gradient, SLSQP | 15 | 1e-9 |
| scipy-root-finding | `root_scalar`, `fsolve`, `root` (hybr, lm) | 8 | 1e-9 |
| scipy-quad | integrate tutorial and `quad`/`dblquad` docs | 22 | 1e-9 |
| scipy-linprog-milp | `linprog` and `milp` docs, tutorial LP, knapsack | 30 | 1e-9 |
| scipy-linalg-tutorial | linalg tutorial: inv, solve, det, norms, eig, svd, eigh, lu, expm | 124 | 1e-9 |
| scipy-sparse-laplacian | `LaplacianNd` spsolve/cg/eigsh, ARPACK tutorial | 44 | 1e-9 |
| scipy-butter-sosfilt | `butter`, `sosfiltfilt`, `freqz_sos` docs | 4246 | 1e-9 |
| scipy-find-peaks | `find_peaks`, `peak_prominences`, `peak_widths` docs | 93 | 1e-9 |
| scipy-fft-welch | fft tutorial, `periodogram` and `welch` docs | 2450 | 1e-9 |
| scipy-solve-ivp | `solve_ivp` docs: decay, cannon with events, Lotka-Volterra dense output | 632 | 1e-9 |
| scipy-interpolate | CubicSpline, Akima/makima, PCHIP, `make_smoothing_spline`, RBFInterpolator | 1708 | 1e-9; GCV fit 1e-4 (2) |
| scipy-stats | `scipy.stats` distributions, t tests, `pearsonr`, `linregress` formulas | 85 | 1e-9 |
| scipy-special | gamma, erf, jv, `jn_zeros`, ellipk, expit/logit, logsumexp | 97 | 1e-9 |
| pandas-groupby-merge | "10 minutes to pandas" and merging guide: groupby, merge with indicator, pivot_table, describe | 161 | 1e-9 |
| pandas-timeseries | time series guide: resample, rolling, shift, diff, pct_change | 699 | 1e-9 |
| statsmodels-ols-longley | OLS docs on Longley, plus the NIST StRD certified values | 55 | 1e-9 |
| statsmodels-glm | Poisson (RAND HIE), binomial logit (ANES 1996), gamma (Engel) | 75 | 1e-9 |
| statsmodels-arima-sunspots | ARMA sunspots notebook: AR(2)/AR(3), forecasts, acf/pacf, adfuller, kpss, Ljung-Box | 149 | 1e-9; AR fits 3e-3 to 1e-8 (3) |
| statsmodels-ols-categorical | formula docs: Grunfeld with C(firm), `anova_lm`, `t_test` | 91 | 1e-9 |
| networkx-paths-flows | weighted graph gallery, `maximum_flow`/`minimum_cut`, `min_cost_flow` | 43 | 1e-12 |
| sklearn-linear-diabetes | user guide OLS/Ridge/Lasso, plot_ols on diabetes | 52 | 1e-9 |
| sklearn-logistic | LogisticRegression reference and StandardScaler pipelines (iris, breast cancer) | 270 | converged fits 5e-5; sklearn defaults up to 1e-1 (4) |
| sklearn-cross-val | `cross_val_score`, StratifiedKFold/KFold with shuffling | 218 | 1e-9 |
| sklearn-pca-iris | PCA vs LDA gallery and PCA reference | 64 | 1e-9 |
| sklearn-kmeans | KMeans reference and the k-means assumptions gallery, iris | 6100 | 1e-9 (labels exact) |
| sklearn-metrics | confusion matrix, classification report, ROC AUC, log loss, regression metrics | 58 | 1e-9 |
| sklearn-nb-lda-knn | GaussianNB guide, feature-scaling gallery (wine), LDA/kNN | 301 | 1e-9 |

Why some tolerances are looser:

1. **curve_fit covariance.** scipy's covariance uses MINPACK's finite-difference Jacobian from the last iteration, while
   lmpy uses the Jacobian at the solution. scipy with the exact Jacobian agrees with lmpy to about 3e-10.
2. **Smoothing spline (GCV).** The cross-validated smoothing parameter is the minimum of a very flat function found to
   1e-5, so the GCV fit is compared through its residual sum of squares and norm. The fixed-lambda fit is compared
   point by point at 1e-9.
3. **AR fits.** Maximum likelihood by different optimisers: lmpy reaches a slightly higher log-likelihood on AR(2) than
   statsmodels' stopping point. acf/pacf/adfuller/kpss stay at 1e-9.
4. **Logistic regression.** sklearn's default lbfgs stops before the optimum: one reference example hits max_iter with
   a ConvergenceWarning, and the others stop at gradient tolerance 1e-4. lmpy solves to the optimum. Every recipe also
   records `*_conv_*` keys against sklearn re-run to convergence, and those agree at 5e-5. Predictions and accuracies
   are compared at 1e-9.

## Known API differences met while writing these

The recipes work around these without loosening anything:

- **pycli conventions kept by lmpy:**
  - `glm` defaults to t p-values for gaussian and gamma (statsmodels: z; pass `use_t false`).
  - `arima` trend "c" is the regression intercept (statsmodels `const` is the process mean).
- **Not in lmpy yet:**
  - **scipy:** `dblquad` (nested `quad`), `jn_zeros`, eigenvectors from `Mat.eig`, complex arrays, `linregress`,
    `fresnel`/`expn`, `sosfilt_zi`, analog filter design.
  - **statsmodels:** `anova_lm`, `t_test`/`f_test`, `ols` covariance matrix.
  - **pandas:** multi-column `pivot` index, time-based rolling windows.
  - **scikit-learn:** per-class metrics in one call, multiclass `log_loss`, SVC, LogisticRegressionCV.
- **JSON output:** infinity is written as `null` in lmpy's JSON result (JSON has no infinity).

Data files and their licences: [data/README.md](data/README.md).
