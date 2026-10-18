# Recipes

Every `.lms` file here runs through the host shim from a directory that contains `cli.schema.dev.json` and this `examples/` tree
(`export PATH=/path/to/pycli/bin:$PATH; lmscript examples/<name>.lms`; `lmscript --check` parses without running). `scripts/run-recipes.sh`
runs all of them in real, `--check` and `--dry-run` modes and is a CI gate on both architectures. Recipes that write files create them in
the current directory and remove them before exiting; the `oracle-*` recipes compare pycli with sibling tools (rcli, statsrust, lmda)
and print a setup hint when the sibling is not on PATH.

## Tutorial conversions (`tut-*`)

Classic online data-analysis tutorial topics rewritten as original recipes against the pycli operations. Every recipe has a header
(Topic / What it does / Data / Usage), chains at least two operations, prints a human-readable result and proves itself with
self-checks that exit 1 on failure (fitted versus generating parameters within a stated tolerance, closed forms, recomputations). The
data are small synthetic tables under `examples/data/`, generated with fixed seeds by `scripts/generate-tutorial-data.py` (the header
says what each table mimics; `--check` regenerates them and fails on drift, and CI runs that check inside the test image).

| Recipe | Topic (after) | Data | Operations used | What is checked |
|---|---|---|---|---|
| `tut-ols-diagnostics.lms` | OLS regression with diagnostics (statsmodels OLS notebook) | `tut-ols.csv` | `sm_ols`, `stats_dist` (F sf, t ppf) | truth inside every 95% CI; coefficients pinned to 1e-4; sigma within 20% of 0.5, Durbin-Watson in [1.5, 2.5], condition number < 30; residuals sum to 0 and reproduce ssr; F p-value and t intervals reproduced from stats_dist |
| `tut-glm-poisson.lms` | Poisson counts with an exposure offset (R `?glm`, statsmodels GLM) | `tut-poisson.csv` | `sm_glm`, `stats_dist` (chi2 sf) | estimates within 2 SE of the generating log rates (0.75 instead of 0.5 fails) and pinned to 1e-4; LR test p < 0.01; deviance/df in [0.5, 1.6] |
| `tut-logistic-dose.lms` | Logistic dose-response and LD50 (R `?glm` budworm) | `tut-dose.csv` | `sm_glm`, `special_fn` (expit), `num_root` | closed-form and num_root LD50 agree to 1e-6; coefficients within 3 SE of the truth; LD50 within half a doubling; fitted curve monotone |
| `tut-group-tests.lms` | t-test, one-way ANOVA, chi-square independence (PlantGrowth / ToothGrowth, scipy-lectures statistics) | `tut-groups.csv` | `frame_import`, `frame_select`, `frame_groupby`, `sm_ols`, `stats_dist` (t, F, chi2 sf); no dedicated test operation exists, the tests are built from regression and the reference distributions | t, F and chi-square p-values reproduced from the reference distributions; coefficients equal group-mean differences and sit within 2.5 SE of the generating -0.4 / +0.8; all effects detected (t-test also after Bonferroni); expected counts >= 5 |
| `tut-bootstrap-ci.lms` | Seeded bootstrap confidence interval for a mean (scipy-lectures statistics) | `tut-bootstrap.csv` | `frame_import`, `frame_to_array`, `stats_dist` (rvs uniform, t ppf, gamma moments) | every row resampled; bootstrap mean within 0.25 SE of the sample mean; bootstrap SE within 15% of s/sqrt(n); percentile interval covers the true mean 10 and its width is within 20% of the t interval |
| `tut-curve-fit-peak.lms` | Gaussian peak fit by nonlinear least squares (scipy-lectures optimisation, R `?nls`) | `tut-peak.csv` | `frame_import`, `frame_to_array_to`, `opt_least_squares`, `opt_minimize`, `frame_to_array` | parameters within 4 SE of (2.0, 0.3, 4.5, 0.8); least squares and BFGS agree to 1e-3; rss recomputed from the table |
| `tut-lp-diet.lms` | Diet problem LP and MILP, shadow prices, SLSQP constrained minimisation (Stigler diet, scipy.optimize tutorial) | `tut-diet.csv` | `frame_import`, `frame_inspect`, `frame_to_array`, `opt_linprog`, `opt_minimize` | LP = 2.669565 and servings as derived by hand, MILP = 2.75, calorie shadow price 0.40/230 (all known answers); feasibility; strong duality; re-solve confirms the dual; SLSQP optimum (1.4, 1.7) |
| `tut-integrate-root.lms` | Numerical integration and root finding (scipy-lectures, scipy.integrate / optimize tutorials) | none | `num_quad`, `num_root`, `opt_root`, `stats_dist` (norm cdf) | three integrals within 1e-9 of closed forms; Kepler residual < 1e-9; system root = (sqrt 3, 1); quadrature of the normal density equals the cdf difference |
| `tut-ode-predator-prey.lms` | Lotka-Volterra predator-prey with events and a conserved quantity (scipy-lectures ODE, odeint tutorial) | none | `ode_solve`, `special_fn` (xlogy), `sig_peaks` | each event is a true maximum of the sampled prey curve; peak 13.67603 at t = 10.07736 / 20.86624 pinned; first integral conserved to 1e-6; peak spacing equals the event spacing |
| `tut-fft-peaks.lms` | Spectrum of a noisy two-tone signal and peak finding (scipy-lectures signal, periodogram / find_peaks) | `tut-signal.csv` | `frame_import`, `frame_to_array_to`, `sig_spectrum` (periodogram, welch), `sig_peaks` | strongest peaks on the 5 and 12 Hz bins; power ratio in [3, 5.5] and 5 Hz power within 0.1 of 0.5; Welch peak at 5 Hz |
| `tut-butterworth.lms` | Butterworth low-pass design and zero-phase filtering (scipy.signal butter / sosfiltfilt) | `tut-signal.csv` | `frame_import`, `frame_to_array`, `sig_filter` | stable; |H| = 1/sqrt 2 at the cutoff; |H(12 Hz)| in (0.1, 0.194), |H(25 Hz)| < 0.01; RMS error to the clean tone < 0.2 and at least halved |
| `tut-arima-acf.lms` | ACF / PACF and an AR(1) forecast (R `?arima` / `?acf`, statsmodels ARMA notebook) | `tut-ar1.csv` | `sm_tsa_acf`, `sm_arima`, `sm_tsa_tools` (ljung_box), `stats_dist` (norm ppf) | ACF(1), ACF(2) within 2-3 SE of phi, phi^2; at most one PACF lag 2-6 outside the plain band; phi within 2 SE of 0.7 (0.6 fails) and pinned to 1e-3; mean within 3 sigma/((1-phi) sqrt n); forecasts converge; interval widths follow the AR(1) formula |
| `tut-kmeans-cv.lms` | k-means purity, stratified cross-validation, classification metrics (scikit-learn user guide, iris) | `tut-flowers.csv` (species means rounded from Fisher's iris, every row a fresh draw) | `frame_import`, `frame_inspect`, `ml_fit` (kmeans, logistic), `ml_predict`, `frame_to_array`, `ml_cv` (knn), `ml_metrics` | k-means purity >= 0.9 with one cluster per species; CV accuracy >= 0.9; training accuracy >= 0.95; confusion-matrix diagonal equals the accuracy |
| `tut-interp-knn.lms` | Interpolation of samples and exact nearest neighbours (scipy.interpolate / spatial tutorials) | `tut-interp-nodes.csv`, `tut-interp-grid.csv`, `tut-flowers.csv` | `frame_import`, `frame_to_array`, `interp_eval` (linear, pchip, cubic + integral), `num_quad`, `spatial_knn`, `frame_inspect` | cubic error < 0.03, pchip < 0.10, linear > cubic; spline integral within 0.01 of quadrature; kNN distances verified by brute force; majority species as expected |
| `tut-graph-roads.lms` | Shortest paths on a road network (networkx shortest-path tutorial, CLRS) | `tut-roads.csv` | `frame_import`, `frame_inspect`, `graph_paths` (dijkstra, bellman_ford, k_shortest, bfs, all distances) | A -> J = 32 km by A-B-G-I-J, 4 segments by BFS and the whole distance table as known answers (independent Dijkstra); lengths recomputed from the road table; Bellman-Ford, Yen and BFS consistent |

The SIR epidemic model from the same tutorial family is already `ode-sir.lms` (below), so the ODE conversion covers predator-prey.
The first line of every recipe, `// access: read` or `// access: write`, says whether it writes files into the working directory
(the `write` recipes create `tut-*.npz` / `.safetensors` next to the schema and remove them before exiting). Provenance of the
fixtures: every table is a seeded draw; `tut-flowers.csv` uses the iris species and column names with means rounded from Fisher's
1936 data (invented standard deviations, no original measurement), the road network is invented, and the SLSQP problem in
`tut-lp-diet.lms` is the one from the scipy.optimize tutorial (credited there).

## Cross-language agreement (`cross-*`)

The same seeded analysis through pycli (Python) and R (rcli shim) from one lmscript, with a side-by-side table and exit 1 on any
disagreement beyond the stated tolerance (README "Cross-language agreement" explains the tolerance principles). Both shims on PATH:
`export PATH="$PWD/bin:$HOME/repos/lmctlhq/rcli/bin:$PATH"`; without rcli the R leg is skipped with a hint and the recipe exits 0.
Each recipe was mutation-checked: perturbing the data fed to one side only makes it exit 1.

| Recipe | Analysis | Data | Python / R operations | Tolerance (what is compared) |
|---|---|---|---|---|
| `cross-ols.lms` | OLS y ~ x1 + x2 | `tut-ols.csv` | `sm_ols` / `fit_lm` | coefficients 64 eps cond(X), standard errors 64 eps cond(X)^2, R^2 and sigma 1e-12 |
| `cross-glm.lms` | logistic (dead ~ ldose) and Poisson with exposure (count ~ group) | `tut-dose.csv`, `tut-poisson.csv` | `sm_glm`, `special_fn` / `fit_glm` | estimates 1e-6, standard errors 1e-5 (penultimate-iterate weights in R), deviances 1e-8 relative |
| `cross-tests.lms` | pooled t-test (regression t vs t.test), chi-square independence (hand statistic + chi2 sf vs chisq.test) | `tut-groups.csv` | `sm_ols`, `stats_dist` / `test_htest` | statistics 1e-10, df exact, p-values 1e-12 relative |
| `cross-arima-acf.lms` | ACF / PACF lags 1-10, ARIMA(1,0,0) with constant by exact ML | `tut-ar1.csv` | `sm_tsa_acf`, `sm_arima` / `ts_acf`, `ts_pacf`, `ts_arima` | ACF/PACF 1e-12; phi and process mean 1e-4; sigma2 1e-4 relative; log-likelihood 1e-6 relative; se(phi) 1e-3 relative |
| `cross-kmeans.lms` | k-means, 3 clusters on the petal measurements | `tut-flowers.csv` | `ml_fit`, `ml_predict`, `frame_to_array` / `cluster_kmeans` | adjusted Rand index >= 0.95 (1 here); within-cluster SS 1e-9 relative when the partitions coincide, 1% otherwise |
| `cross-nls.lms` | Gaussian peak by nonlinear least squares from a data-derived start | `tut-peak.csv` | `opt_least_squares` / `fit_nls` | parameters and standard errors 1e-6 relative, sigma 1e-9 relative, df exact |
| `cross-numerics.lms` | Rosenbrock BFGS, integral of exp(-x^2), Kepler root, twelve distribution values | none | `opt_minimize`, `num_quad`, `num_root`, `stats_dist` / `opt_minimize`, `num_integrate`, `num_root`, `dist_*` | minimisers vs (1, 1): 1e-4 scipy, 1e-3 R optim; integral and root 1e-10; densities / probabilities 1e-13 relative, continuous quantiles 1e-10 relative, discrete quantiles exact |

## Other recipes

| Recipe | What it shows |
|---|---|
| `solve.lms`, `lstsq-fit.lms` | dense solve and polynomial least squares with residual checks |
| `sales-report.lms`, `csv-join-report.lms`, `sensor-pipeline.lms` | typed frames: import, select, groupby, resample, merge, rolling, export |
| `curve-fit.lms`, `minimize-rosenbrock.lms`, `lp-plan.lms`, `ode-fit.lms`, `ode-sir.lms` | S3 / S4 numerics against closed forms and hand optima |
| `ml-classify-cv.lms`, `bridge-ml.lms` | scikit-learn families, saved models, the table / array bridge |
| `sig-denoise-peaks.lms`, `spatial-neighbours.lms`, `sparse-poisson.lms`, `graph-route-plan.lms`, `distributions-tour.lms` | S7 / S8 families with analytic responses |
| `oracle-*.lms` | cross-tool oracles (rcli, statsrust, lmda): docs/parity.md lists every pair and tolerance |
