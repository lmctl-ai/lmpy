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

| Recipe | Topic (after) | Data | Operations used |
|---|---|---|---|
| `tut-ols-diagnostics.lms` | OLS regression with diagnostics (statsmodels OLS notebook) | `tut-ols.csv` | `sm_ols`, `stats_dist` (F sf, t ppf) |
| `tut-glm-poisson.lms` | Poisson counts with an exposure offset (R `?glm`, statsmodels GLM) | `tut-poisson.csv` | `sm_glm`, `stats_dist` (chi2 sf) |
| `tut-logistic-dose.lms` | Logistic dose-response and LD50 (R `?glm` budworm) | `tut-dose.csv` | `sm_glm`, `special_fn` (expit), `num_root` |
| `tut-group-tests.lms` | t-test, one-way ANOVA, chi-square independence (PlantGrowth / ToothGrowth, scipy-lectures statistics) | `tut-groups.csv` | `frame_import`, `frame_select`, `frame_groupby`, `sm_ols`, `stats_dist` (t, F, chi2 sf); no dedicated test operation exists, the tests are built from regression and the reference distributions |
| `tut-bootstrap-ci.lms` | Seeded bootstrap confidence interval for a mean (scipy-lectures statistics) | `tut-bootstrap.csv` | `frame_import`, `frame_to_array`, `stats_dist` (rvs uniform, t ppf, gamma moments) |
| `tut-curve-fit-peak.lms` | Gaussian peak fit by nonlinear least squares (scipy-lectures optimisation, R `?nls`) | `tut-peak.csv` | `frame_import`, `frame_to_array_to`, `opt_least_squares`, `opt_minimize`, `frame_to_array` |
| `tut-lp-diet.lms` | Diet problem LP and MILP, shadow prices, SLSQP constrained minimisation (Stigler diet, scipy.optimize tutorial) | `tut-diet.csv` | `frame_import`, `frame_inspect`, `frame_to_array`, `opt_linprog`, `opt_minimize` |
| `tut-integrate-root.lms` | Numerical integration and root finding (scipy-lectures, scipy.integrate / optimize tutorials) | none | `num_quad`, `num_root`, `opt_root`, `stats_dist` (norm cdf) |
| `tut-ode-predator-prey.lms` | Lotka-Volterra predator-prey with events and a conserved quantity (scipy-lectures ODE, odeint tutorial) | none | `ode_solve`, `special_fn` (xlogy), `sig_peaks` |
| `tut-fft-peaks.lms` | Spectrum of a noisy two-tone signal and peak finding (scipy-lectures signal, periodogram / find_peaks) | `tut-signal.csv` | `frame_import`, `frame_to_array_to`, `sig_spectrum` (periodogram, welch), `sig_peaks` |
| `tut-butterworth.lms` | Butterworth low-pass design and zero-phase filtering (scipy.signal butter / sosfiltfilt) | `tut-signal.csv` | `frame_import`, `frame_to_array`, `sig_filter` |
| `tut-arima-acf.lms` | ACF / PACF and an AR(1) forecast (R `?arima` / `?acf`, statsmodels ARMA notebook) | `tut-ar1.csv` | `sm_tsa_acf`, `sm_arima`, `sm_tsa_tools` (ljung_box), `stats_dist` (norm ppf) |
| `tut-kmeans-cv.lms` | k-means purity, stratified cross-validation, classification metrics (scikit-learn user guide, iris) | `tut-flowers.csv` | `frame_import`, `frame_inspect`, `ml_fit` (kmeans, logistic), `ml_predict`, `frame_to_array`, `ml_cv` (knn), `ml_metrics` |
| `tut-interp-knn.lms` | Interpolation of samples and exact nearest neighbours (scipy.interpolate / spatial tutorials) | `tut-interp-nodes.csv`, `tut-interp-grid.csv`, `tut-flowers.csv` | `frame_import`, `frame_to_array`, `interp_eval` (linear, pchip, cubic + integral), `num_quad`, `spatial_knn`, `frame_inspect` |
| `tut-graph-roads.lms` | Shortest paths on a road network (networkx shortest-path tutorial, CLRS) | `tut-roads.csv` | `frame_import`, `frame_inspect`, `graph_paths` (dijkstra, bellman_ford, k_shortest, bfs, all distances) |

The SIR epidemic model from the same tutorial family is already `ode-sir.lms` (below), so the ODE conversion covers predator-prey.

## Other recipes

| Recipe | What it shows |
|---|---|
| `solve.lms`, `lstsq-fit.lms` | dense solve and polynomial least squares with residual checks |
| `sales-report.lms`, `csv-join-report.lms`, `sensor-pipeline.lms` | typed frames: import, select, groupby, resample, merge, rolling, export |
| `curve-fit.lms`, `minimize-rosenbrock.lms`, `lp-plan.lms`, `ode-fit.lms`, `ode-sir.lms` | S3 / S4 numerics against closed forms and hand optima |
| `ml-classify-cv.lms`, `bridge-ml.lms` | scikit-learn families, saved models, the table / array bridge |
| `sig-denoise-peaks.lms`, `spatial-neighbours.lms`, `sparse-poisson.lms`, `graph-route-plan.lms`, `distributions-tour.lms` | S7 / S8 families with analytic responses |
| `oracle-*.lms` | cross-tool oracles (rcli, statsrust, lmda): docs/parity.md lists every pair and tolerance |
