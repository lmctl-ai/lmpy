# Parity with the sibling tools (rcli, lmstats, lmda, lmrust)

pycli overlaps with other lmctl tools only where the overlap is a cross-tool oracle (docs/decisions.md). This file is the complete
map: for every operation that another CLI also offers, the pair, the recipe that compares them, the tolerance the recipe enforces, and
the status (agree / differ, with the reason). The numbers behind each row are in docs/oracle-findings.md (consolidated table at its top);
the recipes run for real through the host shim in every acceptance (`scripts/run-recipes.sh`) and print a setup hint when the sibling
binary is not on PATH, so a missing tool never fails the suite and never silently passes a comparison either (the leg is reported as
skipped).

Tools: **rcli** (R 4.3.3 through `Rscript --vanilla`), **lmstats** (binary `statsrust`, Rust port of statsmodels conventions), **lmda**
(Rust CSV / series / evaluation tool, commit 36e5285), **lmrust** (the Rust SDK for lmapi: a library, not a CLI; it publishes no
operations, so there is no numerical overlap; pycli's schema is parsed by its TypeScript twin lmts in CI).

## Tolerance tiers (docs/design-opus.md 6.2, applied)

| Class | Same image | Cross-tool / cross-architecture |
|---|---|---|
| codec, artifacts, glue oracles | bit-exact | bit-exact (codec only) |
| distribution functions, special functions | bit-exact | 1e-12 relative (pdf / cdf / sf), 1e-10 for iterative continuous quantiles |
| direct linear algebra, OLS | bit-exact | 1e-9 absolute on coefficients and standard errors |
| GLM (IRLS) | bit-exact | 1e-6 (different stopping rules move the last digits of standard errors) |
| ARIMA (likelihood ridge) | bit-exact | 1e-3 on parameters; log-likelihood 1e-8 |
| interpolation, spectra, metrics, joins, rolling | bit-exact | 1e-9 relative, exact for row order / integers |
| seeded sampling | bit-exact with seed | never value-by-value (different generators): empirical CDF vs quantile function, 4.5 SE |
| clustering | bit-exact with seed | adjusted Rand index after relabelling |

Tolerances are never raised to hide a mismatch; every exceedance is a row in docs/oracle-findings.md with both values and a diagnosis.

## Operation pairs

| pycli | sibling operation | Recipe | Tolerance | Status |
|---|---|---|---|---|
| `stats_dist` pdf / cdf / sf (norm, t, chi2, gamma, beta, binom, poisson) | rcli `dist_density`, `dist_probability` | `oracle-dist.lms` (217 values) | 1e-12 relative | agree (worst 6e-15) |
| `stats_dist` ppf, continuous | rcli `dist_quantile` | `oracle-dist.lms` | 1e-10 relative | agree (worst 3.4e-12; both quantile routines are iterative, t tails) |
| `stats_dist` ppf, discrete (binom, poisson) | rcli `dist_quantile` | `oracle-dist.lms` | exact integers | agree |
| `stats_dist` rvs (norm, gamma, beta, chi2, t) | rcli `rng_draw`, `dist_quantile` | `oracle-rvs.lms` | empirical CDF within 4.5 binomial SE of R's quantiles | agree (worst 2.85 SE); streams differ by design (PCG64 vs Mersenne-Twister) |
| `stats_dist` long tail (genextreme, genpareto, skewnorm, johnsonsu, nct, ncf, vonmises, truncnorm, levy_stable, zipf, betabinom, ...) | none in rcli (R needs add-on packages) | `distributions-tour.lms` (closed forms), tests/s8b2_dist.py (mpmath) | 1e-12 closed forms; 1e-6 levy_stable | no cross-tool pair; parameterisation differences vs R documented in docs/s8b2.md |
| `sm_ols` | rcli `fit_lm`; lmstats `regression` | `oracle-sm-regression.lms` | 1e-9 absolute | agree (< 5e-13); rcli `fit_lm` accepts additive numeric formulas only, the recipe passes dummy and product columns |
| `sm_glm` (poisson, binomial, gamma, gaussian) | rcli `fit_glm`; lmstats `glm` | `oracle-sm-glm.lms` | 1e-6 | estimates agree to 12 digits; standard errors differ up to 4.1e-8 (R stops IRLS on relative deviance 1e-8, statsmodels on its own tol); lmstats identical to pycli |
| `sm_sarimax` (ARMA(1,1) + forecasts) | rcli `ts_arima`; lmstats `arima` | `oracle-sm-arima.lms` | 1e-3 parameters | agree within tolerance: 1.5e-5 vs R, 6.2e-6 vs lmstats after the Nelder-Mead polish; the optimisers stop at different points of a likelihood ridge (llf within 1.4e-8) |
| `sm_tsa_acf`, `sm_tsa_tools` acf / pacf | rcli `ts_acf`; lmstats `acf_pacf` | `oracle-tsa-tools.lms` | 1e-9 | agree (<= 1e-15); lmstats defaults to `ywadjusted` PACF, pycli to `ywm`: the recipe sets the method on both |
| `sm_tsa_tools` ljung_box, kpss | lmstats `ljung_box`, `kpss` | `oracle-tsa-tools.lms` | 1e-9 | agree (4.4e-14) |
| `sm_tsa_tools` adfuller | lmstats `adf` | `oracle-tsa-tools.lms` | 1e-9 statistic, 1e-8 p-value | statistic, lag and nobs agree; p-value differs by 1.4e-9 because lmstats' normal CDF is approximate (mpmath confirms pycli / statsmodels) |
| `num_root` (brentq) | rcli `num_root` (uniroot) | `oracle-root.lms` | 1e-12 | agree (<= 5e-16) |
| `opt_root` (hybr, lm systems) | none (rcli has one-dimensional roots only) | `oracle-root.lms` (closed form) | 1e-8 | no cross-tool pair; success is re-verified by the residual |
| `opt_minimize`, `num_quad`, `opt_least_squares` | rcli `opt_minimize`, `num_integrate`, `fit_nls` (same expression grammar) | `curve-fit.lms`, `minimize-rosenbrock.lms` (closed forms); the shared expression corpus runs on both tools (tests/s3_expr.py, expr-spec.json) | semantics bit-exact for the grammar; optima 1e-8 | grammar parity (shared corpus); numerical optima are compared with closed forms, not with R's optimisers (different algorithms) |
| `opt_linprog`, `ode_solve` | none (rcli has no LP / MILP / generic ODE) | `lp-plan.lms`, `ode-sir.lms`, `ode-fit.lms` (hand optima, analytic solutions) | 1e-9 objective | no cross-tool pair (docs/decisions.md S4) |
| `interp_eval` linear, cubic natural, cubic periodic | rcli `interp_approx`, `interp_spline` | `oracle-interp.lms` | 1e-9 | agree (<= 6.7e-16); R `fmm` is not not-a-knot and R `monoH.FC` is not PCHIP: different interpolants, reported as information |
| `sig_spectrum` (periodogram, boxcar, no detrend) | lmda `series_spectrum` | `oracle-spectrum.lms` | 1e-9 relative | agree (< 1e-18 of the largest bin); lmda's default `detrend: mean` = pycli `constant` |
| `sig_filter`, `sig_peaks`, `spatial_*` | none | `sig-denoise-peaks.lms`, `spatial-neighbours.lms` (analytic responses, hull volumes) | 1e-9 | no cross-tool pair |
| `ml_fit` linear regression | rcli `fit_lm`; lmstats `regression` | `oracle-ml-linear.lms` | 1e-9 | agree (0 on the printed doubles) |
| `ml_fit` kmeans | rcli `cluster_kmeans` (Hartigan-Wong); lmda `cluster_kmeans` | `oracle-kmeans.lms` | ARI = 1 after relabelling | agree on separated blobs; identical seeds do not give identical centres across libraries (different RNGs), so only the partition is compared; the lmda leg needs the lmda binary |
| `ml_fit` logistic, ridge, lasso, trees, `ml_cv`, `ml_predict` | none | `ml-classify-cv.lms`, `bridge-ml.lms` (saved vs in-process predictions) | bit-exact save / load | no cross-tool pair |
| `ml_metrics` (accuracy, balanced accuracy, precision, recall, F1, ROC AUC, Brier) | lmda `eval_classify`, `eval_roc` | `oracle-metrics.lms` | 1e-12 | agree (< 1e-15), ties count 1/2 in both |
| `frame_merge` (inner / left / outer, duplicate keys, NA keys) | lmda `table_join` | `oracle-frame.lms` | exact rows and order | agree; pandas would match NaN keys, pycli and lmda (SQL) never do |
| `frame_rolling` (mean, sum, std, min, max, median) | lmda `series_rolling` | `oracle-frame.lms` | 1e-12 | agree (1e-15); lmda skips blank CSV rows, pycli keeps a masked missing row (recipe uses complete data) |
| `frame_groupby`, `frame_select`, `frame_resample`, `frame_import` / `frame_export` | lmda `table_groupby` and CSV operations (overlap by design: lmda is the CSV tool, pycli keeps only the typed handoff) | `sales-report.lms`, `csv-join-report.lms` (hand values); no lmda leg yet | exact | not compared (docs/decisions.md S2: a future lmda recipe on the four-row region fixture) |
| `special_fn` (34 functions) | rcli `math_special` (gamma, beta, digamma, trigamma, choose, Bessel) | none yet; tests/s8b_ops.py uses mpmath at 50 digits | 1e-13 .. 1e-10 per function | not compared with rcli (mpmath is the stricter oracle); a recipe is a candidate |
| `la_solve`, `la_lstsq`, `la_props`, `la_decomp`, `la_matrix_fn`, `la_sparse` | none (lmda / rcli have no linear algebra operations) | NIST StRD, Hilbert, closed forms (tests/s1_la.py, s8b_ops.py), `sparse-poisson.lms` | per docs/s1.md | no cross-tool pair |
| `graph_paths`, `graph_flow` | none | `graph-route-plan.lms` (CLRS hand values), pure-Python algorithms and the `opt_linprog` LP as oracles (tests/s8b2_graph.py, s8b2_review.py) | exact integers | no cross-tool pair |
| `array_import` / `array_inspect` safetensors | torchcli tensors (same safetensors reader / writer lineage) | `bridge-ml.lms` writes and reads F64 safetensors; no torchcli leg on this host | bit-exact bytes | format parity by construction; a torchcli round-trip recipe is a candidate |

## Deliberate protocol differences

| Topic | pycli | sibling | Why |
|---|---|---|---|
| missing file / tensor key | `NOT_FOUND`, exit 4 | rcli: exit 2 | torchcli convention (docs/decisions.md S1); the one intentional incompatibility |
| missing interpreter | `SERVICE_UNAVAILABLE` / `PYTHON_NOT_FOUND`, exit 8 | rcli: same | fleet convention kept |
| seeds | required and echoed, PCG64, prefix property | rcli: R's Mersenne-Twister, `rng_draw` has no PCG64 | streams are never compared value by value |
| int64 above 2^53 | refused (`INT64_NOT_EXACT`), never rounded | lmscript flags them | docs/design-opus.md 7.2 |
| expressions | the shared grammar of expr-spec.json, parsed in C++ | rcli: the same grammar, parsed in C++ | one spec, one corpus on both tools |
