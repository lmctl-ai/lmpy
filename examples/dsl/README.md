# `_dsl_` examples for lmpy

Every file here is an lmscript program with one `_dsl_` block. Running one compiles the block against
[`../../native/schema/lmpy.dsl.json`](../../native/schema/lmpy.dsl.json), saves the bytecode, runs the native host
`lmpy --dsl`, and reads back the JSON values and checks. No Python, R or Docker is involved: `lmpy` is one executable
whose only shared dependency is glibc.

```sh
cmake -S native -B native/build -G Ninja && cmake --build native/build      # see native/README.md for the dependency fetch
export LMPY=$PWD/native/build/lmpy                                           # or put lmpy on PATH
cd examples/dsl && lmscript task-delivery-plan.lms
```

`native/test.sh` runs every example and fails when any one fails. The checks inside an example are `out.check` /
`out.close`. A failed check exits 4 and lmscript prints the host's message, which names the check. A missing `lmpy`
prints `SKIPPED` and exits 0. [`lmpy.lms`](lmpy.lms) is the shared runner: `savedsl`, then `system`, then `json_parse`.

## Multi-step tasks

These are workflows that chain several families in one native program. With the pycli CLI, each step was one process and
one artifact file.

| Example | Steps |
|---|---|
| [task-heat-equation](task-heat-equation.lms) | sparse Laplacian → Crank-Nicolson operators → 200 time steps (sparse LU each) → analytic comparison, energy and maximum-principle checks → CSV |
| [task-sales-trend](task-sales-trend.lms) | store CSV → groupby date → ISO times → least-squares trend + weekly seasonality → 14-day forecast → weekly resample → CSV |
| [task-flower-classifier](task-flower-classifier.lms) | CSV → label encoding → scaler → PCA → k-means (ARI) → CV model selection (3 kinds) → refit → confusion matrix → save/reload → predictions CSV |
| [task-delivery-plan](task-delivery-plan.lms) | road graph → all-pairs shortest paths → transportation LP (HiGHS) → strong duality → per-shipment routes |
| [task-epidemic-fit](task-epidemic-fit.lms) | SIR simulation → seeded Poisson noise → golden-section search written in the DSL around 255 ODE solves → R0 and peak prediction |
| [task-quality-control](task-quality-control.lms) | seeded data → CSV → groupby summary → nested OLS F test → Welch t tests with Bonferroni → Cpk → out-of-spec export |
| [task-sensor-cleanup](task-sensor-cleanup.lms) | synthetic recording → CSV → Hampel despiking → band-stop + low-pass filtfilt → Welch before/after → peaks → frequency and amplitude → CSV |
| [task-portfolio](task-portfolio.lms) | Cholesky-simulated returns → covariance algebra → closed-form vs SLSQP minimum variance → long-only frontier loop → VaR |

## Ports of the pycli recipes

`<name>.lms` here is the `_dsl_` version of [`../<name>.lms`](../). Ports keep the original's data, known answers and
tolerances. The `oracle-*` and `cross-*` originals compared pycli with sibling tools (lmda, statsrust, rcli/R) at run time.
Their ports assert the pinned values those comparisons established, and each header states the source (R 4.3.3, scipy
1.16.2, statsmodels 0.14.6, scikit-learn 1.7.2, pandas 2.3.3).

| Family | Examples |
|---|---|
| Linear algebra, sparse | solve, lstsq-fit, sparse-poisson, bridge-ml |
| Tables | csv-join-report, sales-report, oracle-frame, sensor-pipeline |
| Distributions, RNG, tests | distributions-tour, oracle-dist, oracle-rvs, tut-bootstrap-ci, cross-tests |
| Optimisation, roots, quadrature, LP | minimize-rosenbrock, curve-fit, cross-nls, oracle-root, lp-plan, tut-lp-diet, tut-curve-fit-peak, tut-integrate-root, cross-numerics |
| ODE, interpolation | ode-sir, ode-stiff-robertson (new), ode-fit, tut-ode-predator-prey, oracle-interp, tut-interp-knn |
| Signal | sig-denoise-peaks, oracle-spectrum, tut-fft-peaks, tut-butterworth |
| Spatial, graphs | spatial-neighbours, graph-route-plan, tut-graph-roads |
| Statistical models | oracle-sm-regression, oracle-sm-glm, oracle-sm-arima, oracle-tsa-tools, cross-ols, cross-arima-acf, tut-ols-diagnostics, tut-glm-poisson, tut-arima-acf, tut-group-tests, cross-glm, tut-logistic-dose |
| Machine learning | oracle-kmeans, oracle-ml-linear, oracle-metrics, ml-classify-cv, cross-kmeans, tut-kmeans-cv |

## Writing a new one

Start from any task file. Notes on the language:

- **Literals:** float literals are `digits.digits`, with no exponent (`0.000001`, not `1e-6`). Write `float(i)` to widen
  an int.
- **Collections:** the language has no lists. Use the host types `Vec`, `StrVec`, `Mat` and `Frame`, and loop with
  `for (x in v)` or `for (i in range(a, b))`.
- **Comments:** inside the block, use whole-line `#` comments.
- **Names:** a local cannot reuse a schema function name such as `dist` or `full`.
- **Options:** go in a trailing `opts().set(...)`. Expression constants and data use `$name` keys.

The full API is [`../../native/API.md`](../../native/API.md).
