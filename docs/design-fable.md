# pycli: the Python scientific stack as an lmapi native tool (design, Fable)

Status: design only, 2026-10-14. Nothing here is implemented or measured unless marked **verified**.

Decision in one paragraph: `pycli` is an MIT C++17 lmcpp front end that validates a fixed catalogue of typed
operations and runs each one in a fresh, isolated CPython child (`python3 -I`) executing an **embedded, fixed
worker script**. The worker imports only what the requested op needs, reads data from private `.npy`/JSON files
written by C++, never evaluates caller text, and writes a JSON result that C++ re-emits. This is rcli's shape
(layer "worker process"), chosen for the same reason: the value of pycli is *being numpy/scipy/pandas/sklearn/
statsmodels*, bit-for-bit for the pinned image, with a clean crash and licence boundary. Where I diverge from
rcli/torchcli I say so explicitly (§1.4, §2.6).

Verified host facts (2026-10-14): aarch64 Ubuntu 24.04, Python 3.12.3 host, no numpy/scipy/pandas apt packages
installed, Docker present, **no amd64 emulation** (`docker run --platform linux/amd64 alpine` fails with exec
format error, so amd64 images must be built on an amd64 CI runner), `pip download` of
`numpy-2.3.3-cp312-manylinux_2_28_aarch64.whl` (14.3 MB) works on the host, 54 GB free, 4 cores, 15 GB RAM.
lmcpp exposes `run_child`, `Context::temp_file_path`, `read_result_file`, `Type::Integer`,
`allow_special_numbers` (used below). lmstats = statsmodels port (45 ops), lmda = pure-Rust data analysis (199
ops), rcli = 28 R ops with a C++ expression parser + `expr-spec.json`.

---

## 1. Architecture

### 1.1 Options

| Option | Verdict | Why |
|---|---|---|
| (a) C++ lmcpp front end + fixed embedded Python worker run as a child per call | **Chosen** | Exact library semantics; crash isolation (segfault in a Cython kernel or OpenBLAS kills only the child); no GIL/ABI coupling; `--schema` needs no Python; identical to rcli's proven shape, so the team already knows the review checklist |
| (b) Embed libpython (Python C API / pybind11) | Rejected | Saves only `exec` + interpreter boot (~30-50 ms); import of numpy/pandas dominates and is unchanged. Ties the binary to one CPython ABI per arch, puts the interpreter in our address space (a kernel crash is our crash, `Py_Finalize` leaks), `fork` after import is unsafe with OpenBLAS threads, and `--schema` would load libpython. All cost, no fidelity gain |
| (c) Call BLAS/LAPACK/ARPACK/Qhull/pocketfft directly from C++ | Rejected for v1 | Reproduces none of the Python-level glue (pandas groupby semantics, sklearn defaults, statsmodels MLE drivers) which is the whole point; it would be a worse lmda/lmstats, both of which are already pure Rust. The oracle role needs the Python stack itself |
| (d) Persistent worker / daemon with a request protocol | Deferred, protocol kept compatible | lmscript calls tools as processes; the shim starts a fresh container per call anyway (~0.7 s **verified** by rcli). A daemon buys back import time only for tight loops; it adds lifecycle, state leakage between calls, and replay bugs. Design the file-based request/response so a later `pycli serve` (Unix socket, same request files) is a drop-in |

### 1.2 Process model

```
pycli <op> … ──lmcpp validate+policy──▶ handler (C++)
   writes  <tmp>/req.json          # options, scalars, strings, expression ASTs, file refs
           <tmp>/in_<name>.npy     # every numeric array/matrix/column, float64/int64/bool, exact binary
           <tmp>/worker.py         # embedded worker source, materialized per call (as rcli does)
   execs   python3 -I -B -X utf8 -X faulthandler <tmp>/worker.py <op> <tmp>/req.json <tmp>/res.json
   env     PATH=/usr/bin:/bin HOME=<tmp> TMPDIR=<tmp> LC_ALL=C.UTF-8 PYTHONHASHSEED=0
           OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
           PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg (matplotlib is not installed; belt and braces)
   waits   lmcpp run_child, timeout default 50 s (PYCLI_TIMEOUT_S may only lower it), stdout/stderr capped 1 MiB
   reads   read_result_file(res.json,"json")  → prints raw result; nonzero/timeout/signal → protocol errors
```

- `-I` (isolated) removes cwd and `PYTHON*` env from `sys.path`/config, so a planted `numpy.py` in the workspace
  cannot be imported. The worker additionally pins `sys.path` to the interpreter's stdlib + site-packages and
  asserts `numpy.__file__` lives under `/usr/local/lib/python3.12/site-packages` before doing anything.
- Lazy imports: the worker imports `numpy` + `json` always, and `scipy.<submodule>`, `pandas`, `sklearn.<module>`,
  `statsmodels`, `networkx`, `pyarrow` only inside the op that needs them. Import cost is the dominant per-call
  cost and must be **measured in slice 0** (estimates, aarch64, not measured: numpy 0.15 s, scipy submodule 0.2-0.4 s,
  pandas 0.6-1.0 s, sklearn linear_model 0.8-1.2 s, statsmodels.api 1.5-2.5 s). The image precompiles `.pyc`
  (`python -m compileall -j0 site-packages`) so no call pays bytecode compilation.
- Result from the worker is `{"ok":true,"result":{…}}` or `{"ok":false,"tool_code":…,"message":…,"warnings":[…]}`;
  the Python exception type/message is data, never a stack trace to the caller (traceback goes to stderr diagnostics).
- The worker is a single source file (`py/worker.py`, ~1500-2500 lines over all slices) embedded via CMake
  `file(READ)` into a C++ raw string, exactly like `rcli/src/worker.hpp.in`. One closed `OPS = {"linalg_solve": op_linalg_solve, …}`
  dispatch dict. No `eval`, `exec`, `compile`, `pickle`, `importlib.import_module(caller_text)`, `getattr(module, caller_text)`
  anywhere; a CTest greps the worker for those tokens and fails on any hit outside an allowlisted comment.

### 1.3 Determinism and reproducibility

| Concern | Rule |
|---|---|
| BLAS threads | numpy and scipy wheels each bundle their own OpenBLAS (`scipy_openblas64`); both honour `OPENBLAS_NUM_THREADS=1`. The worker also calls `threadpoolctl.threadpool_limits(1)` (threadpoolctl is a sklearn dependency) and records the actual BLAS name/threads in `meta.blas`. Single thread removes reduction-order nondeterminism within an image |
| Cross-arch | OpenBLAS kernels differ between aarch64 and amd64 (FMA use, blocking), so `linalg_*`, fits and anything through BLAS are **not** promised bit-identical across architectures. Promise: bit-identical for the (image digest, arch) pair recorded in `meta`; goldens for BLAS-dependent ops are tolerance-based (§6). This is a conscious weakening versus rcli's "bit-exact" claim, which rcli could make only because reference BLAS is scalar |
| RNG | Every stochastic op has required integer `seed`. Ops we control use `numpy.random.default_rng(seed)` (PCG64). sklearn ops pass `random_state=seed` (sklearn builds a legacy `RandomState`/MT19937 from it: documented per op in the schema description). Never `np.random.seed`; the global state is never touched. `meta.rng` names the generator per op |
| Hashing | `PYTHONHASHSEED=0` so set/dict iteration over strings is stable (affects pandas groupby on object keys only in `sort=False` paths, which we never use) |
| Numbers out | Python `repr(float)` is shortest round-trip (David Gay); numpy scalars are converted with `float()` first. Int64 stays integer. Result doubles are therefore exact across the boundary; nlohmann re-emits shortest round-trip. Optional `precision: n` (1-17) rounds *display* values for token economy; default exact (correctness first) |
| Numbers in | JSON numbers parsed by nlohmann (`strtod`, exact) and written to `.npy` as float64/int64, never through text again |
| Warnings | Worker runs each op under `warnings.catch_warnings(record=True)`, `warnings.simplefilter("always")`, `np.errstate(all="warn")`; messages are deduplicated and returned in `result.warnings[]` (category + message). `pandas.errors.*`, `sklearn.exceptions.ConvergenceWarning`, `statsmodels.tools.sm_exceptions.*` all flow through this. Never dropped to stderr |
| Convergence | Non-convergence is a domain status in the result (`converged:false`, `status`, `message`) per protocol §6.1, not an exit code |
| meta | Every result carries `meta:{python, numpy, scipy, pandas, sklearn, statsmodels (versions actually imported), blas, platform:"aarch64"|"x86_64", image:"<label from env PYCLI_IMAGE>", seed, elapsed_ms, import_ms}` |

### 1.4 Where this differs from rcli/torchcli (and why)

1. **Binary `.npy` transport instead of one-number-per-line text** (rcli): exact, no float parsing on either side,
   `np.load(allow_pickle=False, mmap_mode=None)` is one syscall-ish, and the same file format is a public artifact
   format for large arrays. Text files cost rcli seconds at 200k values (its timing table).
2. **Columnar table envelope** `{"columns":[...],"dtypes":[...],"n_rows":n,"data":{"col":[...]}}` instead of rcli's
   row arrays: fewer tokens (no per-row brackets of mixed types), and `r.result.data.x` is what an lmscript author
   wants. Propose lifting one `lmcpp::table` helper so rcli can emit the same shape in its next slice.
3. **Nonfinite policy kept identical to rcli**: `"NaN"`, `"Inf"`, `"-Inf"` as strings; `null` means *missing*
   (pandas `NA`/`NaT`/`None`). In numpy array results there is no missing concept, so `null` never appears there.
   Inputs accept `null` → NaN in float arrays, → missing in table columns.
4. **Write ops exist from the start** (torchcli rule): any op that persists a file is `access:"write"` with an
   exclusive `out` and atomic no-replace install. rcli's all-read catalogue was right for R; tables and models
   need files.
5. **No custom JSON codec**: Python's `json` is exact for doubles and strings; rcli needed one because of R.

---

## 2. Data model and handoff

### 2.1 Three value kinds

| Kind | Inline form (small) | File form (large / persistent) | Notes |
|---|---|---|---|
| Vector / matrix / ndarray | JSON array (row-major nested arrays for rank ≥ 2) with `null`/`"NaN"`/`"Inf"` allowed where `allow_special_numbers` is set | `path.npy` (one array), `path.npz#key` (several, zip of `.npy`, `allow_pickle=False`), `path.safetensors#key` (F32/F64/I64 only, read via `safetensors.numpy`, for torchcli interop) | Rank ≤ 3; dtype policy float64, int64, bool, complex128 as `[re,im]` pairs only on output (`fft`, `eig`) |
| Table | `records` (array of objects) or `columns` (object of equal-length arrays) | `path.csv` (header row, UTF-8), `path.parquet` (pyarrow, slice 6), `path.npz` (one array per column, plus `__pycli_table__` JSON metadata key) | Inline caps: 200k cells; file caps: 256 MiB, 2M rows |
| Model | never inline | `path.pycli-model.json` (§4.3) | Only families with closed-form parameter sets |

Inline vs file is chosen by the caller per parameter with exactly one of the alternatives (`x` | `x_ref`), as
torchcli does with refs; the C++ handler normalises both into `.npy` for the worker. Path strings: relative,
resolved against cwd, canonicalised with `realpath`, must stay inside cwd (symlink escape rejected,
`tool_code:"PATH_OUTSIDE_WORKSPACE"`), `#` disallowed in paths, key after `#` must match `^[A-Za-z_][A-Za-z0-9_.-]{0,63}$`.
Extension decides the reader; there is no sniffing. Input file size cap 256 MiB before open; `.npy` headers are
validated for dtype ∈ {`<f8`,`<i8`,`|b1`,`<f4`,`<i4`} and `fortran_order=False`, otherwise `INVALID_ARTIFACT`.

### 2.2 DataFrame semantics (the part that bites)

| Topic | Rule |
|---|---|
| Index | There is no index. Every table in and out is `RangeIndex`; group keys, resample bins and pivot row labels come back as ordinary columns (`as_index=False`, `reset_index()`). Datetime operations take an explicit `on: "<column>"` |
| dtypes in | CSV: pandas default inference with `dtype_backend="numpy_nullable"` so integer columns with missing values stay `Int64` not float. Inline: JSON number with all-integral values → `Int64`, otherwise `Float64`; strings → `string`; booleans → `boolean`; `null` → missing in any column. Optional `dtypes: {"col": "int64|float64|bool|string|category|datetime|timedelta"}` overrides; `datetime` parses ISO-8601 (`utc: true` to force tz-aware UTC) |
| dtypes out | Published label per column, one of `int64 float64 bool string category datetime datetime_tz timedelta`. `category` columns are emitted as their string values plus `categories:[…]` in a sibling `column_meta`. `datetime` is ISO-8601 with nanosecond precision when nonzero (`2024-01-02T03:04:05.000000001`), tz-aware with offset (`…+00:00`, tz name kept in `column_meta.tz`). `timedelta` is ISO-8601 duration string plus `_ns` integer in `column_meta` when exact nanoseconds matter. Anything else (object columns of mixed types, nested, period, interval, sparse) is rejected with `UNSUPPORTED_DTYPE` rather than stringified silently |
| Missing | `null` out, for every dtype. `NaN` in a float column is pandas-missing and therefore `null`, except in explicitly numeric *array* results (§1.4 item 3). `"NaN"` string appears only for float NaN produced by arithmetic in array ops |
| Column names | `^[A-Za-z_][A-Za-z0-9_ .-]{0,63}$`, unique, UTF-8 not allowed in v1 (rejected, not transliterated). Names are *data* in the worker: they index `df[...]`, never reach `eval`/`query`/patsy as text (§4.2) |
| Row order | Stable: `sort_values(kind="stable")`, `groupby(sort=True)`, merge `sort=False` preserving left order (pandas default), documented per op |
| Big outputs | Default `limit: 200` rows for table results with `truncated:true` and `n_rows` total; to get everything, write a file with `out` (write op) and read it back by ref. lmcpp's 16 MiB result cap stands behind this |

### 2.3 Result envelopes

- Scalar/vector/matrix: `{"value": …, "shape":[…], "dtype":"float64", …}`; complex as `{"re":[…],"im":[…]}`.
- Table: §1.4 item 2 plus `column_meta` when needed and `truncated`.
- Fit: `{"coefficients":{"term":[…],"estimate":[…],"std_error":[…],"statistic":[…],"p_value":[…]}, "n":…, "df_resid":…, "loglik":…, "aic":…, "bic":…, "converged":…, "fitted":[…]?, "residuals":[…]?}` with the long vectors opt-in via `return:["fitted","residuals"]`, matching rcli so `oracle-*.lms` recipes can compare field by field.
- File-producing op: `{"out":"path", "bytes":n, "sha256":"…", "shape"/"n_rows"/"columns"…}`; artifact write = sibling temp + fsync + `link`/`rename` with `O_EXCL` semantics, existing destination → `CONFLICT` (torchcli code lifted or, better, moved into lmcpp as `atomic_install`).

### 2.4 lmcpp gaps to fill (generic, upstream)

1. `lmcpp::npy` writer/reader (C++ side writes `.npy` from JSON arrays; ~120 lines; no numpy dependency).
2. `temp_dir()` on `Context` (one private directory per call instead of N `temp_file_path()` calls) with recursive cleanup on exit/signal.
3. `atomic_install(src_tmp, dst)` and `confine_path(cwd, p)` lifted from torchcli.
4. `lmcpp::table` envelope helper (§1.4 item 2).
5. Move rcli's `expr_parser.hpp` + `expr-spec.json` into lmcpp as `lmcpp::expr` (§4.1) so rcli and pycli share one audited parser and one corpus.

---

## 3. Operation catalogue

Priority P0 = slices 0-2, P1 = slices 3-5, P2 = slice 6+. "Oracle for" names the sibling tool this op certifies.
All ops are `read` unless marked **W** (writes a file). Arrays accept inline or `_ref`.

### 3.1 Meta and linear algebra (numpy / scipy.linalg, LAPACK via OpenBLAS)

| # | Op | Params (shape) | Result | Prio | Risk / notes |
|---|---|---|---|---|---|
| 1 | `py_info` | — | versions, BLAS, platform, thread config | P0 | none; the provenance stamp |
| 2 | `linalg_solve` | `a` (n×n), `b` (n or n×k), `assume_a` ∈ gen/sym/pos/upper/lower | `x`, `rcond` estimate | P0 | ill-conditioned → warning + `rcond`, never silent |
| 3 | `linalg_lstsq` | `a` (m×n), `b`, `rcond` | `x`, `residuals`, `rank`, `singular_values` | P0 | gelsd; oracle for lmda `model_regress` design path |
| 4 | `linalg_decompose` | `a`, `kind` ∈ svd/eig/eigh/qr/cholesky/lu/schur, `full_matrices`, `lower` | factor matrices by name (`u,s,vt` / `w,v` / `q,r` / `p,l,u`) | P0 | eig returns complex as re/im; sign/ordering conventions documented (LAPACK) and tested, not promised across arch |
| 5 | `linalg_matrix_fn` | `a`, `fn` ∈ inv/pinv/det/slogdet/norm/rank/cond/expm/logm/sqrtm/trace, `ord` | `value` (scalar or matrix) | P0 | expm/logm from `scipy.linalg`; n ≤ 2000 |
| 6 | `sparse_solve` | `rows`,`cols`,`vals` (COO triples), `shape`, `b`, `method` ∈ spsolve/cg/gmres/bicgstab/eigsh, `k`, `which`, `tol`, `maxiter` | `x` or eigen pairs, `converged`, `iterations` | P2 | ARPACK (`eigsh`) needs seeded `v0`; nnz ≤ 5M |

### 3.2 scipy numerics (optimize / integrate / interpolate / signal / spatial / special / stats)

| # | Op | Params | Result | Prio | Risk / notes |
|---|---|---|---|---|---|
| 7 | `opt_minimize` | `objective` (expr, §4.1), `start{}`, `method` ∈ Nelder-Mead/Powell/BFGS/L-BFGS-B/TNC/trust-constr-box, `bounds{}`, `tol`, `maxiter` ≤ 10000, `data{}` | `par{}`, `value`, `nfev`, `nit`, `success`, `message`, `hessian_inv`? | P0 | expression surface; same grammar as rcli so a recipe can run both (`minimize-rosenbrock.lms` becomes a cross-tool oracle) |
| 8 | `opt_curve_fit` | `model` (expr in params + `x`), `x`,`y`,`sigma`?, `start{}`, `bounds{}`, `method` ∈ lm/trf/dogbox, `maxfev` | `par{}`, `covariance`, `std_error`, `rss`, `converged` | P0 | oracle vs rcli `fit_nls`; distinct algorithms, so tolerance 1e-5 |
| 9 | `opt_root_scalar` | `f` (expr in `x`), `bracket[2]` or `x0`, `method` ∈ brentq/bisect/newton-secant | `root`, `iterations`, `converged` | P1 | vs rcli `num_root` |
| 10 | `num_integrate` | `f` (expr in `x`), `lower`,`upper` (numbers or `"Inf"`), `epsabs`,`epsrel`,`limit` ≤ 1000 | `value`,`abs_error`,`neval` | P1 | QUADPACK; vs rcli `num_integrate` |
| 11 | `num_ode_solve` | `rhs{}` (one expr per state var, over state names, `t`, params), `y0{}`, `t_span`, `t_eval` or `n_points`, `method` ∈ RK45/DOP853/Radau/BDF/LSODA, `rtol`,`atol`, `params{}`, `max_evals` | table `t` + one column per state, `status`, `nfev` | **P0** | **new capability**: rcli has no ODE solver (deSolve not in base R), lmda/lmstats none. The vectorised expr is evaluated once per RHS call; `max_evals` and timeout guard runaway stiff systems |
| 12 | `interp_1d` | `x`,`y`,`xout`, `kind` ∈ linear/nearest/cubic_spline/pchip/akima/bspline(k), `bc_type`, `extrapolate` | `yout`, optional `derivative` order 1-2 | P1 | vs rcli `interp_spline`/`interp_approx` (different spline conventions: document) |
| 13 | `signal_filter` | `x`, `fs`, `kind` ∈ butter/cheby1/bessel, `btype`, `order`, `cutoff[]`, `zero_phase` (sosfiltfilt), or `kind:"savgol"` with `window`,`polyorder` | `y`, `sos` or coefficients | P1 | n ≤ 2M |
| 14 | `signal_spectrum` | `x`, `fs`, `method` ∈ periodogram/welch, `nperseg`, `window`, `detrend`, `scaling` | `f`, `pxx` | P1 | vs lmda `series_spectrum` |
| 15 | `signal_peaks` | `x`, `height`,`distance`,`prominence`,`width` | `indices`, `properties` table | P2 | |
| 16 | `fft` | `x` (real or `re`/`im`), `inverse`, `norm`, `n` | `re`,`im` or real output for `rfft`/`irfft` | P1 | pocketfft; vs rcli `num_fft` (exact-ish, 1e-12) |
| 17 | `spatial_query` | `points` (n×d), `kind` ∈ kdtree_knn/kdtree_radius/convex_hull/delaunay/distance_matrix/voronoi_regions, `queries`, `k`, `r`, `metric` | neighbours+distances / simplices / hull vertices+volume / matrix | P2 | Qhull: `qhull_options` fixed to `"Qbb Qc Qz"` defaults, never caller text; d ≤ 10, n ≤ 200k |
| 18 | `special_fn` | `fn` ∈ gamma/gammaln/digamma/beta/betaln/erf/erfc/erfinv/expit/logit/lambertw/jv/yv/iv/kv/zeta/comb/perm/hyp2f1, `x`,`y`,`z` arrays | `value` | P1 | vs rcli `math_special`; direct oracle for lmda/lmstats special functions |
| 19 | `stats_dist` | `dist` ∈ norm/t/chi2/f/gamma/beta/expon/lognorm/weibull_min/cauchy/logistic/uniform/binom/poisson/nbinom/geom/hypergeom, `fn` ∈ pdf/cdf/ppf/sf/isf/logpdf/logcdf/rvs/moments/fit, params (`loc`,`scale`,`df`,`a`,`b`,`n`,`p`,…), `x`, `n`, `seed` | `value` or `params` (for `fit`) | P0 | vs rcli `dist_*` (1e-13 rel) — **the first cross-tool oracle, cheap and decisive**; `fit` is MLE with `fix_*` options |
| 20 | `stats_test` | `test` ∈ ttest_ind/ttest_rel/ttest_1samp/mannwhitneyu/wilcoxon/kstest/ks_2samp/shapiro/normaltest/chi2_contingency/fisher_exact/pearsonr/spearmanr/kendalltau/levene/bartlett/kruskal/anova_oneway, `x`,`y`,`table`, options (`alternative`,`equal_var`,`correction`,`method`) | `statistic`,`p_value`,`df`,`ci`,`estimate`,`method` | P1 | three-way oracle rcli ↔ pycli ↔ lmda; scipy's exact/asymptotic choices differ from R's and are documented, not harmonised |
| 21 | `stats_describe` | `x` (array or table columns), `quantiles[]`, `method` (quantile method ∈ numpy's 9 + linear) | moments, quantiles, skew, kurtosis (bias options) | P1 | vs rcli `summary_quantile` types; vs lmda `summary_describe` |

### 3.3 pandas (tables)

Overlap with lmda is deliberate for `info/describe/filter/groupby/merge/pivot` (oracle role); `resample`,
`rolling` with datetime semantics, categoricals, timezones and parquet are capabilities lmda lacks or only
partially has.

| # | Op | Params | Result | Prio | Risk / notes |
|---|---|---|---|---|---|
| 22 | `table_info` | table, `sample` ≤ 20 | dtypes, missing counts, memory, head sample | P1 | |
| 23 | `table_describe` | table, `columns[]`, `percentiles[]`, `include` ∈ number/all | describe table | P1 | vs lmda `summary_describe` |
| 24 | `table_filter` | table, `where` (predicate, §4.1), `select[]`, `sort_by[]`, `descending[]`, `limit`, `offset`, **W** `out`? | table (inline, truncated) or file | P1 | `where` is the structured/expr predicate, never `DataFrame.query` |
| 25 | `table_groupby_agg` | table, `by[]`, `aggs: [{"col","fn" ∈ sum/mean/median/min/max/count/nunique/std/var/first/last/quantile(q), "as"}]`, `dropna` | table | P1 | vs lmda `table_groupby`; `std` ddof=1 documented |
| 26 | `table_merge` | `left`,`right` (two table inputs), `on[]`/`left_on[]`/`right_on[]`, `how` ∈ inner/left/right/outer/cross, `suffixes`, `validate` ∈ 1:1/1:m/m:1/m:m | table | P1 | `validate` default `null`; indicator column optional |
| 27 | `table_pivot` | table, `index[]`,`columns`,`values`,`aggfunc`, `fill_value`, or `melt` mode (`id_vars`,`value_vars`) | table (flattened column names `value_level`) | P1 | MultiIndex flattened with `_`; collision → error |
| 28 | `table_resample` | table, `on` (datetime col), `rule` (pandas offset alias from an **enum list**: `1min 5min 15min h D W ME QE YE` etc., validated by regex `^\d+(s|min|h|D|W|ME|QE|YE)$`), `aggs`, `closed`,`label`,`origin` | table | P1 | **new vs siblings**; tz handling via §2.2 |
| 29 | `table_rolling` | table, `columns[]`, `window` (int or time string), `on`?, `fns[]`, `min_periods`, `center`, `ewm` (`span`/`halflife`/`alpha`) | table with new columns | P1 | vs lmda `series_rolling`/`series_ewma` |
| 30 | `table_convert` **W** | table in (csv/parquet/npz/inline), `out` (csv/parquet/npz), `dtypes{}`, `compression` | file descriptor + dtypes | P1 (csv/npz), P2 (parquet) | the bridge op: everything large goes through it |

### 3.4 scikit-learn

| # | Op | Params | Result | Prio | Risk / notes |
|---|---|---|---|---|---|
| 31 | `ml_preprocess` | `x` (n×d) or table+`columns`, `kind` ∈ standard/minmax/robust/quantile_uniform/onehot/ordinal, **W** `model_out`?, `apply_to`? | transformed `x` (or file), `params` (means/scales/categories) | P1 | persisted as pycli-model (§4.3): closed form, safe |
| 32 | `ml_pca` | `x`, `n_components` (int or variance fraction), `whiten`, `svd_solver` ∈ full/arpack, `seed` (arpack only), **W** `model_out`? | `components`, `explained_variance[_ratio]`, `singular_values`, `scores` (opt-in), `mean` | P1 | vs lmda `multi_pca` (sign convention: sklearn `svd_flip` — document) |
| 33 | `ml_kmeans` | `x`, `k`, `init` ∈ k-means++/random, `n_init`, `max_iter`, `tol`, `algorithm` ∈ lloyd/elkan, `seed`, **W** `model_out`? | `centers`, `labels`, `inertia`, `n_iter` | P1 | vs lmda `cluster_kmeans`, rcli `cluster_kmeans` (ARI=1 check, not label equality) |
| 34 | `ml_fit_predict` | `family` ∈ linear/ridge/lasso/elasticnet/logistic/svm_linear/knn/decision_tree/random_forest/gradient_boosting/naive_bayes, `x_train`,`y_train`,`x_test`?,`y_test`?, `hyper{}` (closed per-family set: `alpha`,`l1_ratio`,`C`,`penalty`,`max_depth`,`n_estimators`,`learning_rate`,`n_neighbors`,`class_weight`), `seed`, **W** `model_out`? (linear families only) | per family: `coef`,`intercept`,`classes`, `feature_importances`, train/test `predictions` (opt-in, capped), `metrics` (auto by task), `n_iter`, `converged` | P1 | tree/forest/GBM/SVM/kNN models **cannot be persisted safely** (sklearn internals are Cython objects; the only sklearn save format is pickle) → these families fit and predict in one call only; `model_out` for them is a validation error with `tool_code:"MODEL_NOT_PERSISTABLE"`, not a silent fallback |
| 35 | `ml_predict` | `model` (pycli-model path), `x` or table, `proba` | `predictions`, `probabilities`? | P1 | only families in §4.3 |
| 36 | `ml_cv` | as `ml_fit_predict` minus test, `cv` ∈ kfold/stratified/timeseries, `folds`, `scoring[]` (enum of sklearn scorer names), `seed`, `shuffle` | per-fold scores, mean, std, fold sizes | P1 | |
| 37 | `ml_metrics` | `y_true`,`y_pred` or `y_score`, `task` ∈ regression/binary/multiclass, `metrics[]` enum, `average`, `pos_label` | metrics, `confusion_matrix`, `roc` points (opt-in) | P1 | vs lmda `eval_classify`/`eval_roc` |

### 3.5 statsmodels (the lmstats oracle)

lmstats is a port of statsmodels 0.14.x. These three ops are therefore the **reference implementation** of
lmstats itself and the highest-value oracle in the fleet; rcli's `fit_glm`/`ts_arima` are a second opinion.

| # | Op | Params | Result | Prio | Risk / notes |
|---|---|---|---|---|---|
| 38 | `sm_ols` | table, `response`, `terms[]` (§4.2 grammar), `weights`?, `cov_type` ∈ nonrobust/HC0-HC3/HAC(maxlags), `return[]` | coefficient table, `r2`,`r2_adj`,`f`,`f_p`,`loglik`,`aic`,`bic`,`df`, `ssr`, `durbin_watson`, opt-in `fitted`,`residuals`,`vif`,`influence` | P1 | vs lmstats `regression`, rcli `fit_lm`, lmda `model_regress` |
| 39 | `sm_glm` | table, `response`, `terms[]`, `family` ∈ gaussian/binomial/poisson/gamma/negative_binomial(alpha)/tweedie(var_power), `link` (enum per family), `offset`,`exposure`,`weights`, `maxiter`, `tol` | coefficient table, `deviance`,`null_deviance`,`pearson_chi2`,`loglik`,`aic`,`bic`,`converged`,`n_iter`, opt-in `fitted` | P1 | vs lmstats `glm`/`poisson`/`logit`, rcli `fit_glm` |
| 40 | `sm_arima` | `series`, `order[3]`, `seasonal_order[4]`, `trend` ∈ n/c/t/ct, `exog`,`exog_future`, `method` ∈ statespace(SARIMAX)/innovations_mle/hannan_rissanen/burg/yule_walker, `enforce_stationarity`,`enforce_invertibility`, `maxiter`, `forecast_steps`, `confidence_level` | params table, `sigma2`,`loglik`,`aic`,`bic`,`converged`, `forecast` (mean, lower, upper), opt-in `residuals` | P1 | **mirrors lmstats `arima` request field-for-field** so `oracle-arima.lms` can call both with one object; SARIMAX MLE is tolerance 1e-4 on loglik vs lmstats |

### 3.6 networkx

| # | Op | Params | Result | Prio | Risk / notes |
|---|---|---|---|---|---|
| 41 | `graph_analyze` | `edges` (table `source,target[,weight]`) or `adjacency`, `directed`, `metrics[]` ∈ degree/betweenness/closeness/eigenvector/pagerank/clustering/components/diameter/density/triangles/louvain(seed) | per-node table (opt-in) + graph scalars + communities | P2 | vs lmda `graph_summary`/`graph_centrality`; n ≤ 100k nodes, betweenness ≤ 5k nodes (O(nm)); louvain is seeded; pure Python so slow is fine |
| 42 | `graph_shortest_path` | graph, `source`,`target`?, `weight`, `method` ∈ dijkstra/bellman_ford/astar(no heuristic)/all_pairs(n ≤ 2000) | paths, lengths | P2 | |

Total: 42 ops listed; the first ~30 by priority are 1-5, 7, 8, 11, 19 (P0: 9 ops) and 9, 10, 12-14, 16, 18, 20-30,
31-40 (P1). P2 (6, 15, 17, 41, 42, parquet) is the cut line if time runs out.

Not in the catalogue, by decision: generic `apply`/`map`/`eval`/`query`, user-defined functions of any kind,
plotting, deep learning (torchcli), text/NLP, image I/O, Excel, HDF5, SQL, anything that needs network, model
hyper-parameter search beyond `ml_cv`, pipelines, `.pkl`/`.joblib` in either direction.

---

## 4. Safety

### 4.1 Expressions: one grammar, one parser, zero `eval`

Every expression-like input (`objective`, `model`, `f`, `rhs`, `where`) uses rcli's expression design verbatim
(design-expr.md): ASCII-only string ≤ 2048 bytes, hand-written C++ recursive-descent parser enforcing
`expr-spec.json` (allowlisted functions with fixed arity, operators, `ifelse`, constants `pi`/`e`, caps on tokens,
nodes, depth, idents), canonical JSON AST in the request. **Proposal: move `expr_parser.hpp` and the spec into
lmcpp (`lmcpp::expr`) so rcli and pycli share one parser and one 45+-case negative corpus**; until then, copy it
unchanged with the corpus. The function allowlist maps to numpy objects in the worker:

```python
FN = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pnorm": scipy.special.ndtr, "ifelse": np.where,
      "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "^": np.power,
      "<": operator.lt, ..., "sum": np.sum, "mean": np.mean, ...}   # objects, never names
def build(node, names):            # returns a closure env -> ndarray; re-validates kind/arity/declared names
    k = node["t"]
    if k == "num":  v = float(node["v"]); return lambda env: v
    if k == "var":  n = node["n"]; assert n in names; return lambda env: env[n]
    if k == "call": f = FN[node["f"]]; args = [build(a, names) for a in node["a"]]; return lambda env: f(*(a(env) for a in args))
    ...
```

`env` is a plain dict of ndarrays (parameters broadcast as 0-d, data columns as 1-d, `x`/`t` for the 1-D ops);
`pandas.eval`, `numexpr`, `DataFrame.query`, `sympy`, `ast.literal_eval` on caller text are never used. The
C++ side already rejects undeclared identifiers, so a mismatch in Python is `INTERNAL_ERROR/EXPR_SPEC_MISMATCH`.
Runtime guards: evaluation counter with `max_evals` (default 500k, cap 2M → `EVAL_LIMIT`), scalar-objective check
at `start` (`EXPR_NOT_SCALAR`), vectorised-length check for 1-D ops, `nonfinite` policy ∈ inf/error for minimisers,
`np.errstate` warnings captured.

`where` for `table_filter` is the same grammar extended with the relational layer at top level (the AST is a
boolean expression over column names and literals, plus `isnull(col)` and `in(col, [literals])` as two extra
allowlisted forms); strings compare only via `==`/`!=`/`in` with string literals that the C++ parser accepts in
this one context (escape-free, ≤ 256 bytes, printable ASCII). This replaces `DataFrame.query` entirely.

### 4.2 Formulas and column names

No formula strings. `sm_ols`/`sm_glm` take `response: "y"` and `terms: ["x1", "x2", "x1:x2", "C(g)",
"I(x1^2)"]` where each term must match the grammar `term := factor (':' factor)* ; factor := IDENT | 'C(' IDENT ')' |
'I(' expr ')'` with `expr` from §4.1. The worker builds the design matrix itself (`pandas.get_dummies(drop_first=True)`
with treatment coding against the first sorted level, products for `:`, the expression evaluator for `I()`,
intercept column unless `intercept:false`), then calls `sm.OLS(y, X)` / `sm.GLM(y, X, family=…)`: **patsy and
formulaic are never invoked**, because patsy evaluates column names and `C()`/`I()` contents as Python. The
design-matrix convention (treatment coding, reference level = first sorted level, term order as given) is
documented and oracle-tested against `smf.ols` in the test image only. Column names are only ever dict keys.

### 4.3 Models: no pickle, ever

The worker never calls `pickle`, `joblib`, `np.load(allow_pickle=True)`, `torch.load`, `cloudpickle`;
`sys.modules["pickle"]` is replaced by a stub that raises at import time of the worker so that any library path
that pickles implicitly (sklearn `clone` does not; `joblib.Memory` would) fails loudly. Persistent model format
`*.pycli-model.json`:

```json
{"pycli_model": 1, "family": "logistic", "created_by": {"pycli": "0.1.0", "sklearn": "1.7.2"},
 "task": "binary", "features": ["x1","x2"], "classes": [0,1],
 "params": {"coef": [[…]], "intercept": […], "scaler": {"kind":"standard","mean":[…],"scale":[…]}},
 "hyper": {"C": 1.0, "penalty": "l2"}, "sha256_train_x": "…"}
```

Persistable families (closed-form parameter sets, reconstructed by assigning public fitted attributes to a fresh
estimator, which sklearn supports for linear models, or by our own numpy forward pass): linear, ridge, lasso,
elasticnet, logistic, scalers, PCA, KMeans, ordinal/onehot encoders, naive_bayes (gaussian). Not persistable in
v1: trees, forests, GBM, SVM (support vectors are persistable in principle; cut for time), kNN (would require
storing the training set). `ml_predict` validates `pycli_model` version, family, feature list and shapes before
use; unknown keys are rejected. Loading a model file is `NOT_FOUND`/`PARSE_ERROR`/`VALIDATION_ERROR`, never code.

### 4.4 Files, process, resources

| Surface | Policy |
|---|---|
| Paths | §2.1: cwd-confined canonical paths, no `#` in paths, extension-dispatched readers, size caps before open, `.npy` header validation, `allow_pickle=False`, parquet via pyarrow (no code), CSV via pandas C engine (`engine="c"`, `encoding="utf-8"`, `on_bad_lines="error"`), **no** `.pkl .joblib .h5 .xlsx .json-with-nan .feather`. Outputs: exclusive `out`, atomic install, `write` access, mode 0644 |
| Python process | `-I -B -X utf8 -X faulthandler`, scrubbed env (§1.2), `HOME`/`TMPDIR` in the private temp dir, stdin `/dev/null`, `resource.setrlimit(RLIMIT_AS, 4 GiB)`, `RLIMIT_CPU` = timeout + 5, `RLIMIT_NPROC` small, `RLIMIT_FSIZE` 1 GiB, `sys.setrecursionlimit(2000)`, `signal.alarm(timeout - 2)` raising `TIMEOUT` inside Python so partial results and warnings survive; lmcpp kills the process group at the hard timeout |
| Container | shim from torchcli/rcli: `--network=none`, `--cpus=2 --memory=5g --pids-limit 128 --read-only --tmpfs /tmp:size=1g`, `--security-opt no-new-privileges`, `--cap-drop ALL`, cwd bind mount only, host uid/gid, `PYCLI_READONLY/PYCLI_DRYRUN/PYCLI_TIMEOUT_S` passthrough. No network at runtime is enforced by Docker, and the worker additionally monkeypatches `socket.socket` to raise, so even a native (non-Docker) run cannot open connections |
| Native run | `PYCLI_PYTHON=/path/to/venv/bin/python3 build/bin/pycli …` for development only; goldens run in the image |
| Caps | per-op complexity caps validated in C++ before spawning (n·d for kdtree, n² for betweenness, n³ for dense decompositions ≤ 2000³ ≈ 8e9 flops, nnz for sparse, rows×cols for tables); output caps: 200k numbers inline, 16 MiB JSON |

Residual risks I cannot fully close: a bug in a C library reached through a validated numeric path (Qhull on
degenerate input, LAPACK on NaN input) can crash or hang the child; that is contained by process isolation,
timeout, and `rlimit`, and surfaces as `TOOL_CRASH`/`TIMEOUT`, never as corrupted output.

---

## 5. Packaging

### 5.1 Runtime image

- Base: `python:3.12-slim-bookworm@sha256:<multi-arch manifest digest>` (already cached on this host for 3.11;
  3.12 is the host Python, so a developer venv and the image agree). Debian glibc 2.36 ≥ manylinux_2_28 floor.
  No conda (size, solver nondeterminism, licence of the default channel), no `python3-numpy` from apt (Debian's
  numpy is old, built against system OpenBLAS with different threading, and bookworm has no sklearn ≥ 1.4).
- Wheels only: `pip install --only-binary=:all: --require-hashes --no-deps -r docker/requirements.lock`.
  The lock is produced by `pip-compile --generate-hashes --allow-unsafe` on the host from
  `docker/requirements.in`; pip-tools records **every** published file hash for each version, so one lock file
  covers aarch64 and amd64. A CTest asserts the lock hashes match the installed `RECORD` files in the image.
- Pins (`requirements.in`; exact patch versions resolved and hash-locked at slice 0, numpy 2.3.3 **verified** to
  exist for cp312 aarch64): `numpy~=2.3`, `scipy~=1.16`, `pandas~=2.3`, `scikit-learn~=1.7`, `statsmodels~=0.14.5`,
  `networkx~=3.5`, `threadpoolctl`, `safetensors` (numpy API only), `pyarrow~=21` (slice 6, separate lock stage so
  the first slices do not pay 130 MB). Nothing else: no matplotlib, numexpr, bottleneck, numba, joblib beyond
  sklearn's hard dependency (present but never called by us), no jupyter.
- Size estimate (not measured): base 150 MB + numpy 60 + scipy 110 + pandas 70 + sklearn 45 (+ joblib/threadpoolctl 5)
  + statsmodels 40 + networkx 10 + pycache ≈ **550-650 MB unpacked without pyarrow, ~800 MB with**. Builder stage
  adds g++/cmake/ninja (~400 MB, discarded). Disk budget on this host: < 3 GB including build cache. Strip
  `tests/` directories from site-packages (`numpy/tests`, `scipy/**/tests`, `pandas/tests`, `sklearn/**/tests`
  are ~150 MB) in the runtime stage only, keep them in the test stage.
- Stages: `builder` (lmcpp + pycli C++ build, 2 jobs), `test` (site-packages + oracle scripts + ctest + sibling
  tool binaries for cross-oracles when present), `runtime` (python + site-packages + `/usr/local/bin/pycli`
  entrypoint; no compiler, no pip). Build context is the repos root with a strict allowlist `.dockerignore`
  (torchcli pattern) so lmcpp comes from the sibling checkout at a pinned commit.
- amd64: the host cannot run amd64 containers (**verified**), so the amd64 image is built and tested only in CI
  (`ubuntu-24.04` runner); aarch64 in CI on `ubuntu-24.04-arm` and locally. Both publish `cli.schema.dev.json`
  and the golden report as artifacts; the schema must be byte-identical across arches (CI diff).

### 5.2 Host shim and CI

`bin/pycli` is rcli's shim with the name changed, image `pycli:runtime`, extra hardening flags from §4.4,
`PYCLI_*` env passthrough, same Docker-failure → `SERVICE_UNAVAILABLE` and crash → `TOOL_CRASH` mapping, and
`.pycli-tmp` under the mounted cwd for lmscript `--input-file` visibility. CI: build test image → ctest (oracle +
sandbox corpus + envelope tests) → build runtime → `lmscript --check` all recipes → run recipes that need only
pycli → run cross-tool recipes when the sibling binaries are present (rcli image, `statsrust`, `lmda`) else
mark skipped **and fail the job if `PYCLI_REQUIRE_CROSS_ORACLE=1`**, so the nightly cannot silently skip.

### 5.3 Licences and notices

Our code MIT (Lattice Systems LLC). `THIRD_PARTY_NOTICES.md` is generated by `scripts/notices.py` in the test
image from `importlib.metadata` (name, version, licence classifier, homepage) plus a hand-maintained section for
libraries vendored inside wheels that metadata does not list: OpenBLAS (BSD-3, inside numpy/scipy), LAPACK
(BSD-3-modified), pocketfft (BSD-3, inside numpy/scipy), ARPACK-NG (BSD-3), Qhull (Qhull licence — must be
reproduced verbatim), QUADPACK/ODEPACK/MINPACK (public domain / BSD), SuperLU (BSD-3), libgfortran and libgomp
(GPL-3 with GCC Runtime Library Exception: redistribution inside the image is permitted, note it), Cython-generated
code (Apache-2.0 for the generator, output under the project's licence), Python (PSF-2.0), pandas (BSD-3), scipy
(BSD-3), scikit-learn (BSD-3), statsmodels (BSD-3), networkx (BSD-3), pyarrow/Arrow (Apache-2.0, bundles
Thrift/Snappy/zstd/lz4 — Apache/BSD), safetensors (Apache-2.0), threadpoolctl (BSD-3), joblib (BSD-3). A CTest
fails when a package in the image is missing from the notices file or carries a licence classifier outside
{MIT, BSD, Apache, PSF, Qhull, GPL-with-runtime-exception (listed explicitly)}. Nothing here is GPL-linked into
our binary: the C++ binary links only lmcpp and nlohmann (MIT). We do not push images in v1.

---

## 6. Test strategy and slice plan

### 6.1 Oracle layers

1. **Glue oracle (every op, CTest, in the test image).** `tests/oracle/<op>.py` computes the expected value by
   calling the library directly (independent code path from `worker.py`, written by the test author from the
   docs), 3-6 fixtures per op including edge cases (NaN/Inf inputs, n=1, singular matrices, ties, empty groups,
   tz-aware datetimes, unseen categories). Expected JSON is produced in the same image; assertions:
   - **exact** (`==` after JSON round trip) for pure-Python/integer paths, RNG draws, `stats_dist` pdf/cdf/ppf,
     `special_fn`, pandas ops, `table_convert` round trips, model file round trips;
   - **rel 1e-13** for BLAS-touching closed forms on the *same* image/arch (solve, lstsq, svd values, fft);
   - **rel 1e-8** IRLS/QR fits; **rel 1e-5** iterative optimisers; ARI = 1 for cluster labels; `np.allclose(rtol=1e-6)`
     on forecasts.
   The same fixture set runs on amd64 in CI with the tolerance tiers, which doubles as the cross-arch statement.
2. **Envelope tests** (CTest, no library needed beyond numpy): nonfinite in/out mapping, `null` vs `"NaN"` rules,
   warnings capture, `PY_ERROR` mapping with exception class in `data`, timeout, crash (`kill -SEGV` injected
   via a hidden test op), missing interpreter → `SERVICE_UNAVAILABLE`, `OUTPUT_TOO_LARGE`, readonly/dryrun
   floors, `CONFLICT` on existing `out`, path escape, `.npy` header fuzz (dtype, shape overflow, fortran order,
   truncated), `--schema` offline with Python removed from PATH.
3. **Sandbox corpus** (shared with rcli): `tests/expr-corpus.json` plus pycli-specific additions
   (`__import__`, `os.system`, `().__class__`, column named `__builtins__`, term `C(os.system("id"))`, `I(open("x"))`,
   `where` with a string containing `"` or newline, model file with extra key `"__reduce__"`, `.npy` with
   `dtype: object`, `.npz` containing a pickled member). Each must fail with the expected `tool_code` and
   **never spawn Python** (asserted by a wrapper that fails if the interpreter path is touched).
4. **Cross-tool oracle recipes** (`examples/oracle-*.lms`, report + gate per tolerance, extend rcli's
   `oracle-findings.md` pattern): `stats_dist` ↔ rcli `dist_*` (1e-13); `opt_minimize` Rosenbrock ↔ rcli (1e-6 on
   value); `sm_arima` ↔ lmstats `arima` ↔ rcli `ts_arima` (loglik 1e-4; coefficients 1e-3); `sm_glm` ↔ lmstats
   `glm` ↔ rcli `fit_glm` (1e-8); `table_groupby_agg`/`table_describe` ↔ lmda (exact for sums/counts, 1e-12 for
   means); `ml_kmeans` ↔ lmda/rcli (ARI); `ml_pca` ↔ lmda `multi_pca` (abs components up to sign, 1e-10);
   `linalg_matmul`-shaped check ↔ torchcli via a safetensors file written by torchcli and read by `linalg_matrix_fn`
   (`norm`), proving the `#key` ref path. Tolerances are never raised to hide a mismatch; a mismatch gets a row in
   `docs/oracle-findings.md` with a diagnosis.
5. **lmapi conformance** inherited from lmcpp; `cli.schema.dev.json` parsed by pinned lmts; `lmscript --check` on
   every recipe; `--dry-run` tolerated by every recipe (`status:"simulated"` branches).

### 6.2 Slices (each test-first, independently verifiable by a cheaper coding model)

| Slice | Deliverable | Reference oracle | Reviewer must check |
|---|---|---|---|
| **0** vertical slice | lmcpp link, `py_info`, `linalg_solve`, `stats_dist` (norm/t only), worker protocol, `.npy` writer, nonfinite rules, meta, Docker test+runtime (numpy+scipy only), shim, `examples/solve.lms` | `tests/oracle/{linalg_solve,stats_dist}.py`; `solve` of a 3×3 with known exact inverse; `norm.ppf(0.975)` = 1.959963984540054 | worker has no `eval/exec/pickle` (grep test passes); `--schema` works without Python; `-I` and scrubbed env in `run_child` argv/env (inspect the test's captured argv); temp dir removed on success, failure and SIGTERM; **measured** timings recorded in `docs/timings.md`: interpreter boot, numpy import, scipy.stats import, docker overhead; `null`/`"NaN"` rules in both directions |
| **1** numpy/scipy core | ops 2-5, 16, 18, 19 (all families), 21; `.npz`/safetensors refs; `oracle-dist.lms` vs rcli | glue oracles + rcli cross recipe | decomposition conventions documented (sign, ordering) and tested against `scipy.linalg` directly; complex output shape; caps enforced before spawn; cross-oracle report committed |
| **2** expressions | `lmcpp::expr` (or copied parser) + corpus, ops 7, 8, 9, 10, 11 (ODE); `minimize-rosenbrock.lms` (shared with rcli), `ode-lotka-volterra.lms` | scipy direct; Rosenbrock minimum; Lotka-Volterra conserved quantity drift < 1e-6 with DOP853 rtol 1e-10; rcli `fit_nls` ↔ `opt_curve_fit` | corpus passes on both tools; function table contains objects only; `max_evals` and `EXPR_NOT_SCALAR` paths; ODE `rhs` evaluation count bounded; no `lambda` built from caller strings |
| **3** pandas | ops 22-30 (csv/npz), dtype policy, datetimes/tz/categorical, `where` predicate; `oracle-tables.lms` vs lmda | pandas direct; lmda | every dtype in the policy table has an in/out fixture; MultiIndex flattening; `limit`/`truncated`; merge `validate`; resample `rule` regex rejects `pd.offsets` escape attempts; atomic write + CONFLICT |
| **4** statsmodels | ops 38-40, design-matrix builder with term grammar; `oracle-arima.lms`, `oracle-glm.lms` vs lmstats and rcli | `smf.ols`/`smf.glm` formula path in the test image (patsy allowed **only** in tests) must equal our hand-built design matrix fits bit-for-bit; lmstats | patsy/formulaic absent from the runtime import graph (test asserts `"patsy" not in sys.modules` after each op); treatment coding reference levels documented; `sm_arima` request mirrors lmstats fields |
| **5** sklearn | ops 31-37, pycli-model format, `ml_predict`; `train-eval.lms` | sklearn direct; model round trip predicts identically to in-memory estimator (exact) | `MODEL_NOT_PERSISTABLE` for tree families; `pickle` stub active; `random_state` plumbing; metrics enums closed; `sha256_train_x` recorded |
| **6** breadth | ops 6, 13-15, 17, 41, 42; pyarrow + `table_convert` parquet; runtime image trimming | scipy/networkx direct; parquet round trip exact incl. tz | Qhull options are constants; sparse caps; image size and per-call timings re-measured and committed |
| **7** release | CI (arm64 + amd64), `THIRD_PARTY_NOTICES.md` + notices test, `docs/parity.md`, `docs/oracle-findings.md`, README with timing table | — | schema identical across arches; amd64 goldens pass at tolerance tiers; notices test green; no image push |

Each slice ends with `cli.schema.dev.json` regenerated and diffed, all recipes `--check`ed, and the timing table
updated. A slice is reviewable by: run ctest in the test image; read the oracle scripts (are they independent of
the worker?); grep the worker diff for forbidden tokens; run the sandbox corpus; run the cross-tool recipe.

---

## 7. Risks, open questions, cuts

### 7.1 Risks (ordered)

1. **Per-call latency.** Docker start (~0.7 s) + interpreter + imports could reach 2-4 s for pandas/sklearn/
   statsmodels ops and ~1 s for numpy-only ops; an lmscript loop of 50 calls is minutes. Mitigations in v1: lazy
   imports, precompiled bytecode, vectorised ops (arrays, `aggs[]`, `metrics[]`, `t_eval`). Decision gate at
   slice 0 with real numbers; if statsmodels import exceeds ~3 s, split `sm_*` into a second worker that imports
   `statsmodels.tsa.statespace` only, and schedule the persistent-worker mode (§1.1 d) for v2.
2. **Sandbox surfaces.** Three remain: the expression AST evaluator, the term grammar for designs, the `where`
   predicate with string literals. All reduce to "validated identifiers used as dict keys + function objects from
   a closed table". The failure mode is a *library* that evaluates text given a string (patsy, `DataFrame.query`,
   `pd.eval`, `np.fromstring`, `numexpr`); the mitigation is the forbidden-token test plus `sys.modules` assertions
   after each op in tests. I rate residual risk low but non-zero; the Docker/`--network=none`/cwd-only mount is
   the second wall.
3. **Cross-arch and cross-version drift.** OpenBLAS kernels, scipy's QUADPACK/MINPACK compile flags, and sklearn's
   default changes make amd64 vs aarch64 and version bumps non-bit-identical. Mitigated by tolerance tiers and
   `meta`; residual risk is a tolerance that is too loose to catch a real bug — reviewers must challenge every
   tolerance above 1e-8 with a reason in the fixture.
4. **pandas semantic breadth.** Datetime/tz/categorical/nullable-int behaviour has many corners (DST transitions,
   `NaT` in groupby keys, categorical ordering, `object` columns of mixed types). Strict rejection of anything
   outside the dtype table turns most corners into `UNSUPPORTED_DTYPE` rather than wrong output; the cost is
   rejected files that lmda would accept.
5. **Model persistence gap.** Only closed-form families are saved; a recipe that wants a forest must fit and
   predict in one call. Honest limitation; alternative (ONNX via skl2onnx + onnxruntime) adds ~100 MB and another
   parser and is deferred until a recipe needs it.
6. **Disk and build time.** ~800 MB runtime image plus a ~1.5 GB builder/test layer, in addition to rcli and
   torchcli images on a 54 GB-free host; pyarrow is the single biggest optional cost (deferred to slice 6, lock
   split so earlier slices never download it).
7. **Library crashes on hostile numeric input** (Qhull degenerate points, LAPACK on NaN, ARPACK non-convergence
   hangs): contained by process isolation and timeouts; surfaced as `TOOL_CRASH`/`TIMEOUT` with the op and sizes
   in `error.data` so the fixture can be added.

### 7.2 Open questions

- Should the table envelope be columnar (this design) or rcli's row arrays? I prefer columnar and propose the
  shared `lmcpp::table` helper; the Lead should pick one shape for the fleet before slice 3.
- Should `stats_test`/`stats_describe` ship at all given rcli + lmda cover them? I keep them as P1 because the
  three-way oracle is cheap and scipy is what most Python-trained users expect; cut first if the budget is tight.
- Lift `expr_parser` into lmcpp now (touches rcli's build) or copy now and lift at slice 7? I recommend lifting
  now: the corpus is the most valuable security artifact and should have one home.
- Python 3.12 vs 3.11: 3.12 matches the host and has wheels for everything pinned; 3.13 free-threaded builds are
  irrelevant (single thread). Any objection is only about torchcli already using a 3.11 base (separate image,
  no conflict).
- Do we want `precision` (display rounding) at all, or leave token economy entirely to lmscript `sprintf`?
  I include it as opt-in because a 17-digit 200-row table is ~3× the tokens of a 6-digit one.

### 7.3 Cuts (not in v1)

libpython embedding; persistent worker; GPU; any `eval`/UDF/lambda/formula-string path; pickle/joblib/ONNX;
tree/forest model persistence; plotting; parquet before slice 6; sparse formats beyond COO-in/CSR-internal;
`scipy.spatial` Voronoi vertex geometry (regions only); multi-output/multi-label sklearn tasks; pipelines and
grid search; statsmodels beyond OLS/GLM/ARIMA (no VAR/ETS/mixed in v1: lmstats has them, rcli has none, add
later as oracles if lmstats parity work needs them); networkx beyond two ops; Excel/HDF5/SQL/feather I/O;
Windows/macOS; image publishing; conda; distro python packages.
