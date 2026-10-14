# pycli: the Python scientific stack as an lmapi native tool (design, Opus)

Status: design only, no code. Author: Opus, 2026-10-14. Licence of our code: MIT, Lattice Systems LLC.

**Decision in one paragraph.** `pycli` is an **MIT C++17 lmcpp front end** that validates every call (schema, caps,
paths, expression ASTs). It then runs **one fresh CPython child per call**, which executes a **fixed, embedded worker**
(a zipapp built from `worker/`) through a closed `dispatch[op]` table. This is rcli's layer (e) with a Python worker.
The stack comes from **hash-locked manylinux wheels** on the `python:3.11-slim-bookworm` base that torchcli already
uses (shared layers save disk). It runs in a per-call `--network=none` container through a `bin/pycli` shim. Data
moves between calls as **artifacts**: **Parquet** for frames, **safetensors** for arrays and models (torchcli-
compatible), and `.npy` read-only. Inline JSON carries only small inputs and compact results. No pickle is ever
loaded. No caller text reaches `eval`, `pandas.query/eval`, patsy formulas or QHull option strings. Expressions use
rcli's grammar, parsed in C++ and rebuilt in Python from a table of function objects. Value: (1) scipy/sklearn
algorithms no sibling has (LP/MILP via HiGHS, ODE, constrained optimisation, signal processing, QHull/KD-trees,
sparse/ARPACK, ML); (2) **real statsmodels 0.14.6 as the ground-truth oracle for lmstats**, which is a port of it;
(3) a third leg in the numerical oracle triangle with rcli and lmstats.

---

## 0. Facts checked on this host (2026-10-14)

| Fact | Consequence |
|---|---|
| aarch64, 4 cores, 15 GiB RAM, 54 GB free on `/` | 2 CPUs / 4 GiB per container. Image budget ≤ 1 GB, with no build-from-source |
| Host has Python 3.12.3 but **no numpy** | Dev "native" mode needs a venv (`PYCLI_PYTHON=`). Goldens run only in the image |
| `python:3.11-slim-bookworm@sha256:dabc8c…` is already pulled (torchcli base) | Reuse it, so the base layer costs 0 extra bytes |
| `rcli:runtime` is 354 MB and `torchcli:dev` is 1.01 GB | pycli's estimated ~0.75–0.9 GB sits between them |
| Ubuntu noble apt has numpy 1.26.4, scipy 1.11.4, sklearn 1.4.1, pandas 2.1.4 and no pyarrow | Distro route is old and incomplete (rejected, §5.2) |
| lmstats pins parity to **statsmodels 0.14.6** (PORTING.md) | pycli pins the same version so the oracle is exact |
| lmcpp has `run_child` (argv only, env replaced, process-group kill), `read_result_file` (16 MiB), one `temp_file_path()` and `allow_special_numbers` | Reuse all of them. Gaps are listed in §1.6 |
| rcli ships `src/expr_parser.hpp` and `expr-spec.json` (C++ recursive descent, AST → R builder) | Lift the parser into lmcpp and share it (§4.1) |

Not measured yet, but slice 0 must measure (all are estimates): python start ~30 ms; `import numpy` ~0.1 s;
`pandas` ~0.35 s; `scipy.optimize` ~0.3 s; `sklearn.ensemble` ~0.8 s; `statsmodels.api` ~1.2 s;
`docker run --rm` ~0.7 s (rcli measured 0.69 s).

---

## 1. Architecture

### 1.1 Options

| Option | Verdict | Why |
|---|---|---|
| (a) C++ lmcpp front end + fixed embedded Python worker, **fresh child per call** | **Chosen (v1)** | Crash isolation: a segfault in BLAS/QHull/ARPACK kills the child, and C++ still emits a protocol envelope. A hard timeout can kill the process group. Clean state per call: no RNG, warning-filter, BLAS-thread or monkeypatch leakage between calls. Licences are trivially separate. Same proven shape as rcli |
| (a′) same worker, **persistent** (session daemon) | Deferred behind a measured gate (§1.5) | Saves 0.3–1.5 s of imports per call. But the daemon needs lifecycle, concurrency, a state-reset protocol and replay/determinism proofs. Docker start (~0.7 s) dominates anyway unless the container is persistent too, which is a bigger change |
| (b) embed libpython (Python C API / pybind11) | **Reject** | No crash isolation: a numpy/BLAS fault takes down the lmapi envelope. The GIL plus C-extension threads make in-process timeouts impossible (you cannot kill a thread inside LAPACK). Linking libpython3.11 ties our binary to an exact Python ABI. Signal handlers (SIGINT) conflict. It gains only the ~30 ms interpreter start, because imports still cost the same per process |
| (c) call BLAS/LAPACK/FFTPACK/ARPACK/QHull directly from C++ | **Reject for v1**, keep as a note | Most of the value is the Python glue: scipy's `solve_ivp` drivers, the HiGHS front end, sklearn estimators, pandas semantics, statsmodels' likelihoods. Raw LAPACK duplicates what lmda/lmstats already get from faer/nalgebra, and results would not match the scipy oracle. Fortran ABI and hidden-symbol issues come from the wheels' bundled `libscipy_openblas` |
| (d) pure-Python CLI (argparse) with lmapi re-implemented in Python | Reject | It duplicates lmcpp's validator and schema publisher, which then drift from the conformance vectors. `--schema` would also need Python to start |
| (d′) Pyodide/WASM | Reject | Huge, slow, and missing parts of the stack (no threading, partial scipy) |

**Where I disagree with the precedents:** rcli validates nearly all domain constraints in C++ before R starts. For
pycli the data mostly lives in Parquet/safetensors artifacts. C++ cannot read Parquet cheaply and should not try.
C++ therefore validates **schema, enums, caps, path policy, expression ASTs, and safetensors/npy headers** (shape and
dtype, using the torchcli reader). **Content-dependent** checks (column exists, dtype compatible, NA present, matrix
square) happen in the worker *before any computation*, through a `precheck()` phase per op. They map to
`VALIDATION_ERROR` with `tool_code` and exit 2, the same classification. The worker never performs a side effect
before its precheck passes.

### 1.2 Call lifecycle

```
pycli <op> --flags…            (or --input-json)
 └─ lmcpp: parse → validate schema → policy (readonly/dryrun) → handler
     ├─ C++ op validator: caps, enums-in-objects, path canonicalisation, ref parsing,
     │    safetensors/npy header check, expression → canonical AST JSON
     ├─ write <ctx>/req.json  {op, params(canonical), inputs:{resolved abs paths}, out:{tmp path, final path},
     │                          limits, seed, ast…}
     ├─ run_child([python, -I, -B, -X, faulthandler, <ctx>/worker.pyz, req.json, resp.json],
     │            env = fixed allowlist (§1.4), timeout = 50 s)
     ├─ read_result_file(resp.json, "json")        (≤16 MiB; we target ≤256 KiB)
     ├─ if op writes: verify tmp output exists, fsync, link-no-replace into final path (torchcli writer)
     └─ print raw result JSON / error envelope, exit code per protocol §6
```

- **The worker is embedded** in the binary as a zipapp (CMake builds `worker.pyz` with `zipfile` at build time and
  embeds it as bytes). It is materialised into the private temp dir per call, as rcli does. A native host binary
  then carries its own catalogue, and `--schema/--list/--help` never start Python. `py_info` reports the
  `worker_sha256`, which CTest asserts equals the build-time hash.
- **Python flags:** `-I` (isolated: ignore `PYTHON*` env, no user site, no cwd/script dir on `sys.path`), so a
  planted `numpy.py` in the workspace is never imported. `-B` stops bytecode writes (rootfs is read-only; the image
  precompiles with `compileall --invalidation-mode unchecked-hash`, otherwise every call recompiles in memory and
  costs time). `-X faulthandler` writes a C-level traceback to stderr on a segfault, and C++ attaches its last 4 KiB
  to `TOOL_CRASH` data.
- **Lazy imports per op group:** `dispatch` maps op → `(module_path, fn)` and imports only that group. `la_solve`
  imports numpy+scipy.linalg (~0.3 s), not sklearn/statsmodels/pandas. This is the cheapest latency lever and is
  required.
- **Request/response files, not pipes.** C++ reads one bounded result file, as rcli does. Stdout/stderr capture is
  bounded (1 MiB) and used only for diagnostics.

### 1.3 Determinism and reproducibility rules (worker prelude, before the first numeric import)

| Concern | Rule |
|---|---|
| BLAS threads | `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` (MKL is absent; set defensively), `VECLIB…`, `NUMEXPR_MAX_THREADS=1`. After import, `threadpoolctl.threadpool_limits(1)` plus an assert in `py_info` that every pool is at 1. Parallel reductions are the main source of run-to-run digit drift; slowness is acceptable |
| BLAS kernel selection | Wheels ship OpenBLAS with `DYNAMIC_ARCH`, so kernels differ by CPU (Neoverse-N1 vs V1, Haswell vs SkylakeX), and so do last-ulp results. Default `OPENBLAS_CORETYPE` pinned per arch (`ARMV8` generic on aarch64, `Haswell` on amd64), overridable for speed through `PYCLI_BLAS_CORETYPE`, echoed in meta. **Uncertain:** slice 0 must verify that the scipy-openblas build honours the variable and measure the slowdown |
| numpy SIMD dispatch | Record `np.show_runtime()` (CPU features, dispatched targets) in `py_info`. Do not disable features by default. Document that float64 transcendental ufuncs may differ by an ulp across CPUs |
| joblib / sklearn | Force `n_jobs=1` and never expose it. The loky backend would fork worker processes, which hits the pids limit and makes timing nondeterministic |
| RNG | Every stochastic op takes a **required integer `seed`** (0 ≤ seed < 2⁶³). numpy: `np.random.Generator(np.random.PCG64(seed))`, never the global `np.random.*` or `RandomState`. scipy.stats `rvs(random_state=gen)`. sklearn takes `random_state=int(seed)`; sklearn internally uses legacy `RandomState`/MT19937, which we document instead of hiding. The result echoes `{seed, bit_generator:"PCG64"|"MT19937(sklearn)"}`. Streams are reproducible only within a pinned lock (numpy guarantees stream stability for `Generator` methods only loosely across versions) |
| Hashing | `PYTHONHASHSEED=0` (set dict/set iteration over strings is stable, which matters for networkx node order) |
| Locale/time | `LC_ALL=C.UTF-8`, `TZ=UTC`. tz database from the pinned `tzdata` wheel (`PYTHONTZPATH` unset, `zoneinfo` uses the tzdata package), so tz results do not depend on the base image's `/usr/share/zoneinfo` |
| Warnings | `warnings.catch_warnings(record=True)` + `simplefilter("always")` around the op, and `np.errstate(all="warn")` so FP exceptions become warnings. Results carry `warnings:[{category, message, count}]`, deduplicated with first-seen order kept and capped at 20 entries plus `warnings_truncated`. `ConvergenceWarning`/`IntegrationWarning`/`OptimizeWarning` additionally set domain `status` fields. `--strict-warnings` is not offered in v1 (rcli has it; nobody has used it yet, so I'd cut it) |
| Accelerators that change results | **Do not install** numexpr or bottleneck (pandas would silently use them; numexpr also has an `evaluate()` string surface). Assert their absence in CTest |

### 1.4 Child environment (complete allowlist)

`PATH=/usr/local/bin:/usr/bin`, `HOME=<ctx tmp>`, `TMPDIR=<ctx tmp>`, `LC_ALL`, `TZ`, `PYTHONHASHSEED`,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, the thread variables above, `OPENBLAS_CORETYPE`,
`MPLBACKEND=Agg` (defensive; matplotlib is not installed), `PYCLI_LIMIT_*` (worker-side caps). Nothing else is inherited. `PYCLI_TIMEOUT_S`
can only lower the 50 s timeout, as in rcli.

### 1.5 Persistent worker: the gate, not the plan

Ship v1 per-call. Measure p50/p95 of `docker + import + op` for each op group in slice 0, and record them in the
README timing table. A session mode is justified only if a real recipe makes ≥20 sequential pycli calls and spends
>60 % of its wall time in start-up. Even then, the preferred first step is a **`batch` op**: a JSON list of
`{op, params}` steps executed in **one** child, in order. Each step is validated by the same C++ validators. Steps
reference earlier outputs only by artifact path, with no variables, conditionals or loops, so it is not a language.
A step failure aborts the batch and returns per-step results so far. This removes the per-step docker/import cost
with no daemon. A true daemon (UDS, state reset per request, crash restart) remains stage-3-or-never.

### 1.6 lmcpp gaps (upstream, MIT)

1. `Context::temp_dir()` (a private 0700 directory) or `temp_file_path(name)`. pycli needs req, resp, worker.pyz and
   tmp outputs per call. rcli hit this gap too.
2. Lift torchcli's **cwd-confined path canonicaliser** and **no-replace atomic writer**, plus its **safetensors
   reader/writer**, into lmcpp (`lmcpp/fs.hpp`, `lmcpp/safetensors.hpp`). Then pycli, torchcli and rcli share one
   audited copy.
3. Lift rcli's expression parser into `lmcpp/expr.hpp` with the spec as data (`expr-spec.json` passed at
   construction). Then rcli and pycli share one parser with a per-tool function table.
4. `run_child` option `rlimits` (`RLIMIT_CPU`, `RLIMIT_NOFILE`, `RLIMIT_FSIZE`, `RLIMIT_CORE=0`), applied
   between fork and exec. `RLIMIT_AS` is unusable with OpenBLAS/numpy, which reserve large virtual memory, so memory
   is capped by docker's cgroup.
5. Publish integer-ness: `Type::Integer` publishes as `number` today. Acceptable, but it is noted in the schema
   descriptions.

---

## 2. Data model and handoff

### 2.1 Principle: artifacts by default, inline only when small

Token economy is a first-order requirement. An LLM-authored lmscript should pass **refs**, not numbers; results
an LLM will read must be **compact summaries**. rcli returns up to 200k numbers inline; **I disagree for pycli**. Any
per-row or per-element output above `inline_max` (default 1,000 values, hard max 20,000) must go to an `out`
artifact. Ops are designed so that this is static:

- **Read ops** (`access:"read"`) return only bounded inline JSON (scalars, coefficient tables, ≤`inline_max`
  vectors). If the output would exceed the cap, the op fails with `COMMAND_FAILED/OUTPUT_TOO_LARGE` and
  `hint:"use <op>_to or pass out"`. In practice, ops whose natural output is large are **write ops** with a
  required `out`.
- **Write ops** (`access:"write"`) require `out` (an exclusive path; an existing file → `CONFLICT` 5, as in
  torchcli), and return a **descriptor** `{ref, path, kind, dtype/shape | n_rows/columns, nbytes, sha256,
  summary:{…}}` plus small inline extras (e.g. `converged`, `n_iter`). `summary` is computed by the worker
  (min/max/mean/NaN-count per column or array) so the caller rarely needs an inspect call.

### 2.2 Artifact kinds

| Kind | Format | Read | Write | Why |
|---|---|---|---|---|
| Array / tensor set | **safetensors** (F64, F32, I64, I32, U8, BOOL) | ✔ | ✔ (canonical) | Header-validated in C++ with the torchcli reader. Directly interoperable with torchcli. No pickle. Multiple named tensors per file (`X`, `y`, `U`, `S`, `Vt`) |
| Array | `.npy` | ✔ (`allow_pickle=False`, v1/v2 header, C++ header check, object dtype rejected) | ✘ | Interop with numpy users. Writing a second array format adds surface but not value |
| Archive | `.npz` | ✘ | ✘ | Zip (bomb risk, many entries, implicit pickle path for object arrays). Not worth it |
| Frame | **Parquet** (pyarrow, zstd level 3, one row group per ≤1 M rows) | ✔ | ✔ (canonical) | Typed: nullable ints, bool, string, float with **NaN ≠ null**, timestamp[ns, tz], date, duration, dictionary (categorical, ordered flag). Compact (disk is tight). Readable by DuckDB/lmsql/polars |
| Frame | CSV (UTF-8, header, RFC 4180) | ✔ via `frame_import` only | ✔ via `frame_export` only | The lingua franca of lmda/lmcreator. Every other frame op takes Parquet, so dtype inference happens exactly once, under an explicit spec |
| Frame | Arrow IPC/Feather | ✘ (v1) | ✘ | Parquet covers it. Add later if a sibling needs zero-copy |
| Model | **pycli model = safetensors + JSON metadata** (§4.3) | ✔ | ✔ | Safe replacement for pickle/joblib |
| Graph | Parquet edge list (`source, target[, weight, capacity]`) and an optional node table | ✔ | ✔ | No GraphML/GML (parsers, XML entity risk); it is just a frame |
| Inline | JSON | ✔ (≤`inline_max` values in, ≤1 MiB doc) | small results | Small vectors, matrices as `{"shape":[r,c],"data":[row-major]}` |

### 2.3 Refs

`ref := path ['#' key]`, following torchcli. `#` is forbidden in path components. Paths are relative to cwd or
absolute, and are canonicalised (realpath, symlinks resolved) in C++. They must stay inside cwd (the shim mounts
only cwd). Keys:

- safetensors: `a.safetensors#X` selects a tensor. Required when a file holds more than one tensor.
- Parquet: `t.parquet#price` selects one column as a 1-D array (numeric/bool only; NA → error unless
  `na:"drop"|"nan"`). Multi-column selection uses a separate `columns:[…]` param, never a key mini-syntax.
- Every array-taking param accepts **either** a ref string **or** inline JSON. The lmcpp param type is `json`, and the
  C++ validator distinguishes a string (ref) from an array/object (inline).

### 2.4 Value encoding (inline JSON)

| Value | Encoding | Note |
|---|---|---|
| float64 | Shortest round-trip decimal (`repr(float)` in Python; nlohmann re-emits the shortest form) | Bit-exact across the boundary, as in rcli. The worker uses its own encoder with `json.dumps(allow_nan=False)` as the final guard |
| NaN / +Inf / −Inf | `"NaN"`, `"Inf"`, `"-Inf"` | **Matches implemented rcli** (rcli design-fable's `null` was not adopted). Input accepts the same strings through `allow_special_numbers` |
| missing (pandas NA, None, NaT, Arrow null) | `"NA"` in typed numeric/temporal columns, JSON `null` in string/bool columns | Keeps NA ≠ NaN, which matters to oracles. Plain float64 numpy NaN stays `"NaN"`; it is *not* reinterpreted as missing |
| int64 | JSON integer, exact | If any value has abs > 2⁵³, the result adds `"int_exceeds_2^53": true` for that field, because lmscript numbers are doubles |
| bool | JSON bool | |
| complex | Separate `re`/`im` arrays (or `X_re`, `X_im` tensors) | Eigenvalues of non-symmetric matrices, `lambertw`, FFT |
| datetime | ISO-8601 with ns precision: tz-aware `2026-03-29T01:30:00+00:00` with tz name in column meta; naive without offset | Never converted silently. Naive and aware values cannot mix in one column |
| duration | ISO-8601 duration or integer ns plus a `unit` field (output: ns integer + `unit:"ns"`) | |
| categorical | String values; column meta `{categories:[…], ordered}` | |

Frame inline output (only for `frame_inspect` head/tail and small aggregates) is **columnar**:
`{"columns":[{"name","dtype","tz?","categories?"}], "data":{"col":[…]}, "n_rows"}`. The pandas index is never
implicit. pycli frames are written with `preserve_index=False`. Any meaningful index (groupby keys, resample bins) is
materialised as ordinary columns, so cross-call semantics have no hidden state.

### 2.5 Sizes

Caps are enforced in C++ from headers/stat, and again in the worker, with per-op complexity budgets in the op
table. Defaults: artifact ≤ 512 MiB; frame ≤ 5 M rows × 500 columns; dense matrix ≤ 8,000² for O(n³) ops (≈ 30 s
single-thread on this host, to be measured); sparse nnz ≤ 20 M. Memory: the container has 4 GiB, so the worker
estimates peak bytes before computing (e.g. SVD workspace) and fails early with `RESOURCE_LIMIT`.

---

## 3. Op catalogue

Naming: flat `group_verb`. **P0** = first slice set (≈30 ops); P1 = next; P2 = later or cut. "Overlap" names the
sibling that already has it and why pycli still includes it, or why it does not.

### 3.1 Meta and artifacts

| op | P | access | args → result | Notes / risk |
|---|---|---|---|---|
| `py_info` | P0 | read | — → versions of python/numpy/scipy/pandas/pyarrow/sklearn/statsmodels/networkx, lock sha, worker sha, BLAS config (threadpoolctl), CPU features, coretype | Provenance stamp. Other ops only echo `lock_id` (8 hex) to save tokens |
| `array_import` | P0 | write | `values` inline matrix \| `frame`+`columns[]` \| `npy` ref; `dtype`; `name`; `out` → descriptor | Bridge from frames/inline/npy into canonical safetensors. `na` policy: error\|nan\|drop_rows |
| `array_inspect` | P0 | read | `ref`, `limit≤32` → dtype, shape, stats, first values | Like torchcli `tensor_inspect` |
| `frame_import` | P0 | write | `csv` path \| `records` inline \| `csv_text`; `schema:{col: dtype}` (enum: int64, float64, bool, string, category, datetime, date, duration); `parse_dates:{col:{format?, tz?}}`; `na_values[]`; `out` → descriptor | **The only place types are inferred**. An unspecified column is inferred by pandas with `dtype_backend="numpy_nullable"` (simpler semantics than the pyarrow backend), and the inferred schema is returned so a script can pin it next time. Risk: date parsing ambiguity, so `dayfirst` is never guessed; a format is required whenever a datetime is mixed or ambiguous |
| `frame_inspect` | P0 | read | `frame`, `head≤20`, `describe:bool` → schema, n_rows, null counts, head rows, numeric describe | Overlap: lmda `table_schema/summary_describe`. Needed because lmda cannot read Parquet |
| `frame_export` | P0 | write | `frame`, `format: csv`, `float_format: "repr"`, `out` | CSV for lmda/lmcreator. Doubles written with repr (17 sig) so nothing is lost (rcli noted R's 15-digit `write.csv` trap) |

### 3.2 Frames (pandas). Deliberately narrow.

lmda already has `table_filter/select/sort/join/groupby/pivot` and `series_rolling` on CSV. pycli's frame ops exist
so pycli pipelines stay in typed Parquet, and for what lmda lacks: **time-zone-aware datetime resampling, nullable
dtypes, and categorical semantics**. If lmda gains Parquet, revisit and cut `frame_merge`/`frame_pivot`.

| op | P | args | Risk |
|---|---|---|---|
| `frame_select` | P0 | `frame`, `where` (structured predicate tree, §4.2), `columns[]`, `sort:[{col, desc}]`, `limit`, `out` | None, provided `where` is structured and never `query()` |
| `frame_mutate` | P1 | `frame`, `assign:[{name, expr, vars:{alias: column}}]`, `out` | Expressions: rcli grammar evaluated vectorised with numpy (§4.1). Column aliases avoid name-escaping problems |
| `frame_groupby` | P0 | `frame`, `by[]`, `aggs:[{col, fn, as}]`, fn ∈ {count, size, sum, mean, median, min, max, std(ddof), var(ddof), first, last, nunique, quantile(q)}, `dropna`, `out` | `fn` strings are mapped through our own dict to bound methods. Never `agg("<caller string>")`, because pandas would `getattr` it |
| `frame_merge` | P1 | `left`, `right`, `on[]`/`left_on/right_on`, `how` ∈ inner/left/right/outer, `validate` ∈ 1:1/1:m/m:1/m:m, `suffixes`, `out` | Row explosion → C++ cannot know, so the worker computes the result size from key counts first (cap) |
| `frame_pivot` | P2 | `frame`, `index[]`, `columns`, `values`, `agg`, `out` | Overlap with lmda; wide-column explosion is capped |
| `frame_resample` | P0 | `frame`, `time` col, `rule` (validated offset alias: `\d+(min|h|D|W-MON..|MS|ME|QS|YS)`), `tz` (IANA name from tzdata), `aggs`, `closed`, `label`, `origin`, `out` | **Differentiator.** DST edges (non-existent/ambiguous local times) take explicit `ambiguous`/`nonexistent` policies and never default silently. Oracle: hand-computed DST fixtures |
| `frame_rolling` | P1 | `frame`, `time`? \| row window `n`, `window` ("7D" or int), `fn`, `min_periods`, `center`, `by[]`, `out` | Overlap with lmda `series_rolling`; kept for time-based windows and nullable types |

### 3.3 Linear algebra (numpy.linalg / scipy.linalg / scipy.sparse)

No sibling exposes general dense LA as a tool. torchcli has matmul only. Inputs are array refs or inline matrices.

| op | P | access | args → result | Risk |
|---|---|---|---|---|
| `la_solve` | P0 | read (x inline ≤ `inline_max`; larger → `OUTPUT_TOO_LARGE`, hint `la_solve_to`, a P1 write twin) | `a`, `b`, `assume_a` ∈ gen/sym/pos/her/banded? (no banded in v1), `check_finite` → `x`, `rcond` estimate, residual norm | Ill-conditioning → `warnings` + `rcond`. Singular → domain `status:"SINGULAR"` (exit 0), not a failure |
| `la_lstsq` | P0 | read | `a`, `b`, `rcond`, `driver` ∈ gelsd/gelsy/gelss → `x`, `residues`, `rank`, `singular_values` | Oracle: NIST StRD Longley/Filip/Wampler (certified to 15 digits) |
| `la_decomp` | P0 | write | `a`, `kind` ∈ svd (full_matrices), eig, eigh (subset by index), qr (mode, pivoting), cholesky (lower), lu, schur, `out` → tensors per kind (`U,S,Vt` / `w_re,w_im,v_re,v_im` / `Q,R,P` …) | Sign/order ambiguity of eigenvectors and singular vectors. The result documents the convention, and oracles compare subspaces (|uᵀv| = 1) rather than raw signs. Complex output is split into re/im |
| `la_props` | P0 | read | `a`, `what[]` ⊂ {det, slogdet, rank (tol), cond (p), norm (ord), trace, eigvals, singular_values, is_symmetric, is_pd} | `det` overflow → recommend slogdet. Results are scalars or short vectors only |
| `la_matrix_fn` | P1 | write | `a`, `fn` ∈ inv, pinv (rtol), expm, logm, sqrtm, (no `funm`: it needs a callable), `out` | `logm/sqrtm` may return complex → split + warning. Matrix-function accuracy depends on conditioning; oracle against closed forms (rotation generator expm) |
| `la_sparse` | P1 | write/read | `coo` frame/arrays (`row, col, val`, `shape`), `task` ∈ spsolve (SuperLU), cg/gmres (`rtol`, `maxiter`), eigsh/eigs (ARPACK: `k`, `which`, `sigma`), svds; `out` for vectors | ARPACK starting vector: `v0` derived from seed via PCG64. Otherwise ARPACK uses LAPACK's internal random start, which is not reproducible. Non-convergence → domain status |

### 3.4 Numerics: optimisation, roots, integration, ODEs (scipy.optimize / integrate)

rcli has `opt_minimize` (R optim), `num_root`, `num_integrate` and `fit_nls`. pycli earns its place with
**constraints, LP/MILP, least_squares with bounds/robust loss, systems of equations, and ODEs**. Ops that only
re-run what rcli does (1-D root, 1-D quad) are P1 and serve as a cross-oracle with rcli.

| op | P | args → result | Risk |
|---|---|---|---|
| `opt_linprog` | P0 | `c`, `a_ub`, `b_ub`, `a_eq`, `b_eq`, `bounds` (per var or global, ±Inf allowed), `integrality[]` (→ `scipy.optimize.milp`), `time_limit≤45`, `mip_rel_gap` → `x`, `fun`, `status` (OPTIMAL/INFEASIBLE/UNBOUNDED/TIME_LIMIT), duals (LP only), `mip_gap` | **Highest-value new op** (HiGHS, MIT). Matrices inline or refs. A degenerate LP can have multiple optima, so oracles compare `fun` and feasibility, not `x` |
| `opt_minimize` | P0 | `objective` (expr), `start{name: value}`, `method` ∈ Nelder-Mead/Powell/BFGS/L-BFGS-B/TNC/SLSQP/trust-constr/COBYLA, `bounds{}`, `constraints:[{type: eq\|ineq, expr}]` (ineq means `expr ≥ 0`), `data{alias: ref/inline col}`, `tol`, `maxiter`, `max_evals` → `par{}`, `fun`, `status`, `message`, `nit`, `nfev`, `jac?`, `constraint_violation` | Expression surface (§4.1). Finite-difference gradients only. A non-finite objective follows rcli's `nonfinite` policy. Constraint support differs per method, so C++ rejects incompatible combinations |
| `opt_least_squares` | P0 | `model` (expr over params+data cols), `response` col, `data`, `start{}`, `bounds{}`, `loss` ∈ linear/soft_l1/huber/cauchy/arctan, `f_scale`, `method` ∈ trf/dogbox/lm, `sigma` col? → `par`, `cov` (from J, `absolute_sigma`), `std_err`, `cost`, `status`, residual stats; `out` for residuals | Essentially `curve_fit`, with robust losses as the addition. Cross-oracle with rcli `fit_nls` on the Gauss-Newton-equivalent case (tolerance 1e-6) |
| `opt_root` | P1 | `equations[]` (n exprs in n named unknowns), `start{}`, `method` ∈ hybr/lm, (1-D: `bracket` → brentq) → `x{}`, `fun[]`, `success` | Systems are new. 1-D overlaps rcli `num_root` and serves as a cross-oracle |
| `num_quad` | P1 | `f` (expr in `x`), `a`, `b` (±Inf), `epsabs`, `epsrel`, `limit`, `points[]` → `value`, `abserr`, `neval`, `status` | Overlaps rcli `num_integrate`; both use QUADPACK, so cross-oracle at 1e-12. 2-D/3-D (`nquad`) is P2 |
| `ode_solve` | P0 | `states[]` names, `rhs{state: expr}` (in `t`, states, `params{}`), `y0{}`, `t_span`, `t_eval[]`\|`n_eval`, `method` ∈ RK45/DOP853/Radau/BDF/LSODA, `rtol`, `atol`, `max_step`, `events?` (P2), `out` (write; Parquet `t, state…`) or inline when small → status, nfev, njev, final state | New for the whole toolset (physim has fixed models only; this one is generic). Stiff problems with an explicit method are slow, so the time cap → `TIMEOUT`, mapped to domain `status:"TIME_LIMIT"` via the eval counter |

### 3.5 Interpolation, signal, spatial, special functions, distributions

| op | P | args → result | Overlap / risk |
|---|---|---|---|
| `interp_eval` | P0 | `x`, `y` (refs/inline), `kind` ∈ linear, cubic (bc_type), pchip, akima, makima, smoothing (`make_smoothing_spline`, `lam`), rbf (`kernel`, `epsilon`, `smoothing`, n-D `points`), `x_new`, `nu` (derivative), `integrate:[a, b]?` → values (inline ≤ cap or `out`) | rcli has `interp_spline/approx` (R methods); pycli adds PCHIP/Akima and n-D RBF. RBF is O(n³), so n ≤ 5,000 |
| `sig_filter` | P0 | design: `family` ∈ butter/cheby1/cheby2/ellip/bessel, `order≤20`, `btype`, `wn`, `fs`, (`rp`,`rs`) → `sos` inline; apply: `x` ref, `mode` ∈ sosfilt/sosfiltfilt, `axis`, `out` | Always SOS, never `ba` (numerically unstable at high order). New in the toolset |
| `sig_spectrum` | P1 | `x`, `fs`, `method` ∈ welch/periodogram/csd/coherence, `window`, `nperseg`, `noverlap`, `detrend`, `scaling` → `f`, `pxx` (inline ≤ cap) | lmda `series_spectrum` overlaps; Welch is the addition |
| `sig_peaks` | P0 | `x`, `height`, `threshold`, `distance`, `prominence`, `width`, `wlen` → indices + properties (capped count) | New |
| `sig_resample` | P2 | `x`, `num` \| `up/down` (polyphase), `window` | |
| `spatial_knn` | P0 | `points` ref (n×d), `queries` ref, `k`, `r` (radius alternative), `p` (Minkowski), `out` → indices + distances tensors | cKDTree. Ties: index order is implementation-defined, so oracles compare distances |
| `spatial_hull` | P1 | `points` (n×d, d ≤ 6), `kind` ∈ convex_hull/delaunay/voronoi, `out` → simplices, volume/area, vertices | **QHull options are never caller-supplied**: a fixed table per kind (`Qbb Qc Qz Q12`, `Qt` etc.). QHull has had crash bugs on degenerate input, which child isolation absorbs. `QhullError` → domain `status:"DEGENERATE"` |
| `spatial_dist` | P1 | `a`, `b?`, `metric` ∈ euclidean/sqeuclidean/cityblock/chebyshev/cosine/correlation/mahalanobis(VI)/jaccard/hamming, `out` | n·m cap. Metrics come from an enum; never pass a callable |
| `special_fn` | P1 | `fn` ∈ allowlist (erf, erfc, erfcx, erfinv, erfcinv, gammainc, gammaincc, gammaincinv, betainc, betaincinv, zeta, polygamma, lambertw(k), expit, logit, log_ndtr, ndtri, ellipk, ellipe, hyp2f1, jv, yv, iv, kv, spherical_jn, airy, struve, owens_t, boxcox, inv_boxcox, xlogy, xlog1py, logsumexp), `args{}` vectors with numpy broadcasting rules made explicit (equal length or scalar) | rcli `math_special` overlaps on gamma/beta/Bessel (cross-oracle). Oracle: **mpmath at 50 digits** in the test image |
| `stats_dist` | P1 | `dist` ∈ allowlist (~40 scipy.stats families incl. genextreme, genpareto, skewnorm, levy_stable, johnsonsu, nct, ncf, vonmises, truncnorm, zipf, betabinom…), `params{}`, `fn` ∈ pdf/logpdf/cdf/logcdf/sf/logsf/ppf/isf/moments/rvs(seed, n) → values | rcli covers R's d/p/q/r families. pycli adds the long-tail families and **a cross-oracle on the shared ones** (norm/t/gamma/beta at 1e-12 rel). Param names are per-family tables (C++). levy_stable is slow and has accuracy issues, so warn |
| `stats_fit` | P2 | MLE `dist.fit` with fixed params | lmda `dist_fitdist` overlaps; cut unless needed |

Deliberately **not** in pycli: hypothesis tests (lmda has ~40 and rcli's `test_htest` covers R; scipy.stats tests
would be a fourth copy, so add only as P2 oracle ops if an lmda discrepancy needs a tie-breaker); FFT (rcli
`num_fft` and lmda have it).

### 3.6 Graphs (networkx, plus scipy.sparse.csgraph where faster)

lmda has `graph_summary`, `graph_centrality`. pycli adds algorithms, not metrics.

| op | P | args → result | Risk |
|---|---|---|---|
| `graph_paths` | P1 | `edges` frame, `directed`, `weight` col, `source`, `target?`, `method` ∈ dijkstra/bellman_ford/bfs, `k_shortest≤10` → path(s), length; or all-pairs `out` (n ≤ 5,000) | Negative cycles → domain status |
| `graph_flow` | P1 | `edges` with `capacity` (+`cost`), `source`, `sink`, `kind` ∈ max_flow/min_cut/min_cost_flow → value, flow edges (`out` when large), cut sets | Integer capacities recommended; float flow uses tolerance |
| `graph_algos` | P2 | `algo` ∈ mst (kruskal/prim), components (weak/strong), topo_sort, louvain (seed), pagerank (alpha, tol) | Louvain is stochastic, so seed is required. PageRank is a cross-oracle with lmda centrality |

### 3.7 Machine learning (scikit-learn). Model format in §4.3.

| op | P | access | args → result | Risk |
|---|---|---|---|---|
| `ml_fit` | P0 | write | `frame`, `features[]`, `target`, `family` ∈ {linear, ridge, lasso, elastic_net, logistic, linear_svc, gaussian_nb, lda, knn_classifier/regressor, kmeans, pca, standard_scaler, random_forest_*, hist_gradient_boosting_*}, `params{}` (per-family typed table), `seed`, `out` (model; trees: see §4.3), `preprocess:{scale?, one_hot[]?}` → train metrics, `n_iter`, `converged`, feature names | Hyperparameters are validated against per-family tables of primitive types and enums **only**; no callables or nested estimators. sklearn may warn and change defaults across versions, so pin |
| `ml_predict` | P0 | write | `model`, `frame`, `kind` ∈ predict/predict_proba/decision_function/transform, `out` (Parquet; input row order preserved, optional `id_column` copied through) → descriptor + summary | Feature-name/dtype mismatch is a precheck error |
| `ml_cv` | P0 | read | `frame`, `features`, `target`, `family`, `params`, `cv` ∈ kfold/stratified/group/timeseries (`n_splits≤20`, `shuffle`, `seed`), `metrics[]` enum → per-fold and mean/std per metric | The best "tool value" op: one call gives an honest estimate. Budget: rows × folds × model cost |
| `ml_metrics` | P1 | read | `y_true`, `y_pred`/`y_score` refs, `metrics[]` ∈ accuracy, balanced_accuracy, f1(average), precision, recall, roc_auc, log_loss, brier, confusion_matrix, r2, mse, mae, mape, … → values | lmda `eval_classify/eval_roc` overlap; included because ml outputs are already artifacts here |
| `ml_search` | P2 | read | grid ≤ 64 combos over `params` of one family + cv | Cost explosion, so it is capped |

### 3.8 statsmodels: the lmstats oracle

lmstats is a pure-Rust port of statsmodels 0.14.x, with parity pinned to 0.14.6. **pycli runs the real
statsmodels 0.14.6.** These ops exist for differential testing of lmstats, not as user features, which mirrors rcli's
"oracle ops". **Formulas are never accepted:** statsmodels' formula API uses patsy, which `eval`s formula terms as
Python. Design matrices are built by our code from `y`, `x[]`, `categorical[]` (treatment coding with an explicit
`reference`), `add_constant`, `interactions:[[a, b]]`, then passed to the array API (`sm.OLS(y, X)`).

| op | P | args → result |
|---|---|---|
| `sm_ols` | P0 | as above + `weights` (WLS), `cov_type` ∈ nonrobust/HC0–HC3/HAC(maxlags) → coefficient table (term, coef, se, t, p, ci), r2, adj_r2, f, aic, bic, llf, condition number |
| `sm_glm` | P0 | family ∈ gaussian/binomial/poisson/gamma/inverse_gaussian/negative_binomial(alpha), link enum, offset/exposure, `maxiter`, `tol` → coef table, deviance, pearson_chi2, llf, aic, converged, n_iter |
| `sm_sarimax` | P0 | `y` ref, order, seasonal_order, trend, `exog?`, `method` (lbfgs), `maxiter`, `forecast_steps`, `alpha` → params (named, as statsmodels names them), se, llf, aic, bic, forecast mean+ci, converged |
| `sm_tsa_tools` | P1 | acf/pacf (method enum), adfuller, kpss, ljung-box → values (cross-oracle with lmstats `acf_pacf/adf/kpss/ljung_box` and rcli `ts_acf`) |

Beyond these, **adding more statsmodels families should follow lmstats' porting queue** (each lmstats op gets a
matching pycli `sm_*` oracle op only when its differential test needs one).

### 3.9 P0 list (the first ~30)

`py_info, array_import, array_inspect, frame_import, frame_inspect, frame_export, frame_select, frame_groupby,
frame_resample, la_solve, la_lstsq, la_decomp, la_props, opt_linprog, opt_minimize, opt_least_squares,
ode_solve, interp_eval, sig_filter, sig_peaks, spatial_knn, ml_fit, ml_predict, ml_cv, sm_ols, sm_glm,
sm_sarimax` = **27**, plus the first P1s once green: `frame_mutate, la_matrix_fn, stats_dist` → 30.

---

## 4. Safety

Threat model: a caller (LLM-authored lmscript) is mistaken or adversarial, and artifact files in cwd may be
attacker-crafted. Goals: no code execution beyond our worker, no file access outside the declared inputs/outputs, no
network, and bounded CPU/memory/time. **No caller text is ever interpreted by Python**, with one exception: strings
that our code compares against enum tables.

### 4.1 Expressions (opt objectives, constraints, ODE RHS, least-squares models, `frame_mutate`)

- **Grammar = rcli's `expr-spec` grammar**: ASCII-only, numbers, declared identifiers, `+ - * / ^`, unary minus,
  relational operators only inside `ifelse`, an allowlisted function set with fixed arities, the constants `pi`/`e`,
  and caps (2048 bytes, 512 nodes, depth 32, 64 identifiers). It is parsed by the shared C++ parser (lmcpp gap 3)
  into the same canonical AST JSON. **One spec file, two backends**: rcli's R builder and pycli's Python builder.
  The function table maps spec names to Python **function objects** (`np.exp`, `np.log1p`,
  `scipy.special.ndtr` for `pnorm`, `np.where` for `ifelse`, …). A spec name without a pycli binding is a
  build-time failure.
- **Python builder:** a recursive function turns AST nodes into nested closures, e.g.
  `lambda env: FN["*"](a(env), b(env))`. **No `eval`, `exec`, `compile`, `ast.parse`, `getattr` with caller
  strings, `operator.attrgetter`, `numexpr`, `pandas.eval` or `DataFrame.query` exists anywhere in the worker.**
  CTest greps the worker source for these and fails if it finds any. The `env` is a plain dict of declared names to
  float or ndarray. The builder re-checks node kinds, names and arity (mismatch → `INTERNAL_ERROR/EXPR_SPEC_MISMATCH`).
- **Semantic notes vs R:** `^` → `np.power` with float64 operands (so `(-8)^(1/3)` is NaN, as in R). `ifelse` →
  `np.where`, which evaluates both branches eagerly; R `ifelse` also evaluates both when vectorised, and the
  resulting warnings are captured. `round` is half-even in numpy and IEC 60559 in R; the corpus pins both. A
  **shared conformance corpus** (`expr-corpus.json`: expression, bindings, expected value to 1e-15) runs against
  both builders, so rcli and pycli cannot drift.
- **Extensions needed by pycli** (spec v2, also usable by rcli): vector-valued systems are **lists of scalar
  expressions** (ODE RHS, root systems, constraints), not vector syntax. Declared-name classes:
  params / data columns / ODE state / `t` / `x`, with collision rules as in rcli.
- **Runaway guard:** a closure-level evaluation counter (`max_evals`), the worker's own `signal.alarm`
  (timeout − 5 s), and C++ process-group kill at 50 s.

### 4.2 Structured predicates (instead of `query`)

`where` is a JSON tree, validated in C++:
`{"and":[{"col":"price","op":">=","value":10},{"or":[{"col":"region","op":"in","value":["EU","UK"]},{"col":"x","op":"is_na"}]}]}`.
Ops: `== != < <= > >= in not_in is_na not_na between starts_with ends_with contains` (literal, never regex in v1).
The worker maps each to a pandas vectorised comparison through a dict. Values are typed against the column dtype in
precheck; datetime literals must be ISO with an explicit offset when the column is tz-aware.

### 4.3 Models: no pickle, ever

- `pickle`, `joblib.load`, `np.load(allow_pickle=True)`, `pd.read_pickle`, `torch.load`, `skops.io.load` and
  `shelve` are forbidden. A CTest source grep plus a **runtime audit hook** (§4.5) blocks `pickle.find_class`, so any
  attempted unpickle anywhere in the process (including inside a library) aborts.
- **pycli model v1** = one safetensors file:
  - tensors: the fitted array attributes of an allowlisted family (`coef_`, `intercept_`, `classes_` (I64 or
    string-encoded via a U8 tensor + offsets), `components_`, `explained_variance_`, `mean_`, `scale_`, `var_`,
    `cluster_centers_`, `theta_`, `var_`, `class_prior_`, `means_`, `scalings_`, `xbar_`, …);
  - metadata (safetensors `__metadata__`, string→string): `pycli_model=1`, `family`, `sklearn_version`,
    `params` (JSON of the validated primitive hyperparameters), `features` (JSON names + dtypes), `target`,
    `preprocess` (scaler/one-hot categories), `train_rows`, `train_sha256` of the input frame, `lock_id`.
  - **Load = reconstruct**: `Family(**validated_params)`, then set each attribute from its tensor with dtype/shape
    checks against a per-family table, `n_features_in_`/`feature_names_in_` from metadata, then a smoke check that
    `predict` on one zero row is finite. Loading is refused when the family or attribute set is not exactly as
    expected.
  - Loading a model written by a different sklearn minor version is refused unless `--allow_version_skew`, because
    attribute semantics move between versions.
- **Tree ensembles** (random forest, HistGradientBoosting): fitted state lives in Cython `Tree` objects whose
  only public persistence is pickle. Rebuilding them via `Tree.__setstate__` with node arrays is technically
  possible but relies on private APIs. v1 decision: **trees are fit-and-predict in one call** (`ml_fit` accepts
  `predict_frame` + `predict_out` for those families and writes no model; `ml_cv` works normally). Revisit with a
  version-pinned node-array exporter (P2) only if needed. Saying so is more honest than shipping a fragile loader.
- statsmodels results are not persisted. Ops return the parameters, and a later `sm_predict` (P2) would take params
  inline.

### 4.4 Untrusted file formats

| Format | Hazard | Mitigation |
|---|---|---|
| Parquet / Arrow | **CVE-2023-47248**: pyarrow < 14.0.1 unpickled `PyExtensionType` data from files | Pin pyarrow ≥ 14.0.1 (actually the current release). Read with `pq.read_table` then convert ourselves; refuse unknown extension types. A test fixture with a malicious extension type must fail cleanly. The pickle audit hook is a second layer. Size caps from footer metadata before reading; reject >10k columns and row groups > cap |
| pandas metadata in Parquet | Foreign `pandas` metadata could request extension dtypes / index reconstruction | Read with `pyarrow` and apply our own type mapping. Honour only pycli-stamped metadata (`pycli_frame=1`) |
| CSV | Huge fields, odd encodings, formula injection on export | pandas C engine, `encoding="utf-8"` strict, field-size cap, row/col caps. On export, optionally prefix cells starting with `= + - @` (P2; matters only for spreadsheets) |
| safetensors / npy | Header overflow, offsets, object dtype | C++ header validation (torchcli code, fuzzed) before Python touches the file. The worker re-checks. `np.load(allow_pickle=False, mmap_mode=None)` |
| Zip/npz/GraphML/HDF5/Excel/SQL/HTML/URLs | Bombs, XXE, network, drivers | **Not supported.** `pd.read_*` is called only for csv and parquet, and URLs are rejected by the path policy before Python |

### 4.5 Process sandbox (defence in depth; none of it replaces "no caller code")

1. **Docker shim** (production): `--network=none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m
   --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=64 --cpus=2 --memory=4g --memory-swap=4g
   -u uid:gid`, a bind mount of cwd only (below `$HOME`, as in rcli/torchcli), and a fresh container per call.
2. **rlimits** in `run_child` (lmcpp gap 4): `RLIMIT_CPU` = timeout+5, `RLIMIT_FSIZE` = artifact cap,
   `RLIMIT_NOFILE` = 256, `RLIMIT_CORE` = 0.
3. **CPython audit hook** installed by the worker prelude after its imports for the op group. It denies:
   `socket.*`, `subprocess.Popen`, `os.system`, `os.exec*`, `os.posix_spawn`, `os.fork`, `ctypes.dlopen`,
   `ctypes.*`, `pickle.find_class`, `marshal.loads`, `urllib.Request`, `webbrowser.open`, and `open()` for write
   outside `{ctx tmp dir}`. Read `open()` is allowed only for `sys.prefix` (lazy scipy/pandas submodule imports,
   tzdata) and the declared input paths. CPython documents audit hooks as **not** a sandbox: C extensions can bypass
   them. They catch library behaviour we did not anticipate and turn it into a loud `FORBIDDEN/AUDIT_BLOCKED` error.
4. **Paths:** C++ canonicalises (realpath) every input and output and confines it to cwd. Outputs go to a private
   tmp file and are linked by C++ with no-replace (O_EXCL semantics). The worker never receives the final path, so
   it cannot overwrite user files even if compromised.
5. No network: the image has no proxy settings, and `--network=none` at the shim. Native dev mode has no network
   guarantee, which the README states.

---

## 5. Packaging

### 5.1 Image layout

```
FROM python:3.11-slim-bookworm@sha256:<index digest>   AS wheels   # same base as torchcli → shared layers
  COPY requirements.lock (hash-locked, universal: aarch64 + x86_64 manylinux wheels)
  RUN pip install --no-cache-dir --only-binary=:all: --require-hashes --no-deps -r requirements.lock
  RUN python -m compileall -q -j2 --invalidation-mode unchecked-hash /usr/local/lib/python3.11
  RUN scripts/prune-site-packages.py   # delete */tests/, *.pyx, *.pxd, docs; record pruned list + bytes
  RUN scripts/collect-licences.py      # dist-info LICENSE + bundled libs → /usr/share/doc/pycli/THIRD_PARTY/
FROM python:3.11-slim-bookworm@sha256:<same>           AS builder  # g++ cmake ninja nlohmann-json, lmcpp @ pinned commit
  build pycli (+ worker.pyz embedded)
FROM wheels AS test       # + requirements-test.lock (mpmath only; CTest drives the python oracle scripts)
FROM wheels AS runtime    # pip, setuptools, wheel removed; ENTRYPOINT /usr/local/bin/pycli
```

- **Why not torchcli's lock?** torchcli's lock pins numpy-free torch deps. pycli gets its own lock. If both images
  share numpy later, keep a common version to share bytes, but do not couple the locks.
- **Locking for two arches:** `uv pip compile requirements.in --universal --python-version 3.11 --generate-hashes
  --only-binary :all: -o requirements.lock`. Every requirement gets **every** platform hash (aarch64 and x86_64
  manylinux). `pip --require-hashes` accepts any listed hash, so **one lock serves both arches**. CI installs the lock
  on both arches to prove completeness. Pin by exact `==` versions. Candidates are chosen at slice 0 from what is
  then current with manylinux2014/2_28 aarch64 wheels (expected: numpy 2.3.x, scipy 1.16.x, pandas 2.3.x,
  pyarrow ≥ 21, scikit-learn 1.7.x, networkx 3.5.x, threadpoolctl, joblib, tzdata, python-dateutil, patsy (forced
  by statsmodels, never used), packaging). **statsmodels pinned to exactly 0.14.6** to match lmstats; this pin
  overrides "latest". **I have not verified these exact versions' availability today; slice 0 does that.**
- **Base digest:** pin the multi-arch **index** digest (verify with `docker buildx imagetools inspect`). torchcli
  hard-codes `--platform=linux/arm64` in `FROM`; pycli must not, or amd64 breaks. Let BuildKit pick the native
  platform. The Debian packages in the slim base are unmodified binaries (GPL/LGPL components such as glibc),
  recorded in THIRD_PARTY_NOTICES with the snapshot date. As in rcli, images are not published in v1.
- **No apt in runtime.** All native libs come inside the wheels (auditwheel-bundled `libscipy_openblas`,
  `libgfortran`, `libgomp` for sklearn, Arrow's own libs). This fixes torchcli's "apt not snapshot-pinned" gap by
  construction; the lock + base digest is the full input set.
- **Size estimate** (unpacked, to be measured): base 233 MB (shared with torchcli: 0 new); numpy ~40, scipy ~110,
  pandas ~70, pyarrow ~130, sklearn ~45, statsmodels ~45, networkx ~15, misc ~20 → **~475 MB new, ~710 MB total**,
  minus ~60 MB from pruning tests. Builder ~1.2 GB (transient; `docker builder prune` documented). Cold build
  ~5–10 min (downloads ~200 MB of wheels). **If disk forces a cut, drop pyarrow** (-130 MB) and fall back to frames as
  CSV + a JSON schema sidecar. That is a real loss of type fidelity (tz, categoricals, NA vs NaN), so I would cut
  networkx and HistGB first.

### 5.2 Rejected packaging routes

| Route | Why not |
|---|---|
| Ubuntu/Debian `python3-numpy` etc. | 2023-era versions (scipy 1.11, sklearn 1.4), **no pyarrow**, statsmodels ≠ 0.14.6. Its one real advantage is reference BLAS (bit-reproducible across CPUs) via `update-alternatives`. A pinned `OPENBLAS_CORETYPE` gives most of that |
| conda/conda-forge/micromamba | +500 MB–1 GB, a slower solver, and a second package ecosystem to audit. Anaconda `defaults` has ToS issues; conda-forge is fine licence-wise but buys nothing over manylinux wheels here |
| Build from source (sdist) | Fortran/Meson toolchains, 30–90 min on 4 cores, and it diverges from the wheels everyone else tests |
| One image per op group | Complicates the shim and multiplies layers. Lazy imports already give most of the latency win |

### 5.3 Host shim `bin/pycli`

Copy rcli's shim (the newer one: `--mount`, cwd below `$HOME`, `.pycli-tmp` in cwd, `SERVICE_UNAVAILABLE` on docker
failures, `TOOL_CRASH` on signal/empty output) and add the §4.5 flags. Forward only `PYCLI_TIMEOUT_S` and
`PYCLI_BLAS_CORETYPE`. Image tag `pycli:runtime`. **Better than both precedents:** generate the shim from one lmcpp
template (`lmcpp/scripts/docker-shim.sh.in`, rcli design gap 5), because three hand-copied shims will drift.

### 5.4 CI

- GitHub Actions matrix: `ubuntu-24.04-arm` and `ubuntu-24.04` (amd64). Steps: build test image → `ctest` (C++
  unit, worker unit, oracles) → build runtime → `pycli --schema | diff - cli.schema.dev.json` → lmts
  `parsePublishedSchema` → `lmscript --check examples/*.lms` → run the self-contained recipes.
- Cross-arch golden policy: goldens are generated on aarch64 (the primary). amd64 compares with the **tolerance
  table** (§6.2), not bit equality, and any amd64 bit difference is reported as an informational diff.
- Licence gate: `collect-licences.py` fails the build if any installed distribution's licence is not in an
  allowlist (BSD-2/3, MIT, Apache-2.0, PSF, MPL-2.0, ISC, BSL-1.0, Qhull, "GPL-3 with GCC runtime exception" for
  libgfortran/libgomp). That output becomes `THIRD_PARTY_NOTICES.md` (checked in; CI diffs it).
- Source greps: forbidden identifiers in `worker/` (§4.1/4.3), `--platform=` in FROM lines, `allow_pickle=True`.

### 5.5 THIRD_PARTY_NOTICES (initial content; regenerate from the lock)

| Component | Licence | Bundled native parts (also listed) |
|---|---|---|
| CPython 3.11 (base image) | PSF-2.0 | Debian bookworm runtime libs (various, incl. GPL/LGPL unmodified binaries) |
| numpy | BSD-3 | OpenBLAS (BSD-3), LAPACK (modified BSD), libgfortran (GPL-3 + runtime exception), pocketfft (BSD-3) |
| scipy | BSD-3 | scipy-openblas (BSD-3), ARPACK-NG (BSD-3), SuperLU (BSD-3), QHull (Qhull licence), HiGHS (MIT), Boost.Math (BSL-1.0), pocketfft (BSD-3), PROPACK (BSD-3), libgfortran (as above) |
| pandas | BSD-3 | (vendored klib, ujson-derived code: MIT/BSD) |
| pyarrow | Apache-2.0 | Arrow C++ deps (zstd BSD, lz4 BSD, snappy BSD, brotli MIT, thrift Apache-2.0, re2 BSD, utf8proc MIT, …), read from wheel |
| scikit-learn | BSD-3 | libgomp (GPL-3 + runtime exception), liblinear/libsvm (BSD-3) |
| statsmodels / patsy | BSD-3 / BSD-2 | — |
| networkx | BSD-3 | — |
| joblib, threadpoolctl | BSD-3 | — |
| tzdata (Python pkg) / dateutil / pytz | Apache-2.0 (data public domain) / Apache-2.0 + BSD-3 / MIT | — |
| lmcpp, nlohmann-json | MIT | — |
| mpmath (test image only) | BSD-3 | — |

---

## 6. Testing

### 6.1 Layers

1. **lmcpp/protocol** (inherited) + pycli CLI regressions: schema offline (Python absent from PATH), validation
   errors, readonly blocks write ops, dryrun creates no file, `CONFLICT` on an existing `out`, timeout → 7, worker
   `kill -SEGV` → `TOOL_CRASH`, oversized output → `OUTPUT_TOO_LARGE`, missing python → `SERVICE_UNAVAILABLE`.
2. **Codec round-trip:** 10⁵ random doubles (incl. subnormals, ±0, max/min) plus NaN/±Inf/NA through inline JSON in
   both directions, and through every artifact kind. Frame type matrix: each dtype × {values, NA, NaN} × {tz-aware,
   naive} through `frame_import → frame_export/inspect` and back must be identity.
3. **Glue oracles (same image, bit-exact):** `tests/oracle/<op>.py` computes the expected result by calling
   numpy/scipy/sklearn/statsmodels directly, independently of `worker/` (it imports none of our modules). It runs
   3–6 fixtures per op including edge cases. CTest asserts **bit equality** of the decoded pycli result. This proves
   the plumbing (argument mapping, defaults, encoding), not the libraries. A reviewer must check that the oracle
   script really is independent: no copy of the worker's argument translation.
4. **Ground-truth oracles (library-independent), tolerance-based:**
   - NIST StRD linear regression (Norris, Pontius, Longley, Filip, Wampler1–5): `la_lstsq`, `sm_ols` to the certified
     digits (LRE ≥ 9 except Filip, which is documented).
   - NIST StRD nonlinear (Misra1a, Thurber, MGH09…): `opt_least_squares` (LRE ≥ 6).
   - Closed forms: `expm` of rotation/nilpotent matrices; Hilbert-matrix solves with known conditioning bounds;
     ODEs with analytic solutions (exponential decay, harmonic oscillator, a stiff Robertson reference at 1e-6);
     `quad` of known integrals incl. infinite ranges; LP/MILP textbook instances (Wyndor, knapsack, assignment) with
     known optima; Dijkstra/max-flow on small graphs with hand-computed answers; DST resample fixtures
     (Europe/London 2026-03-29 and 2026-10-25) with hand-written expected bins.
   - `special_fn`/`stats_dist`: **mpmath at 50 digits** on a grid incl. tails; rel tol 1e-13 (erf family) to 1e-10
     (hyp2f1, levy_stable excluded or 1e-6).
5. **Cross-tool oracles** (lmscript recipes; optional, they need sibling binaries, with a setup hint when missing,
   as in rcli):

| Recipe | Compares | Tolerance |
|---|---|---|
| `oracle-ols-triangle.lms` | pycli `sm_ols` vs rcli `fit_lm` vs lmstats `regression` on the same CSV | coef/se 1e-10 rel |
| `oracle-glm.lms` | `sm_glm` (binomial/poisson) vs rcli `fit_glm` vs lmstats `glm` | 1e-8 |
| `oracle-arima.lms` | `sm_sarimax` vs lmstats `arima` (lmstats targets statsmodels: expect ≤1e-6) vs rcli `ts_arima` ML (≤1e-3, different optimiser) | as stated; extend rcli's `oracle-findings.md` format |
| `oracle-dist.lms` | `stats_dist` vs rcli `dist_*` on shared families incl. tails | 1e-12 rel |
| `oracle-acf.lms` | `sm_tsa_tools` acf vs lmstats/rcli | 1e-12 |
| `oracle-torch-la.lms` | pycli `array_import` → torchcli `tensor_matmul` (F64 via safetensors) → pycli `la_lstsq` | 1e-12; proves artifact interop |
| `oracle-kmeans.lms` | `ml_fit kmeans` vs lmda `cluster_kmeans` vs rcli (Lloyd) | ARI = 1 after relabel |

   Policy, copied from rcli's `oracle-findings.md`: tolerances are never raised to hide a mismatch. Each exceedance
   gets a row with inputs, both values and a diagnosis.
6. **Safety corpus:** rcli's ≥45-case negative expression corpus is reused verbatim, plus Python-specific cases
   (`__import__`, `().__class__`, `lambda`, `os.system`, f-string syntax, `**`, attribute access `x.y`, subscripts).
   All must fail in C++ with `EXPR_*` before Python starts (assert no child was spawned via a counter in a test
   build). Malicious artifacts: Parquet with `PyExtensionType`, npy with object dtype, a `.npy` that is actually a
   pickle, safetensors with overlapping offsets, a symlink pointing outside cwd, `out` pre-created as a symlink.
   Audit-hook tests: a test-only op (compiled out of release builds) tries `socket.socket()`, `open('/etc/x','w')`
   and `pickle.loads`, and each must yield `FORBIDDEN/AUDIT_BLOCKED`.
7. **Determinism:** each stochastic op is run twice in separate containers, and outputs must be byte-identical.
   `ml_cv` with a seed is repeated with `OPENBLAS_CORETYPE` overridden to observe (not assert) drift.

### 6.2 Tolerance policy (published as `docs/oracle.md` for consumers)

| Class | Same image | Cross-arch / cross-tool |
|---|---|---|
| Codec, artifacts, glue oracles | bit-exact | bit-exact (codec only) |
| Special functions, distributions | bit-exact | rel 1e-13 … 1e-10 per fn table |
| Direct LA (solve, lstsq, det) | bit-exact | rel 1e-12 × cond estimate (bounded at 1e-6) |
| Decompositions | bit-exact | subspace/sign-invariant, 1e-10 |
| Iterative (optimize, ODE, ARIMA, GLM IRLS) | bit-exact | rel 1e-6 estimates, 1e-4 se; LP/MILP: objective 1e-9, feasibility 1e-7 |
| Stochastic (rvs, kmeans, RF) | bit-exact with seed | not compared across arch (labels: ARI) |

### 6.3 Slice plan (each slice is test-first, independently verifiable, and suits a cheaper coding model)

Every slice: (1) write the oracle/fixture tests first, red; (2) implement; (3) green in `pycli:test` on aarch64; (4)
README op table + limits, schema regenerated, one recipe updated. **Reviewer (Opus/Fable/Astra) checklist per
slice:** oracle independence; no forbidden identifiers; access classification (anything that writes a file is
`write`); caps validated before spawning; NA/NaN/Inf encoding at every new boundary; warnings captured; seeds
required and echoed; error → protocol code mapping; no new pip dependency without a lock + licence update.

| Slice | Content | Verify (gate) | Reviewer focus |
|---|---|---|---|
| **S0** skeleton | Dockerfile (wheels/builder/test/runtime), lock (both arches), shim, `py_info`, worker protocol (req/resp, prelude, env, audit hook stub), codec, lazy dispatch, timing harness | `pycli py_info` in the container on arm64 **and** amd64; schema offline; `kill -SEGV` → TOOL_CRASH; timing table per op group recorded; image size recorded; `OPENBLAS_CORETYPE` effect verified | Env allowlist complete; `-I -B`; pyc precompiled (no recompilation: strace or timing) |
| **S1** arrays + dense LA | lmcpp lifts (fs, safetensors), `array_import`, `array_inspect`, `la_solve`, `la_lstsq`, `la_props`, `la_decomp` | glue oracles bit-exact; NIST StRD LRE; safetensors fuzz (ASan) | Eigen/singular-vector conventions documented; complex split; size/memory precheck |
| **S2** frames | `frame_import`, `frame_inspect`, `frame_export`, `frame_select` (predicate tree), `frame_groupby`, `frame_resample` | dtype matrix round-trip identity; DST fixtures; malicious Parquet rejected | Inference reported, never silent; tz handling; groupby enum mapping (no getattr) |
| **S3** expressions | lmcpp `expr.hpp` (lift from rcli, spec v2), Python builder, shared corpus, `opt_minimize`, `opt_least_squares`, `num_quad` | shared corpus green on **both** rcli and pycli; negative corpus spawns nothing; NIST nonlinear LRE; Rosenbrock/constrained textbook optima | The builder has no eval path; counter/timeouts; method/constraint compatibility validation |
| **S4** LP + ODE | `opt_linprog` (LP + MILP), `ode_solve` | textbook optima; analytic ODE solutions; stiff Robertson; time limit → domain status | Infeasible/unbounded statuses; Parquet output schema; time budget |
| **S5** statsmodels oracle | `sm_ols`, `sm_glm`, `sm_sarimax`, design-matrix builder | glue oracles; triangle recipes vs rcli + lmstats green; findings file | **No patsy/formula path**; term naming matches statsmodels; categorical reference coding explicit |
| **S6** ML | `ml_fit` (linear/logistic/ridge/lasso/nb/lda/knn/pca/kmeans/scaler), model v1 format, `ml_predict`, `ml_cv`; trees fit+predict | glue oracles; model save→load→predict equals in-process predict bit-exact; tampered model (extra attr, wrong shape, version skew) refused | Hyperparameter tables contain primitives only; no pickle anywhere (audit-hook test); n_jobs=1 |
| **S7** signal/interp/spatial | `interp_eval`, `sig_filter`, `sig_peaks`, `spatial_knn` (+P1 `spatial_hull`, `spatial_dist`) | glue oracles; filter frequency response vs analytic Butterworth magnitude; hull volume of a unit cube | Fixed QHull options; degenerate input → domain status |
| **S8** P1 batch | `frame_mutate`, `la_matrix_fn`, `stats_dist` (+mpmath), `special_fn`, `sm_tsa_tools`, `graph_paths`, `graph_flow`, `ml_metrics`, `frame_merge`, `frame_rolling`, `la_sparse`, `opt_root` | as above + cross-tool dist/acf recipes | Overlap justification still holds (or cut) |
| **S9** measure → maybe `batch` | Only if the §1.5 gate trips | recipe wall time before/after | `batch` stays a list, not a language |

Recipes (stories): `examples/lp-plan.lms` (frame_import of demand CSV → frame_groupby → opt_linprog production plan
→ frame_export → lmcreator chart); `examples/ode-fit.lms` (ode_solve a SIR model → add seeded noise via stats_dist rvs
→ opt_least_squares recovering β, γ → compare); `examples/sensor-pipeline.lms` (frame_resample tz-aware sensor log →
sig_filter → sig_peaks → ml_cv classifier); the oracle recipes of §6.1.

---

## 7. Risks, open questions, cuts

### 7.1 Risks (ordered)

1. **Reproducibility is weaker than rcli's.** OpenBLAS `DYNAMIC_ARCH`, numpy SIMD dispatch, sklearn's internal
   threading and library minor-version changes all move digits. Bit-exact holds only for the same image on the same
   CPU family. Mitigation: single-threaded BLAS, pinned coretype (if it works; S0 verifies), a full provenance stamp,
   tolerance policy cross-arch. Being honest about this in `docs/oracle.md` matters more than chasing bit equality.
2. **Latency.** Docker (~0.7 s) + imports (up to ~1.5 s for statsmodels/sklearn) means ~1–2.5 s per call, so a
   50-call recipe costs a minute or more. Mitigation: lazy imports, precompiled pyc, artifact-oriented ops that do
   more per call, and the `batch` op behind a measured gate. Slowness is acceptable per brief, but it must be
   *measured* and published.
3. **Surface area / supply chain.** ~10 large native-code packages parse untrusted files (pyarrow history:
   CVE-2023-47248). Mitigation: hash-locked wheels, only CSV/Parquet/safetensors/npy, C++ header prechecks,
   pickle-blocking audit hook, a no-network/no-caps container, and a lock refresh process (monthly, with the oracle
   suite as the regression gate).
4. **Expression semantics drift between R and Python backends** (round, pow, ifelse warnings). Mitigation: one spec,
   one shared corpus run on both tools in CI.
5. **Disk.** The ~0.7 GB runtime plus a ~1.2 GB builder cache on a host with 54 GB free and other images. Mitigation:
   shared base with torchcli, pruning, a documented `docker builder prune`, and pyarrow as the documented
   last-resort cut.
6. Smaller: sklearn model-format maintenance per family (an attribute table per sklearn version); pandas 3.0
   (string dtype default, copy-on-write) arriving mid-project, so pin 2.x and treat 3.0 as a planned bump with the
   dtype-matrix tests as the gate; tz database updates changing resample results (tzdata pinned in the lock).

### 7.2 Open questions

- Does `OPENBLAS_CORETYPE` work with the `scipy-openblas32/64` builds that numpy 2.x/scipy 1.16 wheels bundle, and
  what does it cost (likely 1.5–3× on GEMM)? (S0)
- Is the per-call container the right unit, or should the shim keep one `--network=none` container alive per lmscript
  run (`docker exec`)? That is a policy question for all three tools, not pycli alone.
- Should lmda gain Parquet input? If so, pycli's frame ops shrink to `frame_import/inspect/export/resample`.
- Should the shared expression spec live in lmcpp (my proposal) or in lmapi as a protocol extension
  (`cli.expr.v1`) so Rust tools can implement it too? I lean lmapi spec + lmcpp implementation.
- Is the statsmodels oracle role worth a P0 slot ahead of ML? I say yes, because it is the cheapest high-value thing:
  lmstats parity tests currently need a Python environment that the host does not have.
- int64 above 2⁵³ in lmscript: flag only (my proposal) or emit as strings?

### 7.3 Cut list (in order, if pressed)

1. `ml_search`, `stats_fit`, `graph_algos`, `sig_resample`, `frame_pivot`, `nquad`, ODE events.
2. Tree-ensemble persistence (already cut; fit+predict only).
3. networkx entirely (scipy.sparse.csgraph covers shortest paths/MST/components/max-flow), saving ~15 MB and a
   dependency.
4. `frame_merge`, `frame_rolling` (lmda covers them on CSV).
5. pyarrow (last resort; see §5.1).

**Never cut:** the process boundary, the no-eval/no-pickle rules, hash-locked wheels, seed requirement and echo,
NA ≠ NaN encoding, glue + ground-truth oracles, and the statsmodels-0.14.6 oracle ops.
