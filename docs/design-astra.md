# pycli — independent design (Astra)

Status: proposed contract, 2026-10-14. No implementation, image build, dependency installation, benchmark, or numerical test was performed for this design. Our source is MIT, copyright Lattice Systems LLC. Linux arm64 and amd64 are release targets; the supplied host facts are Ubuntu 24.04/aarch64, Docker available, no host NumPy, and about 60 GB free disk.

## 1. Decision and scope

Build a C++17 **lmcpp native tool with an embedded, fixed Python worker**, executed in a fresh child for each operation inside a fresh offline container. Python is an implementation language, never an input language. Publish approximately 30 useful operations, not a generic `python_call(module, function, kwargs)` facility. lmscript owns composition; durable typed files own state across invocations.

The strongest additions are SciPy numerical algorithms, typed Arrow/NumPy/safetensors handoff, and small sklearn models that can predict on subsequent calls. pandas operations earn inclusion for explicit dtype/time/null semantics and artifact composition. Retain one statsmodels OLS operation as an independently callable reference for lmstats. Do not recreate its entire statistical catalogue.

### Precedents actually read

- `~/repos/lmctlhq/rcli`: README; `docs/design-opus.md`, `design-fable.md`, `design-expr.md`, `parity.md`, `oracle-findings.md`; regression, ARIMA, Rosenbrock and curve-fitting example recipes.
- `~/repos/cppwrapper/pytorch`: README, host shim, safetensors reference parsing and writer. Actual shared tensor subset is F32/F64/I64, with precisely one `#` in a reference.
- `~/repos/lmctl-ai/lmcpp/README.md`; `~/repos/lmctl-ai/lmapi/docs/protocol.md`, especially §6.1.
- lmstats and lmda were absent at the requested `~/repos/lmctl-ai` paths; their READMEs were found and read under `~/repos/lmctlhq`. lmstats is `statsrust`, a Rust statsmodels port; lmda already has extensive tables, tests, regression, PCA and clustering, but explicitly excludes persisted predictive models.

Treat rcli's implemented README/parity as newer evidence than its design proposals. In particular, lmcpp already has `run_child`, bounded `read_result_file`, temporary paths, and special-number support. Do not plan to reimplement these as missing SDK features.

### Architecture alternatives

| Alternative | Decision | Reason |
|---|---|---|
| C++ front end + Python child | Choose | Reuses lmcpp validation/policy; preserves upstream algorithms; isolates native crashes and process-global state; no host scientific installation. |
| Embed libpython through C API/pybind11 | Reject | Still imports the stack on every CLI invocation; adds interpreter lifetime, GIL, ABI and crash propagation problems. Little benefit without a resident service. |
| Call BLAS/LAPACK/ARPACK/QHull directly | Reject as primary | Only a fraction of the requested semantics; duplicates SciPy's dtype/layout/error handling and sklearn/pandas algorithms. Underlying library versions also differ from wheel builds. |
| Python-only native CLI | Plausible but not chosen | Less glue, but would duplicate the project's lmcpp protocol/policy conventions. C++ is a thin boundary, not a second numerical implementation. |
| Persistent Python service | Defer | Saves imports but adds authentication, cancellation, stale state, memory retention and lifecycle complexity. Slowness is explicitly acceptable. |

### Invocation and worker contract

1. lmcpp validates the operation, transport, policy, enums, nested descriptors, sizes and relationships it can inspect without decoding scientific files. Unknown nested keys and duplicate JSON keys fail; `object` is not permission for arbitrary kwargs. Reject duplicates before a generic JSON parser can silently discard them, including in scientific-file metadata.
2. C++ opens confined regular-file inputs and copies bounded snapshots to a private per-call directory. The child receives internal paths only. This also avoids relying on inherited descriptors, which `run_child` closes.
3. Materialize the embedded worker and a versioned request JSON there; launch an absolute interpreter with `-I -B`, an argv vector, a cleared environment and stdin `/dev/null`. No shell, caller-selected executable, script, import path or command line fragment.
4. The request has `worker_protocol:1`, `catalogue_hash`, `op`, normalized `params`, and explicit internal input/output mappings. The worker validates the version and dispatches through literal function tables. It lazily imports only the required package family.
5. Worker writes one bounded JSON response to a private file, with typed bulk data in staging files. C++ checks response shape/status and validates staged artifact names, types and bounds before publication. Stdout/stderr from the child are diagnostics, never the result channel.
6. `read_result_file(...,"json")` reads the response. Its current 16 MiB ceiling is not a bulk-data transport: target at most 8 MiB response JSON, including errors and metadata. Temporary context files disappear after the call; returned durable references must never point to them.

Use a private directory containing all files, rather than assuming repeated `temp_file_path()` calls return distinct paths. Keep the worker in a maintained `.py` source embedded at build time, not hand-escaped C++ strings. Generated validation tables and worker share a catalogue/spec hash; decoder checks remain necessary even after C++ validation.

`-I` excludes user/script-directory imports and ignores `PYTHON*` environment variables. It does not disable trusted installation startup hooks; audit the image's site-packages and `.pth` files. Do not claim `PYTHONHASHSEED=0` works under `-I`; define explicit sorting/order semantics instead. See [Python isolated mode](https://docs.python.org/3/using/cmdline.html#cmdoption-I).

### Startup, threads, memory and repeatability

No timings are invented. Slice 0 measures Docker startup, empty Python, NumPy, SciPy, pandas/Arrow, sklearn and statsmodels imports separately, plus cold/warm wall time and peak RSS for representative calls. Imports plausibly dominate small operations; they do not justify a daemon in advance. Vectorized arguments, multiple RHS solves and bounded CV inside one fixed operation amortize startup without exposing a scripting engine.

One request, one process, one numerical thread: set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and any detected backend-specific setting before importing. Set sklearn `n_jobs=1`, Arrow CPU/I/O pool limits, and verify loaded pools with threadpoolctl. Verify HiGHS thread control separately using the pinned supported interface; do not assume threadpoolctl discovers every native pool or forward arbitrary solver options. Native extensions may release the GIL; the GIL is neither a resource limit nor a sandbox. Nested joblib workers are disabled. Different thread mechanisms need separate controls; see [sklearn parallelism](https://scikit-learn.org/stable/computing/parallelism.html).

Repeatability means identical inputs and settings in the same tested image/platform/CPU configuration, subject to algorithm limitations. It does **not** promise cross-architecture bit equality or identical eigenvectors in degenerate subspaces. Pin wheel hashes, interpreter/base image, native libraries, operation defaults, timezone database and worker hash. Report runtime fingerprint on every result; `py_info` gives full versions, loaded BLAS providers, threads, architecture, limits and lock hashes. Do not repeat a kilobyte package list in every result.

Every stochastic path requires an explicit uint32 seed. New NumPy-based paths construct `Generator(PCG64(SeedSequence(seed, spawn_key=(stream,))))`, with explicit nonnegative uint32 stream ID, default 0. Do not rely on `default_rng`'s future default. sklearn APIs needing legacy random state receive an explicit integer/locally constructed RandomState; record `rng_api` and the effective seeds, never pretend these are PCG64 draws. CV derives distinct split/fold/model seeds from documented SeedSequence spawn keys; generated uint32 seeds are recorded. No global `numpy.random.seed` and no serialized RNG objects.

Seed alone is insufficient: array shape, sequence of calls, build and machine can affect streams. Goldens are architecture/runtime-specific; see [NumPy compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html). No random-draw operation is needed initially: rcli already supplies that feature.

Warnings use `warnings.catch_warnings(record=True)` around imports and execution, with fixed filters and explicit NumPy error policy. Return bounded `{category,message,count}` records (32 distinct messages, 512 characters each, plus dropped count), including on failure when available. Classify convergence warnings into domain status. Capture bounded native stderr separately; it is not necessarily a Python warning. A killed process cannot promise recovered warnings. Numeric overflow never silently becomes JSON's invalid bare `NaN`.

If persistence is ever justified: same fixed catalogue and immutable-artifact contract; length-prefixed bounded messages over a private same-UID Unix socket, request IDs, version/hash handshake, one outstanding request, absolute deadlines, cancel by terminating/replacing the worker, per-request warning/RNG reset, request-count/RSS recycling. No handles to resident Python objects, network socket, or arbitrary batch DAG. This is a separate reviewed feature, not v1 infrastructure.

## 2. Native protocol and output policy

Publish `tool:pycli`, `inputs:[flags,json]`, snake_case names, `returns:object`, and no `product` block. `--schema`, `--list`, `--help` do not start Python. A shim can still require an already-present image to expose the binary's schema; it must never pull one implicitly. Schema snapshots and lmts parsing are release gates. No legacy Python CLI exists to preserve.

Success is raw result JSON; lmscript supplies its own `{status,result}` wrapper. Common result fields are `domain_status`, `warnings`, compact `meta`, and named operation outputs. Full arrays, fitted values and previews are explicit selections, not unsolicited output. Defaults return scalar diagnostics plus necessary small outputs or artifact descriptors, with preview disabled.

| Situation | Native result |
|---|---|
| Invalid parameter/type/shape/method | `VALIDATION_ERROR`, exit 2, standard issues with synopsis/usage; tool code under `error.data` |
| Malformed JSON/document | `PARSE_ERROR`, exit 3, following lmcpp |
| Missing file/key | `NOT_FOUND`, exit 4 |
| Existing output | `CONFLICT`, exit 5; never overwrite |
| Escaping path or readonly violation | `FORBIDDEN` / `READONLY_VIOLATION`, exit 6 |
| Deadline | `TIMEOUT`, exit 7; kill process group and clean staging |
| Docker/image/interpreter unavailable | `SERVICE_UNAVAILABLE`, exit 8, explanatory tool code |
| Native crash / malformed worker response | `INTERNAL_ERROR`, exit 1, `TOOL_CRASH` / `WORKER_PROTOCOL` |
| Decoder failure / resource allocation failure | `COMMAND_FAILED`, exit 1, specific `INVALID_ARTIFACT` / `RESOURCE_LIMIT` |
| Solver infeasible, singular, not converged, partial eigenpairs | Exit 0 with explicit domain status and nullable/partial outputs; no fabricated success values |

Classify known numerical exceptions explicitly. Unknown Python exceptions are failures, not automatically domain statuses. Preserve diagnostic text in bounded fields without exposing environment dumps. Dependency mismatch is a native runtime failure, not wrapper product exit 10.

**Effect classification:** `py_info`, `ml_metrics`, `integrate_quad`, `stats_ols`, `opt_linprog` are read operations with inline-only bounded results. `array_convert` is write. The other 24 operations permit optional `out` and are statically **dynamic**, even when returning inline results. Protocol §6.1/§8.3 does not permit runtime downgrading to read. Consequently `--readonly` rejects their inline variants too. This is an intentional, visible tradeoff to keep 30 names rather than duplicate every operation into read/write variants. If readonly composition proves important, split specific operations in a breaking revision; do not lie in the schema.

`--dryrun` on write/dynamic operations validates parameters and returns a plan without running the handler or creating artifacts. It cannot promise matrix content validation or convergence. No operation needs confirmation: writes are confined, exclusive new outputs. SDK readonly/dryrun environment floors remain effective through the shim.

## 3. Data contracts and durable handoff

No live DataFrame/model handles. Each request identifies values or immutable files explicitly. Reopening a file must reproduce the same supported logical data, without importable classes or pandas object serialization.

### Typed values

| Kind | Inline shape | File shape / constraints |
|---|---|---|
| Dense array `A` | `{dtype:"float64",shape:[m,n],values:[...]}` in C row-major order | `{format:"npy",path:"a.npy"}` or `{format:"safetensors",ref:"a.safetensors#tensor"}`; exactly one source |
| Sparse CSR `S` | `{format:"csr",shape:[m,n],data:A,indices:A,indptr:A}` | JSON descriptor plus numeric safetensors components, all confined; no SciPy pickle/opaque sparse object |
| Table `T` | `{fields:[{name,type,...}],columns:{name:[...]}}`, equal lengths | Parquet or Arrow IPC file with validated schema; CSV only through explicit import rules |
| Model `M` | `pycli.model.v1` numeric manifest for small models | Manifest plus safetensors arrays in an exclusive bundle directory; same validator as inline |

Support dense float32/float64/int64/bool initially; arithmetic numerical kernels promote to float64 with a documented cast, rejecting int64 magnitudes that cannot be represented exactly unless the caller explicitly permits a lossy cast. Rank at most 4 for transport, rank 1/2 for most operations. No implicit broadcasting beyond explicitly documented vector/multiple-RHS cases. Reject ragged inputs, object/structured/string NumPy dtypes, nonnative endianness unless explicitly converted, and zero dimensions where the algorithm has no useful meaning.

`path#key` means safetensors only. Split exactly once; reject empty components or additional `#`, as torchcli does. No fragments on CSV/Parquet/npy. Emit `{ref,dtype,shape,nbytes,sha256}` descriptors. For torchcli exchange use only F32/F64/I64; bool requires a documented conversion, not a mislabeled tensor. No npz or arbitrary archive extraction in v1. NumPy loading always sets `allow_pickle=False` and a small header cap; see [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

CSR validation: checked shape products; integer indices in bounds; `indptr[0]=0`, monotone, final pointer equals nnz; data/indices lengths agree; sorted columns and no duplicates by default. An explicit normalization option may sort/sum duplicates and reports that transformation. Never silently densify. Dense algorithm input limits are separate from sparse limits.

### Numbers and missingness

- Serialize finite doubles with Python's round-trip JSON representation and verify C++ reserialization preserves bits; no rounding for prettiness. `allow_nan=False`; normalize all NumPy scalars deliberately. Test negative zero, subnormals, extremes and halfway cases end to end through lmscript.
- In numeric value positions, strings `"NaN"`, `"Inf"`, `"-Inf"` denote nonfinite IEEE values. `null` means missing only in nullable table fields; dense arrays reject missing values. Algorithms generally reject nonfinite inputs, except explicitly documented special functions.
- Do not borrow R's `NA` bit pattern or conflate missing float with a present NaN. `"NA"`/`"NaT"` are ordinary strings in string columns and are not accepted numeric tokens. Nullable timestamps use `null`, including conversion of supported pandas NaT values.
- int64 values outside the exact JSON-consumer integer range ±(2^53−1) are decimal strings in fields explicitly typed int64. Never let lmscript silently round them. Timestamp integer values use decimal strings consistently. File formats preserve integer bits.
- Complex values, NaN payload preservation and signed NaN are outside v1. `eigh` is real symmetric only. A later general `eig` needs an explicit complex representation, not accidental JSON arrays of strings.

### Table fidelity

Supported logical fields: nullable bool, int64, float64, UTF-8 string, date32, timestamp with explicit unit and optional IANA timezone, and categoricals over string/int64 with an ordered flag and explicit category order. Unique column names and ordered field lists are required. No arbitrary Python object columns, nested Arrow fields, registered extension types, decimals, interval/period types or MultiIndex initially.

Arrow validity distinguishes null from present NaN. Use explicit Arrow arrays/masks or a tested Arrow-backed pandas dtype path; never assume `DataFrame.to_parquet()` after default NumPy conversion preserved this distinction. Supported pandas operations must pass masks through correctly; reject unsupported dtype/operation combinations. [Arrow's pandas integration](https://arrow.apache.org/docs/python/pandas.html) documents conversion and index metadata; this contract deliberately narrows that surface.

The external index is always an implicit RangeIndex. User index values must be materialized as named columns before import. Reject Parquet pandas metadata that would silently restore a nontrivial index unless `index_policy:"columns"` explicitly flattens supported simple indices; return the resulting field names. Never infer joins from an index. Grouping/resampling outputs ordinary columns; no hidden MultiIndex artifacts.

Categorical manifests preserve unused levels, order and orderedness even if Parquet dictionary storage changes. Validate the manifest against physical data. Prefer Arrow IPC for exact supported schema/dictionary handoff; Parquet is interoperable storage with a small versioned logical-schema manifest in the bundle. External Parquet missing that metadata imports its physical schema and reports any fidelity limits rather than guessing lost categories.

Timezone-aware timestamps store UTC instants plus the timezone identifier and explicit unit. Do not silently localize naive datetimes. Resampling initially requires UTC (or explicit conversion to UTC); local DST resampling is deferred. Reject precision-losing timestamp conversions unless explicitly permitted. Pin tzdata and expose its version.

CSV requires an explicit schema, delimiter, encoding (UTF-8), null-token list and timestamp format. Default null token is empty field; literal `NA`/`NaN` strings are not automatically missing. Duplicate headers fail. No automatic URL reads, compression inference, or “helpful” date/type inference. CSV is a deliberate lossy bridge to rcli/lmda, not the canonical state representation.

### Outputs and budgets

Optional `out:"artifacts/run01/solve"` names a **new bundle directory**. It contains `manifest.json`, named numeric safetensors/npy files and/or table Arrow/Parquet files. Response references use final workspace-relative paths, so another tool can directly consume `arrays.safetensors#solution`. One rename publishes a multi-file bundle atomically; no manifest can appear before its data.

`array_convert` instead has required `out` naming one new `.npy` or `.safetensors` file and an explicit `format`; its single-file output follows the same exclusive publication rules. It provides direct torchcli interop without bundling overhead and explicit table-to-matrix extraction using an ordered numeric column list. Table extraction rejects null/nonfinite values by default and records feature order; it never one-hot encodes or imputes. Casting is exact by default.

With no `out`, return selected inline values, capped at 10,000 scalar cells and 8 MiB JSON. Exceeding the limit fails with a hint to supply `out`; never silently truncate mathematical outputs or silently create a file. A table preview is explicitly requested and labeled incomplete. Full large arrays are not echoed alongside artifact paths.

Initial hard ceilings, to be lowered by slice benchmarks: 16 MiB request document (SDK), 256 MiB per input/output file, 512 MiB total decoded inputs and staging outputs separately, 1 MiB headers/manifests, 1 million table rows, 256 columns, 1 million CSR nonzeros. Dimensions, rows, string bytes, decompressed bytes and output expansion are bounded independently. These are ceilings, not promises that every allowed combination fits memory/time.

## 4. Prioritized first catalogue: exactly 30 operations

Notation: `A`/`S`/`T`/`M` are the validated descriptors above. `out?` follows the shared artifact policy. P0 is foundation, P1 is the useful initial release, P2 completes the first 30 after security/numerical review. Each enum branch has its own required/forbidden parameters and oracle fixtures. Names below are proposed, not existing APIs.

| # / priority | Operation and concrete arguments | Outputs / reason / main risk |
|---|---|---|
| 1 P0 | `py_info(detail=false)` | Runtime fingerprint; full versions/BLAS/limits on request. No unbounded config dump or host paths. |
| 2 P0 | `array_convert(input:A\|T, columns?:[name], dtype?, allow_lossy=false, format:npy\|safetensors, out)` | Tensor descriptor. Tables require ordered numeric columns and reject nulls; adds table→ML and NumPy↔torchcli transport. Overflow and dtype mistakes are the risk. |
| 3 P0 | `table_read(source:{path,format,schema?,csv_options?}, columns?, preview_rows=0, out?)` | Typed schema, counts and optional table. Adds explicit dtype/null preservation; parsing and decompression budgets are critical. |
| 4 P1 | `table_groupby(table:T, by:[name], aggs:[{column,fn:count\|size\|sum\|mean\|min\|max,as}], dropna=false, observed=true, out?)` | Flat typed table. Explicit pandas categorical/NA semantics justify overlap with lmda; avoid Cartesian category explosion. |
| 5 P1 | `table_merge(left:T,right:T,on:[name],how:inner\|left,validate:one_to_one\|many_to_one,suffixes?,out?)` | Stable left-row ordering, matched counts, flat table. Reject many-to-many in v1; null-key matching is pandas behavior and must be documented/tested. |
| 6 P2 | `table_resample(table:T,time:name,every:{count,unit:s\|min\|h\|day},aggs,origin:epoch\|start,closed:left\|right,label:left\|right,out?)` | UTC time bins including declared empty-bin behavior. Justifies overlap for temporal semantics; no arbitrary pandas offset strings or implicit DST rules. |
| 7 P2 | `table_rolling(table:T,columns:[name],window:int,min_periods:int,fn:mean\|sum\|std,ddof=1,center=false,out?)` | Row-aligned columns, numeric only; fixed row windows first. Missing-window/ddof semantics and output size are risky. |
| 8 P0 | `linalg_solve(a:A,b:A,assume:general\|positive_definite,out?)` | Solution and scaled residual; LAPACK solve, no explicit inverse. Verify positive definiteness/symmetry rather than trusting the hint. |
| 9 P1 | `linalg_svd(a:A,compute_vectors=true,out?)` | Thin U,s,Vh; fixed SciPy driver `gesdd`, no full matrices. Value over torchcli is reference diagnostics; vector signs/subspaces are nonunique. |
| 10 P1 | `linalg_eigh(a:A,out?)` | Ascending eigenvalues/eigenvectors of a verified real symmetric matrix. Reject asymmetric input beyond declared tolerance; repeated eigenvalues nonunique. |
| 11 P1 | `linalg_lstsq(a:A,b:A,rcond:nonnegative number,out?)` | x, rank, singular values, explicitly computed residual norms; fixed `gelsd`. Rank threshold must be caller-visible. |
| 12 P1 | `linalg_expm(a:A,out?)` | SciPy matrix exponential; useful numerical functionality beyond tensor arithmetic. Overflow/conditioning; square matrices only. |
| 13 P1 | `sparse_solve(a:S,b:A,method:cg,rtol,atol,maxiter,out?)` | Iterative x, residual, iterations/status. Requires declared SPD input plus symmetry check; declaration is not a proof of SPD. No LU fill-in risk initially. |
| 14 P2 | `sparse_eigsh(a:S,k,which:largest_algebraic\|smallest_algebraic,tol,maxiter,seed,out?)` | Symmetric ARPACK eigenpairs, residuals, partial status. Explicit seeded v0, fixed ncv rule; no shift-invert or implicit dense fallback. |
| 15 P1 | `opt_linprog(c:A,a_ub?:A,b_ub?:A,a_eq?:A,b_eq?:A,bounds:[{lower,upper}],maxiter)` | HiGHS fixed dual-simplex method; x/objective/slacks/status inline, n≤1,000. New structured optimization without expressions; infeasible/unbounded are domain outcomes. |
| 16 P2 | `opt_least_squares(residual:Expr,parameters:[name],start:A,data:{name:A},lower?:A,upper?:A,loss:linear\|soft_l1,max_evals,out?)` | TRF solution/cost/optimality/termination; finite-difference Jacobian. Adds robust box-constrained fitting versus rcli nls; expression resource limits are risky. |
| 17 P2 | `integrate_quad(integrand:Expr,lower:number,upper:number,abs_tol,rel_tol,max_evals,limit)` | Finite-interval scalar QUADPACK integral/error/status. Intentional rcli cross-oracle; no callbacks, infinite bounds or singular weights initially. |
| 18 P1 | `interp_1d(x:A,y:A,x_new:A,method:linear\|pchip,extrapolate=false,out?)` | Evaluated interpolation, not a serialized callable. Require finite strictly increasing unique x; PCHIP adds explicit shape-preserving semantics. |
| 19 P1 | `signal_filter(x:A,sos:A,mode:causal\|zero_phase,axis:int,initial:zero,out?)` | SOS-filtered samples; fixed zero-phase padding policy exposed in metadata. No design callback; edge transients and too-short series must be tested. |
| 20 P1 | `spatial_neighbors(points:A,queries:A,k:int,p:1\|2,distance_upper_bound?,out?)` | cKDTree distances/indices and found mask. Useful beyond existing statistics; missing index sentinel normalized, tied neighbor identity not promised across platforms. |
| 21 P2 | `spatial_hull(points:A,out?)` | QHull 2D/3D vertices/facets/area/volume. No caller QHull options or random joggling; degeneracy gives domain status, cap points/output. |
| 22 P1 | `special_eval(fn:erf\|erfc\|erfcx\|log_ndtr\|gammaln\|gammainc\|gammaincc,a?:A,x:A,out?)` | Elementwise SciPy special functions; scalar parameter broadcast only. Focus stable tails/incomplete gamma rather than duplicating all R distributions. |
| 23 P1 | `ml_fit(family:ridge\|logistic_binary\|gaussian_nb,x:A,y:A,options:{closed family fields},standardize=false,seed?,out?)` | Numeric portable model, fit diagnostics; fixed SVD ridge/LBFGS logistic. New reusable predictions; feature order, preprocessing and class order are critical. |
| 24 P1 | `ml_predict(model:M,x:A,output:prediction\|probability\|transform,out?)` | Validate feature count/names; predict using fixed numeric evaluator. Probability only classification; transform only PCA/KMeans. No estimator restoration or arbitrary object dispatch. |
| 25 P1 | `ml_metrics(task:regression\|binary_classification,y_true:A,y_pred?:A,y_score?:A,metrics:[enum],positive_label?)` | Regression MAE/MSE/R²; binary accuracy/log_loss/ROC_AUC. Explicit class order, thresholds and undefined statuses; no scorer callbacks. |
| 26 P2 | `ml_cross_validate(family,options,x:A,y:A,split:{kind:kfold\|stratified\|time_series,folds,gap?,shuffle?},metrics,standardize=false,seed,out?)` | Per-fold scores/sizes and split fingerprint; fixed families only. Fit preprocessing within each training fold; leakage and nested compute budget are biggest risks. |
| 27 P1 | `ml_pca(x:A,n_components,standardize=false,whiten=false,out?)` | Full-SVD PCA model/variance and optional scores, reusable through predict/transform. Overlap justified by portable transform; whitening convention/zero variance need exact tests. |
| 28 P1 | `ml_kmeans(x:A,k,seed,n_init:int,maxiter,tol,out?)` | Lloyd centers/inertia/labels, portable nearest-center model. Explicit restarts, no changing `n_init=auto`; complements rcli Hartigan-Wong, labels only defined up to permutation. |
| 29 P1 | `stats_ols(x:A,y:A,add_intercept=false,cov:nonrobust\|HC3)` | Rank/coefs/covariance/SE/df/fit stats; inference undefined fields explicit. statsmodels reference for statsrust/rcli, not a new forecast engine. |
| 30 P2 | `graph_analyze(nodes:[string],edges:[{source,target,weight?}],directed=false,method:components\|shortest_paths\|pagerank,source?,alpha?,tol?,maxiter?,out?)` | NetworkX fixed branches; isolates retained, deterministic node order. Nonnegative finite weights, no multigraph or all-pairs output; convergence and graph size bounded. |

Family option fields for `ml_fit` are deliberately small: ridge `{alpha>=0,fit_intercept}`; binary logistic `{C>0,fit_intercept,maxiter,tol}` with fixed L2 and solver; GaussianNB `{var_smoothing>=0}`. Only logistic uses an explicit local random-state setting if required by the pinned implementation; reject meaningless seeds rather than implying all algorithms are stochastic. Require two declared ordered labels for classification; numeric matrix features only, no automatic one-hot encoding or imputation.

Group-specific limits supplement transport caps: dense factorization max dimension 2,048 and checked work estimate; expm dimension 256; least squares at most 32 parameters/10,000 residuals/100,000 actual expression calls; quadrature at most 10,000 calls/1,000 subdivisions; sparse n≤100,000, eigsh k≤64 and bounded n×ncv workspace; hull 10,000 points in 2D/3D; CV 2–10 folds and aggregate training budget; KMeans n×k×features×iterations×restarts cap; graph 100,000 nodes/500,000 edges; resampling max 1 million bins. Initial budgets must be benchmarked on both architectures. A complexity estimate never replaces the process deadline.

### Deliberately not in the first 30

| Candidate | Disposition |
|---|---|
| `qr`, general `eig`, `inv`, `det` | QR is next if scripts need it. General eig waits for complex transport. Inverse is usually a worse solve interface; defer. Prefer later `slogdet` over overflow-prone det. |
| pandas describe/pivot/query/eval | lmda already covers summary/pivot/filter. No raw query/eval ever. A later typed filter predicate can reuse a bounded Boolean AST; pivot must reject duplicate cells or name its aggregator. |
| scipy.optimize.minimize/root, ODEs | rcli already supplies safe minimize/root. Add only a needed SciPy-specific solver; ODE callbacks/state/event semantics are substantial scope. |
| FFT / scipy.stats omnibus | rcli/torchcli/lmda overlap; add normalization-specific FFT or scipy.stats reference families only with a named consumer/oracle need. |
| scipy.cluster hierarchy | rcli/lmda already cover it. SciPy linkage/Ward conventions could justify a later oracle, not an initial duplicate. |
| statsmodels GLM/ARIMA | Add later as explicit reference operations: GLM `(X,y,family,link,offset,weights)` and ARIMA `(series,order,seasonal_order,trend,method)` with optional same-call forecasts. No formulas or saved result pickles. rcli/lmstats already serve users here. |
| sklearn forests/SVM/grid search/auto pipelines | Safe persistence grows much harder; cut until a specific model format/prediction contract is reviewed. No generic estimator strings. |
| networkx arbitrary algorithms/backends | Three fixed branches only; no plugin dispatch, graph class names, GraphML, pickled graphs, GPU backend or all-pairs centrality. |

## 5. No arbitrary code execution

### Expression operations

Use a tiny mathematical DSL, not Python syntax filtered after parsing. C++ recursive descent produces a canonical AST; Python interprets it directly using fixed numeric functions and array operations. No `eval`, `exec`, `compile`, Python `ast` execution, attribute access, dynamic imports or `getattr` from caller text. Unlike old rcli design-fable proposals, do not parse caller text in the implementation language. Follow the later rcli `design-expr` boundary, but do not rebuild executable Python code objects.

Grammar: decimal finite numbers; declared ASCII identifiers; parentheses; unary minus; `+ - * / ^`; fixed unary `abs sqrt exp log log1p expm1 sin cos tanh`; binary `min max`; scalar constants pi/e. Power is right-associative; unary minus has rcli's precedence (`-x^2 = -(x^2)`). No strings, indexing, comparisons, conditionals, reducers, assignment, lambdas, comprehensions or function definitions initially. Least-squares residuals evaluate elementwise over equal-length declared data columns; scalar output may broadcast only when explicitly enabled. Quadrature accepts only scalar x and a scalar result. Never fall back to Python when parsing fails.

Caps: 2,048 bytes, 512 tokens/nodes, depth 32, 32 parameters, 32 data columns; finite literal parsing, bounded argument arity, no undeclared/name-colliding symbols. Python revalidates AST tags and limits. Reject nested variable-array broadcasting. Estimate temporary memory as live vector intermediates, not just input bytes. Count every expression evaluation, including finite-difference Jacobian probes; SciPy `nfev` alone may not count all work. Overflow/nonfinite residuals yield explicit domain termination; do not universally replace them with infinity. Guard evaluation count, algorithm iterations and wall time separately.

No pandas `query`/`eval`, `apply`, `map` with functions, callable aggregators, regex engines supplied by users, formulas, patsy evaluation or custom sklearn scorers. `stats_ols` uses explicit matrices; intercept creation is an option. pandas explicitly warns that query can execute arbitrary code; a restricted `engine` setting is not a sandbox. See [DataFrame.query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).

### Safe predictive models

Reject pickle/joblib/cloudpickle, NumPy object arrays and statsmodels saved result objects **even if produced by a previous pycli call**: provenance claims are not a reason to deserialize code. sklearn's persistence guide documents these execution risks and the differing tradeoffs of ONNX/skops. See [model persistence](https://scikit-learn.org/stable/model_persistence.html).

Choose a narrow `pycli.model.v1` manifest plus safetensors, not a universal estimator archive:

- Closed `kind`: ridge, logistic_binary, gaussian_nb, pca, kmeans. Exact format/producer versions; unknown fields/versions fail.
- Ordered feature names or explicit positional schema, expected input dtype/count, class labels/order, preprocessing mean/scale (zero variance rule recorded), and named numerical arrays.
- Ridge/logistic: coefficients/intercept; logistic fixed sigmoid/threshold and binary class mapping. GaussianNB: means, positive variances and priors; log-space posterior normalization. PCA: mean/components/explained variance and whitening convention. KMeans: centers and fixed Euclidean distance/tie rule.
- Every tensor shape/value/domain checked; references restricted to same model bundle, hashes checked, no external absolute paths. Hashes detect corruption, not trust. No module names, code, closures, object state dictionaries or links to other models.
- A small fixed numeric inference implementation uses NumPy/SciPy primitives. It does not restore private sklearn attributes. Training remains upstream sklearn; independent tests compare exported prediction/probability/transform outputs to the fitted estimator, including large margins, zero variance, class ties and near-boundary values.

This intentionally duplicates a few prediction equations, not training algorithms. It is the price of auditable, cross-call safe models. Do not imply exact portability to every future sklearn version: model format defines inference semantics, and version bumps require oracle review. ONNX might be preferable when richer models arrive, but adds converter/runtime/operator validation and possible float32 changes. skops is not permission to auto-trust discovered types. Neither is needed for five simple numeric model kinds.

### Paths, runtime and resources

Workspace root is the shim's canonical cwd, a strict descendant of HOME. Mount only that directory, never HOME, Docker socket, SSH agent or credentials. Relative paths are preferred; absolute paths allowed only inside the workspace. Reject URLs, URI schemes, `..` escapes, symlinks, magic links, devices, sockets and FIFOs. Reject hard-linked input files if treating the workspace as an untrusted boundary. Internal filesystem reads use already-opened directory handles and `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS|RESOLVE_NO_MAGICLINKS)` on Linux; a string canonicalization followed by reopening is not race-safe.

Snapshot inputs through validated descriptors to private files before Python decodes them, hashing while copying. Concurrent in-place modification is still possible by the host: detect metadata/size changes and require immutable input usage; do not promise snapshot consistency against a malicious host with equal privileges. Reject custom Arrow extensions/serialized objects and disable any automatic pandas-metadata restoration before conversion. Use bounded decoders and resource limits because numeric formats can still contain malicious sizes or trigger native parser bugs.

Worker writes only staging. C++ creates a private sibling output directory on the target filesystem, validates it, fsyncs files and directory, then uses no-replace atomic rename for bundles; single files use an exclusive no-replace install. Output parent must already exist and be confined. Final conflicts never overwrite. Clean staging on failure/timeout. Failure after rename but before final durability confirmation is reported with `publication_state:"committed_or_unknown"`; do not blindly delete a possibly published result or claim all failures are side-effect-free. Requests use fresh output names, and retries inspect existing artifacts.

Canonical Docker runtime: `--pull=never --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges`, default seccomp, nonroot host UID/GID, `--cpus=1 --memory=2g --memory-swap=2g --pids-limit=64`, bounded private `/tmp` tmpfs, no writable dependency locations. Mount workspace read-only for known read operations where shim routing is reliable; dynamic/write calls require read-write. Policy enforcement still belongs to lmcpp. The mounted workspace is the residual blast radius of a native decoder compromise; Docker is defense in depth, not proof of memory safety.

Child timeout 45 s by default, hard cap 50 s; environment may lower only. Shim has a separate bounded startup/overall deadline and removes its own container on cancellation; killing the Docker client alone is insufficient. Set file-size/open-file/core-dump limits; use cgroup memory as primary memory control, since scientific-library virtual address reservations make strict RLIMIT_AS unreliable. Scratch usage counts against memory/storage budgets. A job needing longer runtime is an explicit future contract change, not an unbounded timeout knob.

Direct native execution is a developer mode with no claim of Docker-enforced offline/sandbox guarantees. The supported user entrypoint is the shim. Never download datasets, models or packages at runtime; all inputs must be supplied explicitly.

## 6. Packaging and licensing

Choose Ubuntu 24.04 minimal runtime, digest-pinned multi-arch base, distro CPython 3.12 and a private venv installed from **hash-locked manylinux wheels**. It matches the host family without relying on host packages; wheels supply the tested native kernels. Use the same package versions on both architectures but separate wheel hashes/filenames. Avoid Alpine/musl and avoid source builds of the scientific stack on this disk-constrained host.

Initial packages: numpy, scipy, pandas, pyarrow, scikit-learn, statsmodels, networkx, safetensors and exact transitive dependencies (including threadpoolctl, joblib where sklearn requires it, timezone/date helpers and any statsmodels requirements). Runtime presence of joblib does not authorize its persistence API. Do not install torch, notebooks, matplotlib or optional backend ecosystems. Cython is a build tool, not required simply because a wheel contains Cython-generated kernels. Do not assume current SciPy uses historical FFTPACK everywhere; inventory actual wheel contents/providers.

Exact versions/digests/hashes are **not yet selected or claimed tested**. Slice 0 must resolve a mutually compatible CPython 3.12 set with wheels for both architectures; statsmodels 0.14.6 is the first compatibility candidate because lmstats documents that oracle baseline. If that forces older dependencies, prefer a coherent tested lock over nominal newest releases. Pinning instructions without a successful dual-architecture resolution are not a release lock.

Build/release procedure:

1. Network-enabled dependency preparation on each native architecture selects exact versions and downloads wheels only (`--only-binary=:all:`). Lock every transitive dependency to the exact downloaded wheel SHA-256 and record Python ABI/platform tags. Never silently build an sdist when an arm wheel is absent.
2. Store reviewed `requirements-arm64.lock` / `requirements-amd64.lock` plus wheel inventory; verify version sets agree. Archive exact wheels in release storage. Runtime installation uses `--no-index --find-links=<wheelhouse> --require-hashes`; no ambient pip config. A hash obtained from the same download is integrity pinning, not an independent supply-chain audit. See [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/).
3. Pin base manifest and per-architecture digest, apt snapshot/package versions for interpreter/runtime libraries, lmcpp/lmapi/lmts source commits, compiler and worker hash. Record shared-library inventory and wheel metadata; run `pip check` and import/BLAS smoke tests. Detect vendored BLAS copies rather than assuming system OpenBLAS controls them all.
4. Multi-stage builder compiles only lmcpp/pycli, at most two jobs. Test image has fixtures/oracle helpers; runtime has venv, shared libraries, notices and binary. Exclude compiler, wheelhouse, pip caches and test fixtures from runtime layers. Use BuildKit mounts rather than copying wheels then deleting them in a later layer.
5. Build arm64 locally; build amd64 on a native amd64 CI runner. QEMU can smoke-test installation but is not the numerical/performance acceptance environment. Run all runtime tests offline. No runtime auto-pull or registry lookup.

Planning estimate, not measurement: full installed stack/runtime 1–2 GB, temporary one-architecture build/test/wheel layers 5–10 GB. Set a provisional 2 GB unpacked runtime target and 12 GB project build-cache budget; measure compressed and unpacked size separately. Do not build both architectures or compile NumPy/SciPy locally. Cache only current locks; remove only project-owned obsolete artifacts, never broad Docker pruning. Fail with a space report before a build likely to exhaust available disk.

Conda would provide coherent native dependencies but introduces another solver/distribution and larger environments; reject absent a wheel gap that cannot be resolved. Distro `python3-numpy/scipy` would be simple but ties versions and ABI/kernel choices to apt and does not give the requested wheel-hash reproducibility across the whole stack. Do not mix distro scientific modules into the venv. A later minimal image may drop statsmodels/networkx if size measurements justify separate schemas; no hidden optional-package branches under one advertised schema.

Our C++/worker/schema/tests use MIT with **Lattice Systems LLC**. A required `THIRD_PARTY_NOTICES` is generated/reviewed against actual distributions, not a hand-waved “Python is permissive” assertion. Record component/version/source URL/hash/SPDX expression/full required notice and all bundled native libraries. Expected inventory categories include Python PSF; NumPy/SciPy/pandas/sklearn/statsmodels/NetworkX BSD-family notices; Arrow Apache notices; safetensors; OpenBLAS/LAPACK; QHull; ARPACK; pocketfft; HiGHS; compression libraries; libstdc++/libgfortran runtimes and runtime exceptions; all transitive wheels and base OS packages. Exact licenses must come from the selected artifacts, not these expectations.

Include wheel `.dist-info/licenses`, upstream COPYING/NOTICE material and OS-package notices in distribution; produce an SBOM and corresponding-source/offer materials wherever the shipped licenses require them. A process boundary alone does not settle redistribution obligations. A reviewer verifies the specific release inventory before image publication. No third-party source is copied into our MIT files to save a dependency.

## 7. Test strategy and ordered implementation slices

Each slice is a small test-first coding assignment with input fixtures, direct reference script, failure cases and explicit acceptance numbers. A cheaper coding model should implement only its named slice after the previous gate passes. Do not give it “implement all of SciPy” or ask it to choose numerical tolerances after seeing failures. No agents were delegated during this design task.

### Oracle layers

1. **Independent plumbing oracle:** direct package calls inside the pinned test image, without importing the worker/validator/codec; verify wrapper outputs and supported settings. This establishes faithful wrapping, not that upstream mathematics is correct.
2. **Known values and invariants:** exact diagonal/triangular solves, expm of zero/nilpotent matrix, SVD reconstruction/orthogonality, eigenpair residuals, LP with known vertex/infeasibility, integral x² on [0,1]=1/3, PCHIP knot interpolation/monotonicity, SOS impulse response, unit square hull, graph path/components, hand-computed confusion matrix.
3. **Independent tools:** rcli and lmstats/lmda on the same saved fixture with aligned conventions. Algorithms sharing LAPACK/QUADPACK are not fully independent evidence; retain analytic cases.
4. **Boundary/adversarial:** malformed/nested descriptors, nonfinite/missing distinctions, singular/ill-conditioned matrices, rank thresholds, category/time/index edge cases, decoder limits, model forgery, expression corpus, timeouts, output races and cancellation.

Default smooth numeric comparisons: `abs(a-b) <= 1e-12 + 1e-10*abs(reference)` on well-conditioned cases, with tighter exact checks for serialization and looser explicitly justified optimizer cases. Check backward error for solves instead of imposing the same forward tolerance on ill-conditioned systems. Eigen/SVD tests compare residuals and subspaces, not signs; KMeans compares inertia/partitions after label permutation, not label integers. ROC/NA/status/schema/index semantics are exact. Same-runtime RNG fixtures check exact streams but do not assert that R/sklearn/NumPy share seeds or outputs.

Cross-tool recipes: OLS uses rcli's existing eight-row fixture and identical intercept/nonrobust covariance, coefficient absolute tolerance 1e-9; compare statsrust and statsmodels result fields explicitly. Quadrature uses analytic polynomials and matched rcli finite-interval tolerances; compare estimates against error budgets. PCHIP must not be falsely compared to R natural cubic splines; linear interpolation can cross-check. Special functions test log tails, never just absolute error of tiny probabilities. PCA tests common centering/scaling/subspaces against lmda; KMeans comparison uses objective/partition only when initialization/algorithm are matched. Tensor conversion round-trips torchcli F64 values bit-for-bit. Future ARIMA oracle adopts rcli's no-mean AR(1) recipe and its documented 1e-3 fit tolerance only after matching likelihood/initialization conventions.

Record discrepancies with fixture, versions, parameters, values, differences and diagnosis. Never raise tolerances just to pass; never regenerate goldens silently on dependency updates. Required cross-tool CI jobs fail with a setup error when a tool is absent; optional local jobs may be explicitly skipped with a reason.

| Slice | Test-first implementation scope | Reviewer gate / reference |
|---|---|---|
| 0 | Dependency resolution, image, skeleton `py_info`, generated schema, shim | Both native architectures resolve identical package versions; offline schema; notices/SBOM inventory; measured size/import RSS/time; no host NumPy installation. Reject unsupported wheels before any source build. |
| 1 | Worker protocol, numeric codec, inline `linalg_solve` | Analytic 2×2 and direct SciPy solve; malformed worker/crash/timeout; bit-preserving double round-trip; lmscript recipe. Verify stdout is exactly one JSON document. |
| 2 | Confined files, atomic writes, `array_convert`, artifact solve | Real torchcli interop; object-npy rejection; symlink/FIFO/race/oversize tests; no overwrite; readonly/dryrun floors; cleanup and killed-container tests. Security review before broad decoders. |
| 3 | `table_read`, Arrow/Parquet/CSV contracts | Round-trip null vs NaN, int64>2^53, empty tables, categories/unused levels, timezone/unit, unsupported indices/extensions; independent Arrow reads. No silently inferred types. |
| 4 | SVD/eigh/lstsq/expm, special functions | Direct SciPy plus invariant cases; rank deficiency and nonfinite behavior; bounded allocations on both architectures. Reviewer verifies drivers/defaults and conditioning-aware tolerances. |
| 5 | Sparse CG, LP, interpolation, SOS, neighbors | Direct libraries, analytic residual/LP/impulse/distance fixtures; CSR corruption, no densification, method domain statuses. Measure cap fixtures and reduce limits if needed. |
| 6 | Groupby/merge, then resample/rolling separately | Direct pandas with explicit arguments; lmda overlap examples with semantics aligned; null keys, duplicate join keys, bins, min_periods/ddof. Check output cardinality before expensive expansion. |
| 7 | Ridge model export/predict, then logistic/NB, metrics | Compare each persisted evaluator to direct estimator predictions on separate test rows; tampered shapes/classes/variances/refs; no pickle imports/calls in persistence path. Review preprocessing/class mapping. |
| 8 | PCA/KMeans export/transform, stats_ols | Reconstruction/variance/partition tests; direct sklearn/statsmodels; OLS rcli/statsrust recipe. Full-rank and singular inference cases; explicit seeds/restarts/rcond conventions. |
| 9 | DSL parser/interpreter alone, then least_squares/quad | Shared accept/reject corpus; fuzz parser/AST validation; known integral/quadratic fit; dry evaluation and hidden finite-difference evaluation budget. Dedicated security review before enabling these ops. |
| 10 | eigsh, hull, graph branches, CV in separate small changes | ARPACK partial-eigenpair handling, degenerate hull, graph isolates/weights/convergence; CV folds and leakage canary against manually fitted folds. Reviewer audits total work/memory, not just per-fit caps. |
| 11 | Release hardening and full 30-op recipes | Native arm64+amd64 conformance, schema parsing, offline recipes, artifact failure tests, source/notice inventory, dependency-update diffs; measured resource table and final support boundaries. |

CI on every change: build targeted test stage, lmcpp/lmapi vectors and lmts schema parse, affected reference fixtures, security/codec regression suite, native two-architecture numerical smoke. Broader cap/performance and full cross-tool runs at release/dependency changes. Fuzz C++ parsers under ASan/UBSan; test worker decoders with bounded malformed fixtures inside containers. Tests call Python directly only in the test image, never through a public arbitrary-code endpoint.

Lmscript demonstration goals: (1) torchcli tensor → pycli solve/expm → safetensors back to torchcli; (2) typed CSV/Parquet → groupby/merge → `array_convert` with explicit ordered columns → fit artifact → predict artifact → metrics; (3) rcli/statsrust/pycli OLS oracle; (4) robust nonlinear fit and analytic quadrature. Each recipe checks domain status as well as lmscript transport status. A converged transport is not a converged optimizer.

## 8. Disagreements, risks, cuts and open questions

Disagreements with precedent proposals are intentional: typed Arrow/numeric artifacts belong early because Python's value includes cross-call data/model use; CSV-only loses important semantics. Aggregated nonfinite counts cannot reconstruct missing locations, so preserve validity explicitly. Old bit-exact R-style claims do not extend to arbitrary CPU/BLAS combinations. The actual torchcli shim is a useful structural pattern, not a complete read-only-rootfs/resource/cancellation policy to copy unchanged. Canonicalize-then-open paths are insufficient for hostile races. Existing lmcpp helpers should be reused, while its coarse nested-schema expressiveness needs explicit validators and documentation.

The three largest risks:

1. **Semantic loss disguised as success.** DataFrame validity/categories/timezones, numerical rank, sklearn preprocessing and ambiguous native domain errors can quietly change answers. Narrow supported types, fixed defaults and independent round-trip/prediction oracles are release blockers.
2. **Native decoding and resource exhaustion.** No eval/pickle removes major execution surfaces, but Arrow/QHull/BLAS and malformed sizes can still crash or exhaust resources. Confined snapshots, capped complexity, container restrictions, deadlines and adversarial fixtures limit exposure; they do not prove native parsers safe.
3. **False reproducibility and dependency drift.** Wheel availability, bundled BLAS differences, solver changes and CPU architecture can invalidate goldens or bloat images. Dual-architecture native CI, exact wheel/base locks, explicit provenance and reviewed tolerance/lock updates are mandatory.

Open questions resolved by measurements/review, not optimistic assumptions: exact dual-architecture wheel set; whether statsmodels 0.14.6 remains the best shared oracle baseline; actual 2 GB image/2 GB memory fit; whether Arrow-backed pandas preserves every supported operation's NaN/null distinction; acceptable repeated import latency; whether the static dynamic-access tradeoff warrants later split names; how much safe-model portability is worth maintaining. The design fixes the decisions needed to start, but cannot certify those facts without slice 0 and decoder/model prototypes.

Cut first if schedule or disk is tight: P2 expressions/quadrature, hull/eigsh, NetworkX, CV, resample/rolling, then statsmodels except its test oracle. Keep the published schema truthful to what ships. Do not cut: typed missingness, exclusive file publication, no-code/no-pickle rules, warning/status capture, pinned offline runtime, arm64+amd64 tests, licence inventory, or direct numerical oracles. A smaller trustworthy catalogue is more useful than an unrestricted Python gateway.
