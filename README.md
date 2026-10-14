# pycli

An MIT C++17/lmcpp frontend for pinned NumPy 2.3.3 / SciPy 1.16.2 / pandas 2.3.3 (S0–S3).
Each call starts a fresh Python child. The fixed dispatch and validated inputs
accept no caller code. The Docker container is the isolation boundary.

| Operation | Access | Inputs | Result |
|---|---|---|---|
| `py_info` | read | none | versions, OpenBLAS pools/threads/core, PCG64 RNG kind, architecture, worker SHA256, hash/isolation flags |
| `la_solve` | read | a, b, assume_a=gen/sym/pos | x, rcond estimate, residual norm, domain status |
| `la_solve_to` | write | same plus exclusive out.safetensors | x descriptor; fails if no artifact is produced |
| `stats_dist` | read | dist=norm/t, fn=pdf/cdf/ppf/sf, x vector, params | values and captured warnings |
| `array_import` | write | format=json/csv/csv_file/npy/safetensors, data, key (default x), header, na=nan/error, exclusive out.safetensors | new F64 artifact descriptor, source dtype, finite/NaN/Inf counts, min, max, norm, na_count (pure C++, no Python) |
| `array_inspect` | read | ref (inline, file.npy, file.safetensors#key, or bare file.safetensors), limit<=1000 | dtype, shape, size, counts, min, max, norm, preview; a bare safetensors path lists tensors from the validated header only |
| `la_lstsq` | read | a (m by n), b, rcond, driver=gelsd/gelsy/gelss | x (<=1000 numbers), rank, singular_values, residues, residual_norm, status OK/RANK_DEFICIENT/NO_CONVERGENCE |
| `la_lstsq_to` | write | same plus exclusive out.safetensors | x descriptor and the same fields; RANK_DEFICIENT still writes the minimum-norm x |
| `la_props` | read | a, what (det, slogdet, rank, cond, norm, trace, inverse), norm_ord, cond_p, rank_tol | requested scalars; inverse (<=1000 numbers) with inverse_residual; status OK/SINGULAR/NO_CONVERGENCE |
| `opt_minimize` | read | objective expression, x0 `{name: number\|[numbers]}`, method (Nelder-Mead, Powell, BFGS, L-BFGS-B, SLSQP, trust-constr), bounds, constraints `[{type: eq\|ineq, expr}]`, gradient expressions, data, tol, maxiter, max_evals, nonfinite | par, fun, status OK/NO_CONVERGENCE/EVAL_LIMIT/TIME_LIMIT/NON_FINITE, message, nit, nfev, evals, nonfinite_evals, constraint_violation |
| `opt_least_squares` | read | model + response or residual expressions, x0, data, bounds, method trf/dogbox/lm, loss, f_scale, sigma, absolute_sigma, tol, max_nfev | par, std_err (Jacobian), cov_status, cost, rss, status, nfev |
| `num_quad` | read | f(x), a, b (finite or "Inf"/"-Inf"), epsabs, epsrel, limit, points | value, abserr, neval, subdivisions, status OK/NO_CONVERGENCE/NON_FINITE/EVAL_LIMIT |
| `num_root` | read | f(x), bracket lower/upper, method brentq/bisect/ridder/toms748, xtol, rtol, maxiter | root, f_root, iterations, function_calls, status |
| `la_decomp` | write | a, kind=svd/qr/eig/eigh/cholesky/lu, full, pivoting, lower, exclusive out.safetensors | one descriptor per factor, small eigen/singular values inline, reconstruction_error, status |

Arrays accept nested row-major JSON, `{shape:[r,c],data:[...]}`, or cwd-confined
`file.npy` / `file.safetensors#key` refs, rank 1 or 2. Little-endian C-order `.npy` v1/v2
with dtype `<f8 <f4 <i8 <i4 |b1 |u1` and safetensors `F64 F32 I64 I32 U8 BOOL` are
converted exactly to binary64 inside C++ (an integer beyond 2^53 is refused, never rounded;
Fortran order, object arrays, complex, rank 0/3+ and .npz are rejected). Headers, shapes,
offsets and lengths are validated before Python; safetensors reads only the selected bounded
payload, and the selected tensor's rank and 200000-number cap are checked from the header
before any allocation. The `.npy` header is parsed by a hand-written closed grammar (no regex).
Array artifacts written by pycli are F64 (`I64` for the qr pivot vector `p`); table artifacts use the S2 dtype whitelist below.

Limits: 1000 numbers per inline array, 200000 per referenced array/CSV file (inline CSV text
1000), 256 MiB per artifact, 32 MiB per array CSV (`array_import`; frame CSV is 16 MiB, see Tables), at most 2,000,000 output numbers per call.

Matrix order is **447 by ref / 31 inline** (square). la_solve returns <=1000 solution
numbers; larger solutions require la_solve_to (likewise la_lstsq / la_lstsq_to). stats_dist returns <=1000 values.
Solve requires finite data; sym/pos require exact symmetry. pos explicitly checks
positive definiteness. Read solves return SINGULAR or NOT_POSITIVE_DEFINITE;
write solves return COMMAND_FAILED with that tool_code and produce no artifact.
Outputs install with an anonymous inode and an atomic no-replace link. This
requires a Linux filesystem supporting O_TMPFILE and /proc/self/fd linking.
Cancellation during writing cannot leave a named partial file; a completed
atomic installation cannot be undone by a later cancellation.

### S1 arrays and dense linear algebra

Missing values: `null`, `"NA"` and an empty CSV field are *missing*; `"NaN"`, `"Inf"`, `"-Inf"`
(and CSV `NaN/Inf/-Inf`) are values. Both are stored as NaN in the artifact (`na_count`
reports the missing ones; `na:"error"` refuses them). CSV is strict: comma separated, optional
header, no quotes and no blank lines (a UTF-8 BOM is skipped); a single column gives rank 1,
otherwise rank 2. Numbers use a fixed decimal grammar (`1e999` and underflow-to-zero are errors,
denormals are exact). Inspect statistics: min/max ignore NaN but keep infinities
(`"NA"` if every value is NaN); `norm` is the Frobenius norm computed with scaling so it
neither overflows nor underflows, and is `NaN`/`Inf` when the data contain them.

Statuses (exit 0 in read ops; write ops fail with COMMAND_FAILED, `tool_code` = status, and
produce no artifact): `SINGULAR` (la_props withholds an inverse when numpy's default rank test
says rank < n, conservative: `diag(1,1e-20)` counts), `RANK_DEFICIENT` (la_lstsq: x is the
minimum-norm solution; la_lstsq_to still writes it), `NOT_POSITIVE_DEFINITE` (cholesky),
`NO_CONVERGENCE` (LAPACK SVD/eigen failure; svd retries `gesvd` after `gesdd`),
`NUMERICAL_FAILURE` (non-finite output, or a factorization whose `reconstruction_error`
exceeds 1e-8). Non-finite inputs and non-exactly-symmetric inputs to eigh/cholesky are
validation errors before Python starts. `la_lstsq` recomputes `residues` (per right-hand
side) and `residual_norm` from `A x - b` for every driver; `rcond` is scipy's `cond`.
`det` may be rounding noise for singular matrices (rank/cond are the tests of singularity;
`cond` is `Inf` when rank < min(m,n) under numpy's default tolerance); `trace` needs a square matrix;
the `eig`/`eigh` residual is scaled by `||A||_F ||V||_F`, so nilpotent/defective inputs are not
failures; `eig` returns `vr_cond` and a `NumericalWarning` when it exceeds ~6.7e7 (defective or
nearly defective: `vr` is then not a basis). Results never hold more than 1000 inline numbers:
`la_lstsq` refuses requests whose x + residues + singular values exceed that, `la_lstsq_to`
stores big residues/singular values in the artifact. A missing input file or tensor key is
`NOT_FOUND` (exit 4); malformed references are validation errors (2).
An `{shape,data}` envelope takes flat row-major data or nested data of exactly that shape.
Safetensors files may hold any spec dtype (F16, BF16, ...): only selecting such a tensor fails.
a determinant outside binary64 is `Inf`/`0` plus a `NumericalWarning` pointing at `slogdet`.

`la_decomp` conventions (LAPACK via scipy.linalg; vectors keep LAPACK's signs, which are not
portable across BLAS/CPU, so compare invariants, not signs; magnitudes and the products below
are what the tests check against scipy called directly):

| kind | tensors | convention |
|---|---|---|
| svd | `u`,`s`,`vt` | `A = u[:,:k] * s @ vt[:k]`, k=min(m,n), `s` descending and >=0; `full:true` gives square `u`,`vt`; gesdd, gesvd fallback (reported as `driver`) |
| qr | `q`,`r` (+`p` int64) | economic by default, `full:true` for full Q; `pivoting:true`: `A[:,p] = q @ r`, `abs(diag r)` non-increasing |
| eigh | `w`,`v` | symmetric input; `w` ascending, `v` orthonormal columns, `A v = v diag(w)` |
| eig | `w_re`,`w_im`,`vr_re`,`vr_im` | complex output split into real/imaginary tensors; eigenvalues sorted by (real, imaginary) ascending, columns of the right eigenvectors permuted alike, unit 2-norm; `A vr = vr diag(w)` with `vr = vr_re + i vr_im` |
| cholesky | `l` (or `u` with `lower:false`) | `A = l @ l.T`; `u = l.T` |
| lu | `p`,`l`,`u` | `A = p @ l @ u`; `p` is an m by m permutation matrix, `l` unit lower m by k, `u` k by n |

Writes use the same exclusive anonymous-inode install as `la_solve_to`; an existing target is
CONFLICT (5), `PYCLI_READONLY` rejects writes (6), dry-run creates nothing. Import and inspect
never start Python, so they work (and are tested) with the interpreter missing.

### S3 expressions, optimisation, quadrature, roots

Callers never send code. An expression is an ASCII string (<=2048 bytes, <=512 tokens and nodes, depth 32, <=16
nested calls) in rcli's grammar: numbers, declared names, `+ - * / ^`, unary minus, relational and `& |` only inside
`ifelse(cond, a, b)`, the constants `pi` and `e`, and the allowlisted functions of `expr-spec.json` (`exp log log1p expm1
log2 log10 sqrt abs sin cos tan asin acos atan sinh cosh tanh floor ceiling trunc round sign gamma lgamma digamma pnorm dnorm
qnorm plogis atan2 pmin pmax beta lbeta choose lchoose`; reducers `sum mean prod min max` only where data columns
exist). The spec file and the shared accept/reject corpus (13/57) are byte-identical copies of rcli's (SHA-256 pinned in
the tests); the C++ parser is a copy with one extension: a variable declared as a vector (`x0: {"v": [1, 2]}`) may be used
only as `v[0]`, `v[1]` with an integer literal below its length, and a scalar may not be indexed. Rejections happen in
C++ before any process starts (`EXPR_SYNTAX`, `EXPR_BAD_BYTE`, `EXPR_UNKNOWN_IDENT`, `EXPR_UNKNOWN_FUNCTION`, `EXPR_ARITY`,
`EXPR_TOO_LONG`, `EXPR_TOO_DEEP`, `EXPR_TOO_MANY_NODES`, `EXPR_NAME_COLLISION`, `EXPR_NOT_SCALAR`, `EXPR_INDEX`,
`EXPR_VECTOR_INDEX`, with `position` and `near`). Python receives only the JSON AST and compiles it iteratively
(no recursion, no `eval`/`exec`/`compile`/`getattr` on caller data) into a postfix program of numpy/scipy function *objects*
from a closed table, re-checking node kinds, arities, names and indices (`INTERNAL_ERROR`/`EXPR_SPEC_MISMATCH` otherwise).
`^` is `numpy.power` on float64 (so `(-8)^(1/3)` is NaN as in R); `ifelse` evaluates both branches (`numpy.where`).

Unknowns are scalars or vectors of at most 64 components in total; they are ordered by name (vector components by index),
which fixes the order of `gradient` expressions and of the flat vector. Data columns (`data`) are finite vectors from inline
arrays or `.npy` / `.safetensors#key` refs, equal length, <=200000 rows, <=32 columns; table (`.npz`) refs are not read
directly (export a column with `array_import`). Names are `[a-z][a-z0-9_]{0,31}` and may not collide with functions,
constants or each other (`BAD_NAME`, `EXPR_NAME_COLLISION`).

Statuses (exit 0): `OK`, `NO_CONVERGENCE` (the solver's own verdict, with its message, including "precision loss"),
`EVAL_LIMIT` (`max_evals`, default 500000, cap 2,000,000, counting every objective, gradient, constraint and residual
call), `TIME_LIMIT` (a wall-clock budget of the call timeout minus 10 s, so the worker stops before the hard kill) and
`NON_FINITE`. `EVAL_LIMIT`/`TIME_LIMIT`/`NON_FINITE` return the best point found so far. A non-finite objective at `x0`
is `NONFINITE_OBJECTIVE` (exit 2); later non-finite objectives count as +Inf (`nonfinite: "inf"`, reported as
`nonfinite_evals` plus a warning) or stop the run (`nonfinite: "error"`). Constraint violation above 1e-6 turns a solver
success into `NO_CONVERGENCE`. Gradient expressions are compared with finite differences at `x0` and a mismatch is a
`NumericalWarning`. Method rules are validated up front: bounds not for BFGS, constraints only SLSQP/trust-constr,
gradients not for Nelder-Mead/Powell; the default method is SLSQP with constraints, L-BFGS-B with bounds, else BFGS.

`opt_least_squares` fits `model` against the `response` column (residual = (model - response)/sigma, `sigma` a positive
number or data column) or minimises concatenated `residuals` expressions. Standard errors: with the central-difference
Jacobian J at the solution (scipy's own 2-point Jacobian is only first-order accurate with an absolute step and was off in the
5th digit for NIST Misra1a), `cov = (J'J)^-1`, columns equilibrated, rank decided at 1e-8, multiplied by `2*cost/(m-n)`
(`curve_fit` convention) unless `absolute_sigma`. `cov_status` is `OK`, `SINGULAR_JACOBIAN` or `NO_DOF` (standard errors are
then `"NA"`). With a robust `loss` scipy's loss-modified Jacobian is used and `std_err` is approximate. The solver runs
with `x_scale="jac"` and `tol` (default 1e-10) for ftol/xtol/gtol.

`num_quad` is QUADPACK (`scipy.integrate.quad`, `full_output`): a warning-level failure (oscillatory or divergent integrand,
`limit` exhausted) is `NO_CONVERGENCE` with the QUADPACK message and the unreliable `value`/`abserr`; a non-finite integrand
value (e.g. a pole on a Gauss-Kronrod node) is `NON_FINITE`. `num_root` needs finite `f(lower)`, `f(upper)` of opposite
sign (`NONFINITE_BRACKET`, `BAD_INTERVAL`, exit 2).

Distribution params accept loc (default 0), scale (positive, default 1), and
positive df for t. ppf rejects probabilities outside [0,1], including infinities;
NaN/missing pass through. Extremely small positive df may make SciPy return NaN;
pycli preserves that value with NumericalWarning rather than inventing a df
floor. Warnings are deduplicated/counted, capped at 20 distinct entries, with
warnings_truncated. Nonfinite JSON strings: NaN, Inf, -Inf. Numeric missing
null/NA becomes NA in distribution results. Finite values make an exact binary64
round trip; CLI output uses shortest round-trip decimal representations (the
private worker codec uses 17 significant digits). Cross-architecture bit equality
is not claimed; BLAS/CPU/version differences remain relevant.

```sh
./scripts/build.sh test
docker run --rm --pull=never --network=none pycli:test   # CTest: S0 + S1, including ASan/UBSan runs
./scripts/build.sh runtime
python3 tests/review_shim.py bin/pycli
python3 tests/container_review.py bin/pycli
./scripts/clean-builder.sh
export PATH="$PWD/bin:$PATH"
pycli --schema
pycli --input-json '{"op":"stats_dist","params":{"dist":"norm","fn":"ppf","x":[0.975]}}'
lmscript --check examples/solve.lms
lmscript --dry-run examples/solve.lms
lmscript examples/solve.lms
lmscript examples/lstsq-fit.lms     # polynomial fit with la_lstsq, residual and normal-equation checks
```

The build uses the pinned local lmcpp revision through its installed CMake
package. Separate cp312 wheel hashes are checked in for aarch64/amd64. Builder
and test apt use the signed immutable Debian **20261012T000000Z** snapshot and
exact top-level versions; runtime retains the full package/compiler inventory,
.deb hashes and package copyrights at `/usr/share/doc/pycli/build`. Scientific
licenses are supplemented by hash-verified matching source notices, including
nlohmann JSON and SuperLU. Runtime copies a pruned scientific installation into
a fresh pinned base, so deleted wheel test bytes are absent from its layers.
Pip already supplied by the base is hidden; its shared base-layer bytes remain.
Network acquisition happens only during builds. Invocation uses --pull=never.

The shim rejects cwd outside HOME or containing comma, quote, equals or control
characters before invoking Docker. It mounts only cwd read/write, disables
network, keeps the rootfs read-only, drops all capabilities, enables
no-new-privileges and Docker's default seccomp, and sets 2 CPUs / 4 GiB / 128
process limits. --init forwards signals; direct namespace PID 1 runs also exit
correctly on TERM/INT. The shim forwards signals to its named container, waits
for the Docker client, and exits 143/130. Worker/request/response copies live in
container-only **512 MiB noexec/nosuid/nodev tmpfs**, never the workspace. A
container SIGKILL/OOM discards them. A shim SIGKILL cannot be trapped; the
remaining container still has the 50-second worker timeout and no workspace
request copies. Host diagnostic captures can remain in the host temp directory
after untrappable SIGKILL; this is not a worker/request copy in the workspace.

The child uses **-P -s -B** and a fully replaced environment. No PYTHONPATH,
PYTHONHOME or user site is inherited, cwd/script directory stay off sys.path,
and PYTHONHASHSEED=0 works. py_info reports/verifies these guarantees. The
interpreter override exists only in the CTest development binary; production
ignores PYCLI_PYTHON. PYCLI_TIMEOUT_S can only lower 50 seconds. OpenBLAS uses
one thread with verified ARMV8 on aarch64; amd64 Haswell remains unverified.

The import guard/audit hook is **defence in depth, NOT a sandbox**. Trusted
scientific imports run before it; removing sys.modules entries does not remove
cached references. ctypes/native calls and os.exec remain reachable to trusted
code. Caller code never runs; the container and fixed dispatch are the wall.
Import rejections use ImportError, allowing library fallback paths. S1 does
not widen this: every caller-supplied file (npy, CSV, safetensors) is parsed and
validated by C++, and Python only sees the C++-written float64 `.npy` copies in the
private tmpfs. Extending audit gates for arbitrary file/process/native events stays
deferred; Python audit hooks cannot safely sandbox native code.

PYCLI_READONLY/PYCLI_DRYRUN and CLI policy flags reach lmcpp. Readonly rejects
writes (exit 6); dryrun simulates without files. lmcpp short-circuits handlers
on dryrun, so custom path/conflict checks are not performed in simulation.
Real existing outputs return CONFLICT (5). SIGSEGV maps to INTERNAL_ERROR /
TOOL_CRASH (1); missing interpreter remains SERVICE_UNAVAILABLE (8), matching
rcli. Unexpected numerical ValueError/OverflowError map to PY_ERROR (1); only
explicit worker prechecks map to validation errors (2). Huge invalid JSON such
as 1e999 gives PARSE_ERROR (3). See docs/parity.md for these choices.

Measurements and image sizes after these fixes are in [docs/timings.md](docs/timings.md).
The checked-in S0 raw measurements remain historical. amd64 locks are resolved;
execution/CI and Haswell verification remain pending, with no amd64 pass claimed.
The copied expression parser and rcli corpus have no exposed S0 expression ops.

| Review-fix runtime latency | p50 | p95 |
|---|---:|---:|
| py_info through shim | 0.861 s | 0.951 s |
| la_solve through shim | 1.348 s | 1.523 s |
| stats_dist through shim | 1.927 s | 2.033 s |

S1 (separate run, same method): `array_import` / `array_inspect` 0.35 s (no Python),
`la_lstsq` / `la_props` / `la_decomp` on 100-200 row inputs 1.2-1.3 s p50; see
[docs/timings.md](docs/timings.md). Runtime image unchanged: 446 MB, 97.43 MB content.

Current image sizes: runtime 446 MB / test 667 MB on disk; runtime content
metric 97.45 MB. Startup calibration: boot 17 ms, NumPy 200 ms, scipy.stats
1229 ms, Docker overhead 391 ms (p50, imports include boot). These timings are
slower than historical S0 samples; see the timing document for limitations.

S2 adds six narrow pandas operations:

| Op | Access | Behavior |
|---|---|---|
| `frame_import` | write | Exactly one of inline `columns`, `records`, UTF-8 `csv` path, or table `frame`; explicit `dtypes`, reported inference; exclusive `.npz` `out` |
| `frame_inspect` | read | Shape, dtype policy, null counts, head/tail and optional numeric describe |
| `frame_export` | write | CSV or columnar JSON; optional exclusive file `out`, otherwise <=1000 cells inline |
| `frame_select` | write | Structured literal predicates, stable `sort_by`, column selection, limit; required artifact `out` |
| `frame_groupby` | write | `keys` and column -> aggregation-list `aggs`; sorted keys, observed categories, optional `dropna` |
| `frame_resample` | write | Datetime `time` column, validated `rule`, aggregation lists, left/right `closed` and `label`; DST-aware bins |

Tables allow 200,000 rows, 500,000 cells, 256 columns and 256MiB artifacts.
CSV inputs are limited before spawn to 16MiB and UTF-8 string cells to 65,536 bytes.
Inputs and combined previews allow <=1000 inline cells. Summaries have a 32KiB
byte budget including metadata; the final response is limited to 64KiB. Preview
strings truncate to 64 characters with `preview_truncated:true`; categories retain
at most 20 levels with `n_categories`/`categories_truncated`, or metadata is omitted
with `column_meta_in_artifact:true`. Inline exports are lossless or fail with
OUTPUT_TOO_LARGE: use `out` for larger results. Full values remain in artifacts.
A table artifact is a single **stored-only NPZ** (compressed archives are refused),
with `__pycli_table__` UTF-8 JSON as a uint8 array, `cN` column arrays and `mN`
boolean missing masks. NumPy reads always disable object deserialization. C++ checks
ZIP bounds, sizes, overlaps, member counts and numeric/byte-array NPY headers before
spawn; Python rechecks metadata, column types, masks and category codes. NPZ is
supported only for tables; array ops retain the S1 formats. String columns use
UTF-8 bytes arrays, never object arrays. Names are unique nonempty UTF-8, <=256 bytes.
NUL characters in names and strings are refused.

The dtype whitelist is `int64`, `float64`, `bool`, `string`, `datetime64[ns]`,
`category`. Nullable ints/bools use masks; int64 is preserved exactly, including
values outside ±2^53 (lmscript numbers cannot represent those exactly). JSON null
means missing for every table dtype; numeric `NA` also means missing, while
`NaN`, `Inf`, `-Inf` are numeric values. String `NA` remains a literal string.
CSV empty fields and explicit `na_values` are missing; quoted empty fields also
count as missing. CSV defaults use strict integer/float/bool/string inference,
reported in `inferred`; leading-zero or plus-prefixed CSV numeric tokens infer
string to preserve identifiers. Numeric `NA` participates in nullable integer
inference; numeric overrides reject underscores and non-decimal syntax. Dates require `parse_dates` or a dtype override. Explicit
float overflow fails instead of becoming infinity. CSV exports use 17 significant
digits, explicit `NaN`/`Inf`/`-Inf` tokens and empty missing/NaT fields.
With matching dtype policies, numeric nonfinite values and missing masks round-trip.
CSV cannot preserve category policy or distinguish empty strings from null; NPZ is the lossless handoff. JSON export is columnar, without hidden index.

Dtype objects accept `dtype` plus `tz` for datetime, or `categories`/`ordered` for
category. Closed timezones: UTC, America/New_York, America/Chicago,
America/Los_Angeles, Europe/London, Europe/Berlin, Asia/Tokyo, Asia/Kolkata,
Australia/Sydney. The pinned pytz/tzdata 2025.2 databases supply these names.
Aware ISO input requires explicit timezone; mixed aware/naive input is refused.
Naive local DST ambiguity/nonexistent times fail; offset-bearing input disambiguates.
Resample rejects NaT in its time column. Rules are 1..10000 plus
`min`, `h`, `D`, `W`, `MS`, `ME`, `QS`, or `YS`; default closure and labels are left,
except `ME` and `W`, which use pandas right/right defaults. Explicit options override these. Bins and group keys are materialized as ordinary columns.
Aggregates use a closed function table: sum, mean, min, max, count, std, var,
median, nunique, first, last. std/var use ddof=1; first/last skip missing values.
Integer sums use exact Python integers and fail `INT64_OVERFLOW` (exit 1) if
the result cannot fit int64; no wrapped artifact is written. String/category sums
are rejected. Reductions return missing for undefined results (including singleton
std); float NaN and masked null group keys merge into one missing group.
Datetime cells and predicates accept dashed ISO dates/timestamps only; natural
language and pandas special words are rejected. Empty CSV lines are ignored.
Aggregate names are flattened as `column__function`; collisions fail.

`where` uses `{and:[...]}`, `{or:[...]}`, `{not:...}`, or
`{col:"name",op:"eq",value:"literal"}`. Leaf operators: eq/ne/lt/le/gt/ge,
isin (list), isnull (no value), between (two inclusive bounds), startswith,
contains (literal, no regex). Comparisons exclude masked null cells; isnull selects only the missing mask, not explicit float NaN. Numeric predicates require numbers (float columns also accept NaN/Inf/-Inf tokens), int64 predicates require integral int64 values (50.0 accepted; 50.5 refused), bool predicates require booleans, and string/category predicates require strings. Datetime predicates require ISO literals with matching timezone awareness.
Predicate depth <=32 and nodes <=256. Column names and values are data.
Pandas is imported only by frame ops; other Python ops report `meta.pandas:"unloaded"`.
The audit guard remains defence in depth; the container is the process boundary.
All frame outputs use anonymous atomic no-replace installation and honor readonly/dryrun.

S2 review measurements (aarch64, ten sequential calls; p95=max): runtime
**527 MB** / **115,126,863 content bytes**, test **975 MB**. Restoring checked-hash
stdlib caches adds 4,434,181 content bytes versus an otherwise identical control.
Fresh pandas import compiles no stdlib sources. These are shared-host loaded
samples; an idle-host rerun is deferred because unrelated work remained active.

| Operation | p50 ms | p95 ms |
|---|---:|---:|
| `frame_import` | 1004 | 1047 |
| `frame_inspect` | 983 | 1007 |
| `frame_export` | 982 | 1022 |
| `frame_select` | 998 | 1099 |
| `frame_groupby` | 976 | 1060 |
| `frame_resample` | 1048 | 1175 |
| `la_solve` | 929 | 992 |
| `docker_overhead` | 406 | 434 |

CSV cap probes pass at 200000 rows / 500000 cells / near 16MiB: 1.20–2.53 seconds,
243MiB total cgroup peak under the 4GiB limit, including the probe harness.
See docs/timings.md and the separate measurement-s2-review-* raw files for
workloads, full host traces and the idle-host qualification. CTest passes 17/17
(312.74 s), including 19 review regressions, 17 original S2 groups and ASan/UBSan
parser fuzz. All three recipes pass real execution, --check and --dry-run.

Explicit int64 JSON columns accept integer tokens or decimal strings, not
floating tokens: lmcpp can promote an out-of-range JSON integer to a rounded
float. This prevents a value below INT64_MIN from silently becoming INT64_MIN.
Conversion of already typed float artifacts remains explicit and range-checked.
