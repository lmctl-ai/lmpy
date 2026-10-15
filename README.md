# pycli

An MIT C++17/lmcpp frontend for pinned NumPy 2.3.3 / SciPy 1.16.2 / pandas 2.3.3 / statsmodels 0.14.6 (S0–S5).
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
the tests); the C++ parser is a copy with one extension: a variable declared as a vector (`x0: {"v": [1, 2]}`; bounds `{"v": [[0, 2], [0, 5]]}` per component or one pair for all) may be used
only as `v[0]`, `v[1]` with an integer literal below its length, and a scalar may not be indexed. Rejections happen in
C++ before any process starts (`EXPR_SYNTAX`, `EXPR_BAD_BYTE`, `EXPR_UNKNOWN_IDENT`, `EXPR_UNKNOWN_FUNCTION`, `EXPR_ARITY`,
`EXPR_TOO_LONG`, `EXPR_TOO_DEEP`, `EXPR_TOO_MANY_NODES`, `EXPR_NAME_COLLISION`, `EXPR_NOT_SCALAR`, `EXPR_INDEX`,
`EXPR_VECTOR_INDEX`, with `position` and `near`). Python receives only the JSON AST and compiles it iteratively
(no recursion, no `eval`/`exec`/`compile`/`getattr` on caller data) into a postfix program of numpy/scipy function *objects*
from a closed table, re-checking node kinds, arities, names and indices (`INTERNAL_ERROR`/`EXPR_SPEC_MISMATCH` otherwise).
Functions and operators follow **R's semantics** where rcli claims interchangeability: comparisons and `& |` are three-valued
(a NaN operand gives NA; `NA & FALSE` is FALSE, `NA | TRUE` is TRUE; numbers are true if nonzero), `ifelse` with an NA/NaN test
returns NaN (it does not take the else branch) and evaluates both branches, `^` follows R_pow (`(-8)^(1/3)` is NaN, a negative base
with an infinite exponent is NaN), `choose`/`lchoose` round `k` half-to-even with a one-time `NumericalWarning` and accept negative
`n`, `lchoose` works in the log domain (`lchoose(2000,1000)` = 1382.268 instead of an overflow), `beta`/`lbeta` are NaN for negative
arguments, `gamma(0)` and `digamma(0)` are NaN. `tests/expr-semantics-golden.json` holds R's answers for all 59 shared expressions on
a 20 x 19 grid of edge values (0, negative, non-integer, huge, +-Inf, NaN) and `tests/s3_semantics.py` requires pycli to reproduce
every cell; `scripts/expr-semantics-diff.py [--regen] [--rcli rcli]` regenerates the table from Rscript and runs spot cases through
the real rcli runtime. One known numerical difference: `choose`/`lchoose` with `|k| >= 1e9` agree on finite/non-finite class only.

Unknowns are scalars or vectors of at most 64 components in total; they are ordered by name (vector components by index),
which fixes the order of `gradient` expressions and of the flat vector. Data columns (`data`) are finite vectors from inline
arrays or `.npy` / `.safetensors#key` refs, equal length, <=200000 rows, <=32 columns; table (`.npz`) refs are not read
directly (export a column with `array_import`). Names are `[a-z][a-z0-9_]{0,31}` and may not collide with functions,
constants or each other (`BAD_NAME`, `EXPR_NAME_COLLISION`).

Statuses (exit 0): `OK`, `NO_CONVERGENCE` (the solver's own verdict, with its message, including "precision loss"),
`EVAL_LIMIT` (`max_evals`, default 500000, cap 2,000,000, counting every objective, gradient, constraint and residual
call), `TIME_LIMIT` (a wall-clock budget of the call timeout minus 10 s, checked before every evaluation including the gradient check, the final residual and the standard-error steps, so the worker stops before the hard kill) and
`NON_FINITE`. `EVAL_LIMIT`/`TIME_LIMIT`/`NON_FINITE` return the best point found so far by objective value; with constraints that point can be infeasible (read `constraint_violation`). A non-finite objective at `x0`
is `NONFINITE_OBJECTIVE` (exit 2; a non-finite analytic gradient at `x0` is `NONFINITE_GRADIENT`); later non-finite objectives count as +Inf (`nonfinite: "inf"`, reported as
`nonfinite_evals` plus a warning) or stop the run (`nonfinite: "error"`). Constraint violation above 1e-6 turns a solver
success into `NO_CONVERGENCE`. Gradient expressions are compared with finite differences at `x0` and a mismatch is a
`NumericalWarning`. Method rules are validated up front: bounds not for BFGS, constraints only SLSQP/trust-constr,
gradients not for Nelder-Mead/Powell; the default method is SLSQP with constraints, L-BFGS-B with bounds, else BFGS.

`opt_least_squares` fits `model` against the `response` column (residual = (model - response)/sigma, `sigma` a positive
number or data column) or minimises concatenated `residuals` expressions. Standard errors: with the central-difference
Jacobian J at the solution (scipy's own 2-point Jacobian is only first-order accurate with an absolute step and was off in the
5th digit for NIST Misra1a), `cov = (J'J)^-1`, multiplied by `2*cost/(m-n)` (`curve_fit` convention) unless `absolute_sigma`. Each
column uses a ladder of nine central-difference steps around `eps^(1/3) * max(|x_j|, 1e-4 * rho/||J_j||)` (rho the residual scale,
J_j the solver's column) and keeps the step whose neighbours agree best, so a parameter that is ~0 at the solution or a response of
magnitude 1e6 still gives standard errors accurate to ~1e-6 (closed-form OLS fixtures for y = 1000x, 1e6 x, 1e-3 x; Misra1a). Columns
are equilibrated and the Jacobian is declared singular below a **1e-8 relative singular value** (`SINGULAR_JACOBIAN`, standard errors
`"NA"`): legitimately worse-conditioned fits such as high-degree polynomials land here, and `jacobian_condition` (of the equilibrated
Jacobian) lets you judge; above 1e6 a `NumericalWarning` says the errors may be inaccurate, and `JACOBIAN_UNCERTAIN` means the
finite-difference and solver Jacobians disagree by more than 0.1 % on some column. `cov_status` is `OK`, `SINGULAR_JACOBIAN`,
`NO_DOF` or `UNAVAILABLE_BUDGET` (standard errors are then `"NA"`). The residual count times the number of unknowns is capped at
20,000,000 before spawn (vector residual expressions count `rows` each): `INPUT_TOO_LARGE`. With a robust `loss` scipy's loss-modified Jacobian is used and `std_err` is approximate. The solver runs
with `x_scale="jac"` and `tol` (default 1e-10) for ftol/xtol/gtol.

`num_quad` is QUADPACK (`scipy.integrate.quad`, `full_output`): a warning-level failure (oscillatory or divergent integrand,
`limit` exhausted) is `NO_CONVERGENCE` with the QUADPACK message and the unreliable `value`/`abserr`; a non-finite integrand
value (e.g. a pole on a Gauss-Kronrod node) is `NON_FINITE`. `num_root` needs finite `f(lower)`, `f(upper)` of opposite
sign (`NONFINITE_BRACKET`, `BAD_INTERVAL`, exit 2); an exactly-zero end is returned at once with 0 iterations. `xtol` is a bracket
width, not an `|f|` target. A singularity on the midpoint node of a symmetric interval (`1/sqrt(abs(x))` on [-1, 1]) is `NON_FINITE`;
the message suggests `points`.

S3 latency through the shim is about 1.0-1.1 s per call (p50), of which ~0.43 s is the worker's NumPy/SciPy import; runtime image
unchanged at 527 MB on disk. Details: [docs/s3.md](docs/s3.md), [docs/timings.md](docs/timings.md).

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

### S4: LP/MILP and ODE

| Operation | Access | Contract |
|---|---|---|
| `opt_linprog` | read | HiGHS dense LP/MILP; bounded inline x, slack and LP duals; never writes |
| `opt_linprog_to` | write | Same solve, all nonempty vectors in exclusive `out` safetensors |
| `ode_solve` | read | Closed-expression solve_ivp RHS/events; bounded inline trajectory; never writes |
| `ode_solve_to` | write | Same integration, time/state/event arrays in exclusive `out` safetensors |

These follow S1's read / `_to` pattern. Read forms work under `--readonly`; results
above the conservative inline budget (850 vector numbers or 24KiB vector JSON,
with room for metadata under the overall 1000-number /64KiB cap) fail with
`OUTPUT_TOO_LARGE` and recommend `_to`. They never create an automatic artifact.
Write forms require `out`, honor readonly/dryrun, and fail without an artifact.
`outputs_in_artifact` is explicit: false for inline, true for descriptors in x,
slack, duals.*, t, y or events. Read descriptors with `array_inspect` (limit<=1000).
An LP write without a primal fails; a partial ODE can still be saved with its
non-OK domain status. Inspect `result.status`, not just the protocol envelope.

LP: `c`, paired `a_ub/b_ub`, `a_eq/b_eq`, per-variable `bounds`, `integrality`
(continuous/integer/binary), `maximize`; 512 variables, 4096 combined rows and
200000 combined dense matrix entries. S1 inline/NPY/safetensors-ref inputs.
All finite input magnitudes, including bounds, are <=1e15; null or signed
`-Inf` / `Inf` tokens open bounds. Defaults [0,null]; equal bounds fix variables;
binary intersects [0,1]. Larger finite values fail validation before spawn with
`VALUE_EXCEEDS_SOLVER_LIMIT`, rather than becoming HiGHS infinity or model errors.

Rows are divided by their largest nonzero coefficient; the objective likewise.
Nonzero coefficient ratios per row/objective must be >=1e-9; scaled RHS magnitude
must be <=1e15. Unsafe ranges fail before spawn (`UNSAFE_COEFFICIENT_RANGE`).
The native small-coefficient cutoff is fixed at 1e-12. Marginals and objective/MIP
bounds are restored to original units. Original-data checks use per-row relative
residuals with no unit-sized absolute floor, bounds, integrality, LP dual signs,
stationarity, complementarity and primal/dual objective agreement (tolerance1e-7).
A failed check returns NUMERICAL_FAILURE, never OK. `primal_feasible`,
`primal_relative_violation`, `dual_feasible`, `dual_relative_violation`,
`relative_duality_gap` and `optimality_verified` expose the checks. MILP optimality
uses native gap/bound metadata, not an independently checked integer certificate.
Integer components within 1e-6 are rounded, signed zero normalized, and feasibility
rechecked on the returned x; `integrality_error` retains the pre-rounding error.

LP statuses derive from the preserved `highs_model_status`, not SciPy's ambiguous
status alone: OK, INFEASIBLE, UNBOUNDED, INFEASIBLE_OR_UNBOUNDED, MODEL_ERROR,
TIME_LIMIT, NODE_LIMIT, ITERATION_LIMIT, NUMERICAL_FAILURE. Unknown native statuses
fail conservatively. Original SciPy message/status remain. LP duals follow the
original objective, including maximize. MILP exposes available `mip_gap`,
`mip_node_count`, `mip_dual_bound`; nonzero requested-gap incumbents are labeled
incumbent. Limits: time >0..30s (default10), node 0..100000 (default10000),
iterations1..100000; mip_rel_gap0..0.1 (default0); one native thread.

ODE: one RHS per component in `y[i]`, `t`, optional named params/states; named y0
and RHS objects are sorted by name. At most32 states /16 parameters; S3 reserved
name rules apply (use b instead of beta). Methods RK45, RK23, DOP853, Radau, BDF,
LSODA; **stiff problems: use BDF/Radau/LSODA**. Default RK45 on stiff Robertson
can spend its full10s budget and return TIME_LIMIT/STEP_LIMIT/EVAL_LIMIT with an
accepted partial state. Tolerances are local solver controls, not global bounds.
rtol1e-13..0.1; scalar/per-state atol1e-15..0.1; optional positive max_step<=1e12.
Distinct finite t_span endpoints have abs<=1e12; forward/reverse supported.

Default101 uniform samples, max_output_points2..10000; timestamp duplicates due
to floating resolution are removed, preserving endpoints. `effective_output_points`
reports the grid size. Explicit ordered t_eval accepts inline/refs, with missing
endpoints added/reported. Potential trajectory/event/final output <=200000 numbers
is checked before spawn. Every RHS component, Jacobian RHS and event evaluation
shares max_evals (default/cap200000). Accepted max_steps default10000/cap20000;
wall default10s/cap30, bounded by the process budget with a two-second kill margin.

Up to4 events {expr,direction:-1/0/1,terminal:false}; roots exactly at the initial
time are excluded, including terminal events. Actual NaN/Inf expressions still
stop NON_FINITE. Per-event32-root cap gives EVENT_LIMIT; event_index is integer
(I64 in artifacts), grouped by event. A terminal event returns its localized root
state and time, including with coarse t_eval. Every non-OK result has
`termination:failed`, `partial_trajectory:true`; statuses include EVAL_LIMIT,
TIME_LIMIT, STEP_LIMIT, EVENT_LIMIT, NON_FINITE, STEP_SIZE_TOO_SMALL, NO_CONVERGENCE.
Callback interruption returns initial + last accepted state, never rejected trial
states; earlier samples/events are unavailable, so zero event counts are incomplete.

Examples: lp-plan.lms (CSV to hand-checked cost4 plan), ode-sir.lms (population
conservation and infected-peak event); both remain inline and use read operations.
HiGHS MIT is covered by the shipped SciPy notices. No new wheel.
See [docs/s4.md](docs/s4.md), [review fixes](docs/review-s4-fixes.md) and
[measurements](docs/timings.md). amd64 execution remains unverified.

Review acceptance: both image builds exit0; CTest **24/24, exit0** (592.68s),
including12 review and16 original S4 groups. Seven recipes pass real/check/dry-run
(21 exits0). Hardened cap probes pass with100.34MiB cgroup peak, including the
harness. Runtime527MB /115249079 content bytes; test1.06GB. Current image IDs,
shared-host timing samples and per-finding dispositions are in the linked docs.

### S5: statsmodels (OLS/WLS, GLM, SARIMAX, ACF/PACF)

| Operation | Access | Contract |
|---|---|---|
| `sm_ols` | read | OLS/WLS on a JSON term list; nonrobust, HC0-HC3 or one-way cluster covariance; coefficient table, R², F, llf, AIC/BIC, Durbin-Watson, condition number; optional inline `include` of `fitted`/`residuals` |
| `sm_ols_to` | write | Same fit; `coef` (p x 6), `fitted`, `residuals` (and `row_index` when rows were dropped) in an exclusive safetensors `out` |
| `sm_glm` | read | GLM by IRLS: gaussian, poisson, binomial, gamma with the documented links, offset/exposure/variance weights, deviance, null deviance, Pearson chi2, scale, llf, AIC/BIC |
| `sm_glm_to` | write | Same fit, artifact as `sm_ols_to` (response-scale fitted values and residuals) |
| `sm_arima` | read | SARIMAX exact Gaussian ML: order, seasonal order, trend, exog, optimizer, start parameters, parameter table, sigma2, llf, AIC/BIC/HQIC, forecast with intervals (h <= 100) |
| `sm_arima_to` | write | Same fit; `residuals` and `forecast` (3 x h: mean, lower, upper, h <= 1000) in an artifact |
| `sm_tsa_acf` | read | Sample ACF or PACF with confidence intervals, nlags <= 100 |

**No formulas.** Models are specified by a JSON term list; patsy, formulaic and `statsmodels.formula` are never imported (statsmodels'
package `__init__` imports `patsy.util` unconditionally, so the worker satisfies that single import with an empty stub that is removed
again, and the real packages are on the worker's blocked-import list; every response reports `meta.forbidden_modules_loaded`, tested to be
empty). The wheels for statsmodels 0.14.6, patsy 1.0.3 and packaging 26.3 are hash-locked for both architectures. Image and import costs are
in [docs/s5.md](docs/s5.md) and [docs/timings.md](docs/timings.md).

Data comes from exactly one of `columns` (inline table, <=1000 cells), `csv` (UTF-8 file with header, S2 limits: <=16 MiB, <=200000 rows)
or `frame` (a `.npz` table from `frame_import`, which carries typed and ordered-category columns). The response and other columns are named
in `y` / `series`, `weights`, `cluster`, `offset`, `exposure`, `exog`; names are 1-256 bytes of letters (any language), digits, space and
`_ . - % $ @ / #`, validated before spawn.

Terms (`terms`, 1-64, no strings): `{"const": true|false}`, `{"var": col}`, `{"cat": col, "ref": level}`, `{"interact": [col, col(, col)]}`,
`{"poly": [col, degree 2..6]}`. An intercept is included unless `{"const": false}`; its column position follows the list (default first).
Column names: `const`; `x`; `g[T.level]` (treatment coding, the reference level is mandatory and excluded; levels are the observed ones in
declared category order, sorted string order, numeric order, or False/True); `a:b` for interactions (factors in the order given; a categorical
factor must also have its own `cat` term and contributes its dummy columns); `x`, `x^2`, ... `x^d` for raw powers. Differences from other
tools: patsy writes `C(g, Treatment('b'))[T.a]`, R writes `ga`, and R's `poly()` is orthogonal unless `raw=TRUE`. The design is checked
against statsmodels' formula API in the test image (same coefficients and standard errors). At most 256 design columns and 10,000,000 design
cells; wider tables need `_to` (inline coefficient tables are limited to about 100 coefficients).

Missing values: null/NaN in any used column are *missing*. `na_action: "error"` (default) fails with `MISSING_VALUES` and a row count;
`"drop"` removes the rows and reports `n_dropped` (the kept rows keep their order; `_to` adds a `row_index` tensor). Infinities are always
`NONFINITE_DATA`. For `sm_arima`, missing series values are part of the state-space model and reported as `n_missing`.

Statuses (exit 0 unless the input is invalid): `OK`; `RANK_DEFICIENT` (the design has fewer independent columns than terms: estimates are
the minimum-norm solution, standard errors/tests/intervals are the string `NA`); `NO_CONVERGENCE` (GLM: IRLS hit `maxiter`; ARIMA: the
optimizer reported failure; results are returned but not trustworthy); `PERFECT_SEPARATION` (binomial); `TIME_LIMIT` (the call timeout minus
10 s and the import time elapsed; a cooperative SIGALRM budget around the library loops, no partial result); `NUMERICAL_FAILURE`. When the
model has no residual degrees of freedom `no_dof` is true and uncertainty columns are `NA`. Write twins fail without an artifact unless the
status is `OK` or `RANK_DEFICIENT`. Domain errors are exit 2: `BAD_TERMS`, `BAD_NAME`, `UNKNOWN_COLUMN`, `NOT_NUMERIC`, `BAD_REFERENCE`,
`BAD_RESPONSE` (poisson needs non-negative integers, gamma positive, binomial within [0,1]), `BAD_WEIGHTS`, `DUPLICATE_COLUMN`,
`TOO_MANY_COLUMNS`, `MISSING_VALUES`, `NONFINITE_DATA`, `BAD_START_PARAMS`, `CONSTANT_SERIES`.

OLS: `cov_type` nonrobust|HC0..HC3|cluster (`use_t` defaults to true for nonrobust, false otherwise, as in statsmodels); the robust overall F is
computed on the array API (statsmodels' own `fvalue` path needs patsy) and tested against it; it is `NaN` when the design is rank deficient
or the robust covariance has fewer effective dimensions than coefficients (e.g. fewer clusters than coefficients). Overflowing sums of squares
give `NUMERICAL_FAILURE`. `aic`/`bic` count `k_params` = number of design
columns, statsmodels' (and lmstats') convention; R counts one more (the error variance), so R's AIC = aic + 2 and BIC = bic + ln(n)
(recorded in docs/oracle-findings.md). GLM links: gaussian identity|log|inverse_power, poisson log|identity|sqrt, binomial
logit|probit|cloglog, gamma inverse_power|log|identity (first is the default); `exposure` needs the log link; `weights` are variance
weights; `maxiter` <= 100 (`n_iter` is reported), `tol` is the IRLS deviance tolerance; gaussian/gamma use the Pearson-based scale. GLM
inference follows R's `summary.glm`: t with `df_resid` for gaussian and gamma (estimated scale; `sm_glm` gaussian equals `sm_ols`), z for poisson
and binomial, and z for every robust `cov_type`; `use_t` overrides. `n_iter` counts IRLS updates (R's `iter` for the same fit is one higher: it
counts the final evaluation). A fitted mean outside the family support (e.g. a negative Poisson mean with the identity link) is `NUMERICAL_FAILURE`
with `NA` uncertainty; `rank_deficient: true` is reported next to the more severe status instead of replacing it. Gamma `llf` is evaluated at
statsmodels' Pearson scale, whereas R's `logLik` uses deviance/n, and R's AIC counts the dispersion as a parameter (see docs/oracle-findings.md).
There is no prediction operation: `fitted`/`residuals` cover the training rows only; rebuild the design from `coefficients.names` and `coding`.

ARIMA: `order` p,q <= 5, d <= 2; `seasonal_order` P,Q <= 2, D <= 1, s in 2..24; state dimension <= 64 and <= 24 parameters; series <= 20000
observations and observations x states^2 <= 2.5e7 (the fitted results keep several n x states x states covariance arrays; the worker rejects
larger models with `INPUT_TOO_LARGE` before fitting). The default `lbfgs` fit is followed by a short Nelder-Mead polish for models with at most
8 parameters (`polished` reports it): statsmodels' finite-difference score stalls L-BFGS on the flat likelihood ridge. Parameter names are statsmodels' (`intercept`/`drift`, exog names, `ar.L1`, `ma.L1`, `ar.S.L12`, ..., `sigma2` last). The `c` trend
is the regression intercept, so an AR model's mean is `c / (1 - sum phi)`, whereas R's `arima` reports the mean itself. `cov_type`: `opg`
(statsmodels default), `oim`, or `approx` (numerical Hessian: what R reports). `nobs_effective` excludes the diffuse observations of
differencing (`aic`/`bic` use it). Forecast intervals are normal: `mean +- z(1-alpha/2) * se`. `start_params` must have the model's exact
parameter count.

ACF/PACF: `acf` is the biased (divide by n) estimate like R unless `adjusted`; confidence intervals are Bartlett's for the ACF and
`+-z/sqrt(n)` for the PACF; PACF methods `ywm` (R's), `ywadjusted`, `ols`, `ldb`; `nlags` < n (PACF: < n/2). Constant series are `CONSTANT_SERIES`. The series is rescaled by its largest magnitude first, so
the correlations are exact for any finite scale (1e-300 .. 1e300); undefined results are `NUMERICAL_FAILURE`.

## S6 machine learning

| Op | Access | Result |
|---|---|---|
| `ml_fit` | write | 13 closed numeric model families to exclusive safetensors; four forest/boosting families fit+predict to NPZ without a model |
| `ml_predict` | write | Version/attribute-checked model to NPZ: predict / predict_proba / decision_function / transform; bounded preview |
| `ml_cv` | read | Kfold/stratified/group/timeseries supervised CV, fold scores and population mean/std |

Pinned sklearn 1.7.2, joblib 1.5.2, threadpoolctl 3.6.0. `features` and `{scale,one_hot}` preprocessing are explicit; no missing/NaN/Inf input, unknown features, caller code or executable model state. Seeds required for stochastic families and all CV (sklearn MT19937, echoed). All worker pools and n_jobs are fixed at one.

ML caps checked before spawn: 20000 rows, 64 encoded features, 200000 cells, 32 classes/clusters/components (KNN k<=128), 200M work product across folds, <=20 folds, <=64 trees/depth12, <=5M KNN query pairs. Defaults/ranges and model v1 contract: [docs/s6.md](docs/s6.md), [ml-spec.json](ml-spec.json). CSV <=16MiB, model/NPZ <=32MiB; outputs <=1000 array entries/64KiB inline. Large predictions are artifact refs; previews <=800 cells/24KiB. Models reject version skew unless explicitly accepted. Nonconvergence/timeouts never produce a successful model.

Recipes: `oracle-ml-linear.lms`, `oracle-kmeans.lms`, `ml-classify-cv.lms`; optional rcli/lmstats legs give setup hints. Timing and image growth measurements are in [docs/timings.md](docs/timings.md).

S6 measurements (aarch64, ten sequential fresh calls through the shim; warm filesystem caches):

| Operation | p50 wall | Lazy import p50 |
|---|---:|---:|
| ml_fit logistic, 40 rows | 2.043 s | 1.235 s |
| ml_predict probabilities, 40 rows | 2.044 s | 1.238 s |
| ml_cv logistic, 4 folds | 2.060 s | 1.242 s |

Runtime **634 MB**, approximately **+61 MB** from S5. Hardened cap probes peak at **191 MiB** cgroup memory; the 64-tree/depth-12 forest
probe takes 14.79 s with an explicit 30 s budget. Other recorded probes take 2.0–3.2 s. Default ML compute budget is 10 s; near-cap fits can
need 30 s or report TIME_LIMIT. Full commands, raw data, p95 values and qualifications: [docs/timings.md](docs/timings.md).
