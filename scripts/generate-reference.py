#!/usr/bin/env python3
"""Generate docs/reference.md from the code: every status / tool code string literal in py/, src/ and bin/pycli must be in the curated table
below (and every table entry must still exist in the code), every compile-time cap (`constexpr` in src/*.hpp, src/main.cpp) must have a
meaning. tests/docs_gate.py runs this with --check so the shipped document cannot drift from the code.

  python3 scripts/generate-reference.py > docs/reference.md
  python3 scripts/generate-reference.py --check     # exit 1 when docs/reference.md is stale or a code/cap is unknown
"""
import json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Protocol envelope codes and process exit codes come from lmcpp (src/lmcpp.cpp exit_for / code_name); they are not scanned.
PROTOCOL = [
    ('INVALID_REQUEST', 2, 'The request is not a valid call (unknown operation, malformed flags, or the host shim refused the working directory).'),
    ('VALIDATION_ERROR', 2, 'The schema or a pre-spawn domain check rejected the inputs; `error.data.tool_code` names the rule (tool codes below). Also every explicit worker precheck (`BadData`).'),
    ('PARSE_ERROR', 3, 'The JSON input could not be parsed (including a numeric literal that overflows binary64, such as 1e999).'),
    ('NOT_FOUND', 4, 'A referenced file or tensor key does not exist (torchcli convention; rcli answers 2 here).'),
    ('CONFLICT', 5, 'The `out` artifact already exists; nothing is overwritten.'),
    ('FORBIDDEN', 6, 'The call is not allowed by policy.'),
    ('READONLY_VIOLATION', 6, 'A write operation under `--readonly` / `PYCLI_READONLY`.'),
    ('TIMEOUT', 7, 'The hard child cap (`PYCLI_TIMEOUT_S`, at most 50 s, imports included) killed the worker; the cooperative `time_limit` answers `TIME_LIMIT` long before this.'),
    ('SERVICE_UNAVAILABLE', 8, 'The Python interpreter could not be started (`PYTHON_NOT_FOUND`), or the host shim could not run the container (image missing, daemon down).'),
    ('MISSING_CONFIRMATION', 9, 'A dangerous operation without `--confirm` (pycli has none).'),
    ('UNSUPPORTED_PRODUCT_VERSION', 10, 'The caller required a product version this build does not satisfy.'),
    ('INTERNAL_ERROR', 1, 'The worker crashed or returned an unreadable result (`TOOL_CRASH`), or an internal invariant failed.'),
    ('COMMAND_FAILED', 1, 'The operation ran and could not deliver: a write form with a non-OK domain status (`tool_code` = the status), an inline result beyond its budget (`OUTPUT_TOO_LARGE`), or an unexpected library error (`PY_ERROR`).'),
]
SHIM = [
    ('INVALID_REQUEST', 2, 'bin/pycli: the working directory is not below `$HOME`, or contains a comma, quote, equals sign or control character (it is bind-mounted by path).'),
    ('SERVICE_UNAVAILABLE', 8, 'bin/pycli: docker could not run `pycli:runtime` (build it with `scripts/build.sh runtime`).'),
    ('INTERNAL_ERROR / TOOL_CRASH', 1, 'bin/pycli: the container exited by a signal or without a result.'),
    ('(signal)', '130 / 143', 'bin/pycli: INT / TERM received while the container ran; the container is killed and removed.'),
]

# kind: status = result.status (exit 0 in read operations; write forms fail with COMMAND_FAILED and tool_code = status);
#       tool_code = error.data.tool_code of a VALIDATION_ERROR (exit 2) unless the meaning says otherwise.
CODES = {
    # ---- domain statuses
    'OK': ('status', 'The operation delivered its full answer.'),
    'TIME_LIMIT': ('status', 'The cooperative `time_limit` budget expired; partial results are not returned (ODE: the partial trajectory is, with this status).'),
    'NUMERICAL_FAILURE': ('status', 'The library produced non-finite or inconsistent numbers, or pycli\'s re-verification of the result failed; no value is reported as if correct.'),
    'NO_CONVERGENCE': ('status', 'The iterative solver stopped without meeting its own convergence test (its message is attached).'),
    'EVAL_LIMIT': ('status', 'The expression-evaluation budget (`max_evals`) was reached.'),
    'NON_FINITE': ('status', 'An expression produced a non-finite value during the run (optimisation / quadrature / roots); as a tool code: a weight, capacity or data column holds NaN or Inf.'),
    'STOPPED': ('status', 'The optimiser stopped early on a user limit (iterations or evaluations) with a usable point.'),
    'JACOBIAN_UNCERTAIN': ('status', 'Least squares converged but the Jacobian is too ill-conditioned for the reported standard errors to be trusted.'),
    'SINGULAR': ('status', 'The matrix is singular (or its condition number reaches 1/eps): no inverse, solve, logm or shift-invert factorisation.'),
    'NOT_POSITIVE_DEFINITE': ('status', '`assume_a: pos` was requested but the Cholesky factorisation failed.'),
    'RANK_DEFICIENT': ('status', 'The design or matrix has fewer independent columns than requested; the minimum-norm answer is still returned where one exists.'),
    'ILL_CONDITIONED': ('status', 'The problem is solvable but its condition number makes the digits unreliable (reported, not hidden).'),
    'DEGENERATE': ('status', 'The geometric input is degenerate for the requested computation (collinear or coplanar points, too few points, duplicates).'),
    'DEGENERATE_DATA': ('status', 'The training data cannot support the model (a constant column where variation is required, an empty class, ...).'),
    'DEGENERATE_CLUSTERS': ('status', 'k-means ended with an empty cluster or fewer distinct points than clusters.'),
    'INSUFFICIENT_CLASSES': ('status', 'Fewer than two classes are present where classification needs them.'),
    'UNDEFINED': ('status', 'The requested quantity is mathematically undefined for these inputs (a correlation of a constant series, a metric with an empty denominator): the defined parts are returned, the undefined ones are listed.'),
    'UNDEFINED_METRIC': ('status', 'An `ml_metrics` quantity is undefined for the given labels (for example AUC with one class).'),
    'NO_PEAKS': ('status', '`sig_peaks` found no peak satisfying the constraints.'),
    'UNSTABLE': ('status', 'The designed filter has a pole on or outside the unit circle.'),
    'INFEASIBLE': ('status', 'LP / MILP / min-cost flow: no point satisfies the constraints (supplies cannot be routed within the capacities).'),
    'INFEASIBLE_OR_UNBOUNDED': ('status', 'HiGHS could not separate infeasibility from unboundedness for the model.'),
    'UNBOUNDED': ('status', 'The objective is unbounded (LP), or a negative-cost cycle has unlimited capacity (min-cost flow).'),
    'UNBALANCED': ('status', 'min_cost_flow supplies do not sum to zero.'),
    'MODEL_ERROR': ('status', 'HiGHS rejected the model itself.'),
    'ITERATION_LIMIT': ('status', 'The LP simplex / IPM iteration cap (`maxiter`) was reached.'),
    'NODE_LIMIT': ('status', 'The MILP branch-and-bound node cap (`node_limit`) was reached.'),
    'STEP_LIMIT': ('status', 'The ODE integrator reached its step cap.'),
    'STEP_SIZE_TOO_SMALL': ('status', 'The ODE integrator could not continue: the required step is below machine resolution.'),
    'EVENT_LIMIT': ('status', 'An ODE event that is not terminal fired more often than allowed.'),
    'NEGATIVE_CYCLE': ('status', 'bellman_ford found a negative cycle reachable from the source (or anywhere for all pairs); `cycle` lists it. A negative self loop is the one-node cycle.'),
    'UNREACHABLE': ('status', 'The target cannot be reached from the source; no path, no length.'),
    'OUT_OF_RANGE': ('status', 'A zipf quantile lies beyond loc + 1e5: that point is `NA`, the others keep their values.'),
    'TOLERANCE_UNATTAINABLE': ('status', 'The iterative sparse solver reported convergence, but the recomputed residual does not meet the requested `rtol` (double precision cannot reach it).'),
    'DOMAIN_ERROR': ('status', 'A special function was evaluated outside its mathematical domain (reported per argument).'),
    'PERFECT_SEPARATION': ('status', 'A binomial GLM separates the classes perfectly; coefficients diverge, the fit is reported with this status.'),
    # ---- opt_least_squares covariance status (`cov_status`, next to the parameter estimates)
    'SINGULAR_JACOBIAN': ('cov_status', 'The Jacobian at the solution is singular or too ill-conditioned: no standard errors.'),
    'NO_DOF': ('cov_status', 'No residual degrees of freedom (as many parameters as observations): no standard errors.'),
    'UNAVAILABLE_BUDGET': ('cov_status', 'The cooperative budget ran out before the covariance could be evaluated.'),
    # ---- validation tool codes (VALIDATION_ERROR, exit 2) and failure codes
    'BAD_PARAM': ('tool_code', 'A parameter value is outside its documented domain (ranges, enumerations, combinations).'),
    'BAD_DATA': ('tool_code', 'The data violates a documented rule (empty, wrong length, out of range, missing column).'),
    'BAD_DTYPE': ('tool_code', 'A column or array has a type the operation does not accept (for example string weights, mixed integer and string node ids, an integer id outside int64).'),
    'BAD_SHAPE': ('tool_code', 'An array has the wrong rank or dimensions.'),
    'BAD_INDEX': ('tool_code', 'A sparse row / column index lies outside the matrix.'),
    'BAD_KEY': ('tool_code', 'A safetensors key is invalid or reserved.'),
    'BAD_REF': ('tool_code', 'A file reference has the wrong extension or form (`path#key`), or an `out` path is not of the required kind.'),
    'BAD_NAME': ('tool_code', 'A variable, term or column name does not match `[a-z][a-z0-9_]{0,31}`.'),
    'BAD_NPY': ('tool_code', 'The `.npy` header is not a supported little-endian C-order v1/v2 numeric array.'),
    'BAD_CSV': ('tool_code', 'The CSV text is malformed for `array_import`.'),
    'BAD_BOUNDS': ('tool_code', 'A bounds entry is not `[lower, upper]`.'),
    'BAD_INTERVAL': ('tool_code', 'A root bracket does not change sign (or is not an interval).'),
    'BAD_RANGE': ('tool_code', 'Integration bounds lie outside the interpolation nodes without `extrapolate`.'),
    'BAD_PROBABILITY': ('tool_code', 'A ppf / isf probability is outside [0, 1].'),
    'BAD_TERMS': ('tool_code', 'The statsmodels design terms are invalid (unknown column, bad interaction, duplicate).'),
    'BAD_MODEL': ('tool_code', 'The model artifact is not a valid pycli model (version, shapes, metadata, reconstructed classes).'),
    'BAD_RESPONSE': ('tool_code', 'The GLM response violates the family (negative counts, non-positive gamma responses, non-binary binomial).'),
    'BAD_REFERENCE': ('tool_code', 'The requested reference level of a categorical term does not exist.'),
    'BAD_WEIGHTS': ('tool_code', 'Weights must be positive and finite.'),
    'BAD_START_PARAMS': ('tool_code', 'SARIMAX start parameters have the wrong length or are non-finite.'),
    'BAD_TABLE_OUTPUT': ('tool_code', 'The worker produced a table that fails the NPZ contract (internal consistency check, exit 1).'),
    'INPUT_TOO_LARGE': ('tool_code', 'A pre-spawn cap on rows, columns, numbers, nodes, edges, points or bytes is exceeded (the caps table below).'),
    'OUTPUT_TOO_LARGE': ('tool_code', 'The inline result would exceed its budget (1000 numbers / 64 KiB, or an operation\'s own budget); use the `_to` form. COMMAND_FAILED, exit 1.'),
    'NA_NOT_ALLOWED': ('tool_code', 'A missing value sits where the operation needs every value (node ids, weights, metrics inputs).'),
    'NON_FINITE': ('tool_code', 'see the status entry above (the same code is used for a NaN / Inf weight or data column).'),
    'NONFINITE_DATA': ('tool_code', 'A statsmodels data column contains Inf.'),
    'NONFINITE_BRACKET': ('tool_code', 'f is not finite at a bracket end.'),
    'NONFINITE_OBJECTIVE': ('tool_code', 'The objective is not finite at x0.'),
    'NONFINITE_GRADIENT': ('tool_code', 'A gradient expression is not finite at x0.'),
    'NONFINITE_RESIDUAL': ('tool_code', 'The residuals are not finite at x0.'),
    'MISSING_VALUES': ('tool_code', 'A statsmodels column has missing values and `missing: error` is in force.'),
    'NO_DATA': ('tool_code', 'No rows remain after dropping missing values.'),
    'NOT_NUMERIC': ('tool_code', 'A column used as a number is not numeric.'),
    'NOT_CATEGORICAL': ('tool_code', 'A column used as a factor is not categorical / string.'),
    'INTERACT_NEEDS_CAT_TERM': ('tool_code', 'An interaction with a categorical column needs the main term as well.'),
    'DUPLICATE_COLUMN': ('tool_code', 'A column name is used twice.'),
    'TOO_MANY_COLUMNS': ('tool_code', 'More columns than the table / design cap.'),
    'TOO_SHORT': ('tool_code', 'The series is too short for the requested lags / test.'),
    'CONSTANT_SERIES': ('tool_code', 'The series is constant: the statistic is undefined.'),
    'NOT_SORTED': ('tool_code', 'The datetime column of a rolling / resample operation is not ascending.'),
    'NOT_INCREASING': ('tool_code', 'Interpolation nodes must be strictly increasing.'),
    'DUPLICATE_X': ('tool_code', 'Interpolation nodes contain a duplicate.'),
    'NOT_SYMMETRIC': ('tool_code', 'A symmetric matrix was required (`assume_a: sym/pos`, eigh, cholesky, hull distance matrices).'),
    'NOT_INTEGER': ('tool_code', 'min_cost_flow needs integer capacities and costs.'),
    'DUPLICATE_EDGE': ('tool_code', 'min_cost_flow needs one edge per (from, to) pair (parallel edges cannot merge costs).'),
    'UNKNOWN_NODE': ('tool_code', 'A source / target / sink / supplies node does not occur in the edges.'),
    'UNKNOWN_COLUMN': ('tool_code', 'A named column is not in the table.'),
    'KEY_DTYPE_MISMATCH': ('tool_code', 'frame_merge keys have incompatible types on the two sides.'),
    'MERGE_VALIDATION': ('tool_code', 'frame_merge `validate` (one_to_one, ...) is violated.'),
    'ROW_EXPLOSION': ('tool_code', 'frame_merge would produce more rows than the table cap.'),
    'INT64_NOT_EXACT': ('tool_code', 'An int64 value beyond 2^53 cannot be converted exactly to binary64 (never rounded).'),
    'INT64_OVERFLOW': ('tool_code', 'An integer result does not fit int64 (COMMAND_FAILED, exit 1).'),
    'UNDER_DETERMINED': ('tool_code', 'opt_root needs as many equations as unknowns.'),
    'GRADIENT_ARITY': ('tool_code', 'The gradient list needs one expression per scalar unknown.'),
    'UNSAFE_COEFFICIENT_RANGE': ('tool_code', 'LP coefficient magnitudes span more than 1e9 after scaling.'),
    'VALUE_EXCEEDS_SOLVER_LIMIT': ('tool_code', 'An LP value exceeds 1e15 in magnitude or is not finite.'),
    'MODEL_NOT_PERSISTABLE': ('tool_code', 'Tree ensembles are fit-and-predict only; they are never saved (no pickle).'),
    'MODEL_VERSION_MISMATCH': ('tool_code', 'The model was written by another sklearn version; `allow_version_skew` must be explicit.'),
    'EXPR_SYNTAX': ('tool_code', 'The expression does not parse under the shared grammar (expr-spec.json).'),
    'EXPR_ARITY': ('tool_code', 'A function is called with the wrong number of arguments.'),
    'EXPR_UNKNOWN_FUNCTION': ('tool_code', 'The function is not in the closed table of the shared grammar.'),
    'EXPR_UNKNOWN_IDENT': ('tool_code', 'An identifier is neither an unknown, a data column nor a bound name.'),
    'EXPR_NAME_COLLISION': ('tool_code', 'A name is both an unknown and a data column.'),
    'EXPR_TOO_LONG': ('tool_code', 'The expression text exceeds the length cap.'),
    'EXPR_TOO_DEEP': ('tool_code', 'The expression nests deeper than the cap.'),
    'EXPR_TOO_MANY_NODES': ('tool_code', 'The expression has more AST nodes than the cap.'),
    'EXPR_BAD_BYTE': ('tool_code', 'The expression contains a byte outside the grammar\'s alphabet.'),
    'EXPR_INDEX': ('tool_code', 'A vector index is invalid.'),
    'EXPR_VECTOR_INDEX': ('tool_code', 'Vector indexing is used where a scalar is required.'),
    'EXPR_NOT_SCALAR': ('tool_code', 'The expression must evaluate to one number.'),
    'EXPR_SPEC_MISMATCH': ('tool_code', 'The worker\'s expression spec differs from the front end\'s (INTERNAL_ERROR, exit 1).'),
    'INVALID_PATH': ('tool_code', 'A path is empty, absolute, contains `..`, a NUL byte or a symlink escaping the working directory.'),
    'PATH_OUTSIDE_WORKSPACE': ('tool_code', 'A path resolves outside the mounted working directory.'),
    'PATH_OUTSIDE_CWD': ('tool_code', 'A path resolves outside the current working directory.'),
    'INVALID_ARTIFACT': ('tool_code', 'A safetensors / npz file is structurally invalid.'),
    'INVALID_HEADER': ('tool_code', 'The safetensors header is not valid JSON of the expected shape.'),
    'INVALID_OFFSETS': ('tool_code', 'safetensors data offsets overlap, exceed the file or are not contiguous.'),
    'INVALID_SHAPE': ('tool_code', 'A tensor shape is negative, overflows or disagrees with its byte length.'),
    'INVALID_TENSOR': ('tool_code', 'A tensor entry is malformed.'),
    'INVALID_TENSOR_REF': ('tool_code', 'A `path#key` reference is malformed.'),
    'TENSOR_NOT_FOUND': ('tool_code', 'The key is not in the safetensors file (NOT_FOUND, exit 4).'),
    'ARTIFACT_NOT_FOUND': ('tool_code', 'The referenced file does not exist (NOT_FOUND, exit 4).'),
    'TOO_MANY_TENSORS': ('tool_code', 'The safetensors file holds more tensors than the reader accepts.'),
    'DUPLICATE_KEY': ('tool_code', 'The safetensors header names a tensor twice.'),
    'UNSUPPORTED_DTYPE': ('tool_code', 'The tensor dtype is outside the supported whitelist (F64 F32 I64 I32 U8 BOOL; `.npy` <f8 <f4 <i8 <i4 |b1 |u1).'),
    'OUTPUT_WRITE_FAILED': ('tool_code', 'The artifact could not be installed atomically (COMMAND_FAILED, exit 1).'),
    'PY_ERROR': ('tool_code', 'An unexpected Python exception reached the worker boundary (COMMAND_FAILED, exit 1; the exception type is attached).'),
    'TOOL_CRASH': ('tool_code', 'The worker died by a signal or without a result (INTERNAL_ERROR, exit 1).'),
    'TIMEOUT': ('tool_code', 'The hard child cap killed the worker (TIMEOUT, exit 7).'),
    'PYTHON_NOT_FOUND': ('tool_code', 'The interpreter could not be executed (SERVICE_UNAVAILABLE, exit 8).'),
    'NUMEXPR_MAX_THREADS': ('tool_code', 'Internal: the thread environment of the child is not the pinned one (also the name of that environment variable).'),
}
# Uppercase literals that look like codes but are not: environment names, dtypes, solver / method names, pandas frequencies.
NON_CODES = {'LC_ALL', 'HOME', 'PATH', 'TMPDIR', 'TZ', 'UTC', 'NA', 'MPLBACKEND', 'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED', 'PYTHONNOUSERSITE',
             'OPENBLAS_NUM_THREADS', 'OPENBLAS_CORETYPE', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYCLI_PYTHON',
             'PYCLI_TIMEOUT_S', 'PYCLI_BLAS_CORETYPE', 'PYCLI_LIMIT_NUMBERS', 'PYCLI_READONLY', 'PYCLI_DRYRUN', 'ARMV8', 'HC0', 'HC1', 'HC2', 'HC3',
             'AIC', 'BIC', 'F64', 'F32', 'F16', 'BF16', 'I64', 'I32', 'I16', 'I8', 'U64', 'U32', 'U16', 'U8', 'BOOL', 'F8_E4M3', 'F8_E5M2',
             'RK45', 'RK23', 'DOP853', 'BDF', 'LSODA', 'BFGS', 'SLSQP', 'QS', 'MS', 'ME', 'YS',
             'INTERNAL_ERROR', 'INVALID_REQUEST', 'SERVICE_UNAVAILABLE'}  # protocol codes emitted by the shim: in the PROTOCOL / SHIM tables
SINGLE = {'OK', 'SINGULAR', 'STOPPED', 'DEGENERATE', 'UNREACHABLE', 'UNBALANCED', 'UNBOUNDED', 'INFEASIBLE', 'UNDEFINED', 'UNSTABLE', 'TIMEOUT'}

CAPS = {
    'MAX_NUMBERS': 'numbers per referenced array / CSV file (`.npy`, safetensors key, csv_file)',
    'MAX_BYTES': 'bytes per artifact (256 MiB)',
    'MAX_INLINE': 'numbers inline in a result (and in an inline array input)',
    'MAX_CSV_BYTES': 'bytes of an `array_import` CSV (32 MiB)',
    'MAX_OUTPUT_NUMBERS': 'numbers written per call (2,000,000)',
    'MAX_VARIABLES': 'unknowns of an expression problem',
    'MAX_COLUMNS': 'data columns bound into an expression problem',
    'MAX_CONSTRAINTS': 'constraints of opt_minimize',
    'MAX_JACOBIAN_ENTRIES': 'residual rows x unknowns of opt_least_squares',
    'ROWS': 'table rows (frames, `.npz` tables, CSV import)',
    'BYTES': 'bytes per table artifact (256 MiB)',
    'COLS': 'table columns',
    'CELLS': 'table cells (rows x columns)',
    'CSV_BYTES': 'bytes of a frame CSV (16 MiB)',
    'STRING_BYTES': 'bytes per string cell',
    'LP_VARIABLES': 'LP / MILP variables',
    'LP_ROWS': 'LP / MILP constraint rows',
    'LP_ENTRIES': 'LP / MILP matrix entries',
    'ODE_STATES': 'ODE state dimension',
    'ODE_POINTS': 'ODE output points',
    'S4_OUTPUT_NUMBERS': 'numbers in an LP / ODE artifact',
    'LP_MAGNITUDE': 'largest LP magnitude (1e15)',
    'LP_MIN_RATIO': 'smallest allowed ratio of nonzero LP coefficients after scaling (1e-9)',
    'SM_TERMS': 'statsmodels design terms',
    'SM_NAME_BYTES': 'bytes per term name',
    'SM_ARIMA_STATES': 'SARIMAX state dimension',
    'SM_ARIMA_PARAMS': 'SARIMAX parameters',
    'SM_EXOG': 'SARIMAX exogenous columns',
    'SM_START': 'SARIMAX start parameters',
    'SM_INLINE_FORECAST': 'forecast steps inline',
    'SM_FORECAST': 'forecast steps in an artifact',
    'SM_ACF_LAGS': 'ACF / PACF lags',
    'S7_INLINE': 'numbers inline for S7 results',
    'S7_RBF_POINTS': 'RBF interpolation nodes',
    'S7_RBF_DIM': 'RBF dimension',
    'S7_KNN_DIM': 'spatial_knn dimension',
    'S7_ORDER': 'filter order',
    'S7_NPERSEG': 'spectral segment length',
    'S7_NFFT': 'FFT length',
    'S7_DIST_PAIRS': 'spatial_dist pairs',
    'S7_FREQS': 'frequency-response points',
    'S7_KNN_K': 'spatial_knn neighbours',
    'S7_HULL_DIM': 'hull / Delaunay / Voronoi dimension',
    'S7_ARTIFACT': 'numbers in an S7 artifact (1,900,000)',
    'S7_BRUTE_WORK': 'brute-force distance work (pairs x dimension)',
    'S8_INLINE': 'numbers inline for S8 results',
    'S8_VALUES': 'values of a referenced evaluation array (stats_dist_to points, frame_to_array values)',
    'S8_COLS': 'columns of an S8 table',
    'S8_CELLS': 'cells of an S8 table',
    'S8_RVS_INLINE': 'rvs values inline',
    'S8_RVS_FILE': 'rvs values in an artifact (1,000,000)',
    'S8_WORK': 'frame_merge / rolling work units',
    'S8_CLASSES': 'distinct labels for classification metrics',
    'S8_LEVY_POINTS': 'evaluation points for levy_stable and vonmises (numerical integration)',
    'S8_ASSIGN': 'frame_mutate assignments per call',
    'S8_WINDOW': 'rolling window length',
    'S8_EXACT': 'largest integer converted exactly to binary64 (2^53)',
    'S8B_SPARSE_DIM': 'sparse matrix dimension',
    'S8B_SPARSE_NNZ': 'sparse stored entries',
    'S8B_SPSOLVE_DIM': 'direct sparse solve / shift-invert dimension',
    'S8B_EIG_K': 'sparse eigen / singular values requested',
    'S8B_ARTIFACT': 'numbers in an S8b artifact (1,900,000)',
    'G_EDGES': 'graph_paths edges',
    'G_NODES': 'graph_paths distinct nodes',
    'G_FLOW_EDGES': 'graph_flow edges',
    'G_FLOW_NODES': 'graph_flow distinct nodes',
    'G_INLINE_NODES': 'nodes listed inline without a target',
    'G_ALLPAIRS_NODES': 'all-pairs nodes (and n*n <= G_MATRIX)',
    'G_MATRIX': 'numbers of the all-pairs matrix (about 1377 nodes)',
    'G_RECORD_CELLS': 'cells of inline edge records',
    'G_ID_BYTES': 'bytes per string node id',
    'G_WEIGHT': 'largest |weight| / capacity (1e15)',
    'G_COST': 'largest |cost| of min_cost_flow (1e12)',
}

TOKEN = re.compile(r"""['"]([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)['"]""")
SINGLE_TOKEN = re.compile(r"""['"](%s)['"]""" % '|'.join(sorted(SINGLE)))
CONSTEXPR = re.compile(r'^\s*constexpr\s+(?:size_t|double|uint64_t|int)\s+(.*?);', re.M)


def scan_codes():
    found = {}
    files = sorted(ROOT.glob('py/*.py')) + sorted(ROOT.glob('src/*.hpp')) + sorted(ROOT.glob('src/*.cpp')) + [ROOT / 'bin/pycli']
    for f in files:
        text = f.read_text()
        for token in TOKEN.findall(text) + SINGLE_TOKEN.findall(text):
            found.setdefault(token, set()).add(f.name)
    return found


def scan_caps():
    caps = []
    for f in sorted(ROOT.glob('src/*.hpp')) + [ROOT / 'src/main.cpp']:
        for m in CONSTEXPR.finditer(f.read_text()):
            for item in m.group(1).split(','):
                name, _, value = item.partition('=')
                name, value = name.strip(), value.strip()
                if name:
                    caps.append((name, value, f.name))
    return caps


def problems(found, caps):
    out = []
    unknown = sorted(t for t in found if t not in CODES and t not in NON_CODES)
    if unknown:
        out.append('code literals without a reference entry (add them to CODES or NON_CODES in scripts/generate-reference.py): ' + ', '.join('%s (%s)' % (t, '/'.join(sorted(found[t]))) for t in unknown))
    stale = sorted(t for t in CODES if t not in found)
    if stale:
        out.append('reference entries no longer in the code: ' + ', '.join(stale))
    names = {n for n, _, _ in caps}
    missing = sorted(n for n in names if n not in CAPS)
    if missing:
        out.append('caps without a meaning: ' + ', '.join(missing))
    gone = sorted(n for n in CAPS if n not in names)
    if gone:
        out.append('cap meanings no longer in the code: ' + ', '.join(gone))
    return out


def render(found, caps):
    schema = json.loads((ROOT / 'cli.schema.dev.json').read_text())
    lines = ['# pycli reference: error codes, statuses, tool codes and caps', '',
             'Generated by `scripts/generate-reference.py` from the code (the gate `tests/docs_gate.py` fails when this file is stale). Do not edit by hand.', '',
             'Schema version %s, %d operations (%d read, %d write).' % (schema['version'], len(schema['operations']),
                                                                       sum(o['access'] == 'read' for o in schema['operations']), sum(o['access'] == 'write' for o in schema['operations'])), '',
             '## Protocol error codes and exit codes', '',
             'The envelope is `{"success": false, "error": {"code", "message", "data"}}`; the exit code follows the lmapi protocol (lmcpp `exit_for`). `error.data.tool_code` carries the pycli-specific code from the tables below.', '',
             '| `error.code` | exit | Meaning |', '|---|---:|---|']
    lines += ['| `%s` | %s | %s |' % row for row in PROTOCOL]
    lines += ['', 'The host shim `bin/pycli` adds, before or around the container:', '', '| Envelope | exit | Meaning |', '|---|---:|---|']
    lines += ['| `%s` | %s | %s |' % row for row in SHIM]
    lines += ['', '## Domain statuses (`result.status`)', '',
              'A domain status is a result, not an error: exit 0 in read operations, with the fields the status allows. A write (`_to`) form with a non-OK status writes nothing and fails with `COMMAND_FAILED`, `error.data.tool_code` = the status and the bounded result fields under `error.data`. Inspect `result.status`, not only the envelope.', '',
              '| Status | Meaning | Where |', '|---|---|---|']
    for token, (kind, meaning) in sorted(CODES.items()):
        if kind == 'status':
            lines.append('| `%s` | %s | %s |' % (token, meaning, ', '.join(sorted(found[token]))))
    lines += ['', '## Covariance status of opt_least_squares (`cov_status`)', '', '| cov_status | Meaning |', '|---|---|']
    for token, (kind, meaning) in sorted(CODES.items()):
        if kind == 'cov_status':
            lines.append('| `%s` | %s |' % (token, meaning))
    lines += ['', '## Validation and failure tool codes (`error.data.tool_code`)', '',
              'Unless the meaning says otherwise these are `VALIDATION_ERROR`, exit 2, raised before any Python starts or by an explicit worker precheck.', '',
              '| Tool code | Meaning | Where |', '|---|---|---|']
    for token, (kind, meaning) in sorted(CODES.items()):
        if kind == 'tool_code':
            lines.append('| `%s` | %s | %s |' % (token, meaning, ', '.join(sorted(found[token]))))
    lines += ['', '## Caps (compile-time constants checked before spawn)', '',
              'Every cap is enforced in C++ before a child process starts; the worker repeats the data-level checks. Measured worst cases per cap are in docs/timings.md.', '',
              '| Constant | Value | Meaning | File |', '|---|---:|---|---|']
    for name, value, file in caps:
        lines.append('| `%s` | `%s` | %s | %s |' % (name, value, CAPS[name], file))
    lines += ['', '## Operations', '', '| Operation | Access | Parameters |', '|---|---|---:|']
    for o in schema['operations']:
        lines.append('| `%s` | %s | %d |' % (o['name'], o['access'], len(o['params'])))
    return '\n'.join(lines) + '\n'


def main():
    found, caps = scan_codes(), scan_caps()
    issues = problems(found, caps)
    if issues:
        for issue in issues:
            print('ERROR:', issue, file=sys.stderr)
        sys.exit(1)
    text = render(found, caps)
    if '--check' in sys.argv:
        current = (ROOT / 'docs/reference.md').read_text() if (ROOT / 'docs/reference.md').exists() else ''
        if current != text:
            print('ERROR: docs/reference.md is stale: regenerate with `python3 scripts/generate-reference.py > docs/reference.md`', file=sys.stderr)
            sys.exit(1)
        print('docs/reference.md matches the code: %d codes, %d caps, %d operations' % (len(CODES), len(caps), len(json.loads((ROOT / 'cli.schema.dev.json').read_text())['operations'])))
        return
    sys.stdout.write(text)


if __name__ == '__main__':
    main()
