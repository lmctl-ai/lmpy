# S0 protocol decisions

The missing-interpreter envelope remains SERVICE_UNAVAILABLE / PYTHON_NOT_FOUND,
exit 8, matching rcli and established lmcpp consumers. Switching to 127 would be
a fleet compatibility change, outside this repository-only repair.

Validation errors (exit 2) are C++ schema/domain errors or explicit worker
BadData prechecks. Numerical library ValueError, LinAlgError outside the known
solve status path, and OverflowError are PY_ERROR / COMMAND_FAILED (exit 1).
Oversized requested output uses the private OutputTooLarge exception, not every
OverflowError. Invalid JSON numeric overflow (1e999) is PARSE_ERROR, exit 3.

Read solves use domain statuses SINGULAR and NOT_POSITIVE_DEFINITE (exit 0).
Write solves require a real artifact and fail with the corresponding tool_code
(exit 1) if there is none. Dryrun's explicit simulated envelope is the exception:
it creates no artifact by design. lmcpp returns before handlers, so custom path/
conflict checks in a simulated write are deferred until real execution; fixing
that requires an upstream validation hook rather than duplicating lmcpp parsing.

Python -P -s -B plus the replaced environment preserves cwd/script/user-site/
PYTHONPATH isolation while enabling PYTHONHASHSEED=0. This intentional change
supersedes decisions.md's -I spelling pursuant to the Lead's S0-fix instruction.

Independent distribution references expose actual SciPy accuracy: df=1 Student t
(Cauchy) cdf agrees with 0.5+atan(x)/pi to 5e-16; its .25/.75 inverse differs from
tan(pi*(p-.5)) by about 1.34e-11, so that reference alone uses absolute 2e-11.
Normal pdf/cdf at zero, specified normal quantiles, symmetry and the df=1e300
normal limit have separate tighter references. Tiny positive df is mathematically
valid; no arbitrary floor is imposed. SciPy NaN from valid input is preserved
with NumericalWarning, distinguishing it from deliberate NaN/missing input.

## S1 decisions

Import and inspect run entirely in C++ (no Python process), so they have no oracle in the
sibling tools; their oracles are numpy and math.hypot. Cross-tool overlap: `la_lstsq` against
rcli/lmstats regression design-matrix solutions is a future recipe; the NIST/rational tests
are the certification for now. Dense LA statuses are domain results (exit 0) in read ops and
`COMMAND_FAILED` with the status as `tool_code` in write ops, as for S0 `la_solve_to`.
Missing (`null`/`NA`/empty CSV field) and NaN are distinguished at import only for counting;
both are stored as NaN. No arithmetic is done on I64 beyond exact conversion (|v| <= 2^53).
