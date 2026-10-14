# S2 review corrections

All code/data findings in REVIEW-S2-Opus/Fable/Astra were addressed. An idle-host
timing rerun is deferred: the shared four-CPU host stayed busy with unrelated work
(16–41% idle, median 32%, across a 30-second waiting window). Sequential samples
under observed load are retained honestly; no idle-host performance claim is made.
The reviews are external to this checkout at ~/repos/cppwrapper/pycli/.

| Review finding | Correction / regression |
|---|---|
| Opus 1: CSV OOM/timeout | C++ bounds reads to 16MiB, rows to 200000, cells to 500000 before spawn. Missing-interpreter tests prove rejection; cap measurements are recorded in timings.md. |
| Opus 2 / Fable 4: inline bytes/categories | 32KiB worker summaries, 64KiB final response; 64-character string previews, at most 20 category levels, explicit truncation/metadata-in-artifact flags. Full categories remain in NPZ. Lossless inline export refuses oversized output. |
| Opus 3: quadratic categories | First-observed inferred levels via dict, set membership, fixed pandas categorical construction. 60000 unique-level timeout regression; measured 200000 rows / 100000 levels. |
| Opus 4: special datetime words | Dashed ISO regex precedes pandas for every datetime data and predicate value. now/today/yesterday/tomorrow/NaT/compact dates refused. |
| Opus 5: NUL ZIP differential | ASCII name whitelist and exact cN/mN/metadata member set in C++, before Python. NUL-shadow and punctuation fixtures use missing interpreter oracle. |
| Opus 6: own oversized string output | 65536-byte UTF-8 string limit, validated on CSV before spawn and before storage allocation. Invalid worker output is INTERNAL_ERROR/BAD_TABLE_OUTPUT. |
| Fable 1: wrapped int64 sums | Exact Python integer accumulation; INT64_OVERFLOW error and no artifact for positive/negative groupby and resample overflow. |
| Fable 2: ME defaults | pandas defaults: right/right for ME/W, left/left otherwise; explicit overrides retained. Hand-computed ME/MS/QS/YS/empty-bin/DST-day/week fixtures. |
| Fable 3: nonfinite CSV round trip | Export NaN/Inf/-Inf with documented casing, blank null/NaT; explicit float64/int64/datetime policies round-trip values and masks. |
| Fable 5: inference/override coercion | NA participates in nullable int inference; leading-zero and plus-prefixed numeric tokens infer string; decimal-only override grammar rejects underscores. |
| Fable 6: NaN/null reductions | Explicit documented policy: undefined reductions become missing; NaN/null keys merge. Hand-computed regression preserves that policy. |
| Fable 7: timezone warning | Copy tz-converted Series before mask assignment; warnings regression. |
| Fable 8: usability | Ignore blank CSV lines; reject string/category sums; integral float int predicates accepted, fractional literals rejected. |
| Fable 9: coverage | Exact hand spring-hourly/DST-day/week/calendar labels and values, dtype bool/category/datetime identity, nonfinite round trips and overflow regressions. Existing-correct coverage fixtures need no manufactured red numerical result. |
| Astra 1: stdlib bytecode | Force checked-hash stdlib caches in runtime's final RUN; runtime probe asserts no stdlib compilation on fresh pandas import and checked-hash headers. Controlled size delta and new sequential loaded-host shim samples in timings.md; idle-host rerun deferred as explained above. |

Red tests ran against the original images before implementation (exit 1); a
second category regression exposed a metadata-byte-count vs row-cap mistake,
fixed before final acceptance. Additional leading-zero decimal regression ran
red against the first corrected binary, then was fixed. An exact quoted/unquoted CSV width-boundary test also ran red, then C++ was
corrected to count decoded cell bytes. The original failing
build was explicitly stopped when its inputs became stale (exit 137); this is
not an acceptance result. Final command exit codes are recorded in timings.md.
