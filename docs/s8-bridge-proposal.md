# S8 table / numeric bridge proposal

S7 deliberately does not implement Argon F4. S8 should add these explicit operations:

* `frame_to_array` (read): `frame` table ref, ordered unique numeric `columns`, `missing=error`
  (default) or `nan`, optional `rows`/`limit`; return an inline rank-2 row-major F64
  matrix only if <=1000 values and <=64KiB. Accept int64/float64/bool (bool maps to 0/1); reject datetime/string/category. Preserve row and column order, report
  selected feature names and conversion/NA counts; refuse strings/categories and
  int64 outside exact binary64 range unless an explicit lossy flag is provided.
* `frame_to_array_to` (write): same conversion to exclusive `out.safetensors`, key
  `key` default `x`; canonical `artifacts.x` descriptor. No pickle/object arrays. `missing=nan` explicitly loses the distinction between masked NA and numeric NaN; report both source counts and `missing_mask_lost:true`. The default error policy avoids that loss.
* `array_to_frame` (write): `ref` inline/NPY/safetensors input, explicit unique nonempty
  column names matching rank-2 width (rank-1 treated as one column), optional
  explicit whitelisted dtypes, exclusive `out.npz`. No implicit index, joins or
  inference from a numeric file's filename. `nan_policy=missing` converts NaN to masked missing; the default `nan_policy=value` retains numeric NaN.

Reject zero selected rows or columns as BAD_DATA in all bridge forms, matching the nonempty S1 tensor contract. Bool overrides require values exactly 0/1; integer overrides require finite exact integers in range.

Use existing validators before spawning: <=200000 rows, <=256 columns, <=500000
cells and <=256MiB table file; bridge output additionally <=200000 numeric values
(the shared S1 input cap) and <=32MiB numeric file. Validate NPZ member sizes/names,
NPY/safetensors headers, exact integer conversion, bool and missing masks. Readonly
allows the inline read form only; write twins honor dryrun, conflict and atomic
install. Every file result follows `artifacts[name]` with the common seven fields.
Budget row/column scans, preview <=1000 cells /24KiB, metadata within64KiB.

Tests: every dtype, nullable masks/NaN/Inf, ordering, integer precision boundaries,
empty table/array decisions, hostile headers, over-cap preflight, readonly/dryrun,
and frame_import -> frame_to_array_to -> numerical op -> array_to_frame recipe.
No implicit conversion inside numerical/ML ops is proposed.

Argon F7: `stats_dist` `rvs` with a required explicit seed is **S8 scope**. Add it
then with a closed distribution enum, PCG64 generator/seed echo, <=1000 inline
values or an explicit `_to` twin, pre-spawn sample/work caps, reproducibility tests
and independent distribution-moment checks. No sampling operation is added in S7.
