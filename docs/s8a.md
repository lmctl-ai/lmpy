# S8a: table/array bridge, frame_mutate / frame_merge / frame_rolling, ml_metrics, seeded sampling

README has the parameters, statuses and conventions; docs/s8-bridge-proposal.md is the binding contract for the bridge. This note records the design decisions and
the evidence behind them.

## Shape of the slice
* `src/s8.hpp` validates in C++ and registers 9 operations (`frame_to_array`, `frame_to_array_to`, `array_to_frame`, `frame_mutate`, `frame_merge`, `frame_rolling`, `ml_metrics`,
  and `stats_dist` with fn rvs plus its write twin `stats_dist_to`); `py/s8.py` (assembled after `s7.py`) is the worker. `stats_dist` keeps its norm/t density path unchanged.
* **No pandas anywhere.** Tables are read and written member by member in the NPZ layout of `py/frames.py` (`c<i>` data, `m<i>` NA mask, `__pycli_table__` metadata), so a
  call pays only the numpy import, and a column's dtype, categories, timezone and NA mask survive untouched (a left join keeps an int64 column int64 with masked cells,
  where pandas would promote it to float). `meta.pandas`, `meta.sklearn` and `meta.scipy` (except `frame_mutate`, which needs `scipy.special` for the expression functions)
  stay `unloaded`; tests assert it. The first numpy NPZ call lazily imports modules the import guard blocks, so the worker warms that path in the trusted import phase.
* Every table is preflighted in C++ with the existing bounded stored-only NPZ validator before it is copied into the private temp directory; outputs are validated again
  and installed atomically (exclusive create, `O_TMPFILE`), tensors go through the S1 writer and report the one seven-field `artifacts[name]` descriptor.
* Write twins and table writers honour `PYCLI_READONLY`, `PYCLI_DRYRUN`, the exclusive-create rule and the signal-cleanup list (new staging names added).

## Caps (C++ before spawn; the worker repeats the ones a corrupted request could break)
| Quantity | Cap | Where |
|---|---|---|
| table | 200000 rows, 256 columns, 500000 cells, 256 MiB | every table input and output (mutate/merge/rolling check the *result* size before spawn or before building) |
| bridge array | 200000 values (shared S1 cap); inline 1000 | `frame_to_array(_to)`, `array_to_frame` |
| `assign` | 32 entries, 2048 bytes / 512 nodes per expression | `frame_mutate` |
| join keys | 1..8 columns; result rows <= `max_rows` <= 200000 | `frame_merge`, from key counts before anything is built |
| rolling | window <= 100000 rows; rows x window x columns x functions <= 1e8 | before spawn (count windows), in the worker for time windows |
| metrics | rows <= 200000 (array loader), <= 30 distinct labels | `ml_metrics` |
| sampling | `n` <= 1000 inline, <= 1,000,000 in a file; `seed` in [0, 2^63) required | `stats_dist`, `stats_dist_to` |

## Honest statuses
* Missing values are never dropped or imputed silently: `frame_to_array` fails with `NA_NOT_ALLOWED` unless `missing=nan`, which is reported (`missing_mask_lost`, both counts).
  `frame_mutate` propagates masks and decides NaN/Inf explicitly (`nonfinite`). `ml_metrics` refuses missing/NaN input.
* int64 never rounds silently (`INT64_NOT_EXACT`); float64 keys are refused for joins; an empty join is `EMPTY_RESULT`, not an empty table; a `pos_label` that does not occur is
  `BAD_PARAM`; undefined metrics are listed with their reason under status `UNDEFINED` while the defined ones are still returned.
* The row-explosion cap is checked from the key counts: the 5000 x 5000 same-key join (25,000,000 rows) is refused in 0.4 s with 0.17 GB peak.

## Oracles (independent of the code under test)
* A numpy-only table reader (tests/oracle/s8.py) decodes the NPZ members itself; nothing reads results through pandas or pycli.
* Joins: a pure-Python nested-loop join over the documented order, for inner/left/right/outer with duplicate keys on both sides, NA keys on both sides, composite keys,
  string/category keys, suffix and naming rules, and every `validate` mode.
* Rolling: pure-Python windows with `math.fsum` for sum/mean/variance/std, sorted-list median, min and max, for count and centred windows with several `min_periods`, missing
  values skipped, int64/bool sources and time windows over duplicated timestamps.
* Metrics: confusion counts and every average by hand, ROC AUC by brute-force pair counting (ties 1/2), log loss and Brier by their formulas (with the 1e-15 clip), R^2/MSE/MAE/MAPE
  by formula, every undefined case.
* Sampling: bit-for-bit reproducibility from the seed (prefix property), sample moments against closed forms with 5-sigma tolerances (100000 draws per family), and
  hand-coded Kolmogorov-Smirnov statistics for norm, uniform, expon and lognorm that reject a deliberately wrong scale (so the test has power).
* Cross-tool: `oracle-metrics.lms` and `oracle-frame.lms` (lmda `eval_classify`, `eval_roc`, `table_join`, `series_rolling`), `oracle-rvs.lms` (R quantile functions through rcli);
  results in `oracle-findings.md`.
* Pipelines as recipes: `csv-join-report.lms`, `bridge-ml.lms` (frame_import -> frame_mutate -> frame_to_array_to -> la_lstsq -> array_to_frame), `sensor-pipeline.lms`
  (frame_resample -> frame_to_array_to -> sig_filter_to -> sig_peaks), `ode-fit.lms` (ode_solve + seeded noise -> opt_least_squares).
