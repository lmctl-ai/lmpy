# S8a review fixes

Review baseline: 354b5be; Opus, Fable, Astra, Argon and Kimi. No ranked finding declined or deferred.

| Review findings | Resolution and regression evidence |
|---|---|
| Opus major 1 | Fixed-width join payload and masks projected before index construction, gathers and key unification; OUTPUT_TOO_LARGE above 256 MiB. `test_join_budget_precedes_gather` replaces the gather with a failure sentinel; baseline reaches it, fixed worker refuses first. Hardened 200000 x 65000-byte fan-out measured separately. |
| Astra 1, Argon F2.1, Kimi 1 | Shared post-load finite-score validation for all scored metrics, probability bounds for Brier/log-loss; array/table parity and NaN/Inf/out-of-range fixtures. |
| Astra 2, Argon F2.2 | Regression uses existing INT64_NOT_EXACT guard; classification-only large integer labels remain exact, with mixed-request regression. |
| Astra 3, Argon F2.5 | Extended intermediates for regression norms and means; genuine constancy checked directly. R2 remains 0 at scales 1e-200, 1, 1e200. Metrics outside binary64 range are UNDEFINED individually, preserving valid results. |
| Astra 4, Argon F1 | Direct timestamp comparisons and saturating Python-integer window subtraction; empty rolling produces a valid empty artifact; extended rolling intermediates avoid representable std/mean overflow. Infinite inputs/nonrepresentable outputs are NON_FINITE, never silent masked arithmetic failures. |
| Astra 5, Argon F2.3–4, Kimi 2 | Scalar pos_label checked before spawn; explicit scored pos_label must occur for any class count. Multiclass binary scores are UNDEFINED, without requiring a dummy positive label. Implicit label 1 remains deliberate for all-negative 0/1 Brier/log-loss. |
| Fable 1–2 | README/schema distinguish ifelse-only conditions from bare comparisons and inline JSON F64 from file integer/bool source policies. NaN conditions follow nonfinite explicitly. |
| Fable 3 | Count now honours min_periods window rows (pandas count semantics); other reducers require observed non-missing values. Hand oracle and leading/masked-window regression updated. |
| Fable 4 | Uniqueness validation counts duplicate key tuples including missing components, independently of missing-never-match join semantics. |
| Fable 5 | Human report uses two-decimal inspection totals while exported CSV retains S2's 17-digit contract; average schema links zero_division. |
| Argon F3 | Select/mutate/merge/rolling consistently return OK with valid zero-row artifacts; empty-side joins allocate typed all-masked absent columns. Extraction/metrics still require observations. EMPTY_RESULT removed from these contracts. |
| Argon F4 | Optional array_to_frame.frame appends columns strictly by row position; equal rows, combined names/cells/storage checked before spawn, collisions require allow_replace. Source order/dtypes/masks/timezones/categories preserved. bridge-ml enriches source rows with fitted values/residuals. |
| Argon F5 | NumPy-only previews use stdlib datetime/ZoneInfo with pinned tzdata, local offset and nanosecond precision. |
| Argon F6 | missing=error now guarantees finite bridge output, refusing numeric NaN/Inf as well as masks. missing=nan explicitly permits nonfinite round trips; nan_policy=missing flags nan_mask_changed. Mask-vs-value semantics, isnull and irreversible mask collapse documented and tested. |
| Argon F7 | Generated names validated before spawn; class count checked for mixed and table requests by bounded stored-member reads. All advertised S8a budgets reach the shared guard (including sampling twins and density path); expiration cannot install staged files. Density/sampling share status/dist/fn/params fields and BAD_PARAM for parameters. |
| Kimi 3 | Category/category keys retain deterministic union levels (left then unseen right); same levels preserved. Ordering retained only if both ordered with identical levels, otherwise ordered=false. Category/string outputs remain strings. |
| Kimi 4 (all five mutations) | PCG64 checked against explicit independent NumPy streams for continuous/discrete and artifact draws; nondefault Student t location/scale; exact min/h/D left-window boundaries; same/different category metadata; hand-computed balanced_accuracy=0.5 with predicted-only label. |

Test-first evidence: initial 19-test reproducer run had 15 failures plus one error (exit 1); separate preallocation/budget sentinel baseline run failed both cases; schema baseline failed; missing-count fixture failed; real expired-budget baseline wrongly wrote an artifact and failed. Already-correct behaviours in Kimi's mutation audit are pinned by independent oracle tests rather than introducing defects into production.

Compatibility changes are intentional: zero-row joins now succeed with artifacts, bridge missing=error is stricter, count leading windows can be missing, and uniqueness includes missing-key tuples. No wheels, image base or licences changed. amd64 execution remains unverified.


Final verification: test/runtime builds exit 0; 32/32 ctest groups exit 0 (505.16 s), including 25 S8 review and 46 existing S8 tests;
three additional ASan/UBSan preflight tests exit 0. All 24 recipes passed real/check/dry-run (72/72 exit 0).
The five Kimi scratch mutants were each killed by the selected new test; expected failures and diagnostics are in
`measurement-s8a-review-mutations.json`. The available R leg again agreed at 25 quantiles (worst 2.85 standard errors);
absent optional lmda/statsrust legs were skipped with their setup hints.
Production timing, memory, image size and command evidence are in `timings.md` and `acceptance-s8a-review.json`.
Runtime/test production binary hashes match. A superseded image build was deliberately stopped after the final cap-error regression was fixed;
the final build and acceptance commands all succeeded. No architecture claim beyond aarch64.

Cleanup complete: scratch builds and Python caches removed; superseded build container removed; project builder cache and unused dangling images pruned. Current runtime/test tags retained; unrelated stopped containers preserved.
