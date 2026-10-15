# S6 review fixes

Reviewed baseline: `a9a1156`; reports: REVIEW-S6-Opus, REVIEW-S6-Fable and REVIEW-S6-Astra in the sibling cppwrapper checkout. No finding declined or deferred.

| Finding | Fix and regression evidence |
|---|---|
| Astra 1, major: KNN single-bit label corruption | Strict classifier `classes_` tensor plus training-code set equality before spawn; fresh estimator class-order check; independent probability-width guard before artifact writing. Binary single-bit and missing-middle multiclass reproducers use a nonexistent interpreter to prove preflight rejection as BAD_MODEL. ASan/UBSan also exercises 1000 valid/corrupted KNN models. This is consistency validation, not authentication of arbitrary finite coefficient edits. |
| Astra 2, major: prediction row count mistaken for training count | Fit-only neighbor/cluster/component restrictions; loaded hyperparameters validated against saved training dimensions. Partitioned and one-row queries agree with full batches for both KNN families, PCA, kmeans and explicit multi-component LDA. Query memory/work/output/pair caps remain enforced. |
| Opus 1 / Fable 4: misleading CV failures | Missing-class log-loss -> UNDEFINED_METRIC; single-class training fold -> INSUFFICIENT_CLASSES. Results name fold_index/metric (when applicable), missing_classes, completed_folds and a stratified/shuffle recommendation. Completed prior folds retained. Undefined ROC-AUC/R² have diagnostics. Explicit input checks still validate; numerical backend ValueError -> NUMERICAL_FAILURE. |
| Opus 2: unidentified linear coefficients | RANK_DEFICIENT with rank/rank_deficient, preserving the usable minimum-norm model; prediction checked against the hand solution. Fold statuses and aggregate CV preserve rank warnings too. |
| Opus 3: impossible clustering / fictitious RNG | DEGENERATE_CLUSTERS for too few distinct points/clusters, no model. Logistic fixed lbfgs is deterministic, seed optional, no generator echoed. MT19937 only when estimator/shuffled CV uses randomness. |
| Fable 1: placeholder schema | All ML inputs described; family keys/types/defaults/ranges/enums generated from ml-spec.json. cli.schema.dev.json regenerated and schema gate checks it. |
| Fable 2: objectives | docs/s6.md states ridge/lasso/elastic/logistic objectives, intercept/scale conventions, glmnet mapping, NB variance and PCA sign/variance conventions. Documentation contract test. |
| Fable 3: source-dependent class types | Source-independent canonical decimal integer labels, including numeric strings; leading-zero/+ identifiers remain strings. Homogeneous label types required; CSV/inline/table fit and prediction fixtures. README/schema describe policy. |
| Fable 5: tiny parameters hidden | Eligible fitted tensors <=100 values echoed as parameters only within whole-response budgets; parameters_omitted flags omission. Training arrays/class codes never echoed. Linear oracle recipe uses coefficients directly, avoiding two worker calls. Tiny ridge and wide scaler fixtures. |
| Fable 6: KNN ties | Document sorted-label vote tie and pinned brute training-order kth-distance rule; hand-computed probability, classification and neighbor-boundary fixtures. |
| Fable 7: missing oracles | Independent NumPy lasso/elastic-net KKT and PCA max-absolute-positive SVD assertions, plus class-less-fold fixtures. These mathematical oracles pass the original implementation; they close coverage gaps rather than change numerical code. |
| Astra supplemental constant LDA diagnostic | DEGENERATE_DATA before the SVD for zero within-class variation, replacing bounded but unhelpful IndexError/PY_ERROR; no artifact. |

The first red review suite against the original test-image binary exited 1: 14 methods, 11 assertion failures and 2 errors. The original independent KKT/PCA oracle passed. Further fixtures split out the LDA/documentation branches and retain completed folds. Behavioral regressions were installed before their implementation changes. Tests inspect the actual CLI and independently calculated tables/gradients; no estimator-vs-estimator numerical oracle.

The batch regression additionally exposed rank-reduced LDA with requested two axes but one actual SVD axis: the old persistence shape check rejected its own model. A separate red reproducer (exit 1) precedes the fix; effective axes are now persisted and flagged, with both one- and two-axis fixtures.

A further red fixture (exit 1) catches CV folds containing only the numeric-looking subset of otherwise textual class identifiers. The full dataset fixes the label policy once; folds never infer a different type.

Compatibility: new strict KNN classifier v1 files persist classes_. S6 prototype KNN models missing that tensor fail BAD_MODEL and need refitting. Other family attribute tables and wheel locks are unchanged. Canonical numeric string class labels now return integers; prefix them when textual identity is required. No new dependencies or relaxed resource caps.

## Final acceptance (aarch64)

* `./scripts/build.sh test`: **0**; `./scripts/build.sh runtime`: **0**.
* `docker run --rm --pull=never --network=none pycli:test`: **0**, **27/27**, **1117.65 s**. Includes original S6 **19 methods (124.17 s)** and review **19 methods (179.77 s)**; schema, notices and sanitizer gates green.
* With `PATH=$PWD/bin:/home/ubuntu/repos/lmctlhq/rcli/bin:$PATH`, every `lmscript examples/x.lms`, `lmscript --check examples/x.lms` and `lmscript --dry-run examples/x.lms`: **0**, all **39 invocations /13 recipes**.
* Exact Astra one-bit mutation through final hardened host shim: fit **0**, both predict kinds **2 / BAD_MODEL**, neither artifact exists. Valid one-row KNN prediction **0**, value **2**; mixed-text-label CV **0**, accuracy **1/3**. Probe harness **0**.
* Real cross-tool recipes: OLS intercept/slope **[1,2]** in pycli and rcli; kmeans ARI **1** vs hand blobs and rcli; classification CV accuracy **0.975**, log_loss **0.20075185634341353**. Optional lmstats/lmda legs still skip with setup hints because no prebuilt binary is available.
* Sequential timing harness **0**; runtime **634 MB**, `inspect .Size` **136467787** (+9087 bytes from review start). Fit/predict/CV p50 **1.970 /1.954 /2.072 s**; import p50 **1.188 /1.188 /1.272 s**. See [timings.md](timings.md#s6-review-fixes-aarch64-2026-10-15) and raw JSON; the fit shim p95 is **3.049 s**, not hidden as a worker-import regression.
* Runtime `05976918f958` and test `02e865bf02c3` contain identical binary SHA256 `6e75fcabcb74d634035700e881d0f74e3eb5e124e07f0bf8ed19ab297c1ec246`. No amd64 execution claimed.

An initial build exited 1 because the new documentation contract fixture needed docs admitted by .dockerignore; corrected before final acceptance. An earlier full run was deliberately stopped after 19 passing targets (exit 137) to include the additional fold-type regression; the complete run above supersedes it. Expected red-test failures are described above, not counted as acceptance failures.

Temporary host build trees/bytecode and retired pycli images were removed; project-labelled dangling builder cache was pruned after all builds and measurements. Final test/runtime tags retained. Local commits only; no push.
