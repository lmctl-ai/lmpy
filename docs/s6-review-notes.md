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

Compatibility: new strict KNN classifier v1 files persist classes_. S6 prototype KNN models missing that tensor fail BAD_MODEL and need refitting. Other family attribute tables and wheel locks are unchanged. Canonical numeric string class labels now return integers; prefix them when textual identity is required. No new dependencies or relaxed resource caps.

Acceptance and measurements are recorded below after final builds.
