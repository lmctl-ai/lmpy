# Native pycli and rcli: one static executable, no Python or R at run time (design, Opus)

Status: design only. I changed no repos and made no commits.
- Inputs: the op lists from `pycli/cli.schema.dev.json` (68 ops, 15 of them `_to` twins) and `rcli/cli.schema.dev.json` (28 ops); the existing native code in lmda, lmstats, lmsim and physim; and the licensing audit (`LICENSE-AUDIT-Opus.md`).
- Line numbers below were checked against HEAD on 2026-10-19.

## 0. Decision in one paragraph

- **What:** build a new Rust workspace **`lmnum`** with one engine crate and one multicall binary. Symlinked as `pycli` it serves pycli's schema; as `rcli` it serves rcli's.
- **Reuse:** lmstats (`statsrust`) for every `sm_*` op and the time-series ops. lmda's library crate for tests, clustering, loess and descriptive statistics. physim-core for ODE, sparse and LP helpers. lmsim only where it fits.
- **Gaps:** filled with permissive crates, plus three statically linked C/C++ libraries: **HiGHS** (MIT), **Qhull** (Qhull licence) and **Boost.Math** (BSL-1.0, behind a small C ABI shim).
- **Static linking:** target musl (`x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`) with mimalloc. The result has no glibc LGPL §6 duty (see the audit), no Python, no R and no Docker.
- **Isolation:** the same binary re-executes itself as a sandboxed worker child (rlimits, seccomp, and a hard deadline in the parent). That keeps today's crash isolation and the 50 s kill.
- **Oracles:** the current Docker images are frozen as **oracle images** and drive parity tests. They are never used at run time.
- **What stays fixed:** the public surface — op names, params, result shapes, statuses and artifact conventions.
- **What changes:** numerical results may differ inside documented tolerance tiers. A short, explicit list of R-identical stream and bit-exact promises becomes "tier T4/T5" or is dropped (§5).

---

## 1. Op-by-op mapping

**Risk legend:**
- **E (exact):** a standard algorithm with a unique answer to rounding (closed forms, direct solves, FFT, sorting, graph shortest paths on integers).
- **S (standard):** a well-defined iterative or numerical method; results agree within tolerance (special functions, quadrature, ODE at a given rtol, IRLS).
- **I (implementation-defined):** the answer depends on library defaults or optimizer paths (sklearn defaults, statsmodels/R ARIMA optimizers, LAPACK eigenvector signs, R's RNG streams, QHull simplex order, random forests).

**Licence and static-link notes, stated once:**
- faer, nalgebra, statrs, rustfft/realfft, petgraph, argmin, linfa, kiddo, safetensors and zip are MIT/Apache-2.0 pure Rust. Static linking is trivial.
- HiGHS (MIT, C++) builds with CMake under the `cc`/`cmake` crate. Static linking is fine.
- Qhull (C, Qhull licence: permissive, requires the notice) is static-link OK.
- Boost.Math is header-only (BSL-1.0). I would wrap it in a ~300-line `extern "C"` shim, which is static-link OK.
- SuperLU (BSD-3, C) is optional, static-link OK.
- Anything GPL, LGPL-only or NR is excluded (§4).

### 1a. pycli (68 ops)

`_to` twins share the row of their base op. They differ only in the artifact writer (§2.4).

| pycli op | Closest native code (reuse) | Gap filler, licence | Risk |
|---|---|---|---|
| `py_info` | — (new `native_info`: versions, build hash, crate list, CPU features) | — | E |
| `array_import`, `array_inspect` | pycli C++ readers to port (`src/safetensors.cpp`, npy grammar in `src/main.cpp`) | `safetensors` crate (Apache-2.0) or our own reader; `npyz` (MIT) for npy | E |
| `la_solve(_to)` | lmstats `src/linalg.rs` (nalgebra); lmda uses faer | **faer** LU/Cholesky/LDLT (MIT) | E (S if ill-conditioned) |
| `la_lstsq(_to)` | lmstats `linalg.rs` lstsq | faer SVD lstsq (gelsd-like); `driver` gelsy/gelss map to QR with pivoting or SVD | E/S (I for rank decisions near the threshold) |
| `la_props` | — | faer (det, slogdet, rank, cond, norms, inverse) | E |
| `la_decomp(_to)` | — | faer svd/qr/eigh/eig/cholesky/lu | E for values; **I for eigenvector/singular-vector signs and eig order**. Already documented as "compare invariants"; keep the documented sort |
| `la_matrix_fn(_to)` | — | expm: Padé scaling-and-squaring (Higham 2005, paper). logm/sqrtm: Schur-based (Higham; faer has complex Schur). inv/pinv: faer | S |
| `la_sparse(_to)` | physim-core `solve_conjugate_gradient` (lib.rs:1774), `solve_gmres` (:1927), `solve_lanczos_sparse_spd` (:2422) | `sprs` (MIT/Apache) for CSR. **eigs/eigsh/svds:** implicitly restarted Lanczos/Arnoldi written from Lehoucq & Sorensen (paper; ARPACK-ng is BSD and may be consulted) or the `faer` sparse module. **spsolve:** faer sparse LU/Cholesky (MIT), with a fill estimate before factorising (fixes S8b-1 finding 1) | S (I: ARPACK `which=SM` convergence paths) |
| `special_fn(_to)` | lmda/lmstats `dist.rs` have a few functions (but lmstats' NR code must be rewritten first; see §4) | **Boost.Math** via shim (erf/erfinv, gamma_p/q and inverses, ibeta and inverse, zeta, polygamma, lambert_w, ellint, hypergeometric_2F1, cyl_bessel_j/y/i/k, sph_bessel, airy, owens_t). Struve: port Cephes (BSD via scipy). logsumexp, xlogy, boxcox: trivial | S (1e-12…1e-14 relative versus scipy) |
| `stats_dist(_to)` (34 families, pdf…isf, moments, rvs) | lmda `dist_fitdist` family code; statrs | Boost.Math distributions (norm, t, chi2, F, gamma, beta, nct, ncf, ncchi2, weibull, pareto, cauchy, laplace, logistic, gumbel, rayleigh, skew_normal, lognormal, poisson, binomial, nbinom, geometric). genextreme/genpareto/johnsonsu/truncnorm/vonmises/betabinom/zipf: closed forms plus Boost primitives. levy_stable: Nolan's integral (paper), as a numeric quadrature. **rvs:** `rand` with PCG64 (`rand_pcg`, MIT/Apache), using numpy's algorithms (BSD, may be consulted) for the samplers | S. rvs: **I** (see §5: bit-identical numpy streams are a stretch goal) |
| `interp_eval(_to)` | lmda `isotonic_regression`, no splines | Write our own: CubicSpline (all bc), PCHIP (Fritsch–Carlson), Akima/makima, the Reinsch smoothing spline (paper), RBF via faer solve. All textbook | E/S (smoothing spline GCV: S) |
| `sig_filter(_to)` | — | Butterworth/Chebyshev/elliptic/Bessel prototype → bilinear → zpk → SOS (Parks & Burrus; scipy BSD may be consulted for pole pairing). sosfilt/sosfiltfilt: our own (Gustafsson edge padding as in scipy) | S (I for SOS pairing order; compare the response, not coefficients) |
| `sig_spectrum(_to)` | lmda `spectrum` (stats.rs:2822, rustfft) | realfft (MIT/Apache); Welch/CSD/coherence code; windows table | E/S |
| `sig_peaks(_to)` | — | Our own find_peaks / prominences / widths, with a bounded wlen (fixes S7 finding 2) | E |
| `spatial_knn(_to)` | — | **kiddo** (MIT/Apache, kd-tree) or a hand-written kd-tree; brute force under a threshold | E (ties: I, already documented) |
| `spatial_hull(_to)` | physim-core `convex_hull_graham_scan` (2-D only, :4976) | **Qhull** C, static, fixed option table | E for volumes; I for simplex order (already documented) |
| `spatial_dist(_to)` | lmda distance code (multi_mds / cluster) | Our own cdist (metric enum) | E |
| `frame_import/inspect/export/select/groupby/resample/mutate/merge/rolling`, `frame_to_array(_to)`, `array_to_frame` | lmda `src/table.rs`, `io.rs`, `table_*` ops (filter/select/sort/join/groupby/pivot), `series_rolling`, `table_convert` | NPZ table layout reader/writer (stored zip; `zip` crate MIT, stored mode only, plus our existing preflight rules); typed columns with masks (lmda's `Column` plus a mask); tz handling with **`jiff`** (MIT/Apache, bundles tzdata) for `frame_resample`; the expression engine shared with `opt_*` | E (resample DST rules: S) |
| `opt_linprog(_to)` | physim-core `solve_simplex` (:4346), too small | **HiGHS** static (MIT): the same solver scipy uses, so it matches scipy's `method=highs` | E for the objective; I for degenerate x (already documented) |
| `opt_minimize` | physim-core `minimize_unconstrained` (:4784); lmstats `optimize.rs` (BFGS/L-BFGS with strong Wolfe) | **argmin** (Nelder-Mead, BFGS, L-BFGS) plus our own L-BFGS-B projection. **SLSQP:** port Kraft's 1988 algorithm from the paper / scipy's BSD Fortran-derived source (consultable). **trust-constr:** defer | I (optimizer paths; compare the optimum and the status, not iterates) |
| `opt_least_squares` | lmstats `optimize.rs` | Levenberg–Marquardt (`levenberg-marquardt` crate, MIT) plus our own TRF/dogbox from Branch–Coleman–Li (paper; scipy BSD consultable); robust losses are formulas | I/S |
| `num_quad` | — | **QUADPACK** QAGS/QAGI/QAGP: the algorithms and the original Fortran are public domain, so a direct port is allowed | S (matches scipy and R closely, since both use QUADPACK) |
| `num_root` | — | Brent, bisection, Ridder, TOMS748 (Alefeld–Potra–Shi paper) | E/S |
| `opt_root` | — | MINPACK hybrd/lmder: MINPACK licence (BSD-like with attribution); port with notice; brentq | S |
| `ode_solve(_to)` | **physim-core** `integrate_adaptive` (lib.rs:1079, RK45 with SciPy dense coefficients), `integrate_implicit` (:3175) | DOP853 (Hairer; the original code is BSD-like; consultable), Radau IIA (Hairer & Wanner paper), BDF (Shampine–Reichelt NDF, paper), LSODA (ODEPACK, public domain). Event location reuses the dense output | S (step sequences are I; compare at tolerance, §3) |
| `graph_paths(_to)`, `graph_flow(_to)` | lmda `graph_summary`/`graph_centrality` (petgraph) | **petgraph** (Dijkstra, Bellman-Ford, BFS); Yen's k-shortest (paper); Dinic max-flow; min-cost flow via network simplex (Kelly–O'Neill / Ahuja; networkx BSD consultable) or HiGHS LP | E for lengths and values; I for tie paths (already documented) |
| `ml_metrics` | lmda `eval_classify`, `eval_roc` (`roc_metrics` stats.rs:9423) | — | E |
| `ml_fit`/`ml_predict`/`ml_cv`: linear, ridge, lasso, elastic_net | lmda `model_regress` (linreg-core); linfa | Our own coordinate descent (Friedman 2010; sklearn BSD consultable) | E (linear, ridge) / S (CD) |
| … logistic, linear_svc | lmda `model_logit` | L-BFGS on the L2-penalised logistic loss (convex, unique optimum); a liblinear-style dual CD for SVC (liblinear BSD-3, consultable) | S at the optimum (I for `n_iter`) |
| … gaussian_nb, lda, knn, standard_scaler, pca | lmda `multi_lda`, `pca` (:34467), kNN code; linfa | — | E (pca signs: I, already handled) |
| … kmeans | lmda `kmeans_with_restarts` (:35163, linfa) | k-means++ seeding and Lloyd. **Differs from sklearn's RNG stream** | I → tier T4 (§3) |
| … random_forest_*, hist_gradient_boosting_* | — (smartcore RF, Apache-2.0) | **Defer.** Fully implementation-defined | I |
| `sm_ols(_to)` | **lmstats** `regression.rs` (+ HC0–HC3, cluster) | design-matrix builder from the JSON term list (port pycli `py/sm.py::sm_design`) | E |
| `sm_glm(_to)` | **lmstats** `glm_model.rs`, `glm_family.rs` | — | S (IRLS 1e-8) |
| `sm_arima(_to)` | **lmstats** `statespace.rs`, `estimators.rs` (SARIMAX) | — | I (optimizer); lmstats already targets statsmodels parity |
| `sm_tsa_acf`, `sm_tsa_tools` | **lmstats** `stattools.rs`, `diagnostics.rs` (acf/pacf/adfuller/kpss/ljung-box) | — | E (acf), S (adf/kpss p-value tables) |

### 1b. rcli (28 ops)

| rcli op | Closest native (reuse) | Gap filler, licence | Risk |
|---|---|---|---|
| `r_info` | — (`native_info` in rcli mode) | — | E |
| `dist_quantile`, `dist_density`, `dist_probability` | statrs; lmda dist code | Boost.Math. **Do not read R nmath.** Use the published algorithms Boost already implements (Wichura AS241, DiDonato–Morris, Cody) | S (≤ 1e-12 relative; R's documented tail accuracy) |
| `rng_draw`, `rng_sample`, `rng_arima_sim` | lmsim generators | Today's contract is **R-identical streams** (Mersenne-Twister + Inversion + Rejection sampling). MT19937 itself is public (the Matsumoto–Nishimura paper, BSD reference code), but R's seed scrambling and `unif_rand` details live in GPL `RNG.c`. Choose: (a) a clean-room black-box reproduction with a two-person rule (§4), or (b) change the contract to a native PCG64 stream | **I**, highest risk (§5) |
| `math_special` | — | Boost.Math (gamma, lgamma, beta, digamma, trigamma, choose, Bessel) | S |
| `summary_quantile` (types 1–9) | lmda `describe` (stats.rs:595) | Hyndman & Fan 1996 definitions (paper) | E |
| `num_fft` | lmda `spectrum` (rustfft) | realfft/rustfft | E (≤ ulps) |
| `test_htest` (t, wilcox, ks, fisher, shapiro, chisq, binom, prop, cor, var, …) | **lmda** `test_*` ops: `fisher_exact` (stats.rs:20704), `wilcoxon_*` (:22009–22200), `mann_whitney_u` (:22260), `shapiro_wilk_swilk` (:22968; scipy BSD port, notice needed), ks, chisq, binomtest, correlation | The exact Wilcoxon/KS distributions from published recurrences | E/S. Note: R's exact/continuity defaults have to be matched behaviourally |
| `fit_lm` | lmstats `regression.rs`; lmda `model_regress` | the design builder | E |
| `fit_glm` | lmstats `glm_model.rs` | — | S |
| `fit_nls` | lmstats `optimize.rs` | Gauss–Newton with step halving (Bates & Watts book; **R's nls source is GPL, do not read**) | I/S |
| `smooth_spline` | — | A Reinsch/penalised B-spline with GCV. R's `smooth.spline` is implementation-specific (knot placement, `spar` scaling, Fortran `sbart`, GPL), so match **behaviour and documentation only** | **I** (tier T3/T4: compare fitted values within 1e-4 of R, df within 0.05) |
| `smooth_loess` | **lmda** `loess` (stats.rs:38235) | Check how lmda's loess was validated. R's loess is Cleveland's dloess (netlib, public domain), so porting the netlib Fortran is allowed | S |
| `interp_spline`, `interp_approx` | (shared with `interp_eval`) | fmm/natural/periodic splines (Forsythe–Malcolm–Moler book), approx | E |
| `ts_arima` | lmstats `statespace.rs`/`estimators.rs` | R's `arima` CSS-ML optimizer path differs from statsmodels. The repo's own `oracle-findings.md` measured 1.1e-7 on AR(1) | I (T3: 1e-3 on coefficients, as today's recipes already use) |
| `ts_acf`, `ts_pacf` | lmstats `stattools.rs`; lmda `series_acf/pacf` | — | E |
| `ts_stl` | **lmstats** `stl.rs` (port of the statsmodels STL, derived from netlib `stl2.f`; R's `stl` uses the same Fortran) | — | S/E |
| `cluster_kmeans` | lmda `kmeans_with_restarts` | **Hartigan–Wong AS 136** from the published paper (Applied Statistics 1979; not R's C code) | I (restarts and RNG) → T4 |
| `cluster_hclust` | lmda `cluster_hclust` (kodama crate, MIT) | Lance–Williams updates (paper) | E for heights; I for merge order with ties |
| `opt_minimize` (rcli) | argmin and lmstats `optimize.rs` | R's `optim` Nelder-Mead/BFGS/CG/L-BFGS-B/SANN come from Nash's book and Byrd et al. (papers). Don't read R's `optim.c`. SANN needs R's RNG → T4 | I |
| `opt_optimize`, `num_root` (rcli) | (shared) | Brent's `fmin` and `zeroin` (Brent 1973 book, published Algol) | S/E |
| `num_integrate` | (shared QUADPACK) | QUADPACK (public domain): **the same algorithm R uses** | S (near bit-identical) |

**Coverage summary:**
- Of the 53 pycli base ops (68 minus 15 twins), about 26 map onto existing native code: 9 frame ops plus the bridge to lmda tables, 5 `sm_*` ops to lmstats, and ml_metrics, ODE, CG/GMRES, graphs and parts of ml.
- About 27 need new code built from permissive crates or papers.
- About 18 of the 28 rcli ops are covered by lmda or lmstats.
- **True gaps** (new numerical code of medium or high risk): SLSQP, L-BFGS-B bounds, TRF least squares, ARPACK-style eigs, matrix logm/sqrtm, IIR design, RBF, levy_stable, R-identical RNG, smooth.spline, RF/HGB.

---

## 2. Architecture

### 2.1 Language: **Rust** (C++ only behind small C-ABI shims)

**Arguments:**
1. **Reuse.** The native assets are Rust: lmda (~97 k lines in `src/`), lmstats (~53 k), physim-core and lmsim. The lmapi SDK `lmrust` is already used by all four.
2. **Ecosystem.** faer, petgraph, argmin and realfft cover most gaps without C.
3. **Static linking.** musl static builds are routine in Rust. C++ would need us to re-wrap all of that Rust.

**What C++ would give:** Boost.Math, Eigen and Spectra (ARPACK-style). We take Boost.Math anyway, as a header-only shim compiled by `cc` with `-static-libstdc++`. libstdc++ is covered by the GCC Runtime Library Exception; on musl we use `libc++` or libstdc++ static.

### 2.2 Reuse strategy: **crate dependencies at pinned git tags, not vendoring or merging**

**lmstats.** Depend on `statsrust` as a library (`[lib]` exists; modules in `src/lib.rs`). Licence BSD-3 per the audit: we must ship its notice, which is fine.

**lmda.** Depend on `lmda` as a library (it has `src/lib.rs` with `pub mod stats/table/io`).
- `stats.rs` is a 96 k-line single module, so compile time is a concern.
- Step 1: put the CLI and clap behind a feature flag (`default-features = false`).
- Step 2: when stable, split out an **`lmda-core`** crate (stats + table) inside the lmda repo. Do this in lmda itself, not by vendoring.

**physim.** Depend on `physim-core` for integrators and sparse solvers.

**lmsim.** Use it only for the generators (`gen_*`); probably not needed in phase 1.

**Before any of this reuse ships publicly**, these licensing blockers must be fixed in the upstream repos:
- lmda: the GPL-derived Tukey code (`stats.rs:10437`, used by `tukey_hsd`).
- lmstats: the Numerical Recipes routines (`dist.rs:64–190, 340, 362`) and the `weighted_moving_average.rs` and `erlang_a.rs`. lmnum must not link those modules: feature-gate them out in statsrust.

Why not merge everything into one repo: the four projects have their own tool surfaces, users and release cycles. A dependency keeps one source of truth. Vendoring would fork 150 k lines.

### 2.3 One binary or two: **one multicall binary, two tool identities**

**Layout:**
- `lmnum` is a multicall binary, busybox-style; `argv[0]` selects the tool.
- `pycli` → pycli schema and ops; `rcli` → rcli schema and ops.
- `lmnum pycli …` and `lmnum rcli …` are explicit forms.

**Why:**
- Both op sets share about 60 % of the numerical core: dist, special functions, FFT, quadrature, roots, optimizers, splines, GLM, ARIMA, acf, kmeans and the expression parser (identical in both today).
- One binary means one release asset, one notice file and one sandbox.
- **Two schemas** keep `tool: "pycli"` and `tool: "rcli"` and their existing op names and params. Recipes that `import "./cli.schema.dev.json" as pycli` don't change.

**The two personalities keep their own semantics where they differ** (e.g. rcli quantile type 7 by default and R's htest result shape; pycli scipy-style results). This happens through thin per-tool adapters over the shared core: `crates/lmnum-py/` and `crates/lmnum-r/`.

The rcli binary keeps spelling `r_info`, but reports `engine: "lmnum <ver>"` and `r_compat: "<oracle R version>"`. It no longer reports a live R.

**Workspace layout:**
```
lmnum/
  Cargo.toml (workspace; MIT; THIRD_PARTY_NOTICES.md generated by cargo-about + manual C/C++ entries)
  crates/lmnum-core/    numerics shared by both: la, special, dist, quad, roots, optim, ode, interp, signal, spatial, graph, ml
  crates/lmnum-io/      npy/safetensors/npz-table/csv readers+writers, path confinement, artifact descriptor
  crates/lmnum-expr/    the expression parser (port of src/expr_parser.hpp + expr-spec.json) and postfix evaluator
  crates/lmnum-sandbox/ worker spawn, rlimits, seccomp, deadline
  crates/lmnum-py/      pycli op adapters (schema-identical to pycli/cli.schema.dev.json)
  crates/lmnum-r/       rcli op adapters (schema-identical to rcli/cli.schema.dev.json)
  crates/sys-highs/ sys-qhull/ sys-boostmath/   static C/C++ (cmake/cc), each with its LICENSE
  bin/lmnum (main.rs: argv[0] dispatch via lmrust)
  parity/   (§3) corpora, oracle runner, tolerance specs
```

### 2.4 Native IO (artifacts keep today's formats exactly)

| Format | Plan |
|---|---|
| safetensors | Our own reader/writer (port `pycli/src/safetensors.cpp` checks: header ≤ 1 MiB, offsets/coverage, dtype table) or the `safetensors` crate behind our caps |
| npy | Closed header grammar (port `parse_npy_header`); reject object, Fortran, big-endian |
| npz tables | Stored-only zip, member names restricted to `[A-Za-z0-9_.]`, `__pycli_table__` metadata (port `src/frames.hpp` preflight and `py/frames.py` layout). Strings stay fixed-width `S<w>` for compatibility, **with byte budgets before allocation** (fixes S8a finding 1 and S8b-2 finding 1) |
| CSV | lmda's `csv` crate path, plus pycli's strict number grammar and the cell budget |
| Artifact install | Our own `O_TMPFILE` + `linkat` no-replace (port from `safetensors.cpp:write_bundle`); `openat2(RESOLVE_BENEATH)` for confinement where available, otherwise the canonicalise-and-check approach |
| Descriptors | The same seven-field `artifacts[name]` objects |

### 2.5 Sandbox, caps and time limits without Docker

The current promises (reviews S0–S8) were: no caller code, caps checked before compute, crash isolation, no network, a 50 s hard kill, and a cooperative `TIME_LIMIT`.

| Promise | Native mechanism |
|---|---|
| No caller code | Unchanged: closed op tables and the expression AST evaluator. There is no interpreter at all, so the surface is smaller than today |
| Caps checked first | All existing C++ pre-spawn validators are ported 1:1 into `lmnum-py`/`lmnum-r` `prepare_*`. Byte budgets (strings, NPZ widths) are added as found in the reviews |
| Crash isolation | The parent validates, then `fork`+`exec`s `/proc/self/exe --worker <req.json> <resp.json>`. A worker panic, abort or SIGSEGV becomes `INTERNAL_ERROR/TOOL_CRASH`, exactly as today |
| Memory cap | In the worker: `setrlimit(RLIMIT_AS, 4 GiB)` (works for native code; it didn't for OpenBLAS + Python). Allocation failure → the worker aborts → TOOL_CRASH. Better: wrap mimalloc to return an error, which maps to `RESOURCE_LIMIT` |
| CPU and time | Worker: `RLIMIT_CPU` = limit + 5 s. Parent: wall deadline 50 s, then SIGKILL → `TIMEOUT` (exit 7). Cooperative `time_limit` checks are added in every iterative loop (optimizers, ODE steps, ARPACK restarts, simplex pivots via HiGHS's `time_limit`, LU via a fill estimate before the call), giving `TIME_LIMIT` with partial results |
| No network or exec | Worker installs a **seccomp-bpf allowlist** after opening its files: read, write, mmap, brk, futex, exit, clock_gettime and a few more. No socket, connect, execve or open-for-write outside the pre-opened fds. Linux only; on macOS rely on the rlimits plus "no code" |
| Threads | Single-threaded by default (deterministic). faer parallelism off (`Par::Seq`) |
| Shim | `bin/pycli` and `bin/rcli` become optional: a native binary on PATH works directly. The Docker shim stays for the **oracle** images only |

Native mode still gives the reproducibility promise "bit-exact within one build and CPU family", provided there is no FMA-dependent SIMD. Pin `-C target-cpu=generic` for release builds and record it in `native_info`.

---

## 3. Oracle and parity strategy

### 3.1 The oracle side
- The current images `pycli:runtime` and `rcli:runtime` are frozen at the last Python- and R-based release tag (`oracle-pycli-<tag>`, `oracle-rcli-<tag>`). Their schemas are byte-identical to the native ones by construction, since the native binary publishes the same schema JSON, enforced by a diff test.
- The oracle runner is the existing shim (`bin/pycli` and `bin/rcli`, renamed `bin/pycli-oracle` and `bin/rcli-oracle`) with `--pull=never`.

### 3.2 Corpora (all requests, no new formats)
1. **Unit corpus.** Every request that today's tests send. They are already in `tests/s*_ops.py` (pycli) and `tests/*` (rcli). Extract them into `parity/requests/<op>/*.json` (thousands of requests), including the hostile and cap cases, which must give the same *status/tool_code*.
2. **Recipe corpus.** All ~100 `examples/*.lms` (pycli and rcli, including `tut-*` and `oracle-*`). Run each twice, with PATH pointing at the native binary and then at the oracle shim. Compare exit codes and the "Checks passed" lines. The cross-tool recipes (rcli vs lmstats, pycli vs lmda) keep running as they are.
3. **Fuzz corpus.** Seeded random valid requests per op (shapes and parameters inside the caps), generated by a Rust generator from the schema. This gives broad differential coverage.

### 3.3 Differ and tolerance tiers
`parity/diff.rs` walks two result JSONs and tolerance-checks the referenced artifacts (safetensors/npz arrays). Per-op specs live in `parity/tolerance/<op>.toml`, with a tier per result field.

| Tier | Rule | Typical ops and fields |
|---|---|---|
| T0 exact | Equal bytes and values (status, tool_code, integer results, shapes, sort orders, names, codec) | statuses everywhere; frame_* (non-float columns); graph lengths on integers; ml_metrics counts; summary_quantile; quantile types |
| T1 1e-13 rel (abs floor 1e-300) | special functions, d/p/q, FFT, dense solves (cond < 1e8), acf | special_fn, stats_dist (non-rvs), dist_*, math_special, num_fft, la_solve, la_props |
| T2 1e-9 rel, or invariant | Factorisations compared by invariant (|uᵀu'| = 1, A = QR), lstsq, OLS, spectra, quadrature (both QUADPACK) | la_decomp, la_lstsq, sm_ols, fit_lm, sig_spectrum, num_quad, num_integrate |
| T3 1e-6 rel on estimates, 1e-4 on SEs, optimum-based | Optimum compared at the objective (f_native ≤ f_oracle + 1e-8 · max(1, \|f\|)) and the constraints; status equal | opt_*, fit_nls, sm_glm/fit_glm (1e-8), ode_solve (compare at t_eval, within 10 × rtol × scale), interp smoothing, smooth_loess |
| T4 statistical | Distributional equivalence, not streams. rvs: KS p > 1e-3 and moments within 5σ over 1e5 draws. kmeans/RF/kNN ties: ARI = 1 on separated fixtures, metric within the CV standard error | stats_dist rvs, rng_* (if the contract changes), cluster_kmeans, ml_cv on stochastic families |
| T5 behavioural | The documented invariant only (e.g. sign-insensitive vectors, simplex sets equal as sets, hull volume) | spatial_hull simplices, eig vectors, Voronoi |

**Rules:**
- A tier may only be loosened with a written justification in the tolerance file. Same discipline as `docs/oracle-findings.md`.
- Every divergence beyond tier is logged with inputs, both values and a diagnosis.

### 3.4 CI
- Parity jobs run nightly: they need the oracle images and Docker, about 40 min.
- Per PR, run the fast subset: the unit corpus for changed ops, plus all recipes.
- The native binary's own tests (`cargo test`, with no Docker) gate every PR.

---

## 4. Clean-room rules

1. **R and CRAN (GPL-2/GPL-3).**
   - Do not read, copy or translate R or CRAN source: C, Fortran or R code, including `nmath/*.c`, `stats/src/*`, R function bodies such as `optim.c`, `nls.R`, `smooth.spline`/`sbart.c`, `arima.c` and `RNG.c`.
   - Allowed: the R *documentation* (help pages, read to learn behaviour; don't copy text), the papers R cites, and **black-box observation** (running the oracle image and fitting behaviour to outputs).
   - Record in each module header: "implemented from <paper/book>; validated against R <ver> outputs".
   - For the high-risk behaviour-matching modules (R RNG seeding, `smooth.spline`'s `spar`/knots, `optim` SANN, Hartigan–Wong restarts) use a **two-person rule**. The person who studies R's behaviour writes a behavioural spec from documentation and outputs only; a different implementer writes the code from that spec.
2. **scipy, statsmodels, scikit-learn, numpy, networkx (BSD-3).**
   - They may be consulted and ported, as long as the BSD notice is kept. Port files get an SPDX header and an entry in `THIRD_PARTY_NOTICES.md` with the upstream copyright.
   - **Their vendored parts carry their own licences.** Check them before porting:
     - scipy: QUADPACK and MINPACK (public domain / MINPACK licence), ARPACK (BSD), SuperLU (BSD), Qhull (Qhull licence), HiGHS (MIT), Cephes (BSD-style), specfun (check before porting), Boost (BSL).
     - sklearn: liblinear/libsvm (BSD-3).
     - statsmodels: some code from other projects. Check the file headers.
3. **Never:**
   - Numerical Recipes (the lmstats `dist.rs` routines must be replaced first).
   - any proprietary source (the lmstats WMA/Erlang-A modules are excluded until their provenance is re-verified).
   - TOMS/ACM algorithms under the ACM software licence. ACM licences restrict commercial use (e.g. TOMS 708 `bratio`), so take incomplete beta from Boost.Math instead.
   - The lmda Tukey code until it is rewritten.
4. **Public-domain or permissive Fortran** (netlib QUADPACK, ODEPACK/LSODA, MINPACK, dloess, `stl2.f`) may be ported directly. Credit them in the notices.

---

## 5. Contract changes (unavoidable; decide explicitly)

- **rcli `rng_draw` / `rng_sample` / `rng_arima_sim` / SANN: R-identical streams.**
  - Recommendation: keep the op names and change the stream to **PCG64 seeded per call**. Mark it `generator: "PCG64"` and document the distributional (T4) guarantee.
  - Recipes that assert R's exact draws (e.g. `rng_draw norm seed 42 → 1.3709584471466685…`, rcli README) move to the **parity** suite, where they run against the R oracle only.
  - The alternative, a clean-room black-box reproduction of R's MT + Inversion + Rejection, is feasible but legally delicate. Make it a later, optional slice under the two-person rule.
- **pycli `stats_dist` rvs:** today numpy PCG64. Native PCG64 gives the same uniform stream; for normal, gamma and similar, numpy's samplers (ziggurat etc., BSD) can be ported, so bit-identical streams are possible. That is stretch goal T0, default T4.
- **`ml_fit` model files (pycli model v1):** native models are written with `pycli_model=2` and `engine=lmnum` metadata. v1 sklearn files are accepted for the families whose parameters are semantically identical (linear, ridge, lasso, enet, logistic, nb, lda, knn, pca, scaler, kmeans centres), with the existing validation. `sklearn_version` skew rules become `engine` rules.
- **Results that become "within tier" instead of bit-identical to scipy/statsmodels/R:** documented per op in a regenerated `docs/parity.md`.

---

## 6. Roadmap (rough sizes; one experienced engineer-week ≈ 1 "w")

| Phase | Content | Size |
|---|---|---|
| **P0 foundation (first slice)** | workspace, lmrust multicall dispatch publishing **byte-identical schemas** for pycli and rcli; `lmnum-io` (npy, safetensors, npz tables, CSV, confinement, atomic install, descriptors); `lmnum-sandbox` (worker, rlimits, seccomp, deadline); parity harness with oracle runner and differ; musl static release builds for both architectures with notices. **Ops:** `py_info`, `r_info`, `array_import`, `array_inspect`, `la_solve(_to)`, `la_lstsq(_to)`, `la_props`, `la_decomp`, `num_fft`, `summary_quantile`, `num_quad`/`num_integrate` (QUADPACK), `num_root` (both). All E/S risk | 5–6 w |
| P1 distributions and special functions | sys-boostmath shim; `stats_dist(_to)` (all 34 families, non-rvs), `special_fn(_to)`, `dist_*`, `math_special`; rvs with PCG64 (T4) | 4 w |
| P2 statsmodels and R models through lmstats/lmda | statsrust dependency (after the NR fix and feature-gating the WMA/Erlang-A modules); `sm_ols`, `sm_glm`, `sm_arima`, `sm_tsa_acf`, `sm_tsa_tools`, `fit_lm`, `fit_glm`, `ts_arima`, `ts_acf`, `ts_pacf`, `ts_stl`; the JSON term-list design builder | 4 w |
| P3 frames | NPZ table engine with masks; `frame_import/inspect/export/select/groupby/resample/mutate/merge/rolling`, bridge ops; `jiff` time zones; byte budgets | 5 w |
| P4 expressions and optimisation | `lmnum-expr` (shared rcli/pycli spec + corpus); `opt_minimize` (NM/BFGS/L-BFGS(-B)/Powell; SLSQP), `opt_least_squares` (LM, TRF), `opt_root` (hybr/lm/brentq), rcli `opt_minimize`/`opt_optimize`/`fit_nls`; `opt_linprog` (HiGHS), `ode_solve` (physim RK45 plus DOP853/Radau/BDF/LSODA) | 7 w |
| P5 signal, interpolation, spatial, graph, sparse | `sig_*`, `interp_*`, `smooth_loess`, `interp_spline/approx`, `spatial_*` (kiddo, Qhull), `graph_*` (petgraph + flows), `la_sparse` (CG/GMRES from physim, sparse LU with fill estimate, Lanczos/Arnoldi), `la_matrix_fn` | 7 w |
| P6 ML subset and R-specific | `ml_metrics`, `ml_fit/predict/cv` for linear/ridge/lasso/enet/logistic/linear_svc/nb/lda/knn/pca/scaler/kmeans; `test_htest` (lmda), `cluster_kmeans` (Hartigan–Wong from the paper), `cluster_hclust`; `smooth_spline` (behavioural, T3/T4) | 6 w |
| P7 optional | R-identical RNG (two-person clean room); RF/HGB (smartcore or our own, T4); trust-constr; levy_stable at full accuracy | 4–8 w |

Total: about 38–45 engineer-weeks for parity of all ops except P7, plus licensing fixes in lmda and lmstats (about 2 w, upstream).

**Deferred or dropped:**
- patsy and formula strings: never.
- Parquet: stays deferred.
- The sklearn model zoo beyond the families above: RF/HGB go to P7.
- `trust-constr`, levy_stable at better than 1e-6, R's exact `smooth.spline` `spar` semantics: best effort.
- Windows: later; seccomp is Linux only.

**First slice to build (P0, in order):**
1. `crates/lmnum-io` with fuzzed npy/safetensors readers. Port the existing pycli tests `tests/s1_arrays.py` cases into a Rust test corpus.
2. Multicall `lmnum` publishing `pycli` and `rcli` schemas. CI diffs them against `pycli/cli.schema.dev.json` and `rcli/cli.schema.dev.json` (initially only the P0 ops are implemented; the others return `UNSUPPORTED` behind a `--native-partial` build flag, so the schema stays complete).
3. `lmnum-sandbox` worker with the crash, timeout and rlimit tests (the same cases as `tests/s0.py` / `tests/shim_test.py`).
4. `la_*` with faer, and QUADPACK `num_quad`/`num_integrate`.
5. The parity harness against `pycli:runtime`/`rcli:runtime` for those ops, with T1/T2 tolerance files.
6. A musl static release for aarch64 and amd64, checked with `file` and `ldd` ("statically linked").

**Gate for P0:**
- All P0 requests from the unit corpus pass parity at their tier.
- `examples/solve.lms`, `lstsq-fit.lms`, `tut-integrate-root.lms` (pycli) and `tut-integrate.lms` (rcli) pass natively.
- The binary is under 25 MB, runs with no Docker, Python or R installed, and calls start in under 20 ms (versus 0.4–1.4 s today).

---

## 7. Risks

1. **R-identical behaviour** (RNG streams, `smooth.spline`, `optim`, `arima`) cannot be reproduced without reading GPL source unless we accept tiers T3/T4. The contract change in §5 is the real decision.
2. **Upstream licence debt.** lmda's Tukey code and lmstats' Numerical Recipes routines and WMA/Erlang-A modules must be fixed before lmnum can link them publicly.
3. **Implementation-defined ML/optimizer outputs** will differ from sklearn and scipy. Tolerance by optimum and status, not iterates, has to be accepted by recipe authors.
4. **Compile times and size.** The 96 k-line `stats.rs` and Boost.Math: split `lmda-core` and use LTO. Estimated release size is 15–25 MB, which needs measuring.
5. **Seccomp portability** (aarch64 syscall numbers, musl): test both architectures in CI.
