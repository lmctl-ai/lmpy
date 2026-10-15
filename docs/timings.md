# S0 review-fix measurements — aarch64, 2026-10-14

Runtime: **446 MB on disk**; test: **667 MB on disk**. Docker containerd reports
a content metric of **97,450,776 bytes** for runtime (`image inspect .Size`).
Historical S0 runtime was 528 MB on disk / 115,730,671 content bytes.
The reduction is about 82 MB on disk and 18.28 MB in the content metric, despite
the added complete upstream/source and C++ package notices.

The runtime now starts from a fresh pinned base and copies a pruned scientific
tree. The unpruned wheel-install layer is absent from runtime RootFS.Layers
(verified independently against the test-image ancestry). Hidden pip bytes
already in the shared base remain; whiteouts are not counted as reclaimed bytes.

| Measurement | p50 ms | p95 ms | Samples |
|---|---:|---:|---:|
| Python -P -s -B boot | 17.352 | 59.530 | 20 |
| NumPy import including boot | 200.036 | 389.229 | 20 |
| scipy.stats import including boot | 1229.418 | 1261.999 | 20 |
| Docker run /bin/true overhead | 390.527 | 443.209 | 10 |
| py_info through hardened shim | 860.823 | 951.167 | 10 |
| 3×3 la_solve through shim | 1347.639 | 1523.489 | 10 |
| norm.ppf(0.975) through shim | 1926.958 | 2033.247 | 10 |

Warm-filesystem wall-clock measurements on a shared four-core host, with network
disabled and --pull=never. Measurement phases ran sequentially after builds.
Import rows include interpreter startup; they are not incremental imports.
Production uses -P -s -B -X faulthandler, scrubbed env, PYTHONHASHSEED=0 and
one OpenBLAS thread. The calibration uses -P -s -B, hash seed 0 and ARMV8.
The reported p95 uses sorted[min(n-1, floor(0.95*n))], a conservative upper
order statistic; for these small sample counts it is the maximum.

The new startup/total latencies are slower than historical S0 samples. These
runs differ in flags/hashing, tmpfs/init and filesystem/layer/cache layout; this
is a measured difference, not an attributed causal diagnosis or a speed claim.
The two-call recipe still does not meet the real >=20-call startup gate for
a persistent worker. The original base comparison, coretype probe and timings
remain in measurement-candidates.json, measurement-coretype.json and the other
historical measurement-*.json files. Those -I measurements do not describe the
current -P/-s worker. The base choice remains pinned Python 3.12 slim bookworm.

Reproduction: scripts/measure.py pycli:runtime, then scripts/measure-shim.py
pycli:runtime. Raw updated data: measurement-review-runtime.json,
measurement-review-shim.json, measurement-review-packaging.json.
Runtime image ID: `sha256:4c09d743cfe3e057d516752974675ad4a4911ae40dcb7bf4b83fcd15ca45f260`.

ARMV8 and one thread were verified in both bundled OpenBLAS pools. The original
400×400 direct-solve probe measured 5.529 ms automatic/neoversen1 vs 5.459 ms
ARMV8 median; it remains historical and is not a general performance guarantee.
amd64 wheels remain resolved/hash-locked only; no amd64 execution is claimed.

The builder/test now use the signed immutable 20261012T000000Z Debian snapshot
and exact top-level package versions. Full compiler/package inventories and
.deb SHA256 hashes are retained in runtime documentation. Source-license
archives were downloaded and hash-verified during a build only.

Acceptance (all commands unpiped): both image builds exit 0; CTest 9/9 exit 0;
host shim/API regression suite 5 tests exit 0; real-container suite 4 tests exit
0; runtime layer/prune/license/provenance gates exit 0; schema comparison exit
0; recipe check, dry-run and real execution exit 0. Red reproductions and the
per-finding dispositions/deferred choices are in review-s0-fixes.md.

## S1 operations — aarch64, 2026-10-14

Same method as above (production shim, `--pull=never`, network off, 10 sequential calls,
p95 = sorted[min(n-1, floor(0.95*n))], i.e. the maximum at this sample size), runtime image
`sha256:83b3c9c0e1b5a180e6bddc927321a8642cdc4f171ae3ed5d5a6c671effd6e9ac`. Raw data:
`measurement-s1-shim.json`, produced by `python3 scripts/measure-shim.py pycli:runtime`.

| Operation (input) | p50 ms | p95 ms |
|---|---:|---:|
| py_info | 772 | 823 |
| la_solve 3x3 | 1205 | 1288 |
| stats_dist norm.ppf | 1705 | 1754 |
| `array_import` inline 2x2 (no Python) | 349 | 368 |
| `array_inspect` 100x100 .npy (no Python) | 344 | 361 |
| `la_lstsq` 200x5 by ref | 1237 | 1439 |
| `la_props` 100x100 (det, rank, cond, norm, trace) | 1236 | 1287 |
| `la_decomp` svd 100x100 -> safetensors | 1273 | 1378 |
| `la_decomp` eig 100x100 -> safetensors | 1208 | 1339 |

* Array import/inspect cost only the Docker round trip (about 0.35 s); every Python-backed op is
  about 0.4-0.5 s of NumPy/SciPy start-up plus a few ms of LAPACK at these sizes.
  Inside the container a 447x447 (the largest square by ref) svd/eig/qr/lu/eigh/cholesky
  call measured 0.3-0.6 s worker time (`meta.elapsed_ms` 311-593 ms).
* The earlier rows were measured again in this run and are 10-12 % faster than the S0
  review-fix table above (host load differs between runs); compare only within a run.
* Runtime image: **446 MB on disk, 97,430,471 content bytes** (S0 review-fix: 446 MB /
  97,450,776). No wheel, library or package was added; the 20 KB difference is the rebuilt
  executable and worker. The *test* image grew from 667 MB to 763 MB because it now contains
  the ASan/UBSan fuzz binaries; it is never shipped.
* Test-suite cost: `ctest` takes about 160 s in the test image (s1_la ~80 s, s1_arrays_asan ~20 s,
  safetensors_asan_fuzz ~27 s).

## S2 frames — aarch64, 2026-10-14

Runtime: **509 MB** reported by `docker image ls`, **110,691,742 content bytes**
from image inspect. Test: **973 MB**, **208,038,870 content bytes** (inspect refreshed
in measurement-s2-packaging.json if the content metric differs). Pandas and minimal
dependencies add about 63 MB to the prior 446 MB runtime. The unpruned scientific
install layer is absent from the runtime ancestry; pandas tests and optional
pyarrow/numexpr/bottleneck/openpyxl are absent.

| Operation | p50 ms | p95 ms |
|---|---:|---:|
| `frame_import` | 1429 | 1522 |
| `frame_inspect` | 1462 | 1625 |
| `frame_export` | 1506 | 3306 |
| `frame_select` | 1421 | 1813 |
| `frame_groupby` | 1459 | 1832 |
| `frame_resample` | 1347 | 1599 |
| `la_solve` | 1410 | 1563 |
| `docker_overhead` | 412 | 461 |

Ten sequential production-shim calls per operation, seven rows / three columns
(UTC timestamps, region, sales); import is inline, inspect requests five rows and
describe, select filters sales>=30, groupby computes region sum/mean, resample uses
1W, export writes CSV. la_solve uses a 1x1 identity. p95 is the maximum of ten
samples, following earlier slices. Docker overhead is a network-disabled /bin/true
container. Warm-filesystem shared-host measurements, performed after acceptance
and builds; no causal speed comparison with historical S0/S1 is implied.

Frame worker import medians are 636–661 ms (NumPy + pandas together, excluding
interpreter boot). The frame_export p95 includes one 3306 ms sample; it is retained,
not smoothed away. Numerical la_solve still reports pandas unloaded.

Runtime image: `sha256:9eb2c9df8a93a29589404cf10e8b020d73a3f56fb2005b7d8fc7c0e59c824ad3`. Raw samples: measurement-s2-shim.json and
measurement-s2-packaging.json. Reproduce with `python3 scripts/measure-frames.py`.

Final acceptance commands were unpiped: ./scripts/build.sh test (0),
docker run --rm --network=none pycli:test (16/16, 268.04 s, exit 0),
./scripts/build.sh runtime (0), all three recipes via bin/pycli (0), their
--check and --dry-run modes (0), real-container review 4/4 (0), shim/API review
5/5 (0), S2 shim readonly (6) and dryrun (0), both with no artifact.

## S2 review: resource and packaging measurements

The original S2 table above is historical; review-run samples are recorded
separately rather than overwriting them. Final runtime Docker ls reports 527 MB
(test 975 MB); inspect content is 115,126,863 bytes (test 208,263,484 bytes).
The otherwise identical no-stdlib-bytecode control reports 509 MB / 110,692,682
content bytes. Every layer except the last RUN matches. Restoring caches adds
4,434,181 content bytes (about 18 MB in the rounded Docker ls display). There are
686 checked-hash stdlib caches totalling 12,311,809 file bytes. The runtime probe
observed zero stdlib source compilations on a fresh pandas import. This verifies
cache reuse; historical shared-host wall times alone cannot prove a causal speedup.
Raw image IDs, layers and sizes: measurement-s2-review-packaging.json.

Cap probe ran sequentially with the production memory/CPU/pids/rootfs/capability/
network/tmpfs restrictions: 4 GiB, two CPUs, 128 pids, read-only root, no caps,
no-new-privileges, network none and 512MiB tmpfs. Five fresh CLI imports succeeded
(exit 0), with the harness included in the cgroup peak of 254,980,096 bytes
(243.17 MiB). Per-child RSS maxima are cumulative upper bounds, not per-case
container peaks. These are import-workload measurements, not a guarantee that
every aggregate combination finishes within the existing 50-second timeout.

| Cap workload | Shape | CSV bytes | Total seconds |
|---|---|---:|---:|
| Numeric cell cap | 100000 x 5 | 1000010 | 2.212 |
| Row cap | 200000 x 1 | 400002 | 1.334 |
| String cell cap | 10000 x 50 | 2500190 | 2.527 |
| 100000 category levels | 200000 x 1 | 1177782 | 1.202 |
| Near byte cap | 200000 x 1 | 16600002 | 1.641 |

Caps are deliberately conservative: CSV <=16MiB, <=200000 rows and <=500000
cells; decoded string width <=65536 UTF-8 bytes (quotes/escaping do not count).
Oversized CSV reads stop at the byte budget before allocation; row/cell/width
fixtures reject before spawning, proved with an unavailable interpreter.
Fixed-width output storage retains its independent 256MiB preallocation budget.
Raw cap probe: measurement-s2-review-caps.json. Reproduce inside the hardened
runtime with scripts/measure-frame-caps.py; the container cgroup high-water mark
includes the probe harness.

Final acceptance commands, unpiped: ./scripts/build.sh test (0),
docker run --rm --network=none pycli:test (17/17, 312.74 s, exit 0),
./scripts/build.sh runtime (0), all three real recipes through bin/pycli (0),
all three --check/--dry-run modes (0), runtime_bytecode.py in final runtime (0),
and cap probe (0). S2 contains 19 review regressions plus 17 original test groups;
ASan/UBSan parser fuzz and all S0/S1/schema/notices/packaging gates pass.
The corrected weekly recipe prints Sunday-ended bins March 8: 260/4 rows and
March 15: 690/6 rows; region sums remain 450/500.

Runtime image: sha256:c1c1fe01ba03b1d321a12f2c7ca3b907234f8d17a74f5c6c1e3dacf6f6f9839c.
Test image: sha256:d463f355b03d2513b218c8770718963ecbc22e1b80c7f6322e7cf453fbc7af3a.

## S2 review: sequential per-call samples under observed load

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

Same seven-row/three-column production-shim workload as the original S2 script,
ten sequential fresh calls per operation after all own builds/tests/cap probes
stopped. Every sample is retained; p95 is the maximum of ten. Numerical la_solve
still asserts pandas unloaded. Resample uses corrected pandas weekly defaults.
An idle-host rerun is deferred: unrelated work on the shared four-CPU host did
not stop. A 30-second waiting trace showed 16–41% idle (median 32%); immediately
before that, two snapshots showed 22–36% idle. These are loaded-host observations,
not an idle-host promise or a controlled causal comparison to the historical
~1.4-second S2 results. No other Docker container, own build or own test was
running at measurement start. vmstat telemetry covers the full sample run.

Frame NumPy+pandas import medians were 372.9–379.8 ms.
Raw samples: measurement-s2-review-shim.json. Host traces: measurement-s2-review-
host.txt, measurement-s2-review-host-wait.txt and measurement-s2-review-host-during.txt.
Reproduce: python3 scripts/measure-frames.py docs/measurement-s2-review-shim.json.
This preserves the historical measurement-s2-shim.json instead of overwriting it.

## S3 operations — aarch64, 2026-10-14

Production shim (`--pull=never`, network off, sequential, 10 calls each; p95 is the maximum at this sample size), runtime image
`sha256:89a5d023c9d319f4a4a6521d172709fde9610b2cea4e69aaf62005f9f636cf76` (`pycli:runtime`, 527 MB on disk, **115,206,817 content bytes**; S2 runtime was 527 MB / 115 MB: no wheel
or package was added, S3 is two Python files and parser code). Raw data: `measurement-s3-shim.json` from
`python3 scripts/measure-numerics.py pycli:runtime`. "import" is the worker's own boot + NumPy/SciPy import time, "worker" its
total, both from the response `meta`.

| Operation (input) | wall p50 ms | wall p95 ms | worker import p50 | worker total p50 |
|---|---:|---:|---:|---:|
| py_info (baseline) | 616 | 672 | 102 | 107 |
| opt_minimize Rosenbrock BFGS (117 objective calls) | 1022 | 1103 | 437 | 449 |
| opt_minimize SLSQP, equality constraint | 1033 | 1066 | 435 | 437 |
| opt_minimize, `sum((t*a-1)^2)` over 200000 rows (npy ref) | 1113 | 1143 | 442 | 543 |
| opt_least_squares, 100-row exponential decay, 3 parameters | 1032 | 1205 | 445 | 451 |
| num_quad exp(-x^2) over (-Inf, Inf) | 1020 | 1065 | 431 | 434 |
| num_root brentq cos(x)-x | 1010 | 1075 | 434 | 434 |

Import cost: `scipy.optimize` + `scipy.integrate` + `scipy.special` add about **330 ms** over the py_info baseline (worker import
~435 ms against ~102 ms; `scipy.stats` was ~1.2 s in the S0 measurements, `scipy.linalg` ops ~300 ms). The solver time itself is
negligible for these sizes (<=100 ms even with 200000 data rows). Every S3 call is therefore dominated by the fixed process
start; this does not trip the persistent-worker gate (still no 20-call sequences in the recipes). The host was not idle
(shared four-core machine); the numbers are sequential samples, not an idle-host claim. Test suite: `ctest` now has 21 tests
(~8 minutes in the container; `s3_ops` ~77 s, `expr_parser_asan_fuzz` ~19 s).

## S4 LP/MILP and ODE — aarch64, 2026-10-14

Final runtime **527 MB** in Docker ls / **115241534 content bytes**;
test **1.05 GB** / **222291947 content bytes**. No new wheel or package.
Runtime content is 34717 bytes above the historical S3 measurement; the rounded
527 MB display is unchanged. Final runtime image:
`sha256:a5d2b45586475842eed754999da11467ab37534b6e74c833b15208af34735b16`.
Test image: `sha256:bfc00d15a7e61af53017f83f8c94077d758776809c0a99e00f0c9ad1022c450d`.
Raw IDs, sizes and layer inventories: measurement-s4-packaging.json.

| Workload | Wall p50 ms | Wall p95 ms | Worker import p50 ms | Worker total p50 ms |
|---|---:|---:|---:|---:|
| LP | 1020.1 | 1060.2 | 442.2 | 446.3 |
| MILP | 995.8 | 1086.0 | 431.2 | 435.6 |
| ODE_decay_RK45 | 1002.3 | 1009.2 | 429.0 | 432.3 |
| ODE_Robertson_BDF | 1239.8 | 1296.6 | 427.0 | 676.5 |

Ten sequential fresh calls per workload through the production bin/pycli shim,
network off, --pull=never, two CPUs/4 GiB/128 pids, read-only root, dropped caps,
no-new-privileges, init and tmpfs. Own builds, acceptance suite, recipes and cap
probes had finished before measurement; no other Docker container was running.
The four-core host was shared and loaded, **not idle**. Start/end load averages:
[3.47412109375, 5.3212890625, 4.462890625] / [2.166015625, 4.71484375, 4.296875].
vmstat telemetry spans the sample run in measurement-s4-host-during.txt.
Warm-filesystem samples are retained without filtering; p95 is the maximum of
ten, consistent with earlier slices. No causal speed claim versus S3 is made.
Worker import and total come from each response's meta; these are medians of
those fields, not differences of wall-clock medians.

LP is the hand-solved Wyndor production example (three constraints/two variables,
maximize, optimum36). MILP is a four-item binary knapsack with capacity8, checked
by exhaustive enumeration in the tests. RK45 solves y'=-y, y(0)=1 to t1 at default
tolerances with 101 output samples. BDF solves Robertson to t40, rtol1e-10,
atol1e-13. Its primary ODEPACK reference and loose-source tolerance qualification
are in docs/s4.md; a close numerical result is not presented as certified digits.
Raw samples: measurement-s4-shim.json; reproduce with
`python3 scripts/measure-s4.py` (exit 0; all 40 calls exit 0, domain OK).

The separate hardened cap run exercised 200000 dense entries, 512 variables,
4096 rows and 32 ODE states with the largest permitted grid at that state count
(5926 points /195558 trajectory numbers). All four calls exit 0, domain OK,
responses <2 KiB; cgroup peak **103346176 bytes (98.56 MiB)** includes NumPy probe
arrays. Large vectors/trajectories are artifacts, including a 1564608-byte ODE
safetensors file. These simple hand-checked cases establish allocation at the
caps, not worst-case MILP complexity. Probe times were under concurrent acceptance
load and are not shim latency comparisons. Raw command/results:
measurement-s4-caps.json, details in docs/s4.md.

Final unpiped acceptance: test build (0), `docker run --rm --network=none pycli:test`
(23/23, 532.30 s, exit 0), runtime build (0), all seven recipes real/check/dry-run
(21 exits 0), cap probe (0), final-runtime schema/docs check (0). S4 has 16 test
groups; expression budgets include every RHS component, event and finite-difference
Jacobian call. HiGHS MIT inventory/notices and parser ASan/UBSan gates pass.
Temporary helper files/images, superseded S4 images and project-labeled disposable
builder cache were removed; final runtime/test retained. amd64 remains unverified.

## S4 review fixes — aarch64

Runtime remains **527 MB** in Docker ls, **115249079 content bytes**; test
**1.06 GB**, **222443444 content bytes**. No wheel or package added.
Runtime: `sha256:0ae50b946a833b13a4c9d5e694c936aebe004abf792d4917d2d992a7c6908abd`.
Test: `sha256:65b859060cb6f124d28c05f9f768d2383d2e8e9da67617b08a02e35379a2b7e4`.
Full size/layer inventory: measurement-s4-review-packaging.json.

The hardened cap harness now invokes the required `_to` write twins with explicit
exclusive paths. All four cases return domain OK / exit0, independently checked
objectives/final states; responses1774–1779 bytes for LP and1744 bytes for ODE.
Peak cgroup memory including the NumPy harness is **105209856 bytes (100.34 MiB)**.

| Cap case | Shape | Seconds | Expected result |
|---|---|---:|---|
| Matrix entries | 500 x 400 =200000 | 1.299 | objective -400 |
| Variables | 390 x 512 =199680 | 0.875 | objective -512 |
| Rows | 4096 x 48 =196608 | 1.061 | objective -48 |
| ODE states/output | 32 states /5926 points /195558 numbers | 0.751 | all final states exp(-1) |

Same wall as the production shim: network none,2 CPUs,4GiB,128pids,read-only root,
no capabilities,no-new-privileges,default seccomp,init and512MiB noexec tmpfs.
The original-data certificate is active at every LP cap. The ODE safetensors
artifact is1564608 bytes. These simple fixtures exercise allocation at the caps,
not worst-case MILP complexity. Probe times were under concurrent acceptance
load; no speed comparison to the historical samples is justified.
Raw command/results: measurement-s4-review-caps.json.

All seven recipes pass real/check/dry-run through bin/pycli (21 exits0); source
and commands were unpiped. Raw outputs: measurement-s4-review-recipes.json.
Production-shim readonly small LP/ODE both exit0; read overflow exits1 with
OUTPUT_TOO_LARGE and no file; write readonly exits6. An event artifact is I64
and its time agrees with log(2). Raw results: measurement-s4-review-policy.json.

### S4 review sequential shim samples

| Workload | Wall p50 ms | Wall p95 ms | Import p50 ms | Worker total p50 ms |
|---|---:|---:|---:|---:|
| LP | 1175.8 | 1238.7 | 441.2 | 446.3 |
| MILP | 1161.7 | 1219.2 | 444.0 | 449.0 |
| ODE_decay_RK45 | 1109.1 | 1185.6 | 443.1 | 446.4 |
| ODE_Robertson_BDF | 1383.6 | 1447.2 | 437.9 | 691.9 |

Same four workloads and production shim as the initial S4 table: Wyndor LP,
four-item binary knapsack,default RK45 decay and Robertson BDF to40 at rtol1e-10
/atol1e-13. Ten sequential fresh calls each after all own builds,tests,recipes,
cap probes and cleanup finished. Every call exits0,domain OK; timing harness
exits0. Warm filesystem,shared four-core host; no idle-host or causal speed
comparison is claimed. p95 is the maximum of10; all samples retained.
Start/end load averages: [6.3984375, 6.361328125, 5.70556640625] / [5.63671875, 6.1806640625, 5.67724609375].
Host telemetry:measurement-s4-review-host-during.txt. Raw samples and image:
measurement-s4-review-shim.json. Generated by scripts/measure-s4.py,copying the
new output to the review filename and restoring the historical raw file.

Final unpiped acceptance: ./scripts/build.sh test (0),
docker run --rm --network=none pycli:test (**24/24,592.68s,exit0**),
./scripts/build.sh runtime (0),all seven recipes real/check/dry-run (21 exits0),
cap probe (0),final-runtime schema/docs gate (0). New review suite12 groups and
original S4 suite16 groups pass; all prior regressions and sanitizer gates pass.
See review-s4-fixes.md for every finding and technical contract. Temporary probes,
superseded images and project-labeled disposable builder cache were removed;
final test/runtime retained. amd64 remains unverified.

## S5: statsmodels (aarch64, 2026-10-14)

Runtime image: 527 -> 573 MB on disk; `image inspect .Size` +9.74 MB content from statsmodels, patsy and packaging (pruned tree).
**Import cost**: the lazy sm import (statsmodels OLS/GLM/SARIMAX/stattools, patsy stub) takes ~1.02-1.05 s median in the worker
(p95 1.3 s) on a shared 4-core host, so no split worker is proposed (threshold 3 s). Non-sm ops pay nothing (py_info import 0.11 s).
End-to-end through the production shim, 10 fresh calls each, 40-row / 120-point inputs (`docs/measurement-s5-shim.json`, `scripts/measure-s5.py`):
py_info 0.74 s; sm_ols 1.92 s; sm_ols_to 1.76 s; sm_glm (Poisson) 1.80 s; sm_arima ARMA(1,1) 1.78 s; sm_tsa_acf 1.75 s (p50 wall).

Caps probe in the hardened container (2 CPU, 4 GiB, read-only; `docs/measurement-s5-caps.json`, `scripts/measure-s5-caps.py`), all exit 0:
OLS 200000 x 50 (10M design cells) 5.2 s (HC3 5.7 s); binomial GLM 7.5 s; Poisson `sm_glm_to` 7.9 s; ARIMA(2,0,1) n=2000 1.5 s;
seasonal (1,1,1)(1,1,1,12) 10.0 s; state-cap ARIMA(5,0,5)(2,0,2,12) 13.8 s -> `NO_CONVERGENCE` at maxiter 50 (a status, not an error).
Cgroup peak memory over the whole probe: 1.0 GB (OLS alone 0.50 GB, GLM 1.0 GB), far below the 4 GiB limit.

### S5 review fixes (2026-10-15)

Re-measured (`scripts/measure-s5.py`, p50 wall): sm_ols 1.74 s, sm_ols_to 1.80 s, sm_glm 1.77 s, sm_arima ARMA(1,1) 1.88 s (was 1.78 s: the
Nelder-Mead polish adds ~0.05 s), sm_tsa_acf 1.80 s; import still ~1.0 s. New ARIMA bound (observations x states^2 <= 2.5e7), measured in the
hardened container (4 GiB): states 55 / n 8264 and states 30 / n 20000 both complete with a cgroup peak of 2.59 GB.
Acceptance: `./scripts/build.sh test` 0, `docker run --rm --pull=never --network=none pycli:test` 25/25 exit 0 (819 s), all 10 examples exit 0.

## S6: scikit-learn (aarch64, 2026-10-15)

Runtime `48279277327e`: **634 MB** (`docker image ls`), versus S5 **573 MB**, approximately **+61 MB**. Docker's content-size field
`image inspect .Size` is 136458812 bytes versus 124998107, +11460705 bytes; this is a different accounting basis from the displayed on-disk size.
Both architecture locks resolve pinned binary wheels; amd64 has not been executed.

Ten fresh processes per case, sequential through the production host shim, after builds/CTest/recipes finished. OS filesystem caches are warm;
these are fresh-worker, not cold-disk, measurements. Raw data: `measurement-s6-shim.json`; generator: `scripts/measure-s6.py`.

| Operation / data | Shim p50 | Shim p95 | Worker import p50 |
|---|---:|---:|---:|
| py_info | 0.646 s | 0.684 s | 0.103 s |
| ml_fit logistic, 40 x 1 | 2.043 s | 2.236 s | 1.235 s |
| ml_predict predict_proba, 40 x 1 | 2.044 s | 2.139 s | 1.238 s |
| ml_cv logistic, 4 folds, 40 x 1 | 2.060 s | 2.321 s | 1.242 s |

The ML import p95 is at most 1.461 s in this sample, below the 3 s threshold: no split worker is proposed. Non-ML ops report sklearn unloaded.
The import includes pandas and all supported ML estimator modules, joblib and threadpoolctl; it never imports statsmodels/patsy for ML.

Resource probes (`measurement-s6-caps.json`, `scripts/measure-s6-caps.py`) used the hardened runtime: offline, init, two CPUs, 4 GiB memory,
128 pids, read-only root, no capabilities/new privileges, default seccomp, uid/gid 1000, 512 MiB noexec tmpfs. Probes were sequential and idle
apart from the measurement harness. Native single-thread compute budgets were explicitly **30 s** (default is 10 s); caps can need the longer
budget, especially forests. Every operation below exited **0**, status **OK**, with no OOM/TOOL_CRASH. Peak cgroup memory over the complete probe
was **200306688 bytes (191 MiB)**; per-case peaks in the JSON are cumulative, not independent RSS maxima.

| Probe | Time inside hardened container |
|---|---:|
| ridge, 20000 x 9 + target (200000 raw cells) | 2.774 s |
| random_forest_regressor, same data, 64 trees / depth 12 | 14.792 s |
| hist_gradient_boosting_regressor, same data, 100 iterations / depth 8 | 3.216 s |
| ridge CV, 20 folds, same data | 2.967 s |
| PCA, 3125 x 64 (200000 cells), 32 components | 2.711 s |
| kmeans, 20000 x 1, k=32 / n_init=1 / max_iter=300 (192M work bound) | 2.216 s |
| KNN prediction, 2000 training x 2500 query rows (5M pairs) | 1.997 s |

Conservative work bounds are not claims that every fit consumes every allowed iteration. Timers are cooperative at Python boundaries; the
container/process hard timeout is still the backstop. The forest cap result is not a claim that it completes within the default 10 s budget.

Acceptance on the final source: `./scripts/build.sh test` **0**; `docker run --rm --pull=never --network=none pycli:test` **0**, **26/26** targets
(958.16 s; S6's 19 unittest methods 128.28 s); `./scripts/build.sh runtime` **0**; all 13 recipes real/check/dry-run **0** (39 invocations).
Runtime and acceptance binaries have the identical SHA256 `b08750accf6d3e01eaba56000f6b77697799be425881bee4858fe892558039e7`.
The measurement harness initially assumed py_info had a status field (exit 1); that assertion was corrected and the complete timing run
restarted, exit 0. Earlier incomplete build/test runs were superseded by the final acceptance above.
