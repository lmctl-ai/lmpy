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


## S6 review fixes (aarch64, 2026-10-15)

Code revision `15d16b5`; runtime `05976918f958`: **634 MB** displayed, unchanged at Docker's rounded precision.
`image inspect .Size` is **136467787 bytes**, **+9087 bytes** versus the review-start image `36d7cf272679` (136458700).
Wheel locks/dependencies and resource caps are unchanged; amd64 was not executed.

Ten sequential fresh processes per case through the hardened production shim, after all builds, CTest, recipes and targeted probes completed;
warm filesystem caches. Raw results: [measurement-s6-review-shim.json](measurement-s6-review-shim.json), generated with
`python3 scripts/measure-s6.py` (exit **0**). With ten samples the reported p95 is the sample maximum.

| Operation / data | Shim p50 | Shim p95 | Worker import p50 |
|---|---:|---:|---:|
| py_info | 0.643 s | 0.662 s | 0.103 s |
| ml_fit logistic, 40 x 1 | 1.970 s | 3.049 s | 1.188 s |
| ml_predict predict_proba, 40 x 1 | 1.954 s | 2.028 s | 1.188 s |
| ml_cv logistic, 4 folds, 40 x 1 | 2.072 s | 2.130 s | 1.272 s |

The fit sample includes a **3.049 s shim maximum**, while its maximum worker elapsed time is **1.361 s**: the slow tail is outside worker
computation/import. ML import sample maxima are <= **1.333 s**, below the 3 s split-worker gate. No speedup claim is inferred from these
small samples. The cap/memory measurements above remain the original S6 measurements; this review run remeasured per-call latency.
The linear oracle recipe now reads its two fitted parameters directly, eliminating two extra array_inspect/shim calls.

Final acceptance: test build **0**; offline `docker run --rm --pull=never --network=none pycli:test` **0**, **27/27** targets (**1117.65 s**);
original S6 **19 methods /124.17 s**, S6 review **19 methods /179.77 s**; runtime build **0**. All **13 recipes x real/check/dry-run =39**
invocations exited **0**. Final runtime and acceptance binary SHA256:
`6e75fcabcb74d634035700e881d0f74e3eb5e124e07f0bf8ed19ab297c1ec246`.

## S7: interpolation, signal, spatial (aarch64, 2026-10-15)

Runtime image **634 MB** on disk (unchanged), `inspect .Size` **136,507,587** bytes (+39,800 over the S6 review figure 136,467,787: the worker source only;
no wheel or package was added, so there is no notices change).

**Import cost per op group** (worker import phase, p50 of 10 fresh calls through the production shim; `docs/measurement-s7-shim.json`, `scripts/measure-s7.py`).
No group approaches the 3 s split-worker threshold, so no split worker is proposed.

| Group | Imports | Worker import p50 | Call p50 wall | p95 wall |
|---|---|---|---|---|
| `py_info` (reference) | scipy | 0.10 s | 0.65 s | 0.67 s |
| `interp_*` | `scipy.interpolate` (+ spatial) | 0.42 s | 1.03-1.04 s | 1.14 s |
| `spatial_*` | `scipy.spatial`, `.distance` | 0.35-0.36 s | 0.94-0.96 s | 1.00 s |
| `sig_*` | `scipy.signal` (+ interpolate, spatial, stats) | 0.71 s | 1.36 s | 1.40 s |

(40-node PCHIP and RBF, 8th-order Butterworth + filtfilt on 400 samples, Welch on 400 samples, 400-sample peak search, k=5 neighbours of 60 points, Delaunay of 30
points, 20 x 20 distance matrix. Computation itself is below 10 ms in every case; the wall time is the interpreter, the scipy subpackage and docker.)

**Caps measured in the hardened container** (2 CPUs, 4 GiB, read-only root, `scripts/measure-s7-caps.py`, raw output `docs/measurement-s7-caps.txt`; cgroup peak
over the whole probe **0.39 GB**). Worst cases are cyclic polytopes (points on the moment curve), which maximise the facet and simplex counts:

| Case | Result |
|---|---|
| RBF 5000 nodes, 2-D and 8-D, thin plate and gaussian, 1000 evaluation points | 6.9-7.2 s, 0.28 GB; interpolation error 3e-14..7e-7 (gaussian eps 2, 2-D, the most ill-conditioned) |
| convex hull d=2/3, 20000 points (sphere and moment curve) | 0.7 s |
| convex hull moment curve d=4 n=800 / d=5 n=400 / d=6 n=100 | 1.6 s (318,800 facets) / 1.1 s (157,212) / 1.2 s (152,000) |
| Delaunay moment curve d=3 n=800 / d=4 n=300 / d=5 n=100 / d=6 n=50 | 1.7 s (318,003 simplices) / 0.8 s / 1.3 s (147,440) / 0.7 s |
| Voronoi moment curve d=3 n=800, d=5 n=100 | `OUTPUT_TOO_LARGE` after 12.2 s / 3.0 s (the ridge lists exceed the 1.9 M-number artifact budget; clean status, no crash) |
| Delaunay / Voronoi uniform 20000 points, 2-D | 0.7 s / 0.9 s |
| 60000 x 3 k=10 neighbours with exclude_self / radius search with up to 1 M pairs | 1.0 s / 2.3 s |
| 1000 x 1000 Mahalanobis matrix; Welch on 200000 samples (nperseg 65536); 20th-order elliptic bandpass filtfilt on 200000 samples; peaks and PCHIP on 200000 samples | 0.7-1.1 s each |

How the hull caps were chosen: with the first draft caps (convex hull d=4 n=5000, d=5 n=1000, d=6 n=300; Delaunay d=3 n=5000) the moment-curve inputs ran past the 50 s kill
(`TIMEOUT`) or died on the 2 M-number output limit. The caps now in force (README) keep every measured worst case under 13 s, and the worker fails cleanly with
`OUTPUT_TOO_LARGE` above 1.9 M numbers. The probe script still lists the old sizes for the point-cap groups `hull` and `tri`; they now print `INPUT_TOO_LARGE`, which
is the proof that the front end refuses them before any process starts.


## S7 review fixes: sequential measurements, 2026-10-15

AArch64 runtime `sha256:91704d190d8992b8831320d840f0c258fc7519df30790f30b227bbafdb2ca630`:
Docker inspect `.Size` **136,513,714 bytes**, **+6,014 bytes** over the reviewed S7
image's measured 136,507,700 (the old document's 136,507,587 was 113 bytes stale).
No wheels or runtime dependencies changed. Import groups remain lazy.

`scripts/measure-s7.py pycli:runtime` ran sequentially, ten fresh shim/container
calls per case, with no concurrent build, test or recipe workloads. Host filesystem
cache was warm. Production shim: 2 CPUs, 4 GiB, pids128, init, network none,
read-only rootfs, all caps dropped, no-new-privileges, default seccomp, private
512 MiB tmpfs, one BLAS thread. p95 here is the largest of ten observations,
not a population-tail estimate. These are fresh measurements, not a controlled
before/after speed comparison. Raw: `measurement-s7-review-shim.json`.

| Case | Complete shim wall p50 / p95 (ms) | Worker import p50 (ms) |
|---|---:|---:|
| py_info | 641.832 / 666.792 | 111.035 |
| interp_eval_pchip_40 | 1064.698 / 1118.450 | 460.972 |
| interp_eval_rbf_40 | 1049.591 / 1060.993 | 448.768 |
| sig_filter_butter8_400 | 1420.173 / 1593.414 | 759.648 |
| sig_spectrum_welch_400 | 1411.521 / 1469.043 | 759.853 |
| sig_peaks_400 | 1407.857 / 1430.057 | 757.591 |
| spatial_knn_60 | 963.852 / 1118.274 | 385.457 |
| spatial_hull_delaunay_60 (30 points) | 978.194 / 1318.746 | 378.505 |
| spatial_dist_60 (20 x 20 pairs) | 942.463 / 966.787 | 375.945 |

Resource regressions ran separately in fresh containers with the same hardening.
`scripts/measure-s7-review.py CASE` creates its inputs, times the CLI inside the
container (includes child imports; **excludes Docker startup**), then removes inputs
and output. Child RSS is Linux RUSAGE_CHILDREN high-water KiB, including process
startup; it is not total cgroup memory. Raw: `measurement-s7-review-resources.json`.

| Case | Direct CLI seconds | Exit / result | Child max RSS (KiB) |
|---|---:|---|---:|
| knn_preflight | 0.007399 | 1 / OUTPUT_TOO_LARGE | 27740 |
| knn_at_output_cap | 1.177887 | 0 / OK | 93336 |
| peaks_inline | 1.138658 | 1 / OUTPUT_TOO_LARGE | 113912 |
| peaks_write | 1.178034 | 2 / BAD_PARAM | 113916 |
| peaks_windowed | 1.209331 | 0 / OK | 117908 |
| radius_dense | 0.751389 | 1 / OUTPUT_TOO_LARGE | 94452 |

`knn_preflight` is the 200000-point, k1000 former-OOM request; it is refused before
Python. The accepted boundary is 950 queries x k1000 x two arrays = 1.9M numbers
(15.2 MB payload). The 200000-sample monotone saw has 99999 candidates: inline
refuses before prominence work, write without wlen refuses its work budget, while
explicit wlen199 performs <=19.9M candidate-window work and writes all properties.
`radius_dense` uses 200000 coincident points, radius1, max_pairs100000 in write mode:
it stops on the first query once the cap is exceeded, without counting all pairs.
No resource probe crashed or timed out; every container completed within its enforced
4 GiB limit. RSS samples do not measure aggregate cgroup peak.

Reproduce a resource case (create a fresh writable `measure-work` below HOME):

```sh
docker run --rm --pull=never --init --network=none --cpus=2 --memory=4g --pids-limit=128 \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m \
  --mount "type=bind,src=$PWD/measure-work,dst=/work" \
  --mount "type=bind,src=$PWD/scripts/measure-s7-review.py,dst=/probe.py,readonly" \
  -w /work --entrypoint python3 pycli:runtime /probe.py knn_at_output_cap
```

Historical S7 cap timings above used the earlier budget policy. Guarded S3–S7
operations now share time_limit seconds (default10, >0..30); a historical native
case taking >10 seconds may now return TIME_LIMIT unless given a larger explicit
budget. The independent <=50-second whole-child limit still includes imports.
No amd64 execution is claimed.

## S8a: bridge, mutate, merge, rolling, metrics, sampling (aarch64, 2026-10-15)

Initial S8a image **634 MB** on disk (unchanged), `inspect .Size` **136,575,889** bytes (**+62,175** over the S7 review figure 136,513,714: worker and front-end source only; no
wheel or package was added, so there is no notices change).

**Import cost.** None of the new operations imports pandas, scipy.stats, scikit-learn or statsmodels (tests assert `meta.pandas == "unloaded"`). Worker import phase, p50 of 10 fresh
calls through the production shim (`docs/measurement-s8-shim.json`, `scripts/measure-s8.py`; shared host, so p95 is noisy):

| Operation | Imports | Worker import p50 | Call p50 wall | p95 wall |
|---|---|---|---|---|
| `py_info` (reference) | numpy + scipy | 0.11 s | 0.80 s | 1.02 s |
| `frame_to_array`, `frame_to_array_to`, `array_to_frame`, `frame_merge`, `frame_rolling`, `ml_metrics`, `stats_dist` rvs | numpy only | 0.10 s | 0.64-0.77 s | 0.78-1.10 s |
| `frame_mutate` | numpy + `scipy.special` (expression functions) | 0.34 s | 0.99 s | 1.12 s |

For comparison the pandas-based table operations of S2 take about 1.0 s per call (import 1.0 s), so the numpy-only bridge ops are about 0.3 s faster and `frame_mutate` is on par while importing neither pandas nor scipy.stats. No split worker is proposed (all far below the 3 s threshold).

**Caps measured in the hardened container** (2 CPUs, 4 GiB, read-only root, `scripts/measure-s8-caps.py`, raw output `docs/measurement-s8-caps.txt`; cgroup peak over the whole probe
**0.25 GB**), every case at its cap:

| Case | Result |
|---|---|
| `frame_mutate`, 200000 rows, one 512-node expression / a transcendental expression | 0.6 s each |
| `frame_mutate`, 32 assignments on 15000 rows (495000 cells) | 0.7 s |
| `frame_to_array_to` 200000 values; `array_to_frame` 100000 x 2 | 0.3 s; 0.4 s |
| `frame_merge` 100000 x 100000 rows, 50000 matches, inner / outer (one_to_one validated) | 0.7 s / 0.8 s |
| `frame_merge` 2000 x 2000 rows over 400 keys, 10000 result rows / the same with `max_rows` 9999 | 0.4 s / `ROW_EXPLOSION` in 0.4 s |
| `frame_merge` 5000 x 5000 rows with one shared key (25,000,000 rows) | `ROW_EXPLOSION` in 0.3 s, nothing allocated |
| `frame_rolling` 200000 rows, window 500 mean (work 1e8) / median | 1.4 s / 7.9 s |
| `frame_rolling` 80000 rows window 100 x 5 functions / window 1000 mean / window 1001 x 2 (work above 1e8) | 0.6 s / 1.1 s / `INPUT_TOO_LARGE` before spawn |
| `frame_rolling` time window 600 s over 80000 rows | 0.8 s |
| `ml_metrics` 200000 rows with 30 classes and a confusion matrix / ROC AUC + log loss + Brier | 0.6 s / 0.5 s |
| `stats_dist_to` rvs 1,000,000 values (norm, gamma, binom, poisson) | 0.3-0.6 s each |

The table-size cap is what binds most calls: a 200000-row table can carry 2 columns and a 3-column one is already refused at import (`INPUT_TOO_LARGE`), so mutate/rolling at 200000 rows work on
1-column tables and the 32-assignment case uses 15000 rows.

Acceptance: `./scripts/build.sh test` 0; `docker run --rm --pull=never --network=none pycli:test` 31/31 groups exit 0 (1924 s with four groups in parallel and a concurrent runtime build);
runtime rebuild 0; all 24 recipes passed real, `--check` and `--dry-run` through the host shim, 72/72 exit 0 (lmda, rcli and statsrust on PATH).


## S8a review fixes (aarch64, 2026-10-15)

Runtime `262a7a8c27d5`: **136,587,173 bytes** (`docker image inspect .Size`), **+9,635 bytes** over the actual reviewed runtime
136,577,538 bytes. Docker still displays about **634 MB** on disk; these are different Docker accounting metrics.
The earlier S8a section records the initial image (1,649 bytes below the reviewed image). No new wheels or packages were added.
Runtime and test-image production binaries have identical SHA256 `2de98642052de6aa71d76e57c11fe03416eb0695f591e0d14ed25434aa6fdd94`.

Fresh production-shim calls, ten sequential samples per case, warm filesystem, shared host idle of our builds/tests/recipes.
No concurrent measurement phases. p95 is the largest sample for n=10. All processes remained offline; imports include interpreter boot.
The S8 worker now also warms pinned stdlib ZoneInfo/tzdata resources before installing the import guard.
These are current observations, not an attribution of historical differences to that warmup.
Raw data: [measurement-s8a-review-shim.json](measurement-s8a-review-shim.json).

| Operation | Import p50 ms | Shim wall p50 ms | Shim wall p95 ms |
|---|---:|---:|---:|
| py_info | 105.811 | 681.374 | 715.461 |
| frame_to_array (100 x 2) | 113.658 | 685.851 | 751.390 |
| frame_to_array_to (200 x 3) | 112.519 | 712.463 | 726.182 |
| array_to_frame (100 x 2) | 113.113 | 711.633 | 783.412 |
| array_to_frame append (200 x 1) | 113.629 | 719.660 | 761.758 |
| frame_mutate (2 expressions) | 341.026 | 1003.417 | 1086.501 |
| frame_merge outer (40 x 40) | 114.789 | 721.549 | 737.670 |
| frame_rolling (window 10, 5 functions) | 113.104 | 721.547 | 913.864 |
| ml_metrics (200 binary rows) | 113.768 | 734.696 | 812.557 |
| stats_dist rvs (1000 gamma draws) | 115.203 | 778.304 | 796.743 |

Hardened production-image probes (2 CPUs, 4 GiB, pids 128, read-only root, no capabilities, no-new-privileges,
default seccomp, no network, caller UID/GID, /tmp tmpfs 512 MiB; only a private cwd writable):

* The review's 200000-row repeated-key join with a 65000-byte string projects about **13.0 GB** of fixed-width output.
  Left/right/outer each refused with OUTPUT_TOO_LARGE (expected exit 1) **before gathers**, in **0.777 / 0.871 / 0.845 s**.
  The preallocation sentinel regression separately verifies no gather is reached.
* Rolling **200000 rows x window 500 = 1e8 window-elements**, mean of finite `1e200`, succeeded (exit 0) in **3.250 s**.
* Whole-probe cgroup `memory.peak`: **213,504,000 bytes (203.6 MiB)**. Cumulative waited-child peak RSS reached
  **202,832 KiB (198.1 MiB)**. These are distinct metrics, and neither is an isolated per-operation allocation count.
* Times above include CLI/Python startup inside the already-started container, excluding outer Docker startup.
  Raw data and all exit codes: [measurement-s8a-review-resources.json](measurement-s8a-review-resources.json),
  reproducer: `scripts/measure-s8-review.py` in a fresh scratch cwd.

Acceptance: test build 0; runtime rebuild 0; **32/32 ctest groups**, exit **0**, **505.16 s**; new S8 review group **25 tests**,
existing S8 group **46 tests**; additional **3 ASan/UBSan preflight tests**, exit 0. All **24 recipes x real/check/dry-run = 72 invocations** exited 0.
The R sampler comparison again checked 25 quantiles, worst **2.85 standard errors**; optional absent lmda/statsrust legs printed setup hints.
All **five requested scratch mutations were rejected** by the regression tests (expected test exit 1), not installed in production.
Full command/result evidence: [acceptance-s8a-review.json](acceptance-s8a-review.json). amd64 execution remains unverified.

## S8b (first half): matrix functions, sparse, special functions, root systems, time-series tests (aarch64, 2026-10-16)

Runtime image **635 MB** on disk (634 MB before), `inspect .Size` **136,628,927** bytes (**+41,754** over the S8a review figure 136,587,173: source only; no wheel or package was added to the runtime, so the notices are
unchanged). mpmath 1.3.0 (a pure-Python wheel, hash-locked in `docker/requirements-test.lock`) is installed in the **test image only**; the test image grew by about 18 MB of site-packages, the runtime by nothing.

**Import cost** (worker import phase, p50 of 10 fresh calls through the production shim; `docs/measurement-s8b-shim.json`, `scripts/measure-s8b.py`). No group approaches the 3 s split-worker threshold.

| Operation | Imports | Worker import p50 | Call p50 wall | p95 wall |
|---|---|---|---|---|
| `py_info` (reference) | numpy + scipy | 0.11 s | 0.66 s | 0.69 s |
| `la_matrix_fn` | numpy, `scipy.linalg` | 0.36 s | 0.97 s | 1.09 s |
| `la_sparse` | numpy, `scipy.sparse.linalg` | 0.34 s | 0.95 s | 1.01 s |
| `special_fn` | numpy, `scipy.special` | 0.33 s | 0.93 s | 0.97 s |
| `opt_root` | numpy, scipy.optimize/integrate/special (the S3 group) | 0.45 s | 1.08 s | 1.27 s |
| `sm_tsa_tools` adfuller / ljung_box | pandas + statsmodels (the S5 group) | 1.07-1.09 s | 1.83-1.85 s | 2.1-2.3 s |

**Historical S8b structured probes** (2 CPUs, 4 GiB, read-only root, original `scripts/measure-s8b-caps.py`, raw output `docs/measurement-s8b-caps.txt`; cgroup peak over the whole probe **0.18 GB**). These structured successes did not bound arbitrary SuperLU fill-in; S8b-1 review below supersedes their factorisation limits, and the script now uses the smaller limits:

| Case | Result |
|---|---|
| `la_matrix_fn_to` n = 447 (the input cap): inv / expm / sqrtm / logm (SPD) | 0.8 s / 0.8 s / 0.9-1.1 s / 1.3 s |
| `logm` of a general random 447 x 447 matrix (Schur with complex arithmetic) / `pinv` 400 x 500 | 3.1 s / 0.9 s |
| `spsolve` 1-D n = 20000 / 2-D Poisson n = 19881 (98,841 entries, SuperLU fill-in) | 0.7 s / 0.8 s |
| `eigsh` shift-invert (sigma = 0) k = 5, n = 66000 (200,000 entries), with vectors | 1.1 s |
| `svds` k = 5 on the 2-D Poisson matrix | 3.1 s |
| `eigs` k = 3 on a random 100000 x 100000 matrix with 200,000 entries | 10.5 s (just inside the default 10 s `time_limit`) |
| `cg` / `gmres` / `eigsh` LA and SA on the 1-D Poisson matrix, n = 66000 | `TIME_LIMIT` at 10.7 s (the matrix needs ~n iterations; raise `time_limit` to 30 or use shift-invert) |
| `special_fn_to` 200000 values: jv, hyp2f1, gammaincinv, airy (4 outputs), struve, owens_t | 0.7-1.5 s each |
| `opt_root` 16 unknowns (hybr, x_i^2 = i + 1) / weakly coupled system (lm) | 0.7 s / 0.8 s |

These structured probes used little memory, but were not worst cases: arbitrary SuperLU fill-in escaped cooperative timing into the 50 s hard kill in review. The S8b-1 fix applies lower pre-spawn factorisation limits to both direct and shift-invert paths; measured boundary results follow below.

Acceptance: `./scripts/build.sh test` 0; `docker run --rm --pull=never --network=none pycli:test` 33/33 groups exit 0 (629 s with four groups in parallel); runtime rebuild 0; all 27 recipes passed real,
`--check` and `--dry-run` through the host shim, 81/81 exit 0 (rcli and statsrust on PATH; the optional lmda legs were skipped with their hints, no lmda binary is present).

## S8b-1 review: idle-host remeasurement and arbitrary-fill caps (aarch64)

Runtime Docker display **635 MB**; `docker image inspect .Size` **136,632,374 bytes**, versus the observed pre-fix image **136,629,907 bytes** (+**2,467 bytes**). These Docker size reports use different metrics; they are not interchangeable. No runtime packages or wheel locks changed. Test image Docker display **1.33 GB**; current runtime + test stay below 3 GB. Runtime/test production binary SHA-256 is identical (`db447da8faac40d4d3e897c9543849ade13b53962e79b27dce6d2f001819fd46`).

Sequential calls through the production host shim, after builds/tests/recipes completed with no competing workload: 10 fresh containers/Python processes per case, warm filesystem cache. All 80 calls succeeded (measurement command exit 0). The empirical p95 with only 10 samples is the largest sample. Source: scripts/measure-s8b.py; raw data: [measurement-s8b1-shim.json](measurement-s8b1-shim.json). Wall times include Docker startup; import time is reported by the worker. These measurements do not establish a speedup over the historical run.

| Case | Worker import p50 ms | Shim wall p50 ms | Shim wall p95 ms |
|---|---:|---:|---:|
| py_info | 107.819 | 615.062 | 744.528 |
| la_matrix_fn_expm_8x8 | 353.025 | 995.352 | 1053.813 |
| la_sparse_cg_100 | 341.683 | 975.588 | 1087.889 |
| la_sparse_eigsh_100 | 339.706 | 975.078 | 1092.923 |
| special_fn_gammainc_100 | 336.615 | 1038.931 | 1126.551 |
| opt_root_circle_line | 464.545 | 1216.746 | 1286.954 |
| sm_tsa_tools_adfuller_200 | 1072.671 | 1972.413 | 2056.923 |
| sm_tsa_tools_ljung_box_200 | 1076.001 | 1995.605 | 2094.388 |

Fresh hardened production-container probe (2 CPUs, 4 GiB, pids 128, read-only root, no caps, no-new-privileges, default seccomp, no network, caller UID/GID, /tmp tmpfs 512 MiB): random symmetric diagonally dominant COO, **n=2000, 10000 stored entries**, exactly **n*stored_entries=20M**. Duplicates coalesce to 9990 actual CSR entries. Both direct and shift-invert paths are at the new dimension/work boundaries, unlike the old banded/Poisson-only probes.

| Path | Exit/status | Wall seconds | Verified residual |
|---|---|---:|---:|
| la_sparse_to task=spsolve | 0 / OK | 0.902 | 4.85e-16 |
| la_sparse_to task=eigsh, sigma=0, k=3, vectors | 0 / OK | 1.101 | 2.20e-15 |

The n=20000 direct and n=40000 shift-invert review patterns now exceed pre-spawn limits and return INPUT_TOO_LARGE rather than entering a non-interruptible factorisation. The limits are conservative; larger structured matrices can be fast but do not justify exposing arbitrary fill-in. Unshifted ARPACK/CG/GMRES retain the larger sparse caps.

Whole-probe cgroup memory.peak: **82,644,992 bytes (78.8 MiB)**. Cumulative waited-child peak RSS: **90,404 KiB (88.3 MiB)**; different metrics, not isolated allocation counts. Times include inner CLI/Python startup, excluding outer Docker startup. Raw evidence: [measurement-s8b1-resources.json](measurement-s8b1-resources.json); reproducer: scripts/measure-s8b1-review.py.

Acceptance: test build **0**, runtime rebuild **0**, **34/34 CTest groups**, exit **0**, **809.12 s**; S8b review **24 tests**, existing S8b oracles **39 tests**; additional **3 ASan/UBSan preflight tests**, exit **0**. All **27 recipes x real/check/dry-run = 81 invocations** exited **0** through the host shim; R and statsrust legs ran (optional absent lmda legs gave their setup hints). All **five Kimi mutants** were rejected (expected test exit 1, audit runner exit 0).

The TSA cross-oracle measured largest statistics/correlations difference **2.2e-14**; ADF p-value difference **1.381838e-9**, traced to lmstats' normal-CDF approximation using an independent 65-digit reference (docs/oracle-findings.md). Full commands/exit codes: [acceptance-s8b1-review.json](acceptance-s8b1-review.json); recipe stdout: [measurement-s8b1-recipes.json](measurement-s8b1-recipes.json). amd64 execution remains unverified.

## S8b-2: graph operations and the stats_dist long tail (aarch64)

**Import cost and image delta** (`scripts/measure-s8b2.py pycli:runtime`, production shim, 10 calls each, raw `docs/measurement-s8b2-shim.json`, re-measured on the idle host after the review fixes; worker import p50 / wall p50 in ms): py_info 102 / 652; special_fn (scipy.special) 326 / 921;
**graph_paths 441 to 455 / 1077 to 1079** and **graph_flow 447 / 1091 to 1093** (networkx plus scipy.sparse.csgraph and the trusted-phase warm-up of every algorithm, so nothing imports lazily under the guard): networkx adds about 120 ms over a scipy.special call;
stats_dist new families 732 to 739 / 1393 to 1442 (scipy.stats import and the warm-up of the requested function alone with canonical parameters; before the review fixes the warm-up ran every method on the caller's parameters: 748 to 791 / 1539 to 1578); stats_dist norm rvs (numpy only) 107 / 679. The image is **139,247,426 bytes** (networkx 3.7 is a 2.6 MB pure-Python wheel
with its test directories pruned); both lock files gained one hash-pinned line.

**Worst cases in the hardened container** (2 CPUs, 4 GiB, read-only root, no network, `scripts/measure-s8b2-caps.py`, raw `docs/measurement-s8b2-caps.txt`; whole-run cgroup peak **0.71 GB**; slowest single call **14.5 s**, the 50 s hard kill is never reached):

| Case | Result | Wall seconds |
|---|---|---:|
| graph_paths_to dijkstra, chain, 100000 edges | rc 0 OK | 1.9 |
| graph_paths_to dijkstra, 223x223 grid (99012 edges) | rc 0 OK | 1.8 |
| graph_paths_to bellman_ford, 223x223 grid | rc 0 OK | 3.1 |
| graph_paths_to bellman_ford, 100000-edge negative chain in reverse order | rc 0 OK | 2.0 |
| graph_paths k_shortest=10, grid corner to corner (Yen on 49729 nodes) | rc 0 TIME_LIMIT | 11.3 |
| graph_paths k_shortest=10, random 50000 nodes | rc 0 OK | 2.5 |
| all pairs n = 1377 (the matrix bound), dijkstra, 100000 edges | rc 0 OK | 2.4 |
| all pairs n = 1377, bellman_ford (Johnson) | rc 0 OK | 2.7 |
| all pairs n = 1378 (rejected before the worker starts) | rc 0 OK | 0.9 |
| max_flow, layered network, 20000 edges | rc 0 OK | 1.3 |
| max_flow, 70x70 bidirectional grid (19320 edges) | rc 0 OK | 1.0 |
| max_flow, diamond ladder with 1e9 capacities (augmenting-path adversary) | rc 0 OK | 1.1 |
| min_cost_flow, layered network, 20000 edges, value 5 | rc 0 OK | 3.3 |
| min_cost_flow, 19999-edge chain, value 1e5 (network simplex) | rc 1 TIME_LIMIT | 10.9 |
| vonmises cdf, 200 points (about 40 ms per point) | rc 0 OK | 8.6 |
| levy_stable ppf, 200 points | rc 0 OK | 4.9 |
| betabinom ppf, n = 5000, 1000 points | rc 0 OK | 5.0 |
| zipf cdf, 1000 points up to 1e5, a = 1.05 | rc 0 OK | 3.3 |
| vonmises rvs, n = 1,000,000 | rc 0 OK | 8.7 |
| truncnorm rvs, n = 1,000,000 (0.71 GB peak) | rc 0 OK | 1.7 |
| skewnorm cdf, 200000 points | rc 1 TIME_LIMIT | 11.2 |
| nct ppf, 200000 points | rc 1 TIME_LIMIT | 14.5 |

`TIME_LIMIT` rows are the cooperative budget answering (default 10 s plus the import and write-out) instead of the 50 s kill. The first run of this script (before the caps in docs/oracle-findings.md items 3 to 6) found the real problems: betabinom with n = 1e9 crashed the worker after 17 s,
zipf with a = 1.0001 and x up to 1e9 was killed at 51.7 s, and 1,000,000-point evaluation inputs were refused by the 200,000-value array cap. Those requests are now rejected before any worker starts (`INPUT_TOO_LARGE`) or end in a domain status (`OUT_OF_RANGE`, `NUMERICAL_FAILURE`).

**Acceptance (final tree):** `./scripts/build.sh test` **0**; `docker run --rm --pull=never --network=none pycli:test` **36/36 CTest groups, exit 0, 1235.79 s** (new groups `s8b2_graph` 24 tests, `s8b2_dist` 24 tests); runtime rebuild **0**; all **30 recipes x real/check/dry-run = 90 invocations** through the host shim: **exit 0 for all 90** (R and statsrust legs ran; optional absent lmda legs print their setup hints). The new recipes: `graph-route-plan` (hand values on the CLRS graphs), `oracle-dist` (217 values vs R, worst relative difference 3.4e-12 on a continuous t quantile, 6e-15 on all pdf/cdf/sf values), `distributions-tour`.

**Review fixes (docs/s8b2-review-notes.md), acceptance of the corrected tree:** `./scripts/build.sh test` **0**; `docker run --rm --pull=never --network=none pycli:test` **37/37 CTest groups, exit 0, 1259.86 s** (new group `s8b2_review`, 27 tests: the review regressions and the fixed-seed fuzz against the pure-Python algorithms, the `opt_linprog` LP oracle and mpmath); runtime rebuild **0**; all **30 recipes x real/check/dry-run = 90 invocations** through the host shim: **exit 0 for all 90**. The five scratch mutants of `scripts/check-s8b2-review-mutations.py` each fail their regression (docs/measurement-s8b2-mutations.json). Worst-case rows above are unchanged by the fixes (the caps moved before spawn or stayed); the import figures above are the re-measured ones.

## Release slice (v0.1.0 tree, aarch64, 2026-10-16)

No new per-operation measurement: the host was not idle when the README timing tables were assembled (`uptime` load 9.26 / 10.29 / 8.15
on 4 cores); the tables in README "Timings" are generated by `scripts/timing-table.py` from the recorded `docs/measurement-*.json` files
and cite each one. amd64 is verified by `.github/workflows/ci.yml` only: run 38913844742 (2026-10-16) passed 38/38 CTest groups in 1243.58 s
on `ubuntu-24.04` (arm64 runner: 1003.75 s), both runtime builds, the notices / schema / provenance gates and 90/90 recipe invocations;
the amd64 runner reports the runtime image at 440 MB and the test image at 983 MB (`docker image inspect .Size` on that Docker version
counts differently from this host's content metric). No amd64 timing is measured.

**Acceptance (final tree 59c4dfd, unpiped, exit codes):** `./scripts/build.sh test` **0**; `docker run --rm --pull=never --network=none pycli:test`
**38/38 CTest groups, exit 0, 1214.39 s** (new group `docs_gate`); `./scripts/build.sh runtime` **0**; `scripts/run-recipes.sh`
**30 recipes x real/--check/--dry-run = 90 invocations, exit 0 for all 90** (rcli, statsrust and lmda are not on this host's PATH: their
optional legs print setup hints); `python3 scripts/no-secrets.py` **0** (419 tracked files); `python3 tests/docs_gate.py` **0**.
`pycli py_info` through the shim reports `provenance.git_sha` 59c4dfdbbd36612ca6f8bc05c1767d0d1bdbd9a6 (not dirty), lmcpp
f6c6f51797912a369d6b27cf9e96f148a9a0e4d1 and the lock SHA-256 values of `docker/requirements-{aarch64,amd64}.lock`.
Image sizes (`scripts/image-size-report.py`): `pycli:runtime` 139,248,160 content bytes (648 MB on disk, 11 layers); `pycli:test`
281,375,098 content bytes (1.36 GB, 15 layers). One earlier full run on a loaded host (load about 9) failed `s4_review`'s
`time_limit: 0.001` LP case (TIME_LIMIT returned before HiGHS started, so `highs_model_status` was absent); the group passed on its own
and in the final run on the idle host, so that assertion is load-sensitive, not a defect of the tree. Builder layers and dangling images
were pruned after the run.
