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
