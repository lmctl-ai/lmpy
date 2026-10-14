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
