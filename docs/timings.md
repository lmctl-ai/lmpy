# S0 measurements — aarch64, 2026-10-14

Decision: CPython 3.12 slim bookworm, pinned multi-arch index
`sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.
NumPy 2.3.3 and SciPy 1.16.2 wheels were resolved and installed with hashes.
The two candidate images used identical scientific versions and the same C++
binary. 3.12's comparison image used 26 MB less disk, and its second-pass stats
call median was slightly lower; solve medians differed by 16 ms. Their p95s
overlap, so this is a size/compatibility choice, not a claim that 3.12 is faster
for every operation. 3.11 reuses torchcli base layers. 3.12 matches the host
Python minor version and avoids coupling this tool to torchcli's old base.

All measurements ran offline (`--network=none`) on this shared four-core host.
Times are wall-clock, warm filesystem cache, not isolated-host benchmarks.
Interpreter/import rows include interpreter startup; they are not incremental
import times. Twenty samples per interpreter row, ten per container/shim row;
p95 is the nearest upper empirical order statistic (the maximum for n=10).
Some candidate and final shim measurements overlapped builds; all raw measurements
are retained so variance and that limitation remain visible. The second pass
avoided numerical work in other containers, but the host was still shared.

| Candidate (before runtime pruning) | Python 3.12 | torchcli Python 3.11 |
|---|---:|---:|
| Docker on-disk image size | 805 MB | 829 MB |
| `python -I -B` boot p50 / p95, ms | 16.337 / 27.925 | 16.687 / 34.839 |
| NumPy import p50 / p95, ms | 133.289 / 155.884 | 113.881 / 155.581 |
| scipy.stats import p50 / p95, ms | 985.504 / 1106.511 | 898.782 / 1113.664 |
| `docker run /bin/true` p50 / p95, ms | 414.281 / 633.892 | 410.295 / 451.467 |
| Pruned comparison image size (retains resolver layers) | 821 MB | 847 MB |
| Shim py_info second pass p50 / p95, ms | 872.321 / 1130.359 | 640.183 / 802.936 |
| Shim la_solve second pass p50 / p95, ms | 977.116 / 1170.579 | 961.009 / 1086.638 |
| Shim stats_dist second pass p50 / p95, ms | 1385.227 / 1680.397 | 1451.416 / 1674.098 |

The resolver images deliberately retained downloaded wheels/pip cache in earlier
layers. The final runtime starts from the pinned base and installs locked wheels
without download caches; those intermediate sizes are not final runtime sizes.
Docker here uses the containerd image store: `docker image ls` reports on-disk
size including unpacked content; `docker image inspect .Size` reports a smaller
content metric. Final runtime: **528 MB on disk**, **115,730,671 bytes inspect
.Size** (about 115.73 MB); final test: **657 MB on disk**. Runtime is compiler-
and pip-free. Test directories are pruned, upstream license files preserved.

| Final runtime (import rows before guard enhancement; shim rows final) | p50 ms | p95 ms |
|---|---:|---:|
| Python isolated boot | 17.986 | 154.563 |
| NumPy import including boot | 123.456 | 156.367 |
| scipy.stats import including boot | 888.982 | 982.820 |
| Docker run overhead (`/bin/true`) | 405.632 | 436.691 |
| Shim py_info | 653.912 | 707.755 |
| Shim la_solve, 3×3 | 879.007 | 1873.154 |
| Shim stats_dist, norm.ppf(0.975) | 1427.009 | 1531.042 |

The numerical operations are unchanged; the final shim rows include the stronger
import guard and unique temporary directories. The complete license text and
guard enhancement increased inspect .Size by about 30 KB (exact content metrics
can vary slightly across rebuilds). Import estimates of 0.2–0.4 seconds for scipy.stats
were too optimistic here. The example makes two calls; it does not meet the
20-call/60%-startup gate for a persistent worker or batch mode.

Both wheels expose their own OpenBLAS pools. Automatic selection was
`neoversen1`; `OPENBLAS_CORETYPE=ARMV8` changed both reported cores to `armv8`.
Both report one thread. Twenty direct 400×400 SPD dense solves had medians
5.529 ms automatic and 5.459 ms ARMV8. There was no measured median slowdown,
within host noise; this probe is not a general performance guarantee. ARMV8 is
therefore the aarch64 default. Haswell on amd64 is provisional and unverified.

Unexpected design constraints:

- `python -I` ignores `PYTHONHASHSEED=0`. Two isolated processes gave different
  string hashes. py_info reports hash randomization enabled. S0 uses fixed
  ordered structures and does not claim deterministic set/hash iteration.
- NumPy imports serialization support, and even scipy's numerical import graph
  imports subprocess/network helpers through `_testutils` and other utilities.
  The selected trusted library group is imported first, then protected modules
  are removed from sys.modules and an import finder blocks new imports while the audit hook blocks network/
  process audit events. The finder is necessary: importlib.import_module bypassed
  the import audit event in a regression test. Cached library references may remain; this is a stub
  defense, with fixed dispatch and the Docker boundary providing isolation.
- S0 installs no threadpoolctl (only NumPy/SciPy). py_info verifies the two actual
  OpenBLAS thread counts through their exported C symbols using ctypes.
- The installed lmscript has no `run` subcommand: that form exits 2 trying to
  open a file named run. `lmscript --check examples/solve.lms` and the supported
  `lmscript examples/solve.lms` both exit 0. The schema key `fn` must be quoted
  in LMScript object literals because it is a language keyword.
- Concurrent schema/measurement calls exposed a race in the shared temp directory
  inherited from rcli: one call could remove it before another created its worker
  directory. The shim now uses a unique mode-0700 .pycli-tmp.XXXXXX directory per
  container; each call cleans only its own directory.
- Output arrays use safetensors, following binding decisions; .npy is read-only
  as a public artifact and also used as private exact binary64 transport.

Raw data: measurement-candidates.json, measurement-coretype.json,
measurement-shim-candidates.json, measurement-shim-candidates-quiet.json,
measurement-runtime.json, measurement-shim-runtime.json,
measurement-shim-runtime-final.json. Reproduce with
scripts/measure.py, scripts/coretype.py, scripts/measure-shim.py and the two
measurement Dockerfiles. All temporary comparison/resolver images are removed
after measurement. amd64 wheels were downloaded and hashed by metadata/platform
resolution during a build; no amd64 execution or pass is claimed.

Final acceptance commands (all unpiped):

| Command | Exit / result |
|---|---|
| Initial `python3 tests/s0.py /nonexistent/pycli` | 1, expected test-first missing binary failure |
| `./scripts/build.sh test` | 0 |
| `docker run --rm --network=none pycli:test` | 0, four CTest suites pass |
| `./scripts/build.sh runtime` | 0 |
| `./bin/pycli --schema` then `cmp cli.schema.dev.json /tmp/pycli-schema.json` | 0 / 0 |
| `PATH="$PWD/bin:$PATH" lmscript --check examples/solve.lms` | 0 |
| `PATH="$PWD/bin:$PATH" lmscript examples/solve.lms` | 0, real solve and quantile pass |
| `PATH="$PWD/bin:$PATH" lmscript run examples/solve.lms` | 2, unsupported subcommand |
| Four concurrent real shim py_info calls | all 0, temporary directories removed |

CTest covers independent SciPy and exact inverse oracles, warning capture,
nonfinite/missing input/output, 100000 binary64 codec cases, offline schema,
invalid header/path rejection before spawn, output caps and >1000-value artifact
outputs, readonly/dryrun/conflict, argv/env scrubbing, audit/import-finder probes,
SIGSEGV, timeout, signal cleanup, shim concurrency and the shared expr corpus.

Cleanup: disposable comparison/resolver images and unused S0 builder layers were
removed. The builder label cleanup exits 0. Only pycli:runtime (528 MB) and
pycli:test (657 MB) remain tagged, totaling 1.185 GB before shared-layer savings.
Host staging/build directories and generated bytecode were removed.
