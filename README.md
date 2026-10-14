# pycli

S0: an MIT C++17/lmcpp frontend for pinned NumPy 2.3.3 / SciPy 1.16.2.
Each call starts a fresh Python child. The fixed dispatch and validated inputs
accept no caller code. The Docker container is the isolation boundary.

| Operation | Access | Inputs | Result |
|---|---|---|---|
| py_info | read | none | versions, OpenBLAS pools/threads/core, PCG64 RNG kind, architecture, worker SHA256, hash/isolation flags |
| la_solve | read | a, b, assume_a=gen/sym/pos | x, rcond estimate, residual norm, domain status |
| la_solve_to | write | same plus exclusive out.safetensors | x descriptor; fails if no artifact is produced |
| stats_dist | read | dist=norm/t, fn=pdf/cdf/ppf/sf, x vector, params | values and captured warnings |

Arrays accept nested row-major JSON, `{shape:[r,c],data:[...]}`, or cwd-confined
`file.npy` / `file.safetensors#key` refs. S0 supports little-endian float64 C-order
.npy v1/v2 and F64 safetensors of rank 1 or 2. Mixed/ragged ranks, object arrays
and .npz are rejected. Headers, shapes, offsets and lengths are validated before
Python; safetensors reads only the selected bounded payload.

Limits: 1000 numbers per inline array, 200000 per referenced array, 256 MiB per
artifact, matrix order **447 by ref / 31 inline**. la_solve returns <=1000 solution
numbers; larger solutions require la_solve_to. stats_dist returns <=1000 values.
Solve requires finite data; sym/pos require exact symmetry. pos explicitly checks
positive definiteness. Read solves return SINGULAR or NOT_POSITIVE_DEFINITE;
write solves return COMMAND_FAILED with that tool_code and produce no artifact.
Outputs install with an anonymous inode and an atomic no-replace link. This
requires a Linux filesystem supporting O_TMPFILE and /proc/self/fd linking.
Cancellation during writing cannot leave a named partial file; a completed
atomic installation cannot be undone by a later cancellation.

Distribution params accept loc (default 0), scale (positive, default 1), and
positive df for t. ppf rejects probabilities outside [0,1], including infinities;
NaN/missing pass through. Extremely small positive df may make SciPy return NaN;
pycli preserves that value with NumericalWarning rather than inventing a df
floor. Warnings are deduplicated/counted, capped at 20 distinct entries, with
warnings_truncated. Nonfinite JSON strings: NaN, Inf, -Inf. Numeric missing
null/NA becomes NA in distribution results. Finite values make an exact binary64
round trip; CLI output uses shortest round-trip decimal representations (the
private worker codec uses 17 significant digits). Cross-architecture bit equality
is not claimed; BLAS/CPU/version differences remain relevant.

```sh
./scripts/build.sh test
docker run --rm --pull=never --network=none pycli:test
./scripts/build.sh runtime
python3 tests/review_shim.py bin/pycli
python3 tests/container_review.py bin/pycli
./scripts/clean-builder.sh
export PATH="$PWD/bin:$PATH"
pycli --schema
pycli --input-json '{"op":"stats_dist","params":{"dist":"norm","fn":"ppf","x":[0.975]}}'
lmscript --check examples/solve.lms
lmscript --dry-run examples/solve.lms
lmscript examples/solve.lms
```

The build uses the pinned local lmcpp revision through its installed CMake
package. Separate cp312 wheel hashes are checked in for aarch64/amd64. Builder
and test apt use the signed immutable Debian **20261012T000000Z** snapshot and
exact top-level versions; runtime retains the full package/compiler inventory,
.deb hashes and package copyrights at `/usr/share/doc/pycli/build`. Scientific
licenses are supplemented by hash-verified matching source notices, including
nlohmann JSON and SuperLU. Runtime copies a pruned scientific installation into
a fresh pinned base, so deleted wheel test bytes are absent from its layers.
Pip already supplied by the base is hidden; its shared base-layer bytes remain.
Network acquisition happens only during builds. Invocation uses --pull=never.

The shim rejects cwd outside HOME or containing comma, quote, equals or control
characters before invoking Docker. It mounts only cwd read/write, disables
network, keeps the rootfs read-only, drops all capabilities, enables
no-new-privileges and Docker's default seccomp, and sets 2 CPUs / 4 GiB / 128
process limits. --init forwards signals; direct namespace PID 1 runs also exit
correctly on TERM/INT. The shim forwards signals to its named container, waits
for the Docker client, and exits 143/130. Worker/request/response copies live in
container-only **512 MiB noexec/nosuid/nodev tmpfs**, never the workspace. A
container SIGKILL/OOM discards them. A shim SIGKILL cannot be trapped; the
remaining container still has the 50-second worker timeout and no workspace
request copies. Host diagnostic captures can remain in the host temp directory
after untrappable SIGKILL; this is not a worker/request copy in the workspace.

The child uses **-P -s -B** and a fully replaced environment. No PYTHONPATH,
PYTHONHOME or user site is inherited, cwd/script directory stay off sys.path,
and PYTHONHASHSEED=0 works. py_info reports/verifies these guarantees. The
interpreter override exists only in the CTest development binary; production
ignores PYCLI_PYTHON. PYCLI_TIMEOUT_S can only lower 50 seconds. OpenBLAS uses
one thread with verified ARMV8 on aarch64; amd64 Haswell remains unverified.

The import guard/audit hook is **defence in depth, NOT a sandbox**. Trusted
scientific imports run before it; removing sys.modules entries does not remove
cached references. ctypes/native calls and os.exec remain reachable to trusted
code. Caller code never runs; the container and fixed dispatch are the wall.
Import rejections use ImportError, allowing library fallback paths. Extending
audit gates for arbitrary file/process/native events is deferred before S1
file-reading operations; Python audit hooks cannot safely sandbox native code.

PYCLI_READONLY/PYCLI_DRYRUN and CLI policy flags reach lmcpp. Readonly rejects
writes (exit 6); dryrun simulates without files. lmcpp short-circuits handlers
on dryrun, so custom path/conflict checks are not performed in simulation.
Real existing outputs return CONFLICT (5). SIGSEGV maps to INTERNAL_ERROR /
TOOL_CRASH (1); missing interpreter remains SERVICE_UNAVAILABLE (8), matching
rcli. Unexpected numerical ValueError/OverflowError map to PY_ERROR (1); only
explicit worker prechecks map to validation errors (2). Huge invalid JSON such
as 1e999 gives PARSE_ERROR (3). See docs/parity.md for these choices.

Measurements and image sizes after these fixes are in [docs/timings.md](docs/timings.md).
The checked-in S0 raw measurements remain historical. amd64 locks are resolved;
execution/CI and Haswell verification remain pending, with no amd64 pass claimed.
The copied expression parser and rcli corpus have no exposed S0 expression ops.

| Review-fix runtime latency | p50 | p95 |
|---|---:|---:|
| py_info through shim | 0.861 s | 0.951 s |
| la_solve through shim | 1.348 s | 1.523 s |
| stats_dist through shim | 1.927 s | 2.033 s |

Current image sizes: runtime 446 MB / test 667 MB on disk; runtime content
metric 97.45 MB. Startup calibration: boot 17 ms, NumPy 200 ms, scipy.stats
1229 ms, Docker overhead 391 ms (p50, imports include boot). These timings are
slower than historical S0 samples; see the timing document for limitations.
