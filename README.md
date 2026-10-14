# pycli

S0 of an MIT C++17/lmcpp frontend for a pinned Python scientific worker. Each
call starts a fresh isolated Python child in an offline container. The catalogue,
validation and schema live in C++; no caller code is accepted.

| Operation | Access | Inputs | Result |
|---|---|---|---|
| py_info | read | none | Python/NumPy/SciPy versions, BLAS pools/threads/core, PCG64 RNG kind, architecture, worker SHA256 |
| la_solve | read | a, b, assume_a=gen/sym/pos | x, rcond estimate, residual_norm, status |
| la_solve_to | write | same plus exclusive out.safetensors | x descriptor with ref, dtype, shape, nbytes |
| stats_dist | read | dist=norm/t, fn=pdf/cdf/ppf/sf, x vector, params object | values |

Arrays accept nested row-major inline JSON, `{shape:[r,c],data:[...]}`, or
cwd-confined refs: `file.npy` or `file.safetensors#key`. S0 supports little-endian
float64 C-order .npy v1/v2 and F64 safetensors of rank 1 or 2. C++ validates
headers and file lengths before Python starts. Object arrays and .npz are
rejected. Inputs are capped at 1000 numbers inline, 200000 numbers via refs,
256 MiB per artifact, and solve order 512. la_solve returns at most 1000 solution
numbers; larger solutions require la_solve_to. Distribution vectors return at
most 1000 values. t requires positive df; loc defaults to 0 and scale to 1
(scale must be positive). Only loc/scale and t's df are accepted in params.

Solve requires finite data. Singular systems return status SINGULAR with exit 0.
Warnings are captured, deduplicated and capped at 20 distinct messages, with a
count and warnings_truncated flag. JSON nonfinite values use strings NaN, Inf,
-Inf; numeric missing input null/NA is preserved as NA in stats_dist. Finite
doubles use 17 significant digits and exact binary64 transport. Same-image,
same-CPU results are reproducible; cross-architecture bit equality is not claimed.

Build and validate on this aarch64 host:

```sh
./scripts/build.sh test
docker run --rm --network=none pycli:test
./scripts/build.sh runtime
./scripts/clean-builder.sh
export PATH="$PWD/bin:$PATH"
pycli --schema
pycli --input-json '{"op":"stats_dist","params":{"dist":"norm","fn":"ppf","x":[0.975]}}'
lmscript --check examples/solve.lms
lmscript examples/solve.lms
```

The build stages an allowlisted context from this repository and the pinned
local lmcpp checkout at `/home/ubuntu/repos/lmctl-ai/lmcpp`, as torchcli does.
The base digest and separate cp312 architecture wheel hashes are checked in.
Network is used only during image builds; no image is pushed. Runtime has no
compiler or pip, and installs only NumPy 2.3.3/SciPy 1.16.2 scientific wheels.
Native development uses CMake with an installed lmcpp and PYCLI_PYTHON pointing
to an interpreter with these wheels; --schema needs no interpreter.

The host shim requires a directory below HOME and mounts only cwd read/write.
It disables network, makes the rootfs read-only, drops all capabilities, enables
no-new-privileges, and limits containers to 2 CPUs, 4 GiB, 128 processes. Calls
use private mode-0700 temporary directories, scrubbed child environments, one
BLAS thread and a 50-second timeout. PYCLI_TIMEOUT_S may only lower the timeout;
PYCLI_BLAS_CORETYPE accepts ARMV8/Haswell. Each shim call uses its own cwd temporary directory to permit concurrent calls.
Temp files are removed after success,
errors, timeout and SIGTERM. SIGSEGV maps to INTERNAL_ERROR/TOOL_CRASH (exit 1);
missing interpreter maps to SERVICE_UNAVAILABLE (exit 8). Write operations honor
lmcpp readonly/dryrun policy and install artifacts atomically without replacing
existing files (CONFLICT, exit 5). Expression parser/spec and the rcli corpus are
copied as required by decisions.md; no expression operation is exposed in S0.

| aarch64 runtime measurement | p50 | p95 |
|---|---:|---:|
| Isolated Python boot | 18 ms | 155 ms |
| NumPy import including boot | 123 ms | 156 ms |
| scipy.stats import including boot | 889 ms | 983 ms |
| Docker run overhead | 406 ms | 437 ms |
| py_info through shim | 654 ms | 708 ms |
| la_solve 3×3 through shim | 879 ms | 1873 ms |
| stats_dist normal quantile through shim | 1427 ms | 1531 ms |

Runtime image: 528 MB on disk (Docker inspect content metric 115.73 MB). See
[docs/timings.md](docs/timings.md) for sample counts, both base candidates, raw
data and limitations. Python -I ignores PYTHONHASHSEED; deterministic hashes
are not claimed. Scientific imports need trusted initialization before the
import audit stub is activated; Docker and fixed dispatch remain the isolation
boundary. threadpoolctl is deferred; py_info checks exported OpenBLAS symbols.
ARMV8 is verified locally; amd64 locks are resolved but execution and Haswell
verification remain for CI. See THIRD_PARTY_NOTICES.md for full wheel notices.
