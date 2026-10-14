# S0 review disposition

Read all three reviews from cppwrapper/pycli. Codes below refer to their numbered
findings (O=Opus, F=Fable, A=Astra). No other repository was edited or pushed.

Red runs before fixes: review_numeric failed 12 assertions, review_worker 5,
review_shim failed mount/control/policy/pull/signal cases, review_packaging failed
8 assertions, and real container_review failed policy/PID1/tmp-copy/wall cases.
The no-pull fixture captured /images/create; the exact comma directory variants
were tested against fake Docker so no arbitrary host bind was allowed to occur.
The terminated-writer oracle actually killed the old writer with SIGXFSZ and
found cancel.safetensors.tmpXXXXXX. The header-memory probe imposes RLIMIT_AS
before reading a sparse 160 MB safetensors input to distinguish preallocation
validation from a later cap check.

| Findings | Fix / regression |
|---|---|
| O1 | Reject comma, quote, equals and all control bytes before Docker; preserve trailing newlines when obtaining cwd; exact two exploit structures and control variants tested. |
| F1/A1 | Forward readonly/dryrun policy floors; fake argv and real file-effect tests. |
| A2 | --pull=never; real Docker CLI against a local Unix-socket API mock verifies no /images/create. |
| O2 | --init; C++ signal handler uses _exit even as PID 1; restores previous sigactions; lmcpp PDEATHSIG kills Python. Native anonymous output avoids named partials after kernel termination. |
| O3 | Background Docker client + wait; traps kill the named container, force-remove descendants, stop/reap client, and return 143/130. Real tests signal only shim PID, not its group. |
| O4 | Container-only noexec/nosuid/nodev /tmp tmpfs; SIGKILL discards requests and worker files; no cwd copies. Host diagnostic captures after shim SIGKILL cannot be trapped and may remain in host temp. |
| O5 | ImportError fallback fixed; README explicitly says guards are NOT a sandbox and documents cached references/ctypes/os.exec. Broader audit write/native/process gates deferred: S0 never executes caller code; native Python cannot be sandboxed with audit hooks, and new file ops need a separately tested allowlist. |
| O6/F5 | Real caps 447 ref / 31 inline; selected tensor header/count is checked before payload allocation; unselected payload is not read. |
| O7/F4 | -P -s -B with scrubbed env enables fixed hash seed; repeat hash probe, safe_path/user_site and planted modules/PYTHON* injection tested. Override is compiled only into CTest dev binary. |
| F2 | Exact symmetry for sym/pos before spawn; explicit Cholesky identifies non-PD; SPD/gen/sym/pos and reviewer counterexamples tested. |
| F3 | Reject shape growth after any scalar was appended; all four mixed/ragged/empty reproductions tested. |
| F6 | Shortest round-trip CLI wording; 1000 random/boundary finite doubles cross C++/Python/BLAS/JSON bit-exactly. |
| F7 | Independent normal/Cauchy references, huge-df limit, sf symmetry, SPD and ill-conditioning checks; success print moved last. |
| F8 | ppf [0,1] validation with BAD_PROBABILITY; tiny positive df preserved with NumericalWarning. Decline arbitrary df floor because df>0 is valid mathematically. |
| F9 | Write operations fail without artifact. Defer dryrun custom path/conflict checks: lmcpp short-circuits before handlers; requires upstream validation hook. |
| F10 | Private BadData/OutputTooLarge classes narrow mapping; correct empty-array message. Keep missing interpreter exit 8 for rcli/lmcpp fleet compatibility; document 1e999 parsing. |
| F11 | Recipe branches on simulated envelopes; meaningful dry-run/check/real execution. |
| F12 | Reject non-string keys; explicit complex re/im, named TypeError for unsupported values, serialize failures into PY_ERROR; subprocess probes test cases. |
| A3 | Full nlohmann/pycli/lmcpp licenses, matching source-license inventory/headers and SuperLU included; source hashes verified by generator; runtime retains package copyrights and GCC text. libquadmath remains LGPL-2.1-or-later. |
| A4 | Signed immutable snapshot 20261012T000000Z, exact top-level builder/test package versions, .deb hashes/full compiler/package provenance. |
| A5 | Fresh runtime base receives only pruned scientific tree; remeasure actual stored image layers; account for base pip bytes that cannot be reclaimed by whiteouts. |

The container is the wall: real tests inspect network=none, zero effective
capabilities, no-new-privileges, read-only rootfs, CPU/memory/pid limits, tmpfs and
default seccomp active (Seccomp:2). Test-only readonly NumPy initialization mount
pauses imports so signals can be delivered deterministically without introducing
production test operations. Direct PID 1 and Python-child termination are tested.
