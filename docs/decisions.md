# pycli: Lead decisions (binding for implementers; supersedes the three designs where they differ)

Sources: docs/design-opus.md (base for catalogue, safety, slices), design-fable.md (op detail, table shape, timings), design-astra.md (artifacts, risks). All three agree on the architecture; this file settles the differences.

## Architecture (agreed)
C++17 lmcpp front end validates everything (schema, caps, paths, expression ASTs, file headers) and starts a **fresh `python3 -I -B` child per call** inside a pinned offline docker image (host shim `bin/pycli`, as rcli/torchcli). The child runs one fixed embedded worker (`py/worker.py`) that lazily imports only what the op needs. No libpython embedding, no direct BLAS calls, no caller code, no `eval/exec/pickle/query/patsy/formulas` anywhere. Persistent worker only if the S0/S9 timing gate trips.

## Differences settled
| Topic | Decision |
|---|---|
| Table interchange v1 | CSV (UTF-8, header) + `columns`/`records` JSON envelope + `.npz`-free. **Parquet/pyarrow deferred to a late slice** (130 MB, CVE history), in a separate lock stage. |
| Array interchange | safetensors (`path#key`, torchcli parity) and `.npy` read-only (`allow_pickle=False`, header validated in C++ first). |
| Inline cap | **<= 1000 numbers inline** in results (token economy); larger results are written to a result file and returned as a ref (Opus). Inputs may be larger via files. |
| Expression parser | **Copy** rcli's `expr_parser`/`expr-spec.json` into pycli now (spec v2 if needed), with a shared corpus test; lifting into lmcpp is a later cross-repo task (not in this project's scope without the user). |
| Models | pycli-model JSON/safetensors parameter format for closed-form families; tree ensembles fit+predict in one call (`MODEL_NOT_PERSISTABLE` on save). No pickle ever. |
| Base image | Decide in S0 by measurement: `python:3.12-slim-bookworm` vs torchcli's base. Record image size and per-call timing. Hash-locked manylinux wheels, one lock per arch, no apt in runtime image. |
| Threads/RNG | `OPENBLAS_NUM_THREADS=1` (+ coretype pin if verified), `numpy.random.default_rng(seed)` PCG64, seeds required for stochastic ops and echoed. |
| Repro claim | Bit-exact only within one image+CPU family; across arch/CPUs tolerance tiers (<=1e-8 default, anything looser must be justified in the test). |
| amd64 | Host cannot emulate; amd64 lock validated by CI only; do not claim amd64 results locally. |

## Catalogue and slices
Use Opus section 3.9 (P0 ~30 ops) as the catalogue and Opus section 6.3 slices S0..S9 as the plan, with Fable's op argument shapes where more concrete. Parquet moves to after S7. Overlap with rcli/lmstats/lmda is justified only as a cross-tool oracle (document in parity.md).

## Process
Test-first per slice with an independent oracle; reviewers (Opus, Fable, Astra) review each slice or pair of slices; coders: Sol + Sonnet (hard: S0 skeleton/packaging, S3 expressions, S5 statsmodels), Luna + Haiku (bounded: ops with a direct library oracle). Commit trailers: one Co-Authored-By per contributing model. Private repo lmctlhq/pycli; push/visibility decisions belong to the user. Clean build output after heavy work (disk tight).
