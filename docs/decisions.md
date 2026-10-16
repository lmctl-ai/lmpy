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

## Release decisions (v0.1.0, 2026-10-16)

**Parquet / pyarrow is NOT in v1.** The table interchange of v1 is CSV (UTF-8, header; `frame_import` / `frame_export`), the
`.npz` table artifact (typed columns, masks, categories; the only internal table format) and safetensors / `.npy` tensors.
Reasons, measured and documented in the slices: pyarrow adds about 130 MB of native code to a 139 MB runtime image, has a
CVE history on exactly the code path that would parse untrusted files (CVE-2023-47248, arbitrary code execution through
`PyExtensionType` deserialisation), and nothing in S0-S8b-2 needed it: every recipe moves tables through CSV or `.npz` and
arrays through safetensors. The deferral already recorded above ("late slice, separate lock stage") becomes a v1 exclusion.

**Future (not scheduled): slice "S10 parquet".** Exact shape, so that it can be picked up without redesign:
* one new hash-locked wheel per architecture (`pyarrow`, manylinux, pinned `==` version) in a **separate lock stage**
  `wheels-parquet` of the Dockerfile, so the v1 image is unchanged until the stage is composed in; the licence gate
  (`scripts/notices.py`, `licenses/manifest.json`) gains pyarrow and the Arrow C++ bundled notices;
* `frame_import` gains `format: parquet` and `frame_export` gains `format: parquet`, both behind C++ header validation
  before any Python starts: magic bytes `PAR1` at both ends, footer length bounded, the Thrift footer parsed by a closed
  reader that refuses any `PyExtensionType` / `ARROW:extension` metadata, encrypted files and more than 256 columns or
  200000 rows (the S2 caps), and the same dtype whitelist as `.npz` tables (int64, float64, bool, string, category,
  timestamp[ns] with tz);
* the worker reads with `pyarrow.parquet.read_table(..., use_pandas_metadata=False)` and converts through the existing
  `.npz` writer, so every downstream operation stays unchanged; no Arrow IPC, Feather or dataset APIs;
* acceptance: the S2 dtype matrix round trip (`frame_import` -> parquet -> `frame_import`) is an identity, the malicious
  corpus (extension types, overlapping column chunks, truncated footer, a pickle disguised as parquet) is refused in C++
  with no child spawned, image delta and import cost measured and published, amd64 lock validated by CI.
Until that slice exists, a Parquet file is rejected by `frame_import` like any other non-CSV input.

**Versioning.** The schema `version` (`pycli --schema`), the CMake project version, `CHANGELOG.md` and `py_info.provenance.version`
carry the same string; CI gates them. Nothing publishes images or packages automatically (RELEASE.md lists the human steps).

**amd64.** This host is aarch64 and cannot emulate amd64. The amd64 CI job is the verification. Status: the first run (38901596246,
2026-10-16) failed two CTest groups on amd64 (a tolerance tighter than the fixture's conditioning and a `numpy.longdouble` precision that
differs by platform, both corrected: docs/amd64-notes.md); run 38913844742 (2026-10-16, commit d611f5d) passed every gate on amd64 and
arm64. Any amd64 claim in this repository refers to that CI job; nothing is measured or claimed from this host.
