# Cross-tool schema snapshots

The oracle recipes (`oracle-sm-*.lms`) call pycli, R through rcli, and lmstats' `statsrust`. Their schemas are local snapshots so the
recipes stay importable when the other checkouts are elsewhere:

* `statsrust.cli.schema.v1.json` is copied from rcli's `examples/schemas/` (lmstats commit `3e17ca02fac4a28723d71f9e6f47d9f0c626b2a2`).
* `lmda.cli.schema.v1.json` is lmda's `cli.schema.dev.json` (renamed as below) at lmda commit `36e528529fe090790edc03726f12b047c46be518`; `oracle-spectrum.lms` uses its `series_spectrum`. The binary is built with `cargo build --release` in `lmctlhq/lmda`; put `target/release` on PATH.
* `rcli.cli.schema.v1.json` is rcli's `cli.schema.dev.json` (renamed: lmscript only accepts `*.cli.schema.v1.json` for imported modules); byte-identical to rcli commit `a857035` (CI compares the snapshot with the pinned rcli checkout it builds for the `cross-*` recipes).

Setup (every host-binary leg is optional; a missing binary prints a hint and that leg is skipped):

```sh
export PATH="$PWD/bin:$HOME/repos/lmctlhq/rcli/bin:/path/to/statsrust-dir:/path/to/lmda/target/release:$PATH"   # pycli shim, rcli shim (needs R), statsrust, lmda
lmscript examples/oracle-sm-regression.lms
```

`statsrust` is built from `lmctl-ai/lmstats` (see its README). Run the recipes from a directory below `$HOME`; the pycli and rcli shims
refuse other working directories. Refresh a snapshot by copying the file from the named commit and updating this note.
