#!/bin/sh
# Run every examples/*.lms through the host shim in real, --check and --dry-run modes from a scratch directory below HOME (the shim's
# requirement), with the pycli bin directory on PATH and examples/data/ copied alongside. Optional sibling tools (rcli, statsrust, lmda) print their setup hints when absent.
# Usage: scripts/run-recipes.sh [examples/x.lms ...]   (default: all recipes); exit 1 when any invocation fails.
set -u
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
work=$(mktemp -d "${HOME:?}/pycli-recipes.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
mkdir -p "$work/examples/schemas" "$work/examples/data"
cp "$repo"/examples/*.lms "$work/examples/"
cp "$repo"/examples/data/*.csv "$work/examples/data/"   # the tut-* recipes read their committed datasets from examples/data/
cp "$repo"/examples/schemas/*.json "$work/examples/schemas/" 2>/dev/null || true
cp "$repo/cli.schema.dev.json" "$work/"
export PATH="$repo/bin:$PATH"
cd "$work" || exit 1
fail=0; n=0
if [ $# -gt 0 ]; then set -- "$@"; else set -- examples/*.lms; fi
for f in "$@"; do
  f=examples/$(basename "$f")
  for mode in real --check --dry-run; do
    n=$((n+1))
    if [ "$mode" = real ]; then lmscript "$f" > "$work/last.log" 2>&1; rc=$?; else lmscript "$mode" "$f" > "$work/last.log" 2>&1; rc=$?; fi
    printf '%-36s %-10s rc=%d\n' "$f" "$mode" "$rc"
    if [ "$rc" -ne 0 ]; then fail=$((fail+1)); sed 's/^/    /' "$work/last.log" | tail -n 20; fi
  done
done
echo "invocations=$n failures=$fail"
[ "$fail" -eq 0 ]
