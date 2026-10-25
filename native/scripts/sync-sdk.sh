#!/usr/bin/env bash
# Vendor the lmcode-cpp runtime from one pinned commit: the VM header, the built-in std headers it
# includes and the canonical std schema. sdk.lock pins the commit and every file's content hash.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${LMCODE_CPP:-$HERE/../../lmcode-cpp}"
LOCK="$HERE/sdk.lock"
hash() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }
vendored() { echo "$HERE/cpp/dslvm.hpp"; ls "$HERE"/cpp/stdlib/*.hpp; echo "$HERE/cpp/std/lmcode-std.schema.json"; }
if [ "${1:-}" = "--verify" ]; then
  [ -f "$LOCK" ] || { echo "sdk.lock missing" >&2; exit 1; }
  count=0
  while IFS='=' read -r key want; do
    want="${want%$'\r'}"
    case "$key" in
      sha256.*)
        file="${key#sha256.}"
        [ -f "$HERE/$file" ] && [ "$(hash "$HERE/$file")" = "$want" ] || { echo "$file differs from sdk.lock: run scripts/sync-sdk.sh" >&2; exit 1; }
        count=$((count+1)) ;;
    esac
  done < "$LOCK"
  [ "$count" -gt 0 ] || { echo "sdk.lock has no per-file hashes" >&2; exit 1; }
  while IFS= read -r file; do
    grep -q "^sha256.${file#"$HERE/"}=" "$LOCK" || { echo "${file#"$HERE/"} is not pinned in sdk.lock" >&2; exit 1; }
  done < <(vendored)
  echo "$count SDK files match sdk.lock ($(sed -n 's/^commit=//p' "$LOCK"))"
  exit 0
fi
[ -f "$SDK/include/dslvm.hpp" ] || { echo "lmcode-cpp checkout not found at $SDK (set LMCODE_CPP)" >&2; exit 1; }
ref="${LMCODE_CPP_REF:-origin/main}"
commit="$(git -C "$SDK" rev-parse "$ref")"
rm -rf "$HERE/cpp/stdlib" "$HERE/cpp/std"
mkdir -p "$HERE/cpp/stdlib" "$HERE/cpp/std"
printf 'repo=https://github.com/lmctl-ai/lmcode-cpp\ncommit=%s\n' "$commit" > "$LOCK"
while IFS= read -r source; do
  case "$source" in
    include/dslvm.hpp) target=cpp/dslvm.hpp ;;
    stdlib/*.hpp|std/lmcode-std.schema.json) target="cpp/$source" ;;
    *) continue ;;
  esac
  git -C "$SDK" show "$commit:$source" > "$HERE/$target"
  printf 'sha256.%s=%s\n' "$target" "$(hash "$HERE/$target")" >> "$LOCK"
done < <(git -C "$SDK" ls-tree -r --name-only "$commit" -- include/dslvm.hpp stdlib/ std/lmcode-std.schema.json)
echo "synced lmcode-cpp $commit"
