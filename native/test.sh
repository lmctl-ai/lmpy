#!/usr/bin/env bash
# lmnum gate: build, static-link check, schema drift, signature parity with the lmscript compiler, every _dsl_ example.
# Usage: native/test.sh [BUILD_DIR]   (env: LMNUM_DEPS, LMSCRIPT_BIN, LMNUM_EXAMPLES="a b" to run a subset)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="${1:-$HERE/build}"
LMSCRIPT="${LMSCRIPT_BIN:-lmscript}"
pass=0; failn=0
ok() { echo "PASS $1"; pass=$((pass+1)); }
bad() { echo "FAIL $1"; failn=$((failn+1)); }

"$HERE/scripts/sync-sdk.sh" --verify >/dev/null && ok "vendored dslvm.hpp matches sdk.lock" || bad "dslvm.hpp hash"
cmake -S "$HERE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release ${LMNUM_DEPS:+-DLMNUM_DEPS="$LMNUM_DEPS"} >/dev/null
cmake --build "$BUILD" -j"$(nproc)" >/dev/null
BIN="$BUILD/lmnum"
# Only glibc may be a shared dependency (libstdc++, libgcc and every numerical library are linked in).
deps="$({ ldd "$BIN" 2>&1 || true; } | awk '{print $1}' | grep -v -E '^(linux-vdso\.so|/lib/ld-linux|/lib64/ld-linux|ld-linux|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|not)' || true)"
if [ -z "$deps" ]; then ok "lmnum depends only on glibc ($(stat -c %s "$BIN") bytes)"; else bad "lmnum has non-glibc shared dependencies: $deps"; fi
if "$BIN" --schema --json | cmp -s - "$HERE/schema/lmnum.dsl.json"; then ok "schema/lmnum.dsl.json equals lmnum --schema --json"; else bad "schema drift: run $BIN --schema --json > native/schema/lmnum.dsl.json"; fi
if "$BIN" --api-md | cmp -s - "$HERE/API.md"; then ok "API.md equals lmnum --api-md"; else bad "API.md drift: run $BIN --api-md > native/API.md"; fi
if diff <("$BIN" --dsl-signatures) <("$LMSCRIPT" --dsl-signatures "$HERE/schema/lmnum.dsl.json") >/dev/null; then ok "registry signatures equal lmscript --dsl-signatures"; else bad "signature parity"; fi

cd "$ROOT/examples/dsl"
list="${LMNUM_EXAMPLES:-$(ls *.lms | grep -v '^lmnum.lms$' | sed 's/\.lms$//')}"
for ex in $list; do
  if out="$(LMNUM="$BIN" "$LMSCRIPT" "$ex.lms" 2>&1)"; then ok "example $ex"; else bad "example $ex"; echo "$out" | tail -5 | sed 's/^/    /'; fi
done
echo "lmnum gate: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
