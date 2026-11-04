#!/usr/bin/env bash
# lmpy gate: build, system-libraries-only check, schema drift, signature parity with the lmscript compiler, every _lmcode_ example.
# Usage: native/test.sh [BUILD_DIR]   (env: LMPY_DEPS, LMSCRIPT_BIN, JOBS, LMPY_EXAMPLES="a b" to run a subset)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="${1:-$HERE/build}"
LMSCRIPT="$("$HERE/scripts/select-compiler.sh")"
pass=0; failn=0
ok() { echo "PASS $1"; pass=$((pass+1)); }
bad() { echo "FAIL $1"; failn=$((failn+1)); }

"$HERE/scripts/sync-sdk.sh" --verify >/dev/null && ok "vendored lmcode-cpp files match sdk.lock" || bad "lmcode-cpp files differ from sdk.lock"
cmake -S "$HERE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release ${LMPY_DEPS:+-DLMPY_DEPS="$LMPY_DEPS"} >/dev/null
cmake --build "$BUILD" -j"${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}" >/dev/null
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) os=windows; BIN="$BUILD/lmpy.exe" ;; Darwin) os=macos; BIN="$BUILD/lmpy" ;; *) os=linux; BIN="$BUILD/lmpy" ;; esac
python3 "$HERE/tests/check-host-flags.py" "$HERE/schema/lmpy.dsl.json" "$BIN" || bad "hard host flag cutover"
# Self-contained: only the operating system's own libraries may be shared dependencies (Linux: glibc; macOS: libSystem and
# libc++; Windows: DLLs that ship with Windows). libstdc++, libgcc and every numerical library are linked in.
size="$(wc -c < "$BIN" | tr -d ' ')"
case "$os" in
  linux) deps="$({ ldd "$BIN" 2>&1 || true; } | awk '{print $1}' | grep -v -E '^(linux-vdso\.so|/lib/ld-linux|/lib64/ld-linux|ld-linux|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|not|statically)' || true)" ;;
  macos) deps="$(otool -L "$BIN" | tail -n +2 | awk '{print $1}' | grep -v -E '^/usr/lib/(libSystem\.B|libc\+\+\.1)\.dylib$' || true)" ;;
  windows) deps=""; for dll in $(objdump -p "$BIN" | awk '/DLL Name:/ {print $3}'); do [ -f "/c/Windows/System32/$dll" ] || [ -f "/c/Windows/System32/downlevel/$dll" ] || deps="$deps $dll"; done ;;
esac
if [ -z "$deps" ]; then ok "lmpy needs only system libraries ($os, $size bytes)"; else bad "lmpy has non-system shared dependencies: $deps"; fi
if "$BIN" --lmcode-schema --json | cmp -s - "$HERE/schema/lmpy.dsl.json"; then ok "schema/lmpy.dsl.json equals lmpy --lmcode-schema --json"; else bad "schema drift: run $BIN --lmcode-schema --json > native/schema/lmpy.dsl.json"; fi
if "$BIN" --api-md | cmp -s - "$HERE/API.md"; then ok "API.md equals lmpy --api-md"; else bad "API.md drift: run $BIN --api-md > native/API.md"; fi
if diff <("$BIN" --lmcode-signatures) <("$LMSCRIPT" --lmcode-signatures "$HERE/schema/lmpy.dsl.json") >/dev/null; then ok "registry signatures equal lmscript --lmcode-signatures"; else bad "signature parity"; fi
# the schema is host-only (lmscript injects the lmcode std catalog): the registry minus the canonical std signatures must be
# exactly what the schema declares without std
csort() { LC_ALL=C sort; }
if diff <(LC_ALL=C comm -23 <("$BIN" --lmcode-signatures | csort) <("$LMSCRIPT" --lmcode-signatures "$HERE/cpp/std/lmcode-std.schema.json" | csort)) \
        <("$LMSCRIPT" --lmcode-signatures "$HERE/schema/lmpy.dsl.json" --without-std | csort) >/dev/null; then
  ok "schema is host-only; the registry adds exactly the lmcode std catalog"; else bad "host/std signature split"; fi

# cli="lmpy" examples must bind to this build, not to an installed or downloaded lmpy
export PATH="$(cd "$(dirname "$BIN")" && pwd):$PATH"
# examples/dsl (ports and tasks) and examples/python (Python recipes with their answers pinned); LMPY_EXAMPLES="a b" picks
# names from either directory
for dir in "$ROOT/examples/dsl" "$ROOT/examples/python" "$ROOT/examples/book"; do
  [ -d "$dir" ] || continue
  cd "$dir"
  for ex in $(ls *.lms 2>/dev/null | grep -v '^lmpy.lms$' | sed 's/\.lms$//'); do
    if [ -n "${LMPY_EXAMPLES:-}" ] && ! printf ' %s ' $LMPY_EXAMPLES | grep -q " $ex "; then continue; fi
    tag="$(basename "$dir")/$ex"
    if out="$(LMPY="$BIN" "$LMSCRIPT" "$ex.lms" 2>&1)"; then ok "example $tag"; else bad "example $tag"; echo "$out" | tail -5 | sed 's/^/    /'; fi
  done
done
cd "$ROOT"
export LMSCRIPT_BIN="$LMSCRIPT" LMPY_HOST="$BIN" LMPY_SHIM="$BUILD/invocation_shim"
case "$os" in windows)
  export LMPY_HOST="$(cd "$BUILD" && pwd -W)/lmpy.exe"
  export LMPY_SHIM="$(cd "$BUILD" && pwd -W)/invocation_shim.exe";; esac
python3 "$HERE/tests/protocol.py" && ok "invocation framing, payloads, dry mode and failures" || bad "invocation protocol"
echo "lmpy gate: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
