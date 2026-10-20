#!/usr/bin/env bash
# Download the pinned third-party sources in deps.lock (sha256-verified), extract them, and build the one compiled
# dependency (HiGHS, static). Network is needed only here; the lmnum build itself is offline.
# Usage: scripts/fetch-deps.sh [DEPS_DIR]   (default native/.deps; pass the same dir to CMake as -DLMNUM_DEPS=DIR)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="${1:-$HERE/.deps}"
mkdir -p "$DEPS/src"
while IFS='|' read -r name url sha dir licence; do
  case "$name" in ''|'#'*) continue ;; esac
  archive="$DEPS/$name-$(basename "$url")"
  if [ ! -f "$archive" ] || [ "$(sha256sum "$archive" | cut -c1-64)" != "$sha" ]; then
    echo "fetch $name"
    curl -fsSL -o "$archive.part" "$url"
    mv "$archive.part" "$archive"
  fi
  have="$(sha256sum "$archive" | cut -c1-64)"
  [ "$have" = "$sha" ] || { echo "$name: sha256 $have differs from deps.lock $sha" >&2; exit 1; }
  [ -d "$DEPS/src/$dir" ] || tar -xzf "$archive" -C "$DEPS/src"
done < "$HERE/deps.lock"
if [ ! -f "$DEPS/highs-prefix/lib/libhighs.a" ]; then
  echo "build HiGHS (static)"
  cmake -S "$DEPS/src/HiGHS-1.7.2" -B "$DEPS/highs-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DFAST_BUILD=ON -DBUILD_TESTING=OFF -DZLIB=OFF -DCMAKE_INSTALL_PREFIX="$DEPS/highs-prefix" >/dev/null
  cmake --build "$DEPS/highs-build" --target highs -j"$(nproc)"
  cmake --install "$DEPS/highs-build" >/dev/null
fi
echo "deps ready in $DEPS"
