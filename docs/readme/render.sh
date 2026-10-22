#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

if [[ -z "${GITHUB_TOKEN:-}" ]] && command -v gh >/dev/null 2>&1; then
  GITHUB_TOKEN="$(gh auth token)"
  export GITHUB_TOKEN
fi
cache="$(mktemp -d "${TMPDIR:-/tmp}/lmpy-readme-cache.XXXXXX")"
trap 'rm -rf "$cache"' EXIT
export LMSCRIPT_TOOL_CACHE="$cache"
mkdir -p build/rendered docs/readme examples/renders

# Verify the three public example sources against a fresh download of @latest.
for example in \
  examples/book/normal-quantile.lms \
  examples/book/least-squares-line.lms \
  examples/book/matrix-solve.lms; do
  lmscript "$example"
done

# The gallery drawings consume and display values computed by lmpy itself.
for source in \
  docs/readme/hero-light.lms \
  docs/readme/hero-dark.lms \
  docs/readme/architecture-light.lms \
  docs/readme/architecture-dark.lms \
  docs/readme/gallery-normal-quantile.lms \
  docs/readme/gallery-least-squares-line.lms \
  docs/readme/gallery-matrix-solve.lms; do
  lmscript "$source"
done

cp build/rendered/readme-hero-light.png docs/readme/hero-light.png
cp build/rendered/readme-hero-dark.png docs/readme/hero-dark.png
cp build/rendered/readme-architecture-light.png docs/readme/architecture-light.png
cp build/rendered/readme-architecture-dark.png docs/readme/architecture-dark.png
cp build/rendered/readme-normal-quantile.png examples/renders/normal-quantile.png
cp build/rendered/readme-least-squares-line.png examples/renders/least-squares-line.png
cp build/rendered/readme-matrix-solve.png examples/renders/matrix-solve.png

python3 - <<'PY'
from pathlib import Path

images = [
    Path("docs/readme/hero-light.png"),
    Path("docs/readme/hero-dark.png"),
    Path("docs/readme/architecture-light.png"),
    Path("docs/readme/architecture-dark.png"),
    Path("examples/renders/normal-quantile.png"),
    Path("examples/renders/least-squares-line.png"),
    Path("examples/renders/matrix-solve.png"),
]
for image in images:
    if not image.is_file() or image.stat().st_size == 0:
        raise SystemExit(f"missing generated README image: {image}")
    if image.stat().st_size > 400_000:
        raise SystemExit(f"README image exceeds 400 KB: {image} ({image.stat().st_size} bytes)")
print(f"README images ready: {len(images)} files, each under 400 KB")
PY
