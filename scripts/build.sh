#!/bin/sh
# Legacy Docker on this host cannot honor Dockerfile-specific ignores. Stage an
# allowlisted context including the pinned local lmcpp checkout, as torchcli does.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT HUP INT TERM
for path in CMakeLists.txt cli.schema.dev.json expr-spec.json ml-spec.json LICENSE README.md CHANGELOG.md RELEASE.md licenses examples docs .dockerignore src py tests bin docker scripts THIRD_PARTY_NOTICES.md; do cp -R "$root/$path" "$context/"; done
# The pinned lmcpp checkout: LMCPP_DIR overrides the developer path (CI checks the same revision out next to the repo).
lmcpp=${LMCPP_DIR:-/home/ubuntu/repos/lmctl-ai/lmcpp}
expected=f6c6f51797912a369d6b27cf9e96f148a9a0e4d1
[ "$(git -C "$lmcpp" rev-parse HEAD)" = "$expected" ] || { echo 'lmcpp revision mismatch' >&2; exit 1; }
[ -z "$(git -C "$lmcpp" status --porcelain --untracked-files=no)" ] || { echo 'lmcpp checkout is dirty' >&2; exit 1; }
mkdir "$context/lmcpp"
git -C "$lmcpp" archive "$expected" | tar -x -C "$context/lmcpp"
# Build provenance for py_info: the pycli commit (plus -dirty when the tree has uncommitted changes) and the pinned lmcpp revision.
git_sha=$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)
[ -z "$(git -C "$root" status --porcelain --untracked-files=no 2>/dev/null)" ] || git_sha="$git_sha-dirty"
docker build -f "$context/docker/Dockerfile" --target "${1:-runtime}" -t "pycli:${1:-runtime}" \
  --build-arg "PYCLI_GIT_SHA=$git_sha" --build-arg "PYCLI_LMCPP_REF=$expected" "$context"
