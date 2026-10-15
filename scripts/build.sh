#!/bin/sh
# Legacy Docker on this host cannot honor Dockerfile-specific ignores. Stage an
# allowlisted context including the pinned local lmcpp checkout, as torchcli does.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT HUP INT TERM
for path in CMakeLists.txt cli.schema.dev.json expr-spec.json ml-spec.json LICENSE README.md licenses examples .dockerignore src py tests bin docker scripts THIRD_PARTY_NOTICES.md; do cp -R "$root/$path" "$context/"; done
lmcpp=/home/ubuntu/repos/lmctl-ai/lmcpp
expected=f6c6f51797912a369d6b27cf9e96f148a9a0e4d1
[ "$(git -C "$lmcpp" rev-parse HEAD)" = "$expected" ] || { echo 'lmcpp revision mismatch' >&2; exit 1; }
[ -z "$(git -C "$lmcpp" status --porcelain --untracked-files=no)" ] || { echo 'lmcpp checkout is dirty' >&2; exit 1; }
mkdir "$context/lmcpp"
git -C "$lmcpp" archive "$expected" | tar -x -C "$context/lmcpp"
docker build -f "$context/docker/Dockerfile" --target "${1:-runtime}" -t "pycli:${1:-runtime}" "$context"
