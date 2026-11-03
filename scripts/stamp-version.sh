#!/usr/bin/env bash
set -euo pipefail
version="${1:?version required}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'invalid version' >&2; exit 2; }
root="$(cd "$(dirname "$0")/.." && pwd)"
printf '%s\n' "$version" > "$root/VERSION"
