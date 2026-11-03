#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
base="$(cat "$root/VERSION")"
if [[ "$base" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}" patch="${BASH_REMATCH[3]}"; else echo "invalid VERSION: $base" >&2; exit 2; fi
if [[ "${1:-}" == --diagnostic ]]; then echo "$base"; exit 0; fi
[[ "${1:-}" =~ ^[1-9][0-9]*$ ]] || { echo 'expected positive workflow run number' >&2; exit 2; }
printf '%s.%s.%s\n' "$major" "$minor" "$((10#$patch + $1))"
