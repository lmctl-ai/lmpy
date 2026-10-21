#!/usr/bin/env bash
# Python parity check (development tool, not part of the CI gate): runs each original Python recipe NAME.py with pycli's
# pinned scientific stack and the lmpy version NAME.lms, then compares every value the .lms recorded with out.put against
# the key of the same name in the JSON object the .py printed.
#   Usage: examples/python/check-parity.sh [NAME ...]
#   env: PYTHON (default native/.deps/oracle-venv/bin/python: numpy 2.3.3, scipy 1.16.2, pandas 2.3.3, statsmodels 0.14.6,
#        scikit-learn 1.7.2, networkx), LMPY (default native/build/lmpy), LMSCRIPT_BIN (default lmscript)
# Tolerance: relative 1e-9 (absolute below 1e-12) unless the .lms header says `// parity-rtol: X` (whole file) or
# `// parity-rtol KEY: X` (one key). A key recorded by only one side is listed, never failed: the .lms may add its own values.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$ROOT/native/.deps/oracle-venv/bin/python}"
export LMPY="${LMPY:-$ROOT/native/build/lmpy}"
LMSCRIPT="${LMSCRIPT_BIN:-lmscript}"
cd "$HERE"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=($(ls *.py | sed 's/\.py$//'))
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fails=0
for n in "${names[@]}"; do
  "$PY" "$n.py" > "$tmp/$n.py.json"
  LMPY_VALUES_OUT="$tmp/$n.lms.json" "$LMSCRIPT" "$n.lms" > "$tmp/$n.lms.out" 2>&1 || { echo "FAIL $n: lmscript failed"; tail -5 "$tmp/$n.lms.out"; fails=$((fails+1)); continue; }
  "$PY" - "$n" "$tmp/$n.py.json" "$tmp/$n.lms.json" "$n.lms" <<'PYEOF' || fails=$((fails+1))
import json, math, re, sys
name, pyf, lmf, src = sys.argv[1:]
py, lm = json.load(open(pyf)), json.load(open(lmf))
text = open(src).read()
m = re.search(r'^// parity-rtol:\s*([0-9.eE+-]+)', text, re.M)
default = float(m.group(1)) if m else 1e-9
per = {k: float(v) for k, v in re.findall(r'^// parity-rtol (\S+):\s*([0-9.eE+-]+)', text, re.M)}
def flat(x):
    if isinstance(x, dict) and "data" in x and "columns" in x: x = x["data"]
    if isinstance(x, dict): return [v for k in sorted(x) for v in flat(x[k])]
    if isinstance(x, (list, tuple)): return [v for e in x for v in flat(e)]
    return [x]
bad, compared = [], 0
for k in sorted(set(py) & set(lm)):
    a, b, tol = flat(py[k]), flat(lm[k]), per.get(k, default)
    if len(a) != len(b): bad.append(f"{k}: {len(a)} python values vs {len(b)} lmpy values"); continue
    for i, (x, y) in enumerate(zip(a, b)):
        compared += 1
        if isinstance(x, str) or isinstance(y, str) or isinstance(x, bool) or isinstance(y, bool):
            if x != y: bad.append(f"{k}[{i}]: python {x!r} lmpy {y!r}")
            continue
        if x is None or y is None:
            if (x is None) != (y is None) and not (x is not None and isinstance(x, float) and math.isnan(x)) and not (y is not None and isinstance(y, float) and math.isnan(y)):
                bad.append(f"{k}[{i}]: python {x!r} lmpy {y!r}")
            continue
        if not math.isclose(x, y, rel_tol=tol, abs_tol=1e-12) and not (math.isinf(x) and x == y):
            bad.append(f"{k}[{i}]: python {x!r} lmpy {y!r} (rel {abs(x-y)/max(abs(x),1e-300):.2e} > {tol:g})")
only_py, only_lm = sorted(set(py) - set(lm)), sorted(set(lm) - set(py))
status = "FAIL" if bad or compared == 0 else "PASS"
print(f"{status} {name}: {compared} values compared over {len(set(py) & set(lm))} keys"
      + (f"; python only: {', '.join(only_py)}" if only_py else "") + (f"; lmpy only: {', '.join(only_lm)}" if only_lm else ""))
for b in bad[:10]: print("    " + b)
sys.exit(1 if status == "FAIL" else 0)
PYEOF
done
echo "python parity: $(( ${#names[@]} - fails )) of ${#names[@]} recipes match"
[ "$fails" -eq 0 ]
