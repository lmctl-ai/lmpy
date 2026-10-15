"""Host-side semantics differential: the same expression table through R (the language rcli evaluates in) and the golden file
that tests/s3_semantics.py runs pycli against inside the image.
  python3 scripts/expr-semantics-diff.py            # recompute R's answers and compare with tests/expr-semantics-golden.json
  python3 scripts/expr-semantics-diff.py --regen    # rewrite the golden file from R
  python3 scripts/expr-semantics-diff.py --rcli ~/repos/lmctlhq/rcli/bin/rcli   # also run the spot cases through the rcli runtime
Needs Rscript (and rcli for --rcli, run from a directory below HOME). Exit 1 on any difference."""
import json, math, pathlib, subprocess, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tests/oracle'))
import semantics
spec = json.loads((root / 'expr-spec.json').read_text())
expressions = semantics.semantic_expressions(spec)
golden_path = root / 'tests/expr-semantics-golden.json'


def r_literal(values):
    return 'c(' + ','.join('NaN' if math.isnan(v) else ('Inf' if v > 0 else '-Inf') if math.isinf(v) else repr(float(v)) for v in values) + ')'


R_CODE = '''
X <- %s
Y <- %s
fmt <- function(v) { v <- suppressWarnings(as.numeric(v)); if (length(v) != 1) "ERR" else if (is.na(v)) "NaN" else sprintf("%%.17g", v) }
for (e in readLines(commandArgs(TRUE)[1])) {
  two <- grepl("\\\\by\\\\b", e)
  ys <- if (two) Y else NA_real_
  out <- character(0)
  for (xx in X) for (yy in ys) out <- c(out, tryCatch(fmt(suppressWarnings(eval(parse(text = e), list(x = xx, y = yy, e = exp(1))))), error = function(err) "ERR"))
  cat(e, paste(out, collapse = " "), sep = "\\t"); cat("\\n")
}
''' % (r_literal(semantics.SEMANTIC_X), r_literal(semantics.SEMANTIC_Y))


def r_table():
    with tempfile.TemporaryDirectory() as directory:
        (pathlib.Path(directory) / 'e.txt').write_text('\n'.join(expressions) + '\n')
        (pathlib.Path(directory) / 'eval.R').write_text(R_CODE)
        out = subprocess.run(['Rscript', '--vanilla', str(pathlib.Path(directory) / 'eval.R'), str(pathlib.Path(directory) / 'e.txt')], capture_output=True, text=True, check=True).stdout
    table = {}
    for line in out.splitlines():
        expression, values = line.split('\t')
        table[expression] = values.split(' ')
    return table


def main():
    table = r_table()
    document = {'generator': 'scripts/expr-semantics-diff.py (Rscript, base R)', 'x': [semantics.token(v) for v in semantics.SEMANTIC_X],
                'y': [semantics.token(v) for v in semantics.SEMANTIC_Y], 'expressions': table}
    status = 0
    if '--regen' in sys.argv:
        golden_path.write_text(json.dumps(document, indent=0) + '\n')
        print(f'wrote {golden_path} ({len(table)} expressions)')
    else:
        golden = json.loads(golden_path.read_text())
        if golden != document:
            print('tests/expr-semantics-golden.json differs from what R computes now'); status = 1
        else:
            print(f'golden file matches R ({len(table)} expressions)')
    if '--rcli' in sys.argv:
        rcli = sys.argv[sys.argv.index('--rcli') + 1]
        spot = json.loads((root / 'tests/expr-semantics-spot.json').read_text())
        for case in spot['cases']:
            proc = subprocess.run([rcli, '--input-json', json.dumps({'op': 'num_integrate', 'params': {'f': case['f'], 'lower': spot['lower'], 'upper': spot['upper']}})], capture_output=True, text=True)
            got = json.loads(proc.stdout)
            ok = got.get('domain_status') == case['status'] and (case['status'] != 'OK' or abs(got['value'] - case['value']) <= 1e-12 * max(1, abs(case['value'])))
            print(('ok   ' if ok else 'FAIL ') + case['f'], {k: got.get(k) for k in ('domain_status', 'value')})
            status |= not ok
    sys.exit(status)


main()
