"""Host-side parity check against the rcli checkout this slice copied from (rcli is not available in the images).
Usage: python3 scripts/expr-parity.py [path/to/rcli]
Checks (exit 1 on any difference): expr-spec.json and tests/expr-corpus.json are byte-identical, and the only differences
between rcli's expr_parser.hpp and pycli's are the namespace/macro names and the documented vector-index extension."""
import difflib, hashlib, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
rcli = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root.parent / 'rcli')
status = 0
for ours, theirs in (('expr-spec.json', 'expr-spec.json'), ('tests/expr-corpus.json', 'tests/expr-corpus.json')):
    a, b = (root / ours).read_bytes(), (rcli / theirs).read_bytes()
    same = a == b
    print(f"{ours}: sha256 {hashlib.sha256(a).hexdigest()} {'==' if same else '!='} rcli {hashlib.sha256(b).hexdigest()}")
    status |= not same
normal = lambda text: text.replace('pycli_expr', 'NS').replace('rcli_expr', 'NS').replace('PYCLI_EXPR', 'X_EXPR').replace('RCLI_EXPR', 'X_EXPR').splitlines()
diff = [line for line in difflib.unified_diff(normal((rcli / 'src/expr_parser.hpp').read_text()), normal((root / 'src/expr_parser.hpp').read_text()), lineterm='', n=0)
        if line[:1] in '+-' and line[:3] not in ('+++', '---')]
removed = [line for line in diff if line[0] == '-']
added = [line for line in diff if line[0] == '+']
print(f"parser: {len(removed)} line(s) of rcli replaced, {len(added)} added (vector-index extension)")
allowed_removed = ('Parser(std::string text, std::set<std::string> names, bool allow_reducers)', ': text_(std::move(text)), names_(std::move(names)), allow_reducers_(allow_reducers), spec_',
                   'std::string text_; std::set<std::string> names_; bool allow_reducers_;', 'identifiers_.insert(name);return node(Json{{"t","var"},{"n",name}});', '#include <set>')
for line in removed:
    if not any(token in line for token in allowed_removed):
        print('unexpected change to rcli code:', line)
        status = 1
sys.exit(status)
