#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
examples=(
    'examples/dsl/lmpy.lms',
    'examples/dsl/install-and-run.lms',
    'examples/book/least-squares-line.lms',
    'examples/book/normal-quantile.lms',
    'examples/book/matrix-solve.lms',
)
for relative in examples:
    source=(root/relative).read_text()
    assert 'run_lmcode(' in source, f'{relative} must invoke through the lmcode manifest helper'
    for retired in ('savedsl(', 'dsl_cli('):
        assert retired not in source, f'{relative} retains unsupported compiler helper {retired}'
shared=(root/'examples/dsl/lmpy.lms').read_text()
assert 'run_lmcode(prog, {command:[bin, "--lmcode", "$DSLFILE"]})' in shared
print('PASS lmpy examples use the lmcode invocation manifest')
