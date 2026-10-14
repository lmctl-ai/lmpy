"""Lock only the selected wheel on each architecture, not source distributions."""
import hashlib, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:])
lines = ['# Exact manylinux wheel hashes; generated in a build with pip download.']
for wheel in sorted(root.glob('*.whl')):
    name, version = wheel.name.split('-')[:2]
    lines += ['# '+wheel.name, name+'=='+version+' --hash=sha256:'+hashlib.sha256(wheel.read_bytes()).hexdigest()]
out.write_text('\n'.join(lines)+'\n')
