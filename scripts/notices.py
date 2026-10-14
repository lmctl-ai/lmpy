"""Run in the pinned test image to retain complete wheel license texts."""
import importlib.metadata as metadata
print('''# Third-party notices — S0

pycli is MIT, Copyright (c) 2026 Lattice Systems LLC.
The safetensors reader/writer and expression parser are adapted from torchcli
and rcli (MIT, Lattice Systems LLC). lmcpp is MIT, pinned to
f6c6f51797912a369d6b27cf9e96f148a9a0e4d1; nlohmann JSON is MIT.
CPython is PSF-2.0. The pinned Debian bookworm base retains Debian copyright
files for its unmodified libraries and utilities (including glibc LGPL).

S0 installs only NumPy 2.3.3 and SciPy 1.16.2 as scientific wheel distributions.
Their full wheel notices below cover BSD-3-Clause NumPy/SciPy/OpenBLAS,
modified-BSD LAPACK, BSD pocketfft/ARPACK/SuperLU/PROPACK, MIT HiGHS,
BSL Boost.Math, Qhull, and GCC libraries with the Runtime Library Exception.
All installed dist-info license files are also preserved in the runtime image.
The C++ executable does not link the scientific libraries.
''')
for name in ('numpy','scipy'):
    dist=metadata.distribution(name)
    assert dist.version=={'numpy':'2.3.3','scipy':'1.16.2'}[name]
    print('## '+name+' '+dist.version+' — original wheel license\n\n```text')
    text=dist.read_text('LICENSE.txt')
    assert text and 'OpenBLAS' in text
    print(text.rstrip())
    print('```\n')
# Reproduce Qhull's full original notice explicitly as required by the design.
dist=metadata.distribution('scipy')
for path in dist.files:
    if str(path).endswith('COPYING_QHULL.txt'):
        print('## Qhull — original notice\n\n```text')
        print(dist.locate_file(path).read_text().rstrip());print('```')
        break
else: raise RuntimeError('Qhull license missing')
