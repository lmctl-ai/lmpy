"""Generate redistribution materials in a pinned NumPy/SciPy image."""
import hashlib,importlib.metadata as metadata,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
licenses=root/'licenses'
manifest=json.loads((licenses/'manifest.json').read_text())
# Fail before emitting a misleading complete inventory.
for package in ('numpy','scipy','pandas','python-dateutil','pytz','tzdata','six','statsmodels','patsy','packaging','scikit-learn','joblib','threadpoolctl','networkx'):
    for name,digest in manifest[package]['files'].items():
        assert hashlib.sha256((licenses/package/name).read_bytes()).hexdigest()==digest,name
for name in ('nlohmann/LICENSE.MIT','python/LICENSE'):
    assert hashlib.sha256((licenses/name).read_bytes()).hexdigest()==manifest[name]['sha256'],name
required=(
 'scikit-learn/COPYING','scikit-learn/sklearn/svm/src/liblinear/COPYRIGHT',
 'joblib/joblib/externals/cloudpickle/cloudpickle.py.notice.txt',
 'joblib/joblib/externals/loky/__init__.py.notice.txt','threadpoolctl/LICENSE','networkx/LICENSE.txt',
 'scipy/scipy/sparse/linalg/_dsolve/SuperLU/License.txt',
 'scipy/scipy/_lib/pyprima/LICENCE.txt',
 'scipy/scipy/_lib/decorator.py.notice.txt',
 'scipy/scipy/io/_fast_matrix_market/src/pystreambuf.h.notice.txt',
 'scipy/scipy/optimize/__lbfgsb.c.notice.txt',
 'scipy/scipy/stats/libnpyrandom/logfactorial.h.notice.txt',
 'scipy/scipy/interpolate/_bary_rational.py.notice.txt',
 'scipy/scipy/integrate/_lebedev.py.notice.txt',
 'scipy/subprojects/xsf/LICENSES_bundled.txt',
 'scipy/subprojects/highs/LICENSE.txt',
 'scipy/subprojects/boost_math/math/LICENSE',
 'pandas/pandas/_libs/src/vendored/ujson/lib/ultrajsonenc.c.notice.txt',
 'pandas/pandas/_libs/include/pandas/vendored/klib/khash.h.notice.txt',
 'pandas/pandas/_libs/src/vendored/numpy/datetime/np_datetime.c.notice.txt',
 'statsmodels/LICENSE.txt','statsmodels/statsmodels/LICENSE.txt','statsmodels/statsmodels/stats/libqsturng/LICENSE.txt',
 'statsmodels/statsmodels/multivariate/factor_rotation/LICENSE.txt','patsy/LICENSE.txt','packaging/LICENSE.APACHE','packaging/LICENSE.BSD')
for name in required:assert (licenses/name).is_file(),name
print('''# Third-party notices — S6

pycli is MIT, Copyright (c) 2026 Lattice Systems LLC.
The full pycli/lmcpp MIT notices and nlohmann JSON 3.11.2 MIT notice appear below.
The safetensors reader/writer and expression parser are adapted from torchcli
and rcli (MIT, Lattice Systems LLC). lmcpp is pinned to
f6c6f51797912a369d6b27cf9e96f148a9a0e4d1. CPython 3.12.15 is PSF-licensed.
The pinned Debian base retains Debian copyright files for unmodified components.

Scientific wheels: NumPy 2.3.3 and SciPy 1.16.2. Their licenses below retain
OpenBLAS/LAPACK/libgfortran notices. libquadmath is LGPL-2.1-or-later, not covered
by the GCC Runtime Library Exception; the amd64 wheels ship it, arm64 does not.
Static libstdc++/libgcc use GCC licensing with the Runtime Library Exception.
S2 adds pandas 2.3.3, python-dateutil 2.9.0.post0, pytz 2025.2, tzdata 2025.2 and six 1.17.0, with full source inventories and wheel notices below.
S5 adds statsmodels 0.14.6 (BSD-3-Clause, with bundled third-party notices for libqsturng, factor_rotation and the state-space
test data), patsy 1.0.3 (BSD-2-Clause, with PSF-2.0 terms for the patsy.compat code; a hard dependency of statsmodels that pycli never imports) and packaging 26.3
(Apache-2.0 OR BSD-2-Clause), with source inventories and wheel notices below.
S6 adds scikit-learn 1.7.2 (BSD-3-Clause; bundled liblinear, libsvm and MurmurHash notices), joblib 1.5.2
(BSD-3-Clause with vendored loky/cloudpickle notices) and threadpoolctl 3.6.0 (BSD-3-Clause).
The wheel's libgomp licence and GCC runtime exception are retained in its full wheel notice.
S8b-2 adds networkx 3.7 (BSD-3-Clause; pure Python, imported only by the graph operations).
S1 adds no shipped component. Its ASan/UBSan test binaries statically link GCC's sanitizer
runtimes (same Runtime Library Exception) and exist only in the test image, never in runtime.
The runtime additionally retains exact builder package copyrights (including
nlohmann's embedded third-party notices), GPL-3 text, compiler/package versions
and downloaded .deb hashes at /usr/share/doc/pycli/build/.

The matching source inventories and notices supplement wheel metadata, including
SuperLU, ARPACK, Qhull, xsf, HiGHS, Boost.Math, UNU.RAN, Biasedurn, pybind11,
array-api-compat, prima, and embedded-license headers. Original license-bearing
source files are retained as .notice.txt where the source inventory specifies a
header rather than a separate license. These are documentation, never executed.
The source archive URLs/hashes and per-file hashes are in licenses/manifest.json.
Some source notices also cover build-only/test code; inclusion does not assert
that every such source component is linked into the runtime executable.
''')
def emit(title,text):
    print('## '+title+'\n\n```text');print(text.rstrip());print('```\n')
for name in ('numpy','scipy','pandas','python-dateutil','pytz','tzdata','six','statsmodels','patsy','packaging','scikit-learn','joblib','threadpoolctl','networkx'):
    dist=metadata.distribution(name);assert dist.version==manifest[name]['version']
    texts=[dist.locate_file(f).read_text() for f in dist.files if dist.locate_file(f).is_file() and pathlib.PurePosixPath(str(f)).name.lower().startswith(('license','copying','notice')) and '.dist-info/' in str(f)]
    assert texts,name
    emit(name+' '+dist.version+' — full wheel notice','\n'.join(texts))
for path in sorted(licenses.rglob('*')):
    if path.is_file() and path.name!='manifest.json':
        emit(str(path.relative_to(licenses)),path.read_text(errors='replace'))
