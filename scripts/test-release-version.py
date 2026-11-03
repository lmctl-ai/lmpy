#!/usr/bin/env python3
import shutil, subprocess, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
 repo=Path(d)/'repo'; shutil.copytree(root,repo,ignore=shutil.ignore_patterns('.git','native/.deps','native/build'))
 major,minor,patch=map(int,(repo/'VERSION').read_text().strip().split('.'))
 calc=repo/'scripts/derive-release-version.sh'
 assert subprocess.check_output([calc,'1'],text=True).strip()==f'{major}.{minor}.{patch+1}'
 assert subprocess.check_output([calc,'37'],text=True).strip()==f'{major}.{minor}.{patch+37}'
 expected=f'{major}.{minor}.{patch+37}'
 subprocess.check_call([repo/'scripts/stamp-version.sh',expected])
 assert (repo/'VERSION').read_text().strip()==expected
 assert subprocess.check_output([calc,'--diagnostic'],text=True).strip()==expected
 cmake=(repo/'native/CMakeLists.txt').read_text()
 assert 'file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" LMPY_VERSION' in cmake
 test_script=(repo/'native/test.sh').read_text()
 assert 'LMSCRIPT_EXAMPLE="${LMSCRIPT_EXAMPLE_BIN:-$LMSCRIPT}"' in test_script
 assert '"$LMSCRIPT" --lmcode-signatures' in test_script and '"$LMSCRIPT_EXAMPLE" "$ex.lms"' in test_script
 print('PASS lmpy run-derived version, stamping and CMake source')
