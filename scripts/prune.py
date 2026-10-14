import pathlib, shutil, sysconfig
root = pathlib.Path(sysconfig.get_path('purelib'))
# Preserve wheel dist-info/licenses and all runtime modules.
for package in ('numpy','scipy','pandas'):
    for path in (root/package).rglob('tests'):
        if path.is_dir(): shutil.rmtree(path)
for pattern in ('pip','pip-*.dist-info','setuptools','setuptools-*.dist-info','wheel','wheel-*.dist-info'):
    for path in root.glob(pattern):
        if path.is_dir(): shutil.rmtree(path)
for path in pathlib.Path('/usr/local/bin').glob('pip*'): path.unlink()
