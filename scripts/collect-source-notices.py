"""Build-time only acquisition: hash-verify source archives and retain licenses."""
import hashlib,json,pathlib,tarfile,urllib.request,io
root=pathlib.Path('/notices');root.mkdir(exist_ok=True)
manifest={}
for name,version in (('numpy','2.3.3'),('scipy','1.16.2'),('pandas','2.3.3'),('python-dateutil','2.9.0.post0'),('pytz','2025.2'),('tzdata','2025.2'),('six','1.17.0'),('statsmodels','0.14.6'),('patsy','1.0.3'),('packaging','26.3'),('scikit-learn','1.7.2'),('joblib','1.5.2'),('threadpoolctl','3.6.0')):
    meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+version+'/json'))
    release=next(item for item in meta['urls'] if item['packagetype']=='sdist')
    data=urllib.request.urlopen(release['url']).read()
    digest=hashlib.sha256(data).hexdigest();assert digest==release['digests']['sha256']
    files={}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for entry in archive.getmembers():
            if not entry.isfile():continue
            basename=pathlib.PurePosixPath(entry.name).name.lower()
            embedded = {
                'scipy/_lib/decorator.py', 'scipy/io/_fast_matrix_market/src/pystreambuf.h',
                'scipy/optimize/__lbfgsb.c', 'scipy/stats/libnpyrandom/logfactorial.h',
                'scipy/stats/libnpyrandom/logfactorial.c', 'scipy/interpolate/_bary_rational.py',
                'scipy/integrate/_lebedev.py'}
            relative_name='/'.join(pathlib.PurePosixPath(entry.name).parts[1:])
            source_notice=(name=='joblib' and '/externals/' in relative_name and pathlib.PurePosixPath(relative_name).suffix=='.py') or (name=='scikit-learn' and pathlib.PurePosixPath(relative_name).suffix in ('.h','.c','.cpp','.pyx') and any(part in relative_name for part in ('/src/','/murmurhash','/liblinear','/libsvm'))) or relative_name in embedded or (name=='pandas' and '/vendored/' in relative_name and pathlib.PurePosixPath(relative_name).suffix in ('.h','.c','.cpp','.py'))
            if not source_notice and not basename.startswith(('license','licence','copying','notice','copyright')):continue
            relative=pathlib.PurePosixPath(*pathlib.PurePosixPath(entry.name).parts[1:])
            assert '..' not in relative.parts
            body=archive.extractfile(entry).read()
            target=root/name/(str(relative)+'.notice.txt' if source_notice else relative);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
            files[str(relative)+('.notice.txt' if source_notice else '')]=hashlib.sha256(body).hexdigest()
    manifest[name]={'version':version,'source':release['url'],'sha256':digest,'files':files}
for name,url in (
    ('nlohmann/LICENSE.MIT','https://raw.githubusercontent.com/nlohmann/json/v3.11.2/LICENSE.MIT'),
    ('python/LICENSE','https://raw.githubusercontent.com/python/cpython/v3.12.15/LICENSE')):
    data=urllib.request.urlopen(url).read();target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    manifest[name]={'source':url,'sha256':hashlib.sha256(data).hexdigest()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('license files:',sum(len(x.get('files',{})) for x in manifest.values()))
print((root/'scipy/LICENSES_bundled.txt').read_text())
