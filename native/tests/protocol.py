"""Exercise the real native host through compiler-owned inputs and raw metadata."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from invocation import run_sources

host = os.environ['LMPY_HOST']
schema = Path('native/schema/lmpy.dsl.json').resolve()

def frame(r):
    lines = r.stdout.splitlines()
    assert len(lines) == 3 and lines[0] == 'LMCODE_RETURN_BEGIN' and lines[-1] == 'LMCODE_RETURN_END', r
    return json.loads(lines[1])

def raw(args, env=None):
    return subprocess.run([host, *map(str, args)], env=env, capture_output=True,
                          text=True, encoding='utf-8', errors='replace', timeout=30)

with tempfile.TemporaryDirectory(prefix='lmpy-protocol-') as temp:
    work = Path(temp)
    def source(name, body):
        path = work / (name + '.lms')
        path.write_text('p = _lmcode_ schema=' + json.dumps(str(schema)) + ';\n' + body + '\nend _lmcode_\n', encoding='utf-8')
        return path
    first, last = work/'first.txt', work/'last.txt'
    a = source('a', 'file_write('+json.dumps(str(first))+',"a"); out.put("msg","café λ"); out.check("pass",true); out.log("diagnostic");')
    b = source('b', 'file_write('+json.dumps(str(last))+',file_read('+json.dumps(str(first))+')+"b"); out.put("n",2);')
    r = run_sources([a,b], capture=work/'capture')
    data = json.loads(r.stdout)
    assert r.returncode == 0 and data['parts'] == 2 and data['ok'], r
    assert data['result'][0] == {'values':{'msg':'café λ'},'checks':{'passed':1,'failed':0,'failures':[]}}, r
    assert data['result'][1] == {'values':{'n':2},'checks':{'passed':0,'failed':0,'failures':[]}}, r
    assert last.read_text(encoding='utf-8') == 'ab' and 'diagnostic' in r.stderr, r
    first.unlink(); last.unlink()
    files = [work/'capture'/f'{i}.lmcode' for i in range(2)]
    for file in files:
        payload = file.read_bytes()
        assert payload[:6] == b'DSLB\x00\x06' and payload[6:38] == hashlib.sha256(schema.read_bytes()).digest()
    for name, body, code, phase in [('app','out.fail("stop");',4,'app'),
                                    ('check','out.check("failed-check",false);',4,'app'),
                                    ('run','int_list().at(9);',3,'run')]:
        bad = source(name, body)
        r = run_sources([a,bad,b], capture=work/name)
        data = json.loads(r.stdout)
        assert r.returncode == code and data['phase'] == phase and data['part'] == 1, r
        assert first.exists() and not last.exists(), r
        first.unlink()
    r = raw(['--lmcode',*files,work/'missing.lmcode'])
    assert r.returncode == 3 and frame(r)['phase'] == 'load' and not first.exists(), r
    manifest = work/'invocation.json'
    metadata = {'version':1,'mode':'builder','dry':True,'parts':[{'index':i,'file':str(f)} for i,f in enumerate(files)]}
    manifest.write_text(json.dumps(metadata), encoding='utf-8')
    env = dict(os.environ,LMCODE_INVOCATION_FILE=str(manifest))
    r = raw(['--lmcode',*files],env)
    assert r.returncode == 0 and frame(r) == {'ok':True,'result':[None,None],'parts':2,'dry':True}, r
    assert not first.exists() and not last.exists(), r
    r = raw(['--dry-run','--lmcode',work/'run'/'1.lmcode'])
    assert r.returncode == 0 and frame(r)['dry'] is True, r
    metadata['dry'] = False
    manifest.write_text(json.dumps(metadata), encoding='utf-8')
    r = raw(['--dry-run','--lmcode',*files],env)
    assert r.returncode == 2 and frame(r)['phase'] == 'cli' and not first.exists(), r
    env['LMCODE_INVOCATION_FILE'] = str(work/'absent.json')
    r = raw(['--lmcode',*files],env)
    assert r.returncode == 2 and frame(r)['phase'] == 'cli', r
    r = raw(['--lmcode-schema','--json'],env)
    assert r.returncode == 0 and r.stdout == schema.read_text(encoding='utf-8'), r
    for args in [['--dsl',files[0]],['--dsl-schema'],['--dsl-signatures'],['--lmcode']]:
        r = raw(args)
        assert r.returncode == 2 and frame(r)['phase'] == 'cli', r
    r = raw(['--lmcode',files[0]])
    assert r.returncode == 0 and frame(r)['result']['values']['msg'] == 'café λ', r
    explicit = source('explicit','out.put("legacy",1); lmcode_return_set(42);')
    r = run_sources([explicit])
    assert r.returncode == 0 and json.loads(r.stdout)['result'] == 42, r
print('PASS real-host framing, values/checks, UTF-8, result isolation, explicit returns, load-before-effects, stop-on-error, exit 2/3/4 and dry metadata')
