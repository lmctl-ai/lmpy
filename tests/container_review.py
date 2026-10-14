"""Real Docker acceptance: policies, live wall, PID 1 and direct shim signals."""
import json,os,pathlib,shutil,signal,subprocess,sys,tempfile,time,unittest,uuid
SHIM=str(pathlib.Path(sys.argv.pop(1)).resolve());DOCKER=shutil.which('docker')
def docker(*args,check=True):
    return subprocess.run([DOCKER,*args],capture_output=True,text=True,check=check)
class Containers(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=pathlib.Path.home());self.root=pathlib.Path(self.tmp.name)
        self.token=uuid.uuid4().hex;fake=self.root/'fake';fake.mkdir()
        wrapper=fake/'docker';wrapper.write_text('#!'+sys.executable+'\nimport os,sys\na=sys.argv[1:]\nif a and a[0]=="run":a[1:1]=["--label","pycli-review-run='+self.token+'"]\nos.execv('+repr(DOCKER)+',['+repr(DOCKER)+']+a)\n');wrapper.chmod(0o755)
        self.env=dict(os.environ,PATH=str(fake)+':'+os.environ['PATH'])
        # Test-only readonly mount pauses trusted NumPy initialization, giving
        # enough time to inspect/cancel the real C++ + Python processes reliably.
        source=docker('run','--rm','--pull=never','--network=none','--entrypoint','cat','pycli:runtime','/usr/local/lib/python3.12/site-packages/numpy/__init__.py').stdout
        assert 'from __future__' not in source
        self.pause=self.root/'numpy-init.py';self.pause.write_text('import time; time.sleep(8)\n'+source)
        old=wrapper.read_text()
        old=old.replace('a[1:1]=["--label",', 'a[1:1]=["--mount",'+repr('type=bind,src='+str(self.pause)+',dst=/usr/local/lib/python3.12/site-packages/numpy/__init__.py,readonly')+',"--label",')
        wrapper.write_text(old)
    def tearDown(self):
        for cid in docker('ps','-aq','--filter','label=pycli-review-run='+self.token).stdout.split():docker('rm','-f',cid,check=False)
        self.tmp.cleanup()
    def invoke(self,*args,env=None):return subprocess.run([SHIM,*args],cwd=self.root,env=env or self.env,capture_output=True,text=True,timeout=10)
    def start_call(self):
        # A long vector keeps scipy's t inverse busy long enough to observe the Python child.
        p=subprocess.Popen([SHIM,'la_solve_to','--input-json',json.dumps({'a':[[1]],'b':[2],'out':'cancel.safetensors'})],cwd=self.root,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+5;cid=None;top=''
        while time.monotonic()<deadline and p.poll() is None:
            ids=docker('ps','-q','--filter','label=pycli-review-run='+self.token).stdout.split()
            if ids:
                cid=ids[0];top=docker('top',cid,'-eo','pid,args',check=False).stdout
                if 'worker.py' in top:return p,cid,top
            time.sleep(.01)
        p.communicate(timeout=5);raise AssertionError('could not observe worker before completion')
    def test_FA1_readonly_dryrun_real(self):
        doc=json.dumps({'a':[[1]],'b':[2],'out':'out.safetensors'})
        for flag,rc in (('PYCLI_READONLY',6),('PYCLI_DRYRUN',0)):
            p=self.invoke('la_solve_to','--input-json',doc,env=dict(self.env,**{flag:'1'}))
            self.assertEqual(p.returncode,rc,p.stdout+p.stderr);self.assertFalse((self.root/'out.safetensors').exists())
        p=self.invoke('--readonly','la_solve_to','--input-json',doc)
        self.assertEqual(p.returncode,6);self.assertFalse((self.root/'out.safetensors').exists())
    def test_O2_O3_wall_and_signal(self):
        for sig,rc in ((signal.SIGTERM,143),(signal.SIGINT,130)):
            p,cid,top=self.start_call()
            config=json.loads(docker('inspect',cid).stdout)[0]
            host=config['HostConfig'];self.assertEqual(host['NetworkMode'],'none');self.assertTrue(host['ReadonlyRootfs'])
            self.assertEqual(host['CapDrop'],['ALL']);self.assertEqual(host['PidsLimit'],128);self.assertEqual(host['Memory'],4*1024**3);self.assertEqual(host['NanoCpus'],2000000000)
            self.assertIn('no-new-privileges',host['SecurityOpt']);self.assertTrue(host.get('Init',False));self.assertNotIn('seccomp=unconfined',host['SecurityOpt'])
            self.assertIn('/tmp',host['Tmpfs']);self.assertEqual(len(config['Mounts']),2) # cwd bind and readonly test fixture; tmpfs is HostConfig.Tmpfs
            self.assertEqual(sum(m['RW'] for m in config['Mounts']),1)
            self.assertTrue(any(m['Source']==str(self.root) and m['RW'] for m in config['Mounts']))
            for word in ('CapEff:\t0000000000000000','NoNewPrivs:\t1','Seccomp:\t2'):
                status=docker('exec',cid,'cat','/proc/1/status',check=False).stdout
                self.assertIn(word,status)
            worker_pids=[int(line.split()[0]) for line in top.splitlines() if 'worker.py' in line]
            started=time.monotonic();p.send_signal(sig);out,err=p.communicate(timeout=3)
            self.assertEqual(p.returncode,rc,(out,err));self.assertLess(time.monotonic()-started,3)
            self.assertFalse((self.root/'cancel.safetensors').exists())
            self.assertFalse(list(self.root.glob('.pycli-tmp*')))
            self.assertEqual(docker('ps','-q','--filter','id='+cid).stdout.strip(),'')
            for pid in worker_pids:
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)
    def test_O4_container_kill_no_workspace_copies(self):
        p,cid,top=self.start_call();docker('kill','--signal','KILL',cid);p.communicate(timeout=3)
        self.assertFalse(list(self.root.glob('.pycli-tmp*')))
        self.assertFalse((self.root/'cancel.safetensors').exists())
    def test_O2_pid1_direct(self):
        # Native frontend PID 1 (no --init): observe child, signal only parent.
        args=[DOCKER,'run','--rm','--pull=never','--network=none','--name','pycli-review-'+self.token,'--label','pycli-review-run='+self.token,'--mount','type=bind,src='+str(self.pause)+',dst=/usr/local/lib/python3.12/site-packages/numpy/__init__.py,readonly','--read-only','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=512m','pycli:runtime','stats_dist','--input-json',json.dumps({'dist':'t','fn':'ppf','x':[.975]*1000,'params':{'df':.01}})]
        p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if 'worker.py' in docker('top','pycli-review-'+self.token,'-eo','pid,args',check=False).stdout:break
            time.sleep(.01)
        docker('kill','--signal','TERM','pycli-review-'+self.token)
        out,err=p.communicate(timeout=3);self.assertEqual(p.returncode,143,(out,err))
if __name__=='__main__':unittest.main()
