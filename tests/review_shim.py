"""Mock argv and direct-PID cancellation, plus real Docker API no-pull proof."""
import http.server,json,os,pathlib,signal,socketserver,subprocess,sys,tempfile,threading,time,unittest
SHIM=str(pathlib.Path(sys.argv.pop(1)).resolve())
class Review(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=pathlib.Path.home());self.root=pathlib.Path(self.temp.name)
        fake=self.root/'fake';fake.mkdir();self.log=self.root/'args';self.started=self.root/'started'
        script=fake/'docker'
        script.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$LOG"
if [ "$1" = run ]; then
 : > "$STARTED"
 if [ "${BLOCK:-}" = 1 ]; then sleep 5; fi
 echo '{"success":true}'
fi
''');script.chmod(0o755)
        self.env=dict(os.environ,PATH=str(fake)+':'+os.environ['PATH'],LOG=str(self.log),STARTED=str(self.started),PYCLI_READONLY='1',PYCLI_DRYRUN='1')
    def tearDown(self):self.temp.cleanup()
    def invoke(self,cwd=None):return subprocess.run([SHIM,'--schema'],cwd=cwd or self.root,env=self.env,capture_output=True,text=True)
    def test_O1_comma_exact(self):
        # Opus's exact directory structure; fake Docker guarantees no host bind occurs on red.
        for name in (str(self.root)+'/q/,src=/tmp/rvtarget,dst='+str(self.root)+'/q/',str(self.root)+'/x,src=/tmp/rvtarget'):
            directory=pathlib.Path(name);directory.mkdir(parents=True)
            p=self.invoke(directory);self.assertEqual(p.returncode,2,p.stdout)
            self.assertEqual(json.loads(p.stdout)['error']['code'],'INVALID_REQUEST')
        self.assertFalse(self.log.exists(),'unsafe path invoked Docker')
    def test_O1_quotes_controls(self):
        for suffix in ('quote"','equal=','new\nline','end\n','tab\t'):
            directory=self.root/suffix;directory.mkdir()
            with self.subTest(suffix=suffix):self.assertEqual(self.invoke(directory).returncode,2)
    def test_FA1_policy_and_wall(self):
        self.assertEqual(self.invoke().returncode,0)
        args=self.log.read_text()
        for flag in ('PYCLI_READONLY','PYCLI_DRYRUN','--pull=never','--init','--tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m','TMPDIR=/tmp','--cap-drop=ALL','--security-opt=no-new-privileges','--read-only','--pids-limit=128','--memory=4g','--network=none'):
            with self.subTest(flag=flag):self.assertIn(flag,args)
        self.assertNotIn('seccomp=unconfined',args)
        self.assertFalse(list(self.root.glob('.pycli-tmp*')))
    def test_O3_signal_only_shim(self):
        for sig,rc in ((signal.SIGTERM,143),(signal.SIGINT,130)):
            self.started.unlink(missing_ok=True)
            p=subprocess.Popen([SHIM,'--schema'],cwd=self.root,env=dict(self.env,BLOCK='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            deadline=time.monotonic()+3
            while not self.started.exists() and time.monotonic()<deadline:time.sleep(.01)
            start=time.monotonic();p.send_signal(sig)
            try:out,err=p.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill();p.communicate();self.fail('signal only to shim did not cancel promptly')
            self.assertEqual(p.returncode,rc,(out,err))
            self.assertLess(time.monotonic()-start,2)
            self.assertIn('kill --signal',self.log.read_text())
    def test_A2_real_cli_no_pull_api(self):
        calls=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args):pass
            def do_HEAD(self):
                calls.append(self.path);self.send_response(200);self.send_header('API-Version','1.47');self.send_header('Content-Length','0');self.end_headers()
            def do_GET(self):self.reply()
            def do_POST(self):self.reply()
            def do_DELETE(self):self.reply()
            def reply(self):
                calls.append(self.path)
                self.rfile.read(int(self.headers.get('Content-Length','0')))
                body=json.dumps({'message':'No such image: pycli:runtime'}).encode()
                self.send_response(404);self.send_header('Content-Length',str(len(body)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
        class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):daemon_threads=True
        server=Server(str(self.root/'docker.sock'),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            p=subprocess.run([SHIM,'--schema'],cwd=self.root,env=dict(os.environ,DOCKER_HOST='unix://'+str(self.root/'docker.sock')),capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,8,p.stdout+p.stderr)
            self.assertFalse(any('/images/create' in path for path in calls),calls)
        finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
