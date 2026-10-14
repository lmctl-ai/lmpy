import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

shim = pathlib.Path(sys.argv[1]).resolve()
home = pathlib.Path.home().resolve()
with tempfile.TemporaryDirectory(prefix="pycli-shim-", dir=home) as td:
    root = pathlib.Path(td)
    fake = root / "fakebin"
    fake.mkdir()
    host_tmp = root / "host-tmp"
    host_tmp.mkdir()
    args_file = root / "args.txt"
    block_file = root / "block-started"
    docker = fake / "docker"
    docker.write_text("""#!/bin/sh
printf '%s\\n' "$@" >"$ARGS_FILE"
case "${FAKE_DOCKER_CASE:-ok}" in
  ok) echo '{"success":true,"result":{}}'; exit 0;;
  daemon) echo 'docker: Cannot connect to the Docker daemon' >&2; exit 125;;
  partial) echo '{'; exit 143;;
  block) : >"$BLOCK_FILE"; while :; do sleep 1; done;;
  crash) exit "${FAKE_DOCKER_RC:-139}";;
esac
""")
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=str(fake) + os.pathsep + env["PATH"], ARGS_FILE=str(args_file),
               FAKE_DOCKER_CASE="ok", PYCLI_TIMEOUT_S="15", PYCLI_SECRET="allowed-prefix",
               TMPDIR=str(host_tmp), BLOCK_FILE=str(block_file),
               SECRET_SHIM_TEST="must-not-pass")
    colon = root / "work:area"
    colon.mkdir()
    p = subprocess.run([str(shim), "--schema"], cwd=colon, env=env, text=True, capture_output=True)
    assert p.returncode == 0 and json.loads(p.stdout)["success"] is True, (p.returncode, p.stdout, p.stderr)
    argv = args_file.read_text().splitlines()
    assert argv[argv.index("--mount") + 1] == f"type=bind,src={colon},dst={colon}", argv
    assert "--network=none" in argv and "--cpus=2" in argv and "--memory=4g" in argv
    assert all(flag in argv for flag in ("--read-only", "--cap-drop=ALL", "--pids-limit=128", "--security-opt=no-new-privileges"))
    assert "-v" not in argv and "TMPDIR=" + str(colon / ".pycli-tmp") in argv
    assert "PYCLI_TIMEOUT_S" in argv and "PYCLI_SECRET" not in argv and "SECRET_SHIM_TEST" not in argv
    assert not (colon / ".pycli-tmp").exists()

    env["FAKE_DOCKER_CASE"] = "block"
    blocked = subprocess.Popen([str(shim), "--schema"], cwd=colon, env=env,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    deadline = time.monotonic() + 5
    while not block_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert block_file.exists(), "mock docker did not start"
    assert list(host_tmp.iterdir()), "shim did not create output captures"
    os.killpg(blocked.pid, signal.SIGTERM)
    blocked.communicate(timeout=3)
    assert blocked.returncode == 143, blocked.returncode
    assert not list(host_tmp.iterdir()), "signal trap leaked host capture files"
    assert not (colon / ".pycli-tmp").exists(), "signal trap leaked container temp directory"

    for case, rc, code in (("crash", 1, "INTERNAL_ERROR"), ("partial", 1, "INTERNAL_ERROR"), ("daemon", 8, "SERVICE_UNAVAILABLE")):
        env["FAKE_DOCKER_CASE"] = case
        env["FAKE_DOCKER_RC"] = "139"
        p = subprocess.run([str(shim), "--schema"], cwd=colon, env=env, text=True, capture_output=True)
        body = json.loads(p.stdout)
        assert p.returncode == rc and body["error"]["code"] == code, (case, p.returncode, p.stdout, p.stderr)

    for unsafe in (pathlib.Path("/"), home, pathlib.Path(tempfile.gettempdir())):
        p = subprocess.run([str(shim), "--schema"], cwd=unsafe, env=env, text=True, capture_output=True)
        assert p.returncode == 2 and json.loads(p.stdout)["error"]["code"] == "INVALID_REQUEST", (unsafe, p.returncode, p.stdout)
print("pycli shim boundary, docker arguments, env filtering, cleanup and failure mapping passed")
