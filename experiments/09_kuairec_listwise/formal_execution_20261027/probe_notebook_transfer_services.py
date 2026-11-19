import contextlib, io, json, logging, shlex
from inspire import InspireClient
code = '''import json, socket, os, shutil
out = {"sshd_present": shutil.which("sshd") is not None,
"bootstrap_sentinel": os.path.isfile("/tmp/.inspire_rtunnel_bootstrap_v1"),
"bootstrap_kit": os.path.isdir("/inspire/hdd/global_public/inspire-skill-bootstrap/v1")}
for port in (22222, 31337):
    try:
        with socket.create_connection(("<ip>", port), timeout=2):
            out[str(port)] = "open"
    except OSError:
        out[str(port)] = "closed"
print(json.dumps(out))
'''
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    with InspireClient(account="sii-shared", allow_browser=False, timeout=15, operation_timeout=50) as client:
        result = client.notebooks.exec("nancy-stage09-prep-20261027-r1", workspace="CPU资源空间",
            command="python -B -c " + shlex.quote(code), transport="jupyter", timeout=35)
print(json.dumps({"returncode":result.returncode,"completed":result.completed,
                  "output":json.loads(result.stdout) if result.completed and result.returncode == 0 else None}))
