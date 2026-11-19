"""Upload the probe into a NEW remote dir and start it in the background (nohup + sentinel)."""
import base64
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

B = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/'
ROOT = B + 'tiger_pretrain_probe_20261114_v1'
CODE = B + 'session_wise_ar_five_20261113_r2'
EXPECTED_PARENT = '3404b75ed4d88d365be029da7b6498dd135be13283ed40360a78ac466693a8b3'
LAUNCH = f'''set -euo pipefail
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
export PYTHONPATH={CODE}/ar_deps:{CODE}
cd {ROOT}
grep -q {EXPECTED_PARENT} {CODE}/parents_devicefix/tiger/manifest.json
{jexec.PYTHON} -B {ROOT}/probe_pretrain.py --code-root {CODE} --output {ROOT}/results > probe.log 2>&1 && touch DONE || touch FAILED
'''
files = {'probe_pretrain.py': (HERE / 'probe_pretrain.py').read_bytes(), 'launch.sh': LAUNCH.encode()}
payload = {k: base64.b64encode(v).decode() for k, v in files.items()}
src = f'''
import base64, hashlib, subprocess
from pathlib import Path
r = Path({ROOT!r})
for name, data in {payload!r}.items():
    p = r / name; raw = base64.b64decode(data)
    if p.exists():
        assert p.read_bytes() == raw, 'refuse to overwrite ' + name
    else:
        r.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    print(name, hashlib.sha256(p.read_bytes()).hexdigest())
assert not (r / 'probe.log').exists(), 'already started'
subprocess.Popen(['nohup', 'bash', str(r / 'launch.sh')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
print('started')
'''
print(jexec.python(src, timeout=120))
print('local', {k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
