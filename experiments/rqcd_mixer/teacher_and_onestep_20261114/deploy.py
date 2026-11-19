"""Upload diagnose.py + launch.sh into a NEW remote directory (refuses to overwrite)."""
import base64
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'session_wise_ar/cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

ROOT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_teacher_onestep_20261114_v1'
SOURCE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1'
LAUNCH = f'''set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT={ROOT}
PY={jexec.PYTHON}
cd "$ROOT"
exec > >(tee -a diagnose.log) 2>&1
exec "$PY" -B "$ROOT/diagnose.py" --source {SOURCE} --output "$ROOT/results"
'''
files = {'diagnose.py': (HERE / 'diagnose.py').read_bytes(), 'launch.sh': LAUNCH.encode()}
payload = {k: base64.b64encode(v).decode() for k, v in files.items()}
src = f'''
import base64, hashlib
from pathlib import Path
r = Path({ROOT!r})
for name, data in {payload!r}.items():
    p = r / name
    raw = base64.b64decode(data)
    if p.exists():
        assert p.read_bytes() == raw, 'refuse to overwrite different ' + name
    else:
        r.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    print(name, hashlib.sha256(p.read_bytes()).hexdigest())
'''
print(jexec.python(src, timeout=120))
print('local', {k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
