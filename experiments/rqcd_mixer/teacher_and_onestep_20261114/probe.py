"""Read-only probe for D1/D2: CF source tree, prepared assets, teacher files."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'session_wise_ar/cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

SRC = r'''
import json
from pathlib import Path
R = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1')
print('ROOT', sorted(p.name for p in R.iterdir()))
for sub in ('teacher', 'cf_prepared', 'cf'):
    d = R / sub
    print(sub.upper(), sorted((p.name, p.stat().st_size) for p in d.iterdir() if p.is_file())[:40])
    print(sub.upper()+'_DIRS', sorted(p.name for p in d.iterdir() if p.is_dir())[:40])
p = R / 'cf_prepared' / 'PREPARED.json'
if p.exists():
    m = json.loads(p.read_text()); print('PREP_KEYS', list(m.keys())); print('DATASET', m.get('dataset_path'))
for f in R.rglob('runtime.locked.json'): print('RUNTIME', f)
for f in (R / 'teacher').rglob('*.json'): print('TJSON', f, f.read_text()[:600])
'''
print(jexec.python(SRC, timeout=300))
