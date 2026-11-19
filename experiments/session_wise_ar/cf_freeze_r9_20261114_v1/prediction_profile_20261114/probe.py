"""Read-only probe for steps 2-4: dataset, checkpoint catalog keys, TIGER prediction format."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'paired_bootstrap_20261114'))
import jexec  # noqa: E402

BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
SOURCE = f'''
import json, gzip, glob, os
from pathlib import Path
B={BASE!r}
d=Path(B+'/goal_20261027_r1/inputs/dataset')
print('DATASET', sorted(p.name for p in d.iterdir())[:40] if d.exists() else 'missing')
t0=Path(B+'/session_wise_ar_five_20261113_r2/runs/AR-T0')
print('T0', sorted(str(p.relative_to(t0)) for p in t0.rglob('*') if p.suffix in ('.gz','.json') and 'checkpoint' not in str(p))[:60])
for f in sorted(t0.rglob('predictions*.gz'))[:1]:
    with gzip.open(f,'rt') as h: print('T0ROW', f, h.readline()[:800])
import torch
p=Path(B+'/cf_freeze_r9_20261114_v1/runs/M1/train/checkpoints/R35_T040.pt')
s=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
print('CKPT_KEYS', list(s.keys()))
for k,v in s.items():
    if isinstance(v,dict): print(' ',k,list(v.keys())[:30])
'''
print(jexec.python(SOURCE, timeout=300))
