"""Read-only probe: locate H0/H1 R35/best test+validation per-sample predictions for 3 seeds."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'session_wise_ar/cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

SRC = r'''
import gzip, json
from pathlib import Path
B = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
roots = ['h_freeze35_20261029_r1/runs', 'multiseed_freeze_20261109_v1', 'h0_multiseed_r20_freeze_20261110_v1', 'goal_20261027_r1/runs']
for r in roots:
    base = B / r
    if not base.exists(): print('MISSING', r); continue
    hits = []
    for p in base.rglob('*.jsonl.gz'):
        s = str(p.relative_to(B))
        if 'R35' in s or 'best' in s or 'R32' in s or 'validation' in s or 'test' in s:
            hits.append((s, p.stat().st_size))
    for s, z in sorted(hits)[:80]: print('F', s, z)
    print('COUNT', r, len(hits))
p = next((B / 'h0_multiseed_r20_freeze_20261110_v1').rglob('test_R35/per_sample.jsonl.gz'))
with gzip.open(p, 'rt') as f: row = json.loads(f.readline())
print('KEYS', list(row.keys())); print('ROW', json.dumps(row)[:700])
'''
print(jexec.python(SRC, timeout=600))
