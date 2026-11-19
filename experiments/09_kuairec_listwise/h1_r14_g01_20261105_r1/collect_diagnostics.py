import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from audit_analysis_core import _rows,recompute_row
read=lambda p:json.loads(p.read_text())
manifest=read(ROOT/'validation_subset_8192.json')
wanted={r['sample_id'] for r in manifest['rows']}
path=ROOT.parent/'goal_20261027_r1/runs/H1/validation_R14/per_sample.jsonl.gz'
values={r['sample_id']:recompute_row(r,catalog_size=10728)['list/ndcg@5'] for r in _rows(path) if r['sample_id'] in wanted}
assert set(values)==wanted and len(values)==8192
out={'R14_before_T160':sum(values.values())/8192}
for label in ('after_T160','R15','R16','R17','after_T200','R18','R19','R20'):
    audit=read(ROOT/'acceptance'/f'{label}_audit.json')
    subset=audit['diagnostic_subset']
    out[label]=(subset['metrics'] if subset else audit['metrics'])['list/ndcg@5']
(ROOT/'acceptance/DIAGNOSTICS.json').write_text(json.dumps({'examples':8192,'same_subset':True,'ndcg5':out},indent=2))
print(json.dumps(out))
