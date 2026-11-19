"""Check one completed new-D receipt/summary; does not replay predictions."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--epoch',type=int,required=True)
p.add_argument('--directory',type=Path,required=True)
a=p.parse_args()
assert a.epoch in (8,11,14,17,20,23,26,29,32,35)
read=lambda name:json.loads((a.directory/name).read_text(encoding='utf-8'))
record=read('record.json');summary=read('summary.json')
digest=hashlib.sha256((a.directory/'summary.json').read_bytes()).hexdigest()
assert digest==record['summary_sha256']
assert record['epoch']==a.epoch and record['kind']=='new'
root='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1/runs/H1-Spread200-R35'
for key,name in (('summary','summary.json'),('predictions','per_sample.jsonl.gz')):
    assert record[key]==f'{root}/evaluations/validation_R{a.epoch:02d}/attempt1/{name}'
expected=dict(split='validation',examples=71657,full_split=True,beam=32,
    inference_precision='fp32',architecture='H',seed=2024,
    r_updates=a.epoch*693,t_updates=(a.epoch-5)//3*20)
assert all(summary.get(k)==v for k,v in expected.items())
assert summary.get('subset') is None and summary.get('diagnostic_prefix_limit') is None
assert record['ndcg5']==summary['metrics']['list/ndcg@5']
result=dict(status='source_hash_and_metadata_checked_only',summary_sha256=digest,
    record_sha256=hashlib.sha256((a.directory/'record.json').read_bytes()).hexdigest(),
    expected=expected,production_metrics={k:summary['metrics'][k] for k in ('list/ndcg@5','list/hr@5','list/recall@5')},
    evaluation_seconds=summary['seconds'],
    independent_prediction_replay='pending final unified audit',not_goal_completion=True,
    check_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (a.directory/'SOURCE_CHECK.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
