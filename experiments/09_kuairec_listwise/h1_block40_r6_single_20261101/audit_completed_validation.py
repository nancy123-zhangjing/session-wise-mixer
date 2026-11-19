"""Independently replay one completed new-E full validation; no inference."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
CORE_SHA='4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--epoch',type=int,required=True)
a=p.parse_args()
assert a.epoch in (8,11,14,17,20,23,26,29,32,35)
directory=HERE/('first_validation_R08' if a.epoch==8 else f'saved_validation_R{a.epoch:02d}')
assert not (directory/'audit.json').exists() and not (directory/'ACCEPTED.json').exists()
record=read(directory/'record.json');summary=read(directory/'summary.json')
task=read(directory/'task.json');done=read(directory/'task.done.json')
assert done['status']=='completed_readonly_task' and done['state_before']==done['state_after']
assert done['request_sha256']==sha(directory/'task.json')
assert done['checkpoint_sha256']==task['checkpoint_sha256']==record['checkpoint_sha256']
assert done['runtime_sha256']==read(HERE/'RUN_GATE.json')['runtime_sha256']
expected=dict(split='validation',examples=71657,full_split=True,beam=32,
    inference_precision='fp32',architecture='H',seed=2024,
    r_updates=a.epoch*693,t_updates=(1+(a.epoch-6)//6)*40)
assert record['epoch']==a.epoch and record['kind']=='new'
assert all(summary.get(k)==v for k,v in expected.items())
assert summary.get('subset') is None and summary.get('diagnostic_prefix_limit') is None
assert sha(directory/'summary.json')==record['summary_sha256']
assert sha(directory/'per_sample.jsonl.gz')==record['predictions_sha256']
assert sha(ROOT/'scripts/listwise_diffgrm/analysis.py')==CORE_SHA
from scripts.listwise_diffgrm.analysis import audit_predictions
result=audit_predictions(directory/'per_sample.jsonl.gz',dataset_dir=ROOT/'outputs/kuairec_listwise_v1/run1')
write(directory/'audit.json',result)
assert (result['passed'] and result['mismatches']==0 and result['source_labels_verified']
        and result['source_cold_groups_verified'] and result['examples']==71657)
assert result['predictions_sha256']==record['predictions_sha256']
assert result['summary_sha256']==record['summary_sha256']
accepted=dict(status='passed_one_completed_validation_only',epoch=a.epoch,
    summary_sha256=record['summary_sha256'],predictions_sha256=record['predictions_sha256'],
    audit_sha256=sha(directory/'audit.json'),independent_core_sha256=CORE_SHA,
    dataset_binding=summary['dataset_binding'],examples=result['examples'],users=result['users'],
    scope='Independent CPU metric replay; no training or decoding. Not endpoint/test/Goal acceptance.',
    wrapper_sha256=sha(Path(__file__)))
write(directory/'ACCEPTED.json',accepted)
print(json.dumps(dict(status=accepted['status'],epoch=a.epoch,examples=result['examples'],
    users=result['users'],mismatches=result['mismatches'],max_absolute_error=result['max_absolute_error'],
    ndcg5=result['metrics']['list/ndcg@5'],audit_sha256=accepted['audit_sha256'])))
