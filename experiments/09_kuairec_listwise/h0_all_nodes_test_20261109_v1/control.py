import sys, json, base64, shlex
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'h1_no_code_tag_20261108_v1'))
import control as c
c.HERE = HERE
c.REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h0_all_nodes_test_20261109_v1'
c.JOB = 'nancy-s09-h0nodes-test-1109-v1'

def remote_python(code, receipt=None, timeout=120):
    encoded = base64.b64encode(code.encode()).decode()
    return c.remote(c.PYTHON + ' -B -c ' + shlex.quote(f'import base64;exec(base64.b64decode({encoded!r}))'), receipt, timeout)

def inventory():
    code = '''
from pathlib import Path
import json
root=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
checkpoints=[]
summaries=[]
for p in root.rglob('*.pt'):
    if 'H0' in str(p) and '/code/' not in str(p): checkpoints.append(dict(path=str(p),bytes=p.stat().st_size))
for p in root.rglob('summary.json'):
    if ('H0' in str(p) or 'h0_r14_test' in str(p)) and '/code/' not in str(p):
        try:
            v=json.loads(p.read_text())
            summaries.append(dict(path=str(p),split=v.get('split'),examples=v.get('examples'),r_updates=v.get('r_updates'),t_updates=v.get('t_updates'),beam=v.get('beam'),precision=v.get('inference_precision'),metrics=v.get('metrics')))
        except Exception: pass
print(json.dumps(dict(checkpoints=checkpoints,summaries=summaries)))
'''
    result=json.loads(remote_python(code, 'INVENTORY_READBACK.json'))
    c.write('INVENTORY.json',result)
    print(json.dumps(result,ensure_ascii=False))

def prepare():
    c.resources()
    code = '''
from pathlib import Path
import json,hashlib
root=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
out=root/'h0_all_nodes_test_20261109_v1'
out.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
pending=[]
for epoch,t,relative in [(5,0,'goal_20261027_r1/runs/H0/initial.pt'),(8,40,'h_freeze35_20261029_r1/runs/H0-Freeze20-R35/phase08_R_R08_T040.pt'),(11,80,'h_freeze35_20261029_r1/runs/H0-Freeze20-R35/phase12_R_R11_T080.pt'),(17,160,'h_freeze35_20261029_r1/runs/H0-Freeze20-R35/phase20_R_R17_T160.pt'),(20,200,'h_freeze35_20261029_r1/runs/H0-Freeze20-R35/phase24_R_R20_T200.pt'),(35,400,'goal_20261027_r1/runs/H0/last.pt')]:
    row=dict(epoch=epoch,t_updates=t,checkpoint=str(root/relative),sha256=sha(root/relative),source='original' if epoch in (5,35) else 'validated_replay')
    if epoch in (8,11,17,20):
        gate=root/'h_freeze35_20261029_r1/runs/H0-Freeze20-R35/replay'/f'R{epoch:02d}.json'
        v=json.loads(gate.read_text())
        if v['status']!='passed' or v['top_list_agreement']!=1.0: raise ValueError('Replay mismatch')
        row.update(replay_gate=str(gate),replay_gate_sha256=sha(gate))
    pending.append(row)
manifest=dict(pending=pending,reuse_epochs=[14,32],unavailable_epochs=[23,26,29],no_training=True)
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps(manifest))
'''
    c.write('MANIFEST.json',json.loads(remote_python(code,'PREPARE_READBACK.json')))
    c.cli(['notebook','scp','env-zzl',(HERE/'evaluate.py').as_posix(),c.REMOTE+'/evaluate.py','--workspace',c.CPU,'-t','120'],'UPLOAD.json')
    expected=c.digest(HERE/'evaluate.py')
    actual=remote_python(f"import hashlib;print(hashlib.sha256(open({(c.REMOTE+'/evaluate.py')!r},'rb').read()).hexdigest())")
    if actual.strip()!=expected: raise ValueError('Upload mismatch')
    print('Staged inference-only runner and six checkpoint hashes')

def submit():
    if (HERE/'SUBMISSION_INTENT.json').exists(): raise ValueError('Submission already attempted')
    if c.cli(['job','list','--workspace',c.WORKSPACE,'--keyword',c.JOB,'--all'],'JOB_ABSENCE.json').get('items'): raise ValueError('Job exists')
    script=f'cd {c.REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u evaluate.py > evaluation.log 2>&1'
    req=['job','create','--name',c.JOB,'--workspace',c.WORKSPACE,'--project','公共科研项目','--group',c.GROUP,'--quota','1,20,200','--image',c.IMAGE,'--nodes','1','--shm-size','32','--max-time','4','--priority','4','--no-auto-fault-tolerance','--no-enable-notification','--command',shlex.join(['bash','-c',script])]
    c.cli(req+['--dry-run'],'DRY_RUN.json')
    c.write('SUBMISSION_INTENT.json',dict(job=c.JOB,root=c.REMOTE,request=req))
    print(c.cli(req,'SUBMISSION_RESULT.json'))

def status():
    print(c.cli(['job','status',c.JOB,'--workspace',c.WORKSPACE],'PLATFORM_LATEST.json'),flush=True)
    print(c.remote(f'cd {c.REMOTE} && if test -f evaluation.log; then tail -n 4 evaluation.log; fi'),flush=True)

def accept():
    job=c.cli(['job','status',c.JOB,'--workspace',c.WORKSPACE],'PLATFORM_LATEST.json')
    if job['status']!='SUCCEEDED': raise ValueError('Not complete: '+job['status'])
    workers=c.cli(['job','instances',c.JOB,'--workspace',c.WORKSPACE],'TERMINAL_INSTANCES.json')['items']
    if len(workers)!=1 or workers[0]['status']!='instance_success': raise ValueError('GPU release unconfirmed')
    code=f'''
import json,sys,hashlib
from pathlib import Path
r=Path({c.REMOTE!r})
sys.path[:0]=[str(r.parent/'h_freeze35_20261029_r1/code'),str(r.parent/'h1_no_code_tag_20261108_v1')]
from audit_analysis_core import audit_predictions
v=json.loads((r/'RESULT.json').read_text())
for row in v['nodes']:
    a=audit_predictions(row['predictions'],dataset_dir=r.parent/'goal_20261027_r1/inputs/dataset')
    if not a['passed'] or not a['source_labels_verified'] or a['examples']!=53194: raise ValueError('Metric audit failed')
    for key in ('summary_sha256','predictions_sha256'):
        if a[key]!=row[key]: raise ValueError('Artifact changed')
    row['audit_passed']=True; row['max_absolute_error']=a['max_absolute_error']
v.update(audit_passed=True,gpu_released=True)
(r/'ACCEPTED_RESULT.json').write_text(json.dumps(v,indent=2)+'\\n')
print(json.dumps(v))
'''
    result=json.loads(remote_python(code,'ACCEPTANCE_READBACK.json',timeout=600))
    c.write('ACCEPTED_RESULT.json',result)
    print(json.dumps([dict(epoch=x['epoch'],ndcg5=x['metrics']['list/ndcg@5'],audit_passed=x['audit_passed']) for x in result['nodes']]))

def inventory_h1():
    code='''
import json
from pathlib import Path
r=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
replay=r/'h1_replay_r5_r20_20261105_r1'
files=[str(p) for p in replay.rglob('*.pt')]
verification=[dict(path=str(p),content=json.loads(p.read_text())) for p in replay.rglob('*VERIFICATION*.json')]
tests=[]
for p in r.rglob('summary.json'):
    if 'H1' in str(p) and '/code/' not in str(p):
        v=json.loads(p.read_text())
        if v.get('split')=='test': tests.append(dict(path=str(p),r=v.get('r_updates'),t=v.get('t_updates'),n=v.get('examples'),ndcg=v.get('metrics',{}).get('list/ndcg@5')))
print(json.dumps(dict(checkpoints=files,verification=verification,tests=tests)))
'''
    result=json.loads(remote_python(code,'H1_INVENTORY_READBACK.json'))
    c.write('H1_INVENTORY.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    globals()[sys.argv[1] if len(sys.argv)>1 else 'inventory']()
