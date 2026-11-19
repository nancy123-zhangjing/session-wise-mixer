"""One additional H100 for original H1 missing-node test inference."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'h0_all_nodes_test_20261109_v1'))
import importlib.util
spec=importlib.util.spec_from_file_location('node_control',HERE.parent/'h0_all_nodes_test_20261109_v1/control.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
c=base.c
c.HERE=HERE; base.HERE=HERE
c.REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_all_nodes_test_20261109_v1'
c.JOB='nancy-s09-h1nodes-test-1109-v1'

def prepare():
    c.resources()
    code='''
from pathlib import Path
import json,hashlib,sys,torch
r=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
out=r/'h1_all_nodes_test_20261109_v1';out.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(r/'h_freeze35_20261029_r1/code'))
from scripts.listwise_diffgrm.recovery import fingerprint
h0=torch.load(r/'goal_20261027_r1/runs/H0/initial.pt',map_location='cpu',weights_only=False)
h1=torch.load(r/'goal_20261027_r1/runs/H1/initial.pt',map_location='cpu',weights_only=False)
same_model=fingerprint(h0['model'])==fingerprint(h1['model'])
same_codes=torch.equal(h0['alignment']['hard_codes'],h1['alignment']['hard_codes'])
if not same_model or not same_codes: raise ValueError('R5 cannot share inference')
verification=r/'h1_replay_r5_r20_20261105_r1/REPLAY_VERIFICATION.json'
v=json.loads(verification.read_text())
if not v['exact_replay_verified'] or v['r20_state_differences']: raise ValueError('H1 replay unverified')
pending=[]
for epoch,t in [(8,40),(11,80),(14,120),(17,160),(35,400)]:
    p=(r/'goal_20261027_r1/runs/H1/last.pt') if epoch==35 else r/'h1_replay_r5_r20_20261105_r1/runs/H1-Replay-R20/checkpoints'/f'R{epoch:02d}_T{t:03d}.pt'
    digest=sha(p)
    if epoch!=35:
        gate=next(b for b in v['boundaries'] if b['epoch']==epoch)
        if gate['checkpoint_sha256']!=digest or not gate['predictions']['exact']: raise ValueError('Replay hash mismatch')
    pending.append(dict(epoch=epoch,t_updates=t,checkpoint=str(p),sha256=digest,source='original' if epoch==35 else 'exact_replay'))
manifest=dict(arm='H1',pending=pending,reuse_epochs=[5,20],unavailable_epochs=[23,26,29,32],no_training=True,shared_R5=dict(model_equal=same_model,hard_codes_equal=same_codes,h0_sha256=sha(r/'goal_20261027_r1/runs/H0/initial.pt'),h1_sha256=sha(r/'goal_20261027_r1/runs/H1/initial.pt')),replay_verification_sha256=sha(verification))
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps(manifest))
'''
    c.write('MANIFEST.json',json.loads(base.remote_python(code,'PREPARE_READBACK.json')))
    c.cli(['notebook','scp','env-zzl',(HERE/'evaluate.py').as_posix(),c.REMOTE+'/evaluate.py','--workspace',c.CPU,'-t','120'],'UPLOAD.json')
    expected=c.digest(HERE/'evaluate.py')
    actual=base.remote_python(f"import hashlib;print(hashlib.sha256(open({(c.REMOTE+'/evaluate.py')!r},'rb').read()).hexdigest())")
    if actual.strip()!=expected: raise ValueError('Upload mismatch')
    print('H1 five-node runner staged; R5 identical recommender and SID confirmed')

def verify_shared():
    code='''
import torch,json
from pathlib import Path
r=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
a=torch.load(r/'goal_20261027_r1/runs/H0/initial.pt',map_location='cpu',weights_only=False)['alignment']
b=torch.load(r/'goal_20261027_r1/runs/H1/initial.pt',map_location='cpu',weights_only=False)['alignment']
checks={k:torch.equal(a[k],b[k]) for k in ('mixer.base_codebooks','item_slots','hard_codes')}
checks.update({k:torch.count_nonzero(b[k]).item()==0 for k in ('mixer.attention.out_proj.weight','mixer.attention.out_proj.bias')})
if not all(checks.values()): raise ValueError('Shared R5 catalog not identical')
p=r/'h1_all_nodes_test_20261109_v1/MANIFEST.json'
m=json.loads(p.read_text());m['shared_R5']['catalog_checks']=checks
p.write_text(json.dumps(m,indent=2)+'\\n')
print(json.dumps(m))
'''
    c.write('MANIFEST.json',json.loads(base.remote_python(code,'SHARED_R5_READBACK.json')))
    print('R5 effective codebooks, collision ordering inputs and model verified identical')

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare': prepare()
    elif action=='verify_shared': verify_shared()
    else: getattr(base,action)()
