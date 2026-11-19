"""Inspect preserved cancellation state and exact journal continuity on CPU."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'outputs/stage09_spread200_20261031/code'
OUT=HERE/'cancellation_r1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(CODE))
os.environ['DIFFGRM_ENGINE_ROOT']=str(CODE/'external/DiffGRM')
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(4)
from scripts.listwise_diffgrm.spread200 import build_spread,verify_binding
from scripts.listwise_diffgrm.recovery import fingerprint

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
expected={
 'last.pt':'bd28ba74e82a776a38e567d91c742dfccd1ca4591881151635a9da7efa5267fe',
 'training.jsonl':'348e7a914323ae4fa7e587d0578406c35357ab67c3c005e212a7c7eb65186b1e',
 'train.log':'2923823989eb6cbf665490a401b363a2a62e6f1afafb196fd2f53b231d7d42fb',
 'status.json':'05a57936efe04b10096b2dc50d246a0059ac737da9e8afdbf5bd34f441d4411c'}
assert all(sha(OUT/k)==v for k,v in expected.items())
manifest=read(CODE/'SPREAD200_EXECUTION_MANIFEST.json')
assert sha(CODE/'SPREAD200_EXECUTION_MANIFEST.json')==read(HERE/'EXECUTION_PACKAGE.json')['manifest_sha256']
assert all(sha(CODE/k)==v['sha256'] for k,v in manifest['files'].items())
parent=read(HERE/'PARENT_BINDING.json')
c,assets,data=build_spread(parent['local'],parent['sha256'],'cpu',str(ROOT/'outputs/kuairec_listwise_v1/run1'),str(ROOT/'outputs/stage09_restore/goal_20261027_r1/assets/pse_v1'),exact_rng=False)
state=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
verify_binding(state['parent_binding'],state['execution_binding'])
old=copy.deepcopy(state['execution_binding']);new=copy.deepcopy(c.execution_binding)
relocations={k:dict(saved=old['config'].pop(k),local=new['config'].pop(k)) for k in ('dataset','assets')}
assert old==new
source_fingerprint=fingerprint(state)
clone=copy.deepcopy(state)
clone['execution_binding']=copy.deepcopy(c.execution_binding)
clone['inner']['binding']=copy.deepcopy(c.execution_binding)
clone['protocol']['source_proof']['paths']=copy.deepcopy(c.protocol['source_proof']['paths'])
c.restore(clone,exact_rng=False)
restored=c.checkpoint()
equal={k:fingerprint(state['inner'][k])==fingerprint(restored['inner'][k]) for k in ('model','alignment','r_optimizer','t_optimizer','scheduler','r_updates','t_updates','progress')}
assert all(equal.values()) and c.inspection_only
assert fingerprint(state)==source_fingerprint and sha(OUT/'last.pt')==expected['last.pt']
lines=(OUT/'training.jsonl').read_bytes().splitlines(keepends=True)
prefix=read(HERE/'midpoint_R20/ACCEPTED.json')
assert hashlib.sha256(b''.join(lines[:10495])).hexdigest()==prefix['source_log_sha256']
assert len(lines)==13014
orders=read(HERE/'CPU_INPUT_ACCEPTANCE.json')
orders={x['phase']:x for x in orders['t_blocks']+orders['r_phases']}
r,t,phase,u,indices=13860,100,25,0,None
for line in lines[10495:]:
    row=json.loads(line);kind='T' if (phase-5)%4==0 else 'R';width=20 if kind=='T' else 693
    if indices is None:
        b=(phase-5)//4;source=5+4*(b//2) if kind=='T' else phase
        indices=np.random.default_rng(np.random.SeedSequence([2024,7919,source])).permutation(709443)
        if kind=='T':indices=indices[(b%2)*20480:((b%2)+1)*20480]
        assert hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()==orders[phase]['sha256']
    batch=indices[u*1024:(u+1)*1024]
    assert (row['phase'],row['phase_index'],row['update'],row['lists'])==(kind,phase,u+1,len(batch))
    assert row['sample_indices_sha256']==hashlib.sha256(batch.astype('<i8',copy=False).tobytes()).hexdigest()
    noise=row['noise_used'];lr=row['learning_rates_used']
    assert (noise['optimizer_step'],noise['total_optimizer_steps'],noise['softmax_temperature'])==(t,400,1)
    assert abs(noise['gumbel_noise_scale']-(.2+.4*(1+math.cos(math.pi*t/400))))<1e-14
    assert (lr['codebook'],lr['mixer'])==(1e-5,1e-4)
    assert abs(lr['R']-(.001*.5*(1+math.cos(math.pi*(r-500)/(24255-500)))))<1e-14
    r+=kind=='R';t+=kind=='T';u+=1
    assert (row['r_updates'],row['t_updates'])==(r,t)
    if u==width:phase+=1;u=0;indices=None
assert (r,t,phase,u)==(16339,140,30,400)
assert (state['inner']['r_updates'],state['inner']['t_updates'])==(r,t)
assert state['inner']['progress']==dict(phase=30,update=400,boundary_prepared=False,rng_prepared=True)
assert state['inner']['rng']['cuda'] is not None and state['t_rng'] is not None
result=dict(status='passed_CPU_restore_and_journal_alignment',checkpoint_sha256=expected['last.pt'],
    source_hashes=expected,r_updates=r,t_updates=t,progress=state['inner']['progress'],
    journal_rows=len(lines),cached_prefix_rows=10495,new_tail_rows_verified=2519,
    journal_exactly_matches_checkpoint=True,replayed_rows_required=0,
    restored_equal=equal,source_fingerprint=source_fingerprint,dependency_path_relocation=relocations,
    cuda_rng_preserved_but_not_applied_on_CPU=True,paired_T_rng_present=True,
    script_sha256=sha(Path(__file__)),GPU_resume_not_performed=True,
    limitation='Preserved checkpoint CPU restoration; final fresh-archive cold restore and new CUDA continuation remain separate requirements.')
with (OUT/'RESTORE_READY.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result))
