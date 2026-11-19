"""Strict CUDA round-trip of both retained endpoint checkpoints, no updates."""
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
os.environ['DIFFGRM_ENGINE_ROOT']=str(ROOT/'code/external/DiffGRM')
import torch
from scripts.listwise_diffgrm.recovery_r14_g01 import build_g01
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.training import sha256
inputs=json.loads((ROOT/'INPUTS.json').read_text())
ctrl,_,_=build_g01(inputs['parent'],'cuda:0',inputs['dataset'],inputs['assets'])
out=ROOT/'runs/R14-G01'
records=[]
for epoch,expected_t in ((17,160),(20,200)):
    record=json.loads((out/f'validation_records/R{epoch}.json').read_text())
    path=Path(record['checkpoint'])
    assert sha256(path)==record['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    ctrl.restore(state)
    restored=ctrl.checkpoint()
    assert fingerprint(state)==fingerprint(restored),'Full restore mismatch'
    assert (ctrl.trainer.r_updates,ctrl.trainer.t_updates)==(epoch*693,expected_t)
    records.append(dict(epoch=epoch,checkpoint=str(path),sha256=sha256(path),full_cuda_restore_exact=True))
result=dict(status='passed',records=records,updates_performed=0)
(ROOT/'CHECKPOINT_RESTORE.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
