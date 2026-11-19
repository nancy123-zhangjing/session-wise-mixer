"""Necessary transition check in the single allocated GPU; no optimizer steps."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
os.environ['DIFFGRM_ENGINE_ROOT']=str(ROOT/'code/external/DiffGRM')
import torch
from scripts.listwise_diffgrm.recovery_r17_release1 import build_g01,PARENT_SHA256
from scripts.listwise_diffgrm.recovery import fingerprint
inputs=json.loads((ROOT/'INPUTS.json').read_text())
assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
ctrl,_,_=build_g01(inputs['parent'],'cuda:0',inputs['dataset'],inputs['assets'])
parent=torch.load(inputs['parent'],map_location='cpu',weights_only=False)
saved=ctrl.checkpoint()
for key in ('model','alignment','r_optimizer','t_optimizer','scheduler','rng','r_updates','t_updates','precision'):
    assert fingerprint(parent['inner'][key])==fingerprint(saved['inner'][key]),key
mixer=ctrl.trainer.alignment.mixer
with torch.no_grad():
    effective,_=mixer.crossed_codebooks(False)
    expected=ctrl.release_anchor['C'].to(effective)
    error=float((effective-expected).abs().max())
    assert torch.allclose(effective,expected,atol=1e-7,rtol=0),error
    codes=ctrl.trainer.alignment.hard_codes.clone()
    ctrl.trainer.refresh_catalog()
    assert torch.equal(codes,ctrl.trainer.alignment.hard_codes)
ctrl.restore(saved)
assert fingerprint(saved)==fingerprint(ctrl.checkpoint())
result=dict(status='passed',parent_sha256=PARENT_SHA256,full_state_exact=True,initial_sid_exact=True,
            initial_codebook_max_abs_error=error,roundtrip_exact=True,R=17,T=160,gamma=1.0)
(ROOT/'GPU_GATE.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
