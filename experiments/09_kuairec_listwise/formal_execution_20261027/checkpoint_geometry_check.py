"""Read-only CPU check of saved SID counts and learned effective codebooks."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).absolute().parents[3]
RESTORE=ROOT/'outputs/stage09_restore/goal_20261027_r1'
HERE=Path(__file__).absolute().parent
FREEZE=HERE/'FOUR_BEST_FREEZE_20261028.json'
OUT=HERE/'CHECKPOINT_GEOMETRY_CHECK_20261028.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not OUT.exists()
    assert sha(FREEZE)=='4d0a0a17be2fd087db30c3dbbfe815a1d9243983619e7c8d47f1ca6fcd681ae2'
    freeze=json.loads(FREEZE.read_text(encoding='utf-8'))
    diag_path=HERE/'CHECKPOINT_SID_DIAGNOSTICS_20261028.json'
    assert sha(diag_path)=='fcdfc6d26caa20ecbd164fe42dfc4c12955fd8bda6a413fba6b2dd42136c7d77'
    diag=json.loads(diag_path.read_text(encoding='utf-8'))
    helper=ROOT/'outputs/stage09_delivery/goal_20261027_r1/local_cpu_restore_acceptance.py'
    assert sha(helper)=='52339717ac6971e4b8f4041119a3c4c81197d78374fc7897e06cfef659444850'
    spec=importlib.util.spec_from_file_location('verified_restore_helper',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    source=RESTORE/'code'
    first=torch.load(RESTORE/'runs/H1/best.pt',map_location='cpu',weights_only=False)
    engine=h.verify_source_files(source,first['binding'])
    os.environ['DIFFGRM_ENGINE_ROOT']=str(engine);sys.path.insert(0,str(source))
    assets_mod=importlib.import_module('scripts.listwise_diffgrm.assets')
    align_mod=importlib.import_module('scripts.listwise_diffgrm.alignment')
    assets=assets_mod.load_assets(RESTORE/'assets/pse_v1',dataset_dir=ROOT/'outputs/kuairec_listwise_v1/run1')
    rows={}
    for arm in ('J0','J1','H0','H1'):
        rows[arm]={};sids={}
        for role in ('best','last'):
            p=RESTORE/'runs'/arm/(role+'.pt');expected=freeze['arms'][arm][role]['sha256']
            assert sha(p)==expected
            st=torch.load(p,map_location='cpu',weights_only=False)
            assert st['arm']==arm and st['binding']['code']==first['binding']['code'] and st['binding']['engine']==first['binding']['engine']
            sid=st['alignment']['hard_codes'][1:].clone();sids[role]=sid
            counts=Counter(map(tuple,sid.tolist()))
            coll=sum(n for n in counts.values() if n>1)
            reference=diag['arms'][arm][role]['sid']['collision']
            assert len(counts)==reference['unique_sids'] and coll==reference['collided_items']
            a=align_mod.build_alignment(assets,arm.endswith('1'),config_overrides=st['binding']['config']['alignment'])
            a.load_state_dict(st['alignment'],strict=True);a.eval();a.set_recommender_cache(False)
            with torch.no_grad():
                books,_=a.mixer.crossed_codebooks(False);base=a.mixer.base_codebooks;initial=a.mixer.initial_codebooks
                assert torch.isfinite(books).all()
                predicted=[];predicted64=[];sse=0.
                for start in range(1,len(a.item_slots),512):
                    slots=a.item_slots[start:start+512]
                    predicted.append(a.mixer.squared_distances(slots.float(),books.float()).argmin(-1))
                    # Float64 direct squared differences, independent of the training distance expansion.
                    distances=(slots.double()[:,:,None,:]-books.double()[None,:,:,:]).square().sum(-1)
                    predicted64.append(distances.argmin(-1))
                    sse+=distances.min(-1).values.sum().item()
                p32=torch.cat(predicted);p64=torch.cat(predicted64)
                c64=Counter(map(tuple,p64.tolist()))
                norm=base.norm().item();e0=initial.norm().item()
                rows[arm][role]=dict(checkpoint_sha256=expected,
                    unique_saved_sids=len(counts),collided_saved_items=coll,collision_rate=coll/len(sid),
                    independent_python_counter_matches=True,
                    cpu_fp32_recomputed_sid_item_mismatches=int(sid.ne(p32).any(-1).sum()),
                    cpu_fp64_direct_distance_sid_item_mismatches=int(sid.ne(p64).any(-1).sum()),
                    cpu_fp64_unique_sids=len(c64),
                    cpu_fp64_collision_rate=sum(n for n in c64.values() if n>1)/len(sid),
                    used_codes_per_slot=[len(set(sid[:,i].tolist())) for i in range(4)],
                    mixer_delta_over_base_global_norm=(books-base).norm().item()/norm,
                    effective_displacement_over_initial_global_norm=(books-initial).norm().item()/e0,
                    base_displacement_over_initial_global_norm=(base-initial).norm().item()/e0,
                    effective_codebook_norm=books.norm().item(),base_codebook_norm=norm,
                    mean_nearest_reconstruction_sse=sse/len(sid))
            assert sha(p)==expected
            del a,st
        changed=sids['best'].ne(sids['last'])
        rows[arm]['best_to_last_sid_change']=dict(item_rate=changed.any(-1).double().mean().item(),digit_rate=changed.double().mean().item())
    result=dict(schema='stage09_checkpoint_geometry_readonly_v1',status='PASS',created_at_utc=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha(FREEZE),sid_diagnostics_sha256=sha(diag_path),script_sha256=sha(Path(__file__)),
        gpu_used=False,model_weights_changed=False,checkpoint_written=False,labels_used=False,arms=rows,
        limitation='CPU endpoint diagnostics, not a historical per-T trajectory or causal identification. Global norm ratios are diagnostics, not proof of gradient dominance.')
    with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',output=str(OUT),sha256=sha(OUT),arms=rows)),flush=True)

if __name__=='__main__':main()
