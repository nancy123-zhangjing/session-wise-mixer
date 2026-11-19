"""Post-training CPU artifact audit. Never trains or runs dataset inference."""
import copy, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent/'goal_20261027_r1'
CODE=ORIGINAL/'code'
os.environ['DIFFGRM_ENGINE_ROOT']=str(CODE/'external/DiffGRM')
sys.path.insert(0,str(CODE))
import torch
from scripts.listwise_diffgrm.assets import load_assets
from scripts.listwise_diffgrm.runtime import build_model
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.training import AlternatingTrainer, sha256
from protocol import GridController, PARENT_SHA256, GRID_PREFIX, fingerprint

def check(condition,message):
    if not condition: raise ValueError(message)

def main():
    torch.set_num_threads(4)
    parent_path=ORIGINAL/'runs/H1/initial.pt'
    check(sha256(parent_path)==PARENT_SHA256,'Parent hash changed')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    for name,digest in parent['binding']['code'].items():
        check(sha256(CODE/name)==digest,'Historical training code changed: '+name)
    for name,digest in parent['binding']['engine'].items():
        check(sha256(CODE/'external/DiffGRM'/name)==digest,'Historical engine changed: '+name)
    check(sha256(ORIGINAL/'assets/pse_v1/manifest.json')==parent['binding']['asset_manifest'],'Asset manifest differs from parent')
    assets=load_assets(ORIGINAL/'assets/pse_v1')
    all_results={}
    for arm in ('B','C'):
        run=ROOT/'runs'/arm
        done=json.loads((run/'GPU_COMPLETE.json').read_text())
        check(done['status']=='passed' and done['arm']==arm,'GPU completion missing')
        spec=json.loads((run/'experiment_spec.json').read_text())
        for name,digest in spec['extension_sha256'].items():
            check(sha256(ROOT/name)==digest,'Training source changed: '+name)
        check(sha256(ROOT/f'config_{arm}.json')==spec['config_sha256'],'Config changed')
        cfg=json.loads((ROOT/f'config_{arm}.json').read_text())['base_config']
        trainer=AlternatingTrainer(build_model(cfg,assets.make_tokenizer()),
            build_alignment(assets,True,config_overrides=cfg['alignment']),cfg,709443,precision='bf16')
        binding=json.loads((run/'binding.json').read_text())
        control=GridController(trainer,parent,binding,spec,exact_rng=False)
        validations=[]; files={}; freeze_hash=None; grid_at_freeze=None; state_hashes={}
        for epoch in (8,11,14,17,20,23,26,29,32,35):
            receipt=json.loads((run/'checkpoints'/f'R{epoch:02d}.json').read_text())
            path=Path(receipt['checkpoint'])
            digest=sha256(path)
            check(digest==receipt['sha256'],'Checkpoint hash mismatch')
            state=torch.load(path,map_location='cpu',weights_only=False)
            expected_t=min((epoch-5)//3*40,200)
            check((state['r_updates'],state['t_updates'])==(epoch*693,expected_t),'Wrong R/T budget')
            check(state['progress']['update']==0,'Incomplete node')
            check(state['rng']['cuda'] is not None and len(state['rng']['cuda'])==1,'Missing single-card CUDA RNG')
            # CPU loading proves structural recoverability, not CUDA next-step equality.
            restored=copy.deepcopy(state); restored['rng']['cuda']=None
            control.restore(restored)
            for optimizer in (trainer.r_optimizer,trainer.t_optimizer):
                for parameter,entry in optimizer.state.items():
                    for key in ('exp_avg','exp_avg_sq'):
                        check(entry[key].shape==parameter.shape,'Adam tensor shape mismatch')
            grid_hash=fingerprint({k:v for k,v in state['model'].items() if k.startswith(GRID_PREFIX)})
            if epoch==20:
                freeze_hash=state['freeze_hash'];grid_at_freeze=grid_hash
            if epoch>=20:
                check(freeze_hash is not None and state['freeze_hash']==freeze_hash,'Tokenizer freeze drift')
            if epoch==35:
                check(grid_hash!=grid_at_freeze,'Grid did not learn after R20')
            directory=Path(receipt['validation']);summary_path=directory/'summary.json'
            summary=json.loads(summary_path.read_text())
            check(summary['split']=='validation' and summary['full_split'] and summary['examples']==71657,'Validation scope wrong')
            check(summary['beam']==32 and summary['inference_precision']=='fp32','Validation protocol wrong')
            check(summary['r_updates']==epoch*693 and summary['t_updates']==expected_t,'Validation/checkpoint mismatch')
            check(summary['metrics']['list/ndcg@5']==receipt['ndcg5'],'Validation score mismatch')
            validations.append(dict(epoch=epoch,T=expected_t,metrics=summary['metrics']))
            state_hashes[epoch]=fingerprint(dict(model=state['model'],alignment=state['alignment']))
            files[str(path)]=digest
            for p in (summary_path,directory/'per_sample.jsonl.gz'):
                files[str(p)]=sha256(p)
            del state,restored
        # Include the inherited R5 candidate when checking selection.
        candidates=[dict(epoch=5,score=parent['progress']['best_score'])]
        candidates += [dict(epoch=r['epoch'],score=r['metrics']['list/ndcg@5']) for r in validations]
        chosen=max(candidates,key=lambda r:(r['score'],-r['epoch']))
        best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        check(best['progress']['best_epoch']==chosen['epoch'],'Wrong validation-selected best')
        check(last['r_updates']==24255 and last['t_updates']==200 and last['progress']['phase']==45,'Final cursor wrong')
        for label,state,epoch in [('best',best,chosen['epoch']),('R35',last,35)]:
            if epoch!=5:
                check(fingerprint(dict(model=state['model'],alignment=state['alignment']))==state_hashes[epoch],'Selected checkpoint differs from saved node')
            files[str(run/(label+'.pt' if label=='best' else 'last.pt'))]=sha256(run/(label+'.pt' if label=='best' else 'last.pt'))
        tests=json.loads((run/'TEST_RESULTS.json').read_text())
        check({r['label'] for r in tests}=={'best','R35'},'Missing test endpoint')
        for row in tests:
            if row.get('reused'):
                check(chosen['epoch']==35 and row['reused']=='best','Invalid test reuse');continue
            checkpoint=run/('best.pt' if row['label']=='best' else 'last.pt')
            check(sha256(checkpoint)==row['checkpoint_sha256'],'Test checkpoint mismatch')
            for name in ('summary','predictions'):
                p=Path(row[name]);digest=sha256(p)
                check(digest==row[name+'_sha256'],'Test artifact changed')
                files[str(p)]=digest
            summary=json.loads(Path(row['summary']).read_text())
            epoch=chosen['epoch'] if row['label']=='best' else 35
            check(summary['split']=='test' and summary['full_split'] and summary['examples']==53194,'Test scope wrong')
            check(summary['beam']==32 and summary['inference_precision']=='fp32','Test protocol wrong')
            check(summary['t_updates']==min((epoch-5)//3*40,200),'Test T budget mismatch')
            check(summary['r_updates']==epoch*693 and summary['metrics']==row['metrics'],'Test summary mismatch')
        all_results[arm]=dict(status='passed',best_epoch=chosen['epoch'],validations=validations,tests=tests,
            freeze_hash=freeze_hash,grid_changed_after_freeze=True,artifact_sha256=files)
    report=dict(status='passed',scope='artifact_and_CPU_load_audit',cuda_resume_retested=False,
                historical_source_and_assets_verified=True,shared_R5_validation_ndcg5=parent['progress']['best_score'],
                resource_release_verified=False,arms=all_results)
    (ROOT/'FINAL_ARTIFACT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
