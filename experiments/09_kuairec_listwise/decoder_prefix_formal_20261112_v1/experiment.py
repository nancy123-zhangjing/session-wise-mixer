"""Original runner, model-specific warmup, immutable validation checkpoints, stop at R20."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
os.environ['DIFFGRM_ENGINE_ROOT'] = str(CODE / 'external/DiffGRM')
sys.path.insert(0, str(CODE))
VARIANTS = {'H0-A':('H0','asymmetric'), 'H1-A':('H1','asymmetric'),
            'H0-B':('H0','cross_attention'), 'H1-B':('H1','cross_attention')}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def config(variant):
    arm,mode=VARIANTS[variant]
    cfg=json.loads((CODE/f'configs/09_kuairec_listwise/{arm.lower()}_prefix_{mode}.json').read_text())
    original=ROOT.parent/'goal_20261027_r1'
    cfg['dataset']=str(original/'inputs/dataset')
    cfg['assets']=str(original/'assets/pse_v1')
    return cfg

def verify_source():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if sha(CODE/name)!=digest: raise ValueError('Source changed: '+name)

def prepare_cache():
    import torch
    from scripts.listwise_diffgrm.run import build_context
    from scripts.listwise_diffgrm.execution import TrainingTensorCache
    verify_source()
    _,_,data,binding=build_context(config('H0-A'),'cpu','bf16',False,prepare_training=False)
    cache=TrainingTensorCache(data)
    path=ROOT/'training_cache.pt'
    torch.save({'cache':cache,'count':len(data),'dataset_binding':{'data':binding['data'],'asset_manifest':binding['asset_manifest']}},path)
    write(ROOT/'CACHE_READY.json',dict(count=len(data),sha256=sha(path)))
    print('CACHE_READY',flush=True)

def run(variant):
    import torch
    from scripts.listwise_diffgrm import run as runner
    from scripts.listwise_diffgrm.training import phase_plan, atomic_save
    from scripts.listwise_diffgrm.recovery import fingerprint
    verify_source()
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    arm,mode=VARIANTS[variant];cfg=config(variant)
    out=ROOT/variant;out.mkdir(exist_ok=True)
    ready=json.loads((ROOT/'CACHE_READY.json').read_text())
    assert sha(ROOT/'training_cache.pt')==ready['sha256']
    payload=torch.load(ROOT/'training_cache.pt',map_location='cpu',weights_only=False)
    cache=payload['cache'];assert cache.count==709443==ready['count']
    original_context=runner.build_context
    def context(*args,**kwargs):
        kwargs['prepare_training']=False
        result=original_context(*args,**kwargs)
        assert len(result[2])==cache.count
        binding=result[3]
        assert ({'data':binding['data'],'asset_manifest':binding['asset_manifest']})==payload['dataset_binding']
        result[2]._training_tensor_cache=cache
        return result
    runner.build_context=context
    plan=phase_plan(cfg,cache.count)
    assert len(plan)==45 and plan[24]['epoch']==20 and plan[24]['validate']
    original_publish=runner.publish_checkpoint
    class StopR20(Exception): pass
    def publish(output,state,*,save_best):
        original_publish(output,state,save_best=save_best)
        row=plan[state['progress']['phase']-1]
        if row['validate']:
            folder=Path(output)/'checkpoints';folder.mkdir(exist_ok=True)
            path=folder/f"R{row['epoch']:02d}_T{state['t_updates']:03d}.pt"
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=False)
                assert fingerprint(old)==fingerprint(state)
            else: atomic_save(path,state)
            write(folder/f"R{row['epoch']:02d}.json",dict(epoch=row['epoch'],T=state['t_updates'],checkpoint=str(path),sha256=sha(path),validation=state['progress']['validations'][-1]))
            print(json.dumps(dict(event='validation_complete',variant=variant,R=row['epoch'],T=state['t_updates'],best_score=state['progress']['best_score'])),flush=True)
            if row['epoch']==5:
                pair={k:fingerprint(state[k]) for k in ('model','alignment','r_optimizer','t_optimizer','scheduler','rng')}
                write(out/'R5_PAIR.json',pair)
                other=ROOT/(('H1' if arm=='H0' else 'H0')+'-'+variant[-1])/'R5_PAIR.json'
                if other.exists(): assert json.loads(other.read_text())==pair,'Paired R5 differs'
        if row['kind']=='R' and row['epoch']==20:
            assert state['r_updates']==13860 and state['t_updates']==200
            write(out/'TRAINING_COMPLETE.json',dict(R=20,T=200,best_epoch=state['progress']['best_epoch'],best_score=state['progress']['best_score']))
            raise StopR20()
    runner.publish_checkpoint=publish
    warm=out/'warmup';train=out/'train'
    if not (warm/'complete.json').exists():
        runner.train(SimpleNamespace(command='warmup',output=str(warm),resume=str(warm/'last.pt') if (warm/'last.pt').exists() else None,device='cuda:0',precision='bf16'),cfg)
    if not (out/'TRAINING_COMPLETE.json').exists():
        # A crash after publishing the R20 cursor must not enter T201.
        if (train/'last.pt').exists():
            last=torch.load(train/'last.pt',map_location='cpu',weights_only=False)
            if last['progress']['phase']==25:
                write(out/'TRAINING_COMPLETE.json',dict(R=20,T=200,best_epoch=last['progress']['best_epoch'],best_score=last['progress']['best_score']))
        if not (out/'TRAINING_COMPLETE.json').exists():
            try:
                runner.train(SimpleNamespace(command='train',arm=arm,output=str(train),resume=str(train/'last.pt') if (train/'last.pt').exists() else None,shared_warmup=str(warm/'last.pt'),device='cuda:0',precision='bf16'),cfg)
            except StopR20: pass
    # Stop at the requested research boundary. Testing uses validation-selected best and R20.
    receipts=[]
    for label in ('best','R20'):
        cp=train/('best.pt' if label=='best' else 'last.pt');dest=out/f'test_{label}'
        if not (dest/'summary.json').exists():
            retry=0
            while dest.exists():
                retry+=1;dest=out/f'test_{label}_retry{retry}'
            state=torch.load(cp,map_location='cpu',weights_only=False)
            trainer,assets,_,binding=original_context(cfg,'cuda:0','bf16',arm=='H1',prepare_training=False)
            runner.validate_evaluation_checkpoint(state,trainer.plan)
            trainer.restore(state,binding,arm=arm)
            with torch.autocast('cuda',enabled=False):
                runner.evaluate(trainer,assets,split='test',destination=dest,beam=32)
            del trainer,assets,state
        receipts.append(dict(label=label,checkpoint=str(cp),sha256=sha(cp),summary=str(dest/'summary.json')))
    write(out/'GPU_COMPLETE.json',dict(variant=variant,status='passed',tests=receipts))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.command=='prepare': prepare_cache()
    else: run(a.variant)
