"""Inference only: fixed H0 validation nodes, no optimizer steps or SID refresh."""
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGE=ROOT.parent
CODE=STAGE/'h_freeze35_20261029_r1/code'
os.environ['DIFFGRM_ENGINE_ROOT']=str(CODE/'external/DiffGRM')
sys.path.insert(0,str(CODE))

def main():
    import torch
    from scripts.listwise_diffgrm.recovery35 import build_recovery35, verify_binding
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.recovery_diagnostics import diagnostic_context
    from scripts.listwise_diffgrm.run import evaluate
    from scripts.listwise_diffgrm.training import sha256, restore_rng
    if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('One H100 required')
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    controller,assets,_=build_recovery35(STAGE/'goal_20261027_r1/runs/H0/initial.pt','C','cuda:0')
    trainer=controller.trainer
    records=[]
    reference=json.loads((STAGE/'goal_20261027_r1/evaluations/H0_test_beam32/summary.json').read_text())
    for row in manifest['pending']:
        epoch=row['epoch']; checkpoint=Path(row['checkpoint'])
        if sha256(checkpoint)!=row['sha256']: raise ValueError('Checkpoint changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        inner=state.get('inner',state)
        if state.get('preflight_only') or inner.get('preflight_only'): raise ValueError('Tainted state')
        verify_binding(inner['binding'],controller.execution_binding)
        if (inner['r_updates'],inner['t_updates'])!=(epoch*693,row['t_updates']): raise ValueError('Wrong clocks')
        trainer.model.load_state_dict(inner['model'],strict=True)
        trainer.alignment.load_state_dict(inner['alignment'],strict=True)
        trainer.r_updates=inner['r_updates']; trainer.t_updates=inner['t_updates']
        restore_rng(inner['rng'])
        before=fingerprint(dict(model=trainer.model.state_dict(),alignment=trainer.alignment.state_dict()))
        dest=ROOT/'evaluations'/f'test_R{epoch:02d}'
        if dest.exists(): raise ValueError('Existing evaluation: inspect before retry')
        print(json.dumps(dict(event='node_start',epoch=epoch,checkpoint=str(checkpoint))),flush=True)
        with diagnostic_context(trainer),torch.autocast('cuda',enabled=False):
            evaluate(trainer,assets,split='test',destination=dest,beam=32)
        if fingerprint(dict(model=trainer.model.state_dict(),alignment=trainer.alignment.state_dict()))!=before:
            raise ValueError('Inference mutated model')
        summary=json.loads((dest/'summary.json').read_text())
        if (summary['examples'],summary['full_split'],summary['beam'],summary['inference_precision'],summary['r_updates'],summary['t_updates'])!=(53194,True,32,'fp32',epoch*693,row['t_updates']):
            raise ValueError('Wrong evaluation protocol')
        for key in ('dataset_binding','metric_protocol','seed','architecture'):
            if summary[key]!=reference[key]: raise ValueError('Protocol differs: '+key)
        result=dict(**row,summary=str(dest/'summary.json'),predictions=str(dest/'per_sample.jsonl.gz'),summary_sha256=sha256(dest/'summary.json'),predictions_sha256=sha256(dest/'per_sample.jsonl.gz'),metrics=summary['metrics'])
        (ROOT/f'RESULT_R{epoch:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
        records.append(result)
        print(json.dumps(dict(event='node_complete',epoch=epoch,ndcg5=summary['metrics']['list/ndcg@5'])),flush=True)
    (ROOT/'RESULT.json').write_text(json.dumps(dict(status='passed',nodes=records),indent=2)+'\n')

if __name__=='__main__': main()
