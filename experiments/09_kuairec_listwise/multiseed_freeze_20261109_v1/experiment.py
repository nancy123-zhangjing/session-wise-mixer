"""Four independent single-card runs; immutable historical engine, no architecture edits."""
import argparse, copy, hashlib, json, os, sys
from pathlib import Path
from protocol import SEEDS, ARMS, skipped, counts
ROOT=Path(__file__).resolve().parent
STAGE=ROOT.parent
CODE=STAGE/'h_freeze35_20261029_r1/code'
ORIGINAL=STAGE/'goal_20261027_r1'
os.environ['DIFFGRM_ENGINE_ROOT']=str(CODE/'external/DiffGRM')
sys.path.insert(0,str(CODE))

def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)

def main():
    import torch
    from scripts.listwise_diffgrm.run import build_context, phase_indices, microbatches, evaluate, run_lock
    from scripts.listwise_diffgrm.training import atomic_save, seed_all, sha256, restore_rng, validate_optimizer_clock
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.recovery_diagnostics import diagnostic_context
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('Expected one H100')
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for relative,digest in manifest['runner_sha256'].items():
        if sha256(ROOT/relative)!=digest: raise ValueError('Runner changed')
    for relative,digest in manifest['source_sha256'].items():
        if sha256(CODE/relative)!=digest: raise ValueError('Frozen source changed')
    cfg=json.loads((ROOT/f'config_seed{args.seed}.json').read_text())
    if sha256(ROOT/f'config_seed{args.seed}.json')!=manifest['config_sha256'][str(args.seed)]:
        raise ValueError('Config changed')
    output=ROOT/f'seed{args.seed}_{args.arm}'
    spec=dict(format='stage09_multiseed_freeze_v1',seed=args.seed,arm=args.arm,
              freeze_epoch=20 if args.arm=='H1' else None,runner_sha256=manifest['runner_sha256'])
    with run_lock(output,resume=args.resume):
        # Both arms use the original Mixer-disabled shared-warmup construction.
        trainer,assets,data,binding=build_context(cfg,'cuda:0','bf16',False)
        if len(data)!=709443 or len(trainer.plan)!=45: raise ValueError('Original schedule/data changed')
        progress=dict(phase=0,update=0,best_score=None,best_epoch=None,validations=[])
        freeze_hashes=None
        def freeze_state():
            return {k:fingerprint(v) for k,v in dict(alignment=trainer.alignment.state_dict(),t_optimizer=trainer.t_optimizer.state_dict()).items()}
        def save_state():
            result=trainer.checkpoint(binding,copy.deepcopy(progress),args.arm)
            result.update(experiment_spec=spec,freeze_hashes=freeze_hashes)
            return result
        def restore(state):
            nonlocal progress,freeze_hashes
            if state['binding']!=binding or state['experiment_spec']!=spec or state['precision']!='bf16':
                raise ValueError('Resume binding mismatch')
            clocks=counts(trainer.plan,state['progress'],args.arm)
            for kind,key in [('R','r_updates'),('T','t_updates')]:
                if state[key]!=clocks[kind]: raise ValueError('Clock mismatch')
                validate_optimizer_clock(state[kind.lower()+'_optimizer'],clocks[kind],kind)
            if state['scheduler']['last_epoch']!=clocks['R']: raise ValueError('R scheduler mismatch')
            if int(state['alignment']['diger_optimizer_step'])!=clocks['T'] or int(state['alignment']['diger_total_optimizer_steps'])!=400:
                raise ValueError('DIGER schedule mismatch')
            trainer.model.load_state_dict(state['model'],strict=True)
            trainer.alignment.load_state_dict(state['alignment'],strict=True)
            trainer.r_optimizer.load_state_dict(state['r_optimizer'])
            trainer.t_optimizer.load_state_dict(state['t_optimizer'])
            trainer.scheduler.load_state_dict(state['scheduler'])
            trainer.r_updates=clocks['R'];trainer.t_updates=clocks['T']
            progress=copy.deepcopy(state['progress']);freeze_hashes=state['freeze_hashes']
            if args.arm=='H1' and progress['phase']>=25 and freeze_hashes is None:
                raise ValueError('Missing R20 freeze anchor')
            if freeze_hashes is not None and freeze_state()!=freeze_hashes: raise ValueError('Frozen state changed')
            trainer.alignment.mixer.mixer_enabled=args.arm=='H1' and progress['phase']>=5
            restore_rng(state['rng'])
        if args.resume: restore(torch.load(output/'last.pt',map_location='cpu',weights_only=False))
        else:
            write(output/'resolved_config.json',cfg)
            write(output/'binding.json',binding)
            atomic_save(output/'initial.pt',save_state())
        t=cfg['training']
        while progress['phase']<len(trainer.plan):
            phase=progress['phase']; row=trainer.plan[phase]
            trainer.alignment.mixer.mixer_enabled=args.arm=='H1' and phase>=5
            if skipped(row,args.arm):
                if trainer.t_updates!=200 or freeze_state()!=freeze_hashes: raise ValueError('Freeze violation')
                progress['phase']+=1;progress['update']=0
                atomic_save(output/'last.pt',save_state())
                print(json.dumps(dict(event='T_skipped_frozen',epoch=row['epoch'],T=trainer.t_updates)),flush=True)
                continue
            indices=phase_indices(cfg,len(data),phase,row)
            if progress['update']==0: seed_all(cfg['seed']+100003*(phase+1))
            while progress['update']<row['updates']:
                start=progress['update']*t['effective_batch_size']
                selected=indices[start:start+t['effective_batch_size']]
                values=trainer.update(microbatches(data,selected,t['microbatch_size']),row['kind'])
                progress['update']+=1
                record=dict(phase_index=phase,epoch=row['epoch'],update=progress['update'],**values)
                with (output/'training.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                if progress['update']%100==0:
                    atomic_save(output/'last.pt',save_state())
                    print(json.dumps(record),flush=True)
            if row['kind']=='T': trainer.refresh_catalog()
            save_best=False
            if row['validate']:
                atomic_save(output/'last.pt',save_state())
                dest=output/f"validation_R{row['epoch']:02d}"
                retry=0
                while dest.exists():
                    retry+=1;dest=output/f"validation_R{row['epoch']:02d}_retry{retry}"
                result=evaluate(trainer,assets,split='validation',destination=dest)
                score=result['metrics']['list/ndcg@5']
                progress['validations'].append(str(dest))
                if progress['best_score'] is None or score>progress['best_score']:
                    progress['best_score']=score;progress['best_epoch']=row['epoch'];save_best=True
            progress['phase']+=1;progress['update']=0
            if args.arm=='H1' and row['kind']=='R' and row['epoch']==20:
                if trainer.t_updates!=200: raise ValueError('Wrong freeze boundary')
                freeze_hashes=freeze_state()
                write(output/'FREEZE_R20.json',dict(R=20,T=200,hashes=freeze_hashes))
            if freeze_hashes is not None and freeze_state()!=freeze_hashes: raise ValueError('Frozen state drift')
            saved=save_state()
            if row['validate']:
                checkpoints=output/'checkpoints';checkpoints.mkdir(exist_ok=True)
                cp=checkpoints/f"R{row['epoch']:02d}_T{trainer.t_updates:03d}.pt"
                atomic_save(cp,saved)
                write(checkpoints/f"R{row['epoch']:02d}.json",dict(epoch=row['epoch'],T=trainer.t_updates,checkpoint=str(cp),sha256=sha256(cp),validation=str(dest),ndcg5=score))
                print(json.dumps(dict(event='validation_complete',epoch=row['epoch'],T=trainer.t_updates,ndcg5=score)),flush=True)
                if row['epoch']==5:
                    paired={k:fingerprint(saved[k]) for k in ('model','alignment','r_optimizer','t_optimizer','scheduler','rng')}
                    write(output/'R5_PAIR.json',dict(seed=args.seed,hashes=paired))
                    other=ROOT/f"seed{args.seed}_{'H0' if args.arm=='H1' else 'H1'}"/'R5_PAIR.json'
                    if other.exists() and json.loads(other.read_text())['hashes']!=paired: raise ValueError('Paired warmup mismatch')
            if save_best: atomic_save(output/'best.pt',saved)
            atomic_save(output/'last.pt',saved)
        if (trainer.r_updates,trainer.t_updates)!=(24255,200 if args.arm=='H1' else 400): raise ValueError('Final counts wrong')
        write(output/'TRAINING_COMPLETE.json',dict(seed=args.seed,arm=args.arm,R=35,T=trainer.t_updates,best_epoch=progress['best_epoch'],best_score=progress['best_score']))
        # Only validation-selected best and the fixed R35 endpoint are tested.
        best_epoch=progress['best_epoch']; receipts=[]
        for label,checkpoint in [('best',output/'best.pt'),('R35',output/'last.pt')]:
            if label=='R35' and best_epoch==35:
                receipts.append(dict(label=label,reused='best'));continue
            restore(torch.load(checkpoint,map_location='cpu',weights_only=False))
            dest=output/f'test_{label}'
            before=fingerprint(dict(model=trainer.model.state_dict(),alignment=trainer.alignment.state_dict()))
            if dest.exists(): raise ValueError('Inspect existing test before retry')
            with diagnostic_context(trainer),torch.autocast('cuda',enabled=False):
                result=evaluate(trainer,assets,split='test',destination=dest,beam=32)
            if fingerprint(dict(model=trainer.model.state_dict(),alignment=trainer.alignment.state_dict()))!=before: raise ValueError('Evaluation mutated weights')
            receipts.append(dict(label=label,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),summary=str(dest/'summary.json'),summary_sha256=sha256(dest/'summary.json'),predictions=str(dest/'per_sample.jsonl.gz'),predictions_sha256=sha256(dest/'per_sample.jsonl.gz'),metrics=result['metrics']))
            write(output/'TEST_RESULTS.json',receipts)
        write(output/'TEST_RESULTS.json',receipts)
        write(output/'GPU_COMPLETE.json',dict(status='passed',seed=args.seed,arm=args.arm,best_epoch=best_epoch,tests=receipts))

if __name__=='__main__': main()
