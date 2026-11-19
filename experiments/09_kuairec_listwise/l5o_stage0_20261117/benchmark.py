"""Bounded H100 execution acceptance. Never calls the formal training runner."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE / 'code'
if CODE.exists():
    sys.path.insert(0, str(CODE))
else:
    sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scripts.listwise_diffgrm.run import build_context, training_updates, evaluate
from scripts.listwise_diffgrm.runtime import build_model
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all, rng_state, restore_rng, cpu_copy, atomic_save

OUT = HERE / 'results'


def write(name, value):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(p)


def event(**values):
    print(json.dumps(dict(time=time.time(), **values)), flush=True)


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype),tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(x.tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():
                h.update(str(k).encode()); visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def make(cfg, assets, count, mixer):
    seed_all(cfg['seed'])
    model = build_model(cfg, assets.make_tokenizer()).cuda()
    alignment = build_alignment(assets,mixer,config_overrides=cfg['alignment']).cuda()
    return AlternatingTrainer(model,alignment,cfg,count,precision='bf16')


def next_check(trainer, assets, count, batch, mixer):
    arm='H1' if mixer else 'H0'
    binding={'diagnostic_only':True,'config':trainer.config}
    saved=trainer.checkpoint(binding,dict(phase=0,update=0,preflight_only=True),arm)
    path=OUT/f'{arm}_shorttest_only.pt'
    atomic_save(path,saved)
    a=trainer.update(batch,'R')
    expected=digest(dict(model=trainer.model.state_dict(),alignment=trainer.alignment.state_dict(),
        ropt=trainer.r_optimizer.state_dict(),topt=trainer.t_optimizer.state_dict(),rng=rng_state()))
    replacement=make(trainer.config,assets,count,mixer)
    replacement.restore(torch.load(path,map_location='cpu',weights_only=False),binding,arm=arm,allow_preflight=True)
    b=replacement.update(batch,'R')
    got=digest(dict(model=replacement.model.state_dict(),alignment=replacement.alignment.state_dict(),
        ropt=replacement.r_optimizer.state_dict(),topt=replacement.t_optimizer.state_dict(),rng=rng_state()))
    assert a==b and expected==got, 'GPU next-update restore mismatch'
    del replacement
    trainer.restore(saved,binding,arm=arm,allow_preflight=True)
    return dict(exact_next_update=True,checkpoint_bytes=path.stat().st_size,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(cfg, assets, data, mixer, fast):
    c=copy.deepcopy(cfg)
    c['execution']['compile']=False
    if not fast:
        c['execution']={}
        c['training']['microbatch_size']=256
        c['evaluation']['batch_size']=4
        data=copy.copy(data)
        del data._training_tensor_cache
    label=f'H{int(mixer)}_'+('fast' if fast else 'reference')
    trainer=make(c,assets,len(data),mixer)
    result=dict(label=label,initial=digest(dict(model=trainer.model.state_dict(),alignment=trainer.alignment.state_dict())),
                config=c,blocks=[])
    last_batch=None
    for block,(kind,steps) in enumerate([('R',80),('T',40),('R',80)]):
        order=np.random.default_rng(2024+block).permutation(len(data))[:steps*1024]
        seed_all(2024+100003*(block+1))
        torch.cuda.reset_peak_memory_stats()
        rows=[]; start=time.time()
        with training_updates(data,order,c,0,'cuda:0') as updates:
            it=iter(updates)
            for u in range(steps):
                torch.cuda.synchronize(); t=time.perf_counter()
                _,batches=next(it); batches=list(batches)
                ready=time.perf_counter()
                values=trainer.update(batches,kind)
                torch.cuda.synchronize()
                rows.append(dict(update=u,seconds=time.perf_counter()-t,data_s=ready-t,**values))
                last_batch=batches
                if u%20==0: event(label=label,block=block,update=u,loss=values['loss'],seconds=rows[-1]['seconds'])
        if kind=='T': trainer.refresh_catalog()
        result['blocks'].append(dict(kind=kind,rows=rows,start_unix=start,end_unix=time.time(),
            mean_seconds=statistics.mean(x['seconds'] for x in rows[5:]),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(label+'.json',result)
    if fast:
        result['restore']=next_check(trainer,assets,len(data),last_batch,mixer)
        # Same checkpoint, samples and beam32; only execution batching changes.
        # CPU reference frontier remains the oracle for the GPU transfer change.
        evaluations=[]
        for size,frontier in [(4,False),(128,True),(256,True)]:
            trainer.config['evaluation']['batch_size']=size
            trainer.config['execution']['batched_frontier']=frontier
            directory=OUT/f'{label}_eval_b{size}'
            try:
                torch.cuda.reset_peak_memory_stats()
                ev=evaluate(trainer,assets,split='validation',destination=directory,limit=256)
                import gzip
                with gzip.open(directory/'per_sample.jsonl.gz','rt') as f:
                    p=[json.loads(line)['prediction'] for line in f]
                evaluations.append(dict(batch=size,seconds=ev['seconds'],metrics=ev['metrics'],predictions=p,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            except torch.cuda.OutOfMemoryError:
                evaluations.append(dict(batch=size,oom=True));torch.cuda.empty_cache()
        result['evaluations']=evaluations
        baseline=evaluations[0]['predictions']
        for row in evaluations[1:]:
            if not row.get('oom'):
                row['identical_lists_fraction']=float(np.all(np.array(row['predictions'])==baseline,axis=1).mean())
        result['counts']=dict(r=trainer.r_updates,t=trainer.t_updates)
    write(label+'.json',result)
    del trainer,last_batch
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'resolved_config.json'))
    p.add_argument('--compile-probe',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)
    if args.compile_probe:
        cfg['execution']['compile']=True
        trainer,assets,data,_=build_context(cfg,'cuda:0','bf16',True)
        rows=[]
        for kind in ('R','T'):
            indices=np.arange(5*1024)
            with training_updates(data,indices,cfg,0,'cuda:0') as updates:
                for u,batches in updates:
                    torch.cuda.synchronize();t=time.perf_counter()
                    values=trainer.update(batches,kind)
                    torch.cuda.synchronize()
                    rows.append(dict(kind=kind,update=u,seconds=time.perf_counter()-t,**values))
        write('COMPILE_PROBE.json',dict(completed=True,rows=rows,
            adopted=False,note='Timing probe only; adoption additionally requires paired numerical and resume checks'))
        return
    write('hardware.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0)))
    with (OUT/'gpu.csv').open('w') as gpu:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv','-l','1'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            temporary,assets,data,binding=build_context(cfg,'cuda:0','bf16',False)
            write('binding.json',binding)
            del temporary;gc.collect();torch.cuda.empty_cache()
            records={}
            for mixer in (False,True):
                for fast in (False,True):
                    row=run(cfg,assets,data,mixer,fast);records[row['label']]=row
            summary={}
            passed=True
            for arm in ('H0','H1'):
                a,b=records[arm+'_reference'],records[arm+'_fast']
                assert a['initial']==b['initial']
                comparisons=[]
                for x,y in zip(a['blocks'],b['blocks']):
                    mean_a=statistics.mean(r['loss'] for r in x['rows'][5:])
                    mean_b=statistics.mean(r['loss'] for r in y['rows'][5:])
                    delta=abs(mean_a-mean_b)/max(abs(mean_a),1e-8)
                    comparisons.append(dict(kind=x['kind'],speedup=x['mean_seconds']/y['mean_seconds'],
                        reference_seconds=x['mean_seconds'],fast_seconds=y['mean_seconds'],loss_relative_delta=delta))
                    passed=passed and delta<=0.05 and y['mean_seconds']<x['mean_seconds']
                summary[arm]=dict(blocks=comparisons,restore=b['restore'],evaluations=b['evaluations'])
                passed=passed and all(r.get('identical_lists_fraction',1)>=0.99 for r in b['evaluations'])
            write('COMPLETE.json',dict(passed=passed,summary=summary,
                boundary='200 updates per arm per execution mode; no formal training or SASRec calibration',
                loss_gate='phase mean relative difference <=5%; practical short-run check, not a statistical proof'))
            # Bounded compilation trial; never silently make it a common setting.
            with (OUT/'compile.log').open('w') as log:
                try:
                    child=subprocess.run([sys.executable,__file__,'--config',args.config,'--compile-probe'],
                        stdout=log,stderr=subprocess.STDOUT,timeout=240)
                    if child.returncode:
                        write('COMPILE_STATUS.json',dict(adopted=False,returncode=child.returncode))
                except subprocess.TimeoutExpired:
                    write('COMPILE_STATUS.json',dict(adopted=False,timeout_seconds=240))
        finally:
            monitor.terminate();monitor.wait(timeout=10)


if __name__=='__main__':
    main()
