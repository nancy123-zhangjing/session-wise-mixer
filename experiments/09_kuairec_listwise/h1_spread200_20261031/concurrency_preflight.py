"""Isolated same-H100 scheduling equivalence test; no formal output writes."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('code','dataset','output','manifest-sha','input-proof','input-proof-sha'):p.add_argument('--'+k,required=True)
    for k in ('parent','parent-sha','assets','checkpoint','checkpoint-sha','journal','journal-sha'):p.add_argument('--'+k)
    p.add_argument('--mode',choices=('cpu-check','coordinator','train','eval_a','eval_b'),default='coordinator')
    a=p.parse_args();started=time.monotonic();code=Path(a.code).resolve();out=Path(a.output).resolve()
    if out==code or code in out.parents:raise ValueError('Never write sealed scientific code')
    if a.mode in ('cpu-check','coordinator'):out.mkdir(parents=True,exist_ok=False)
    else:
        if read(out/'REQUEST.json')['preflight_only'] is not True:raise ValueError('Worker needs isolated coordinator')
        (out/a.mode).mkdir(exist_ok=False)
    if sha(code/'SPREAD200_EXECUTION_MANIFEST.json')!=a.manifest_sha:raise ValueError('Manifest changed')
    for name,v in read(code/'SPREAD200_EXECUTION_MANIFEST.json')['files'].items():
        if sha(code/name)!=v['sha256']:raise ValueError('Scientific source changed')
    if sha(a.input_proof)!=a.input_proof_sha:raise ValueError('Input proof changed')
    proof=read(a.input_proof)
    if proof['status']!='passed_CPU_input_only':raise ValueError('Input proof not accepted')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
    os.environ.update(DIFFGRM_ENGINE_ROOT=str(code/'external/DiffGRM'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(code))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from scripts.listwise_diffgrm.data import KuaiRecListDataset
    from scripts.listwise_diffgrm.run import microbatches
    from scripts.listwise_diffgrm.recovery import fingerprint
    indices=np.random.default_rng(np.random.SeedSequence([2024,7919,30])).permutation(709443)
    if a.mode=='cpu-check':
        data=KuaiRecListDataset(a.dataset,'train')
        if data.binding!=proof['data_binding']:raise ValueError('Dataset differs')
        check=[]
        for u in (400,429):
            selected=indices[u*1024:(u+1)*1024];batches=list(microbatches(data,selected,256))
            assert len(batches)==4 and all(len(b['target_item_ids'])==256 for b in batches)
            value=fingerprint(batches)
            if u==400 and value!=proof['records'][0]['batch_fingerprint']:raise ValueError('Next-R input changed')
            check.append(dict(update=u+1,indices_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),batch_fingerprint=value))
        result=dict(status='passed_CPU_concurrency_inputs',preflight_only=True,model_execution=False,records=check,script_sha256=sha(__file__),seconds=time.monotonic()-started)
        write(out/'CPU_CHECK.json',result);print(json.dumps(result),flush=True);return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):raise ValueError('One H100 required')
    original={a.checkpoint:a.checkpoint_sha,a.journal:a.journal_sha}
    if any(sha(k)!=v for k,v in original.items()):raise ValueError('Formal source changed')
    from scripts.listwise_diffgrm.spread200 import build_spread
    from scripts.listwise_diffgrm.recovery_evaluation import evaluate_recovery,load_subset
    c,assets,data=build_spread(a.parent,a.parent_sha,'cuda:0',a.dataset,a.assets)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c.restore(source)
    if (c.trainer.r_updates,c.trainer.t_updates,c.progress['phase'],c.progress['update'])!=(16339,140,30,400):raise ValueError('Wrong recovery cursor')
    subset=load_subset(read(code.parent/'INPUTS.json')['subset'],a.dataset)
    locked=np.asarray([x['index'] for x in subset['rows']],dtype=np.int64)
    if data.binding!=proof['data_binding']:raise ValueError('GPU data differs')
    for x in proof['records']:
        u=x['update']-1;selected=indices[u*1024:(u+1)*1024]
        if hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()!=x['indices_sha256']:raise ValueError('R sample stream differs')
    def train_segment():
        losses=[];t=time.perf_counter()
        for u in range(400,430):
            batches=microbatches(data,indices[u*1024:(u+1)*1024],256)
            row=c.step(batches,'R');losses.append(row['loss'])
        torch.cuda.synchronize()
        return losses,time.perf_counter()-t
    def eval_segment(selected,destination):
        before=fingerprint(c.checkpoint())
        validation=KuaiRecListDataset(a.dataset,'validation',assets=assets)
        t=time.perf_counter();evaluate_recovery(c.trainer,assets,validation,destination,indices=selected,subset={'preflight_only':True})
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        if fingerprint(c.checkpoint())!=before:raise ValueError('Read-only evaluation changed state')
        return seconds
    if a.mode in ('train','eval_a','eval_b'):
        role=out/a.mode
        write(out/('READY_'+a.mode+'.json'),dict(preflight_only=True,ready_unix=time.time()))
        while not (out/'GO.json').exists():
            if time.monotonic()-started>600:raise TimeoutError('Coordinator barrier timeout')
            time.sleep(.1)
        write(out/('ACTIVE_'+a.mode+'.json'),dict(start_unix=time.time()))
        if a.mode=='train':
            losses,seconds=train_segment()
            write(out/('COMPUTE_DONE_'+a.mode+'.json'),dict(end_unix=time.time()))
            result=dict(state_fingerprint=fingerprint(c.checkpoint()),losses=losses,seconds=seconds,r_updates=c.trainer.r_updates,t_updates=c.trainer.t_updates)
        else:
            selected=locked[:1024] if a.mode=='eval_a' else locked[1024:2048]
            seconds=eval_segment(selected,role/'predictions')
            write(out/('COMPUTE_DONE_'+a.mode+'.json'),dict(end_unix=time.time()))
            result=dict(seconds=seconds,examples=1024,predictions_sha256=sha(role/'predictions/per_sample.jsonl.gz'))
        if any(sha(k)!=v for k,v in original.items()):raise ValueError('Formal files modified')
        write(role/'RESULT.json',dict(result,status='passed_worker',preflight_only=True));return
    write(out/'REQUEST.json',dict(preflight_only=True,formal_continuation=False,script_sha256=sha(__file__),checkpoint_sha256=a.checkpoint_sha))
    losses,train_seconds=train_segment();baseline=dict(state_fingerprint=fingerprint(c.checkpoint()),losses=losses,seconds=train_seconds)
    c.restore(source)
    eval_seconds=eval_segment(np.concatenate([locked[:128],locked[1024:1152]]),out/'baseline_predictions')
    baseline_rows={}
    with gzip.open(out/'baseline_predictions/per_sample.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);baseline_rows[row['sample_id']]=row
    del c,assets,data,source;gc.collect();torch.cuda.empty_cache()
    commands=[]
    for mode in ('train','eval_a','eval_b'):
        args=[sys.executable,'-B','-u',str(Path(__file__).resolve())]
        for k,v in vars(a).items():
            if k=='mode':continue
            if v is not None:args.extend(['--'+k.replace('_','-'),str(v)])
        args.extend(['--mode',mode]);commands.append((mode,args))
    processes={};handles=[];samples=[]
    try:
        for mode,args in commands:
            handle=(out/(mode+'.log')).open('xb');handles.append(handle)
            processes[mode]=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT)
        while not all((out/('READY_'+mode+'.json')).exists() for mode in processes):
            if any(p.poll() is not None for p in processes.values()):raise RuntimeError('Worker failed before barrier')
            if time.monotonic()-started>600:raise TimeoutError('Worker initialization timeout')
            time.sleep(.5)
        write(out/'GO.json',dict(start_unix=time.time()))
        while any(p.poll() is None for p in processes.values()):
            if time.monotonic()-started>720:raise TimeoutError('Short test deadline')
            if any(p.poll() not in (None,0) for p in processes.values()):raise RuntimeError('Concurrent worker failed')
            active=[mode for mode in processes if (out/('ACTIVE_'+mode+'.json')).exists() and not (out/('COMPUTE_DONE_'+mode+'.json')).exists()]
            r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if r.returncode==0:
                values=[float(x.strip()) for x in r.stdout.strip().split(',')]
                if len(values)==3:samples.append(dict(unix=time.time(),gpu_percent=values[0],memory_percent=values[1],memory_used_mib=values[2],active_roles=active))
            time.sleep(1)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('Worker nonzero exit')
    finally:
        for proc in processes.values():
            if proc.poll() is None:proc.terminate()
        for proc in processes.values():
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        for handle in handles:handle.close()
        write(out/'GPU_SAMPLES.json',dict(preflight_only=True,samples=samples))
    actual=read(out/'train/RESULT.json')
    if actual['state_fingerprint']!=baseline['state_fingerprint'] or actual['losses']!=baseline['losses']:raise ValueError('Concurrent training differs from sequential 30-step state')
    matched={}
    for mode in ('eval_a','eval_b'):
        n=0
        with gzip.open(out/mode/'predictions/per_sample.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);old=baseline_rows.get(row['sample_id'])
                if old is not None:
                    if row!=old:raise ValueError('Concurrent FP32 prediction, score, label or metrics changed')
                    n+=1
        if n!=128:raise ValueError('Oracle sample coverage differs')
        matched[mode]=n
    overlap=[x['gpu_percent'] for x in samples if len(x['active_roles'])==3]
    if any(sha(k)!=v for k,v in original.items()):raise ValueError('Formal state or journal changed')
    result=dict(status='passed_concurrency_numerical_equivalence',preflight_only=True,formal_continuation=False,
        checkpoint_sha256=a.checkpoint_sha,journal_sha256=a.journal_sha,source_files_unchanged=True,
        sequential_R_seconds=baseline['seconds'],concurrent_R_seconds=actual['seconds'],R_steps_compared=30,
        complete_state_and_loss_sequence_exact=True,FP32_rows_exact=matched,baseline_eval_seconds=eval_seconds,
        overlap_samples=len(overlap),overlap_gpu_mean_percent=statistics.mean(overlap) if overlap else None,
        overlap_gpu_max_percent=max(overlap) if overlap else None,overlap_gpu_min_percent=min(overlap) if overlap else None,
        overlap_samples_at_least_40=sum(x>=40 for x in overlap),peak_sampled_memory_mib=max((x['memory_used_mib'] for x in samples),default=None),
        measured_overlap_adequate=len(overlap)>=10 and statistics.mean(overlap)>=40,
        seconds=time.monotonic()-started,script_sha256=sha(__file__),
        limitation='Short genuine-work overlap test, not proof of future three-hour utilization. No formal checkpoint or prediction is replaced.')
    write(out/'CONCURRENCY_PREFLIGHT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
