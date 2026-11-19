"""Bounded CPU/input or isolated CUDA recovery profiling; never saves model states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,v):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('code','dataset','output','manifest-sha'):p.add_argument('--'+k,required=True)
    for k in ('parent','parent-sha','assets','checkpoint','checkpoint-sha','journal','journal-sha','input-proof','input-proof-sha'):p.add_argument('--'+k)
    p.add_argument('--cpu-input-only',action='store_true')
    a=p.parse_args();started=time.monotonic()
    code=Path(a.code).resolve();out=Path(a.output).resolve()
    if out==code or code in out.parents:raise ValueError('Never write into sealed scientific code')
    out.mkdir(parents=True,exist_ok=False)
    if sha(code/'SPREAD200_EXECUTION_MANIFEST.json')!=a.manifest_sha:raise ValueError('Manifest changed')
    manifest=read(code/'SPREAD200_EXECUTION_MANIFEST.json')
    for name,v in manifest['files'].items():
        if sha(code/name)!=v['sha256']:raise ValueError('Scientific source changed')
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
    if a.cpu_input_only:
        data=KuaiRecListDataset(a.dataset,'train')
        records=[]
        for u in range(400,405):
            selected=indices[u*1024:(u+1)*1024];t=time.perf_counter()
            batches=list(microbatches(data,selected,256));seconds=time.perf_counter()-t
            assert [len(x['target_item_ids']) for x in batches]==[256]*4
            records.append(dict(update=u+1,indices_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),batch_fingerprint=fingerprint(batches),cpu_input_seconds=seconds))
        result=dict(status='passed_CPU_input_only',preflight_only=True,model_execution=False,
            data_binding=data.binding,records=records,script_sha256=sha(__file__),
            seconds=time.monotonic()-started,limitation='Local CPU data-preparation timing, not GPU training speed.')
        write(out/'CPU_INPUT_PROFILE.json',result);print(json.dumps(result),flush=True);return
    for k in ('parent','parent_sha','assets','checkpoint','checkpoint_sha','journal','journal_sha','input_proof','input_proof_sha'):
        if getattr(a,k) is None:raise ValueError('GPU probe requires '+k)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):raise ValueError('Exactly one H100 required')
    original={a.checkpoint:a.checkpoint_sha,a.journal:a.journal_sha}
    if any(sha(k)!=v for k,v in original.items()):raise ValueError('Formal source state or journal changed')
    if sha(a.input_proof)!=a.input_proof_sha:raise ValueError('CPU input proof changed')
    proof=read(a.input_proof)
    if proof['status']!='passed_CPU_input_only' or proof['script_sha256']!=sha(__file__):raise ValueError('Wrong CPU input proof')
    from scripts.listwise_diffgrm.spread200 import build_spread
    c,assets,data=build_spread(a.parent,a.parent_sha,'cuda:0',a.dataset,a.assets)
    if data.binding!=proof['data_binding']:raise ValueError('CPU/GPU datasets differ')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if (state['inner']['r_updates'],state['inner']['t_updates'],state['inner']['progress']['phase'],state['inner']['progress']['update'])!=(16339,140,30,400):raise ValueError('Wrong recovery point')
    prepared=[];input_times=[]
    for entry in proof['records']:
        u=entry['update']-1;selected=indices[u*1024:(u+1)*1024];t=time.perf_counter()
        batches=list(microbatches(data,selected,256));input_times.append(time.perf_counter()-t)
        if fingerprint(batches)!=entry['batch_fingerprint']:raise ValueError('Actual input tensors differ from CPU acceptance')
        prepared.append(batches)
    def step(batches):
        torch.cuda.synchronize();t=time.perf_counter();value=c.step(batches,'R');torch.cuda.synchronize()
        return value,time.perf_counter()-t
    c.restore(state);first,first_seconds=step(prepared[0]);expected=fingerprint(c.checkpoint())
    c.restore(state);second,second_seconds=step(prepared[0])
    if first!=second or fingerprint(c.checkpoint())!=expected:raise ValueError('Next-R CUDA restore is not exact')
    c.restore(state);compute_times=[]
    for batches in prepared:
        _,seconds=step(batches);compute_times.append(seconds)
    c.restore(state)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
        _,profile_seconds=step(prepared[0])
    if fingerprint(c.checkpoint())!=expected:raise ValueError('Profiler altered next-R state')
    prof.export_chrome_trace(str(out/'CUDA_PROFILE_TRACE.json'))
    events=prof.events();kernels=[e for e in events if str(e.device_type).endswith('CUDA')]
    rows=sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)
    top=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,cpu_total_us=e.cpu_time_total,device_total_us=e.device_time_total) for e in rows[:25]]
    kernel_us=sum(e.device_time_total for e in kernels)
    try:
        hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.max.sm,pstate','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except Exception as exc:hardware=type(exc).__name__
    if any(sha(k)!=v for k,v in original.items()):raise ValueError('Probe changed formal checkpoint or journal')
    result=dict(status='passed_isolated_CUDA_recovery_profile',preflight_only=True,formal_continuation=False,
        r_updates_before=16339,t_updates_before=140,next_R_replay_exact=True,profile_next_R_state_exact=True,
        source_files_unchanged=True,checkpoint_sha256=a.checkpoint_sha,journal_sha256=a.journal_sha,
        scientific_manifest_sha256=a.manifest_sha,script_sha256=sha(__file__),input_proof_sha256=a.input_proof_sha,
        next_R_seconds=[first_seconds,second_seconds],input_seconds=input_times,compute_seconds=compute_times,
        input_median_seconds=statistics.median(input_times),compute_median_seconds=statistics.median(compute_times),
        profiled_step_wall_seconds=profile_seconds,cuda_kernel_events=len(kernels),sum_cuda_kernel_duration_us=kernel_us,
        kernel_sum_to_profiled_wall_ratio=kernel_us/(profile_seconds*1e6),top_cpu_operators=top,
        torch_version=torch.__version__,torch_cuda_build=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),hardware_snapshot=hardware,
        seconds=time.monotonic()-started,limitation='Only isolated next-R recovery and timing; profiler adds overhead and kernel-duration ratio is not nvidia-smi utilization. No model checkpoint is saved; these updates cannot enter formal training.')
    write(out/'RECOVERY_PROFILE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
