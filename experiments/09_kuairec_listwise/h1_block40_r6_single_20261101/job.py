"""One bounded formal continuation; genuine-work concurrency, no retry or padding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


class UtilizationGuard:
    def __init__(self,limit=9000):
        self.limit,self.low_since,self.unknown_since=limit,None,None

    def observe(self,now,value):
        if value is None:
            if self.unknown_since is None:self.unknown_since=now
            return 'utilization_unobservable_300s' if now-self.unknown_since>=300 else None
        self.unknown_since=None
        if value>=40:
            self.low_since=None
        elif self.low_since is None:
            self.low_since=now
        if self.low_since is not None and now-self.low_since>=self.limit:
            return 'sustained_low_utilization_save_and_stop'
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','python','self-sha','gate','gate-sha'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args();start=time.monotonic();root=Path(a.root).resolve()
    if root!=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1'):
        raise ValueError('Wrong owned root')
    if sha(__file__)!=a.self_sha or sha(a.gate)!=a.gate_sha:raise ValueError('Runtime package changed')
    gate=read(a.gate)
    if gate['status']!='accepted_single_e_runtime':raise ValueError('E start not bound')
    if (gate['consumed_gpu_seconds_upper_bound']+gate['platform_max_seconds']+gate['release_allowance_seconds']>86400
            or gate['inner_max_seconds']>=gate['platform_max_seconds']):
        raise ValueError('Cumulative budget does not fit')
    runtime=root/'runtime.py'
    if sha(runtime)!=gate['runtime_sha256']:raise ValueError('Runtime changed')
    if sha(root/'CPU_SCHEDULE_CHECK.json')!=gate['schedule_check_sha256']:
        raise ValueError('Schedule acceptance changed')
    run=root/'runs/H1-Block40-R6-R35'
    inputs=read(root/'INPUTS.json');code=root/'code'
    if sha(inputs['parent'])!=gate['parent_sha256']:raise ValueError('Parent changed')
    jobs=root/'jobs/formal_r1';jobs.mkdir(parents=True,exist_ok=False)
    write(jobs/'resume_gate.json',dict(gate,gate_sha256=a.gate_sha,launcher_sha256=a.self_sha))
    env=os.environ.copy()
    for key in list(env):
        if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NODE_RANK','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','NNODES','NPROC_PER_NODE','SLURM_PROCID','SLURM_LOCALID','SLURM_NTASKS'} or key.startswith(('TORCHELASTIC_','OMPI_','PMI_','PMIX_','ACCELERATE_','RDZV_')):
            env.pop(key)
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        DIFFGRM_ENGINE_ROOT=str(code/'external/DiffGRM'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    requested={'at':None,'reason':None};process=None
    def stop(sig,_frame=None,reason='external_signal'):
        if requested['at'] is None:requested.update(at=time.monotonic(),reason=reason)
        if process is not None and process.poll() is None:os.killpg(process.pid,sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    common=['--parent',inputs['parent'],'--parent-sha',gate['parent_sha256'],'--dataset',inputs['dataset'],
        '--assets',inputs['assets'],'--subset',inputs['subset'],'--warmup-predictions',inputs['warmup_predictions'],
        '--output',str(run)]
    commands=[('train',[a.python,'-B','-u',str(runtime),'--root',str(root),'--gate',a.gate,
        '--gate-sha',a.gate_sha,'--self-sha',gate['runtime_sha256']]),
        ('test',[a.python,'-B','-u','-m','scripts.listwise_diffgrm.block40_single_run','test']+common)]
    result=dict(status='running',automatic_retry=False,stages=[],runtime_sha256=gate['runtime_sha256'],
        launcher_sha256=a.self_sha,gate_sha256=a.gate_sha,max_seconds=gate['inner_max_seconds'])
    guard=UtilizationGuard(gate['low_utilization_stop_seconds']);sample_at=0
    try:
        for label,command in commands:
            if requested['at'] is not None:raise RuntimeError('Stopped before '+label)
            with (jobs/(label+'.log')).open('xb') as log:
                process=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None:
                    now=time.monotonic()
                    if now-start>=gate['inner_max_seconds']-120 and requested['at'] is None:
                        stop(signal.SIGTERM,reason='reserved_budget_shutdown')
                    if now>=sample_at:
                        sample_at=now+gate['utilization_sample_seconds'];percent=None
                        try:
                            measurement=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                            if measurement.returncode==0:
                                values=measurement.stdout.strip().split(',')
                                if len(values)==2:percent=float(values[0])
                        except subprocess.TimeoutExpired:
                            pass
                        reason=guard.observe(now,percent)
                        with (jobs/'gpu_utilization.jsonl').open('a',encoding='utf-8') as samples:
                            samples.write(json.dumps(dict(unix=time.time(),elapsed_seconds=now-start,stage=label,
                                gpu_percent=percent,low_since_elapsed=None if guard.low_since is None else guard.low_since-start,
                                guard_action=reason))+'\n')
                        if reason and requested['at'] is None:stop(signal.SIGTERM,reason=reason)
                    if requested['at'] is not None and now-requested['at']>=60:
                        os.killpg(process.pid,signal.SIGKILL)
                    write(jobs/'status.json',dict(result,stage=label,pid=process.pid,elapsed_seconds=now-start,
                        heartbeat_unix=time.time(),stop_reason=requested['reason']))
                    time.sleep(5)
                result['stages'].append(dict(stage=label,exit_code=process.returncode))
                if process.returncode or requested['at'] is not None:
                    raise RuntimeError(label+' stopped: '+str(process.returncode)+' '+str(requested['reason']))
        complete=read(run/'complete.json')
        if (complete['r_updates'],complete['t_updates'])!=(24255,200):raise ValueError('Wrong final clocks')
        if not (run/'test_predictions_complete.json').is_file():raise ValueError('Tests not complete')
        result['status']='training_validation_test_predictions_complete'
    except BaseException as exc:
        result.update(status='incomplete',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if process is not None and process.poll() is None:
            stop(signal.SIGTERM)
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        result.update(elapsed_seconds=time.monotonic()-start,stop_reason=requested['reason'])
        write(jobs/'result.json',result)
        if result['status']=='training_validation_test_predictions_complete':
            with (root/'FORMAL_COMPLETION.json').open('x',encoding='utf-8') as f:
                json.dump(dict(attempt='formal_r1',result_sha256=sha(jobs/'result.json'),
                    resume_gate_sha256=sha(jobs/'resume_gate.json')),f,indent=2);f.write('\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
