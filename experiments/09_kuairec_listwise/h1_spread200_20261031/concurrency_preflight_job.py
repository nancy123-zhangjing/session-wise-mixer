"""Single bounded concurrency-equivalence preflight, no platform submission or automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','python','self-sha','probe-sha','input-proof-sha'):p.add_argument('--'+k,required=True)
    a=p.parse_args();start=time.monotonic();root=Path(a.root).resolve()
    if root!=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1'):raise ValueError('Wrong owned root')
    if sha(__file__)!=a.self_sha:raise ValueError('Supervisor changed')
    probe=root/'concurrency_preflight.py';proof=root/'CPU_PERF_INPUT_PROFILE.json'
    if sha(probe)!=a.probe_sha or sha(proof)!=a.input_proof_sha:raise ValueError('Probe package changed')
    # 348 seconds already used; reserve the entire new 0.25h platform cap plus
    # 180 seconds release allowance. Both preflight and overall caps still fit.
    if 348+900+180>1800 or 35498+900+180>86400:raise ValueError('GPU budget exceeded')
    inputs=json.loads((root/'INPUTS.json').read_text(encoding='utf-8'))
    jobs=root/'jobs/concurrency_preflight_r1';jobs.mkdir(parents=True,exist_ok=False)
    command=[a.python,'-B','-u',str(probe),'--code',str(root/'code'),'--dataset',inputs['dataset'],
        '--assets',inputs['assets'],'--parent',inputs['parent'],'--parent-sha','991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1',
        '--manifest-sha','c875e203a0c07ecc109b0ee6e89502cdcc31ab4887fab0d4506f4173e30b3f07',
        '--checkpoint',str(root/'runs/H1-Spread200-R35/last.pt'),'--checkpoint-sha','bd28ba74e82a776a38e567d91c742dfccd1ca4591881151635a9da7efa5267fe',
        '--journal',str(root/'runs/H1-Spread200-R35/training.jsonl'),'--journal-sha','348e7a914323ae4fa7e587d0578406c35357ab67c3c005e212a7c7eb65186b1e',
        '--input-proof',str(proof),'--input-proof-sha',a.input_proof_sha,'--output',str(root/'concurrency_preflight_r1')]
    env=os.environ.copy()
    for k in list(env):
        if k in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NODE_RANK','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','NNODES','NPROC_PER_NODE'} or k.startswith(('TORCHELASTIC_','OMPI_','PMI_','PMIX_','RDZV_')):env.pop(k)
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    stopped={'at':None};process=None
    def stop(sig,_frame=None):
        if stopped['at'] is None:stopped['at']=time.monotonic()
        if process is not None and process.poll() is None:os.killpg(process.pid,sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    result=dict(status='running',preflight_only=True,formal_updates=0,max_seconds=840,automatic_retry=False)
    try:
        with (jobs/'probe.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=root/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                now=time.monotonic()
                if now-start>=780 and stopped['at'] is None:stop(signal.SIGTERM)
                if stopped['at'] is not None and now-stopped['at']>=30:os.killpg(process.pid,signal.SIGKILL)
                write(jobs/'status.json',dict(result,pid=process.pid,elapsed_seconds=now-start,heartbeat_unix=time.time()))
                time.sleep(5)
            if process.returncode:raise RuntimeError('Probe exited '+str(process.returncode))
            report=root/'concurrency_preflight_r1/CONCURRENCY_PREFLIGHT.json'
            if json.loads(report.read_text(encoding='utf-8'))['status']!='passed_concurrency_numerical_equivalence':raise ValueError('Missing passing profile')
            result.update(status='completed_isolated_concurrency_preflight',report_sha256=sha(report))
    except BaseException as exc:
        result.update(status='incomplete',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if process is not None and process.poll() is None:
            stop(signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        result['elapsed_seconds']=time.monotonic()-start;write(jobs/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
