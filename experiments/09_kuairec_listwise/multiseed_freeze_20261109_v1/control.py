"""Separate one-card jobs with durable submission intents; no duplicate submissions."""
import sys,json,shlex,base64,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'h1_no_code_tag_20261108_v1'))
import control as c
c.HERE=HERE
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/multiseed_freeze_20261109_v1'

def py(code,receipt=None,timeout=120):
    encoded=base64.b64encode(code.encode()).decode()
    return c.remote(c.PYTHON+' -B -c '+shlex.quote(f'import base64;exec(base64.b64decode({encoded!r}))'),receipt,timeout)

def prepare():
    c.resources()
    help_result=subprocess.run([str(c.CLI),'job','create','--help'],env=c.ENV,capture_output=True,encoding='utf-8',check=True)
    (HERE/'CREATE_HELP.txt').write_text(help_result.stdout,encoding='utf-8')
    files=('experiment.py','protocol.py','test_protocol.py','config_seed2025.json','config_seed2026.json','PLAN.md')
    c.remote(f'mkdir -p {REMOTE}')
    for name in files:
        c.cli(['notebook','scp','env-zzl',(HERE/name).as_posix(),REMOTE+'/'+name,'--workspace',c.CPU,'-t','120'],'UPLOAD_'+name+'.json')
    expected={name:c.digest(HERE/name) for name in files}
    code=f'''
from pathlib import Path
import hashlib,json,subprocess,sys
r=Path({REMOTE!r});stage=r.parent;source=stage/'h_freeze35_20261029_r1/code'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
expected={expected!r}
for name,digest in expected.items():
    if sha(r/name)!=digest: raise ValueError('Upload changed '+name)
old=json.loads((stage/'goal_20261027_r1/runs/H0/resolved_config.json').read_text(encoding='utf-8'))
for seed in (2025,2026):
    cfg=json.loads((r/f'config_seed{{seed}}.json').read_text(encoding='utf-8'))
    target=dict(old);target['seed']=seed
    if cfg!=target: raise ValueError('Scientific configuration changed beyond seed')
files=list((source/'scripts/listwise_diffgrm').glob('*.py'))+list((source/'external/DiffGRM/genrec/models/DIFF_GRM').glob('*.py'))+list((source/'external/DiffGRM/genrec').glob('*.py'))
manifest=dict(format='multiseed_freeze_v1',seeds=[2025,2026],arms={{'H0':'alternating_R35_T400','H1':'freeze_R20_T200_continue_R35'}},runner_sha256={{n:expected[n] for n in ('experiment.py','protocol.py')}},config_sha256={{str(s):expected[f'config_seed{{s}}.json'] for s in (2025,2026)}},source_sha256={{str(p.relative_to(source)):sha(p) for p in files}})
(r/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\\n')
check=subprocess.run([sys.executable,'-B',str(r/'test_protocol.py')],cwd=r,capture_output=True,text=True)
if check.returncode: raise ValueError(check.stdout+check.stderr)
manifest['cpu_checks']=dict(passed=True,output=check.stdout+check.stderr)
(r/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps(manifest))
'''
    manifest=json.loads(py(code,'STAGE_READBACK.json'))
    c.write('MANIFEST.json',manifest)
    print('Original scientific config preserved; two seeds staged; schedule checks passed')

def submit(seed,arm):
    name=f'nancy-s09-s{seed}-{arm.lower()}-1109-v1'
    stem=f'{seed}_{arm}'
    if (HERE/f'{stem}_SUBMISSION_INTENT.json').exists(): raise ValueError('Prior submission: inspect before retry')
    if c.cli(['job','list','--workspace',c.WORKSPACE,'--keyword',name,'--all'],f'{stem}_ABSENCE.json').get('items'): raise ValueError('Job exists')
    script=f'cd {REMOTE} && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && {c.PYTHON} -B -u experiment.py --seed {seed} --arm {arm} > {stem}.log 2>&1'
    req=['job','create','--name',name,'--workspace',c.WORKSPACE,'--project','公共科研项目','--group',c.GROUP,'--quota','1,20,200','--image',c.IMAGE,'--nodes','1','--shm-size','32','--max-time','24','--priority','4','--no-auto-fault-tolerance','--no-enable-notification','--command',shlex.join(['bash','-c',script])]
    c.cli(req+['--dry-run'],f'{stem}_DRY_RUN.json')
    c.write(f'{stem}_SUBMISSION_INTENT.json',dict(job=name,root=REMOTE,request=req))
    print(c.cli(req,f'{stem}_SUBMISSION_RESULT.json'),flush=True)

def status():
    for seed in (2025,2026):
        for arm in ('H0','H1'):
            stem=f'{seed}_{arm}'
            if not (HERE/f'{stem}_SUBMISSION_INTENT.json').exists(): continue
            name=json.loads((HERE/f'{stem}_SUBMISSION_INTENT.json').read_text(encoding='utf-8'))['job']
            print(c.cli(['job','status',name,'--workspace',c.WORKSPACE],f'{stem}_STATUS.json'),flush=True)
    code=f'''
from pathlib import Path
import json
r=Path({REMOTE!r})
for seed in (2025,2026):
 for arm in ('H0','H1'):
  stem=f'{{seed}}_{{arm}}';log=r/(stem+'.log');run=r/('seed'+stem)
  print(json.dumps(dict(run=stem,log_tail=log.read_text(encoding='utf-8')[-1800:] if log.exists() else '',training_tail=(run/'training.jsonl').read_text().splitlines()[-1] if (run/'training.jsonl').exists() else None,warmup_pair=json.loads((run/'R5_PAIR.json').read_text(encoding='utf-8')) if (run/'R5_PAIR.json').exists() else None)))
'''
    print(py(code,'PROGRESS_READBACK.json'),flush=True)

def stage_analysis():
    name='finalize.py'
    c.cli(['notebook','scp','env-zzl',(HERE/name).as_posix(),REMOTE+'/'+name,'--workspace',c.CPU,'-t','120'],'UPLOAD_finalize.json')
    digest=py(f"import hashlib;print(hashlib.sha256(open({(REMOTE+'/'+name)!r},'rb').read()).hexdigest())")
    if digest.strip()!=c.digest(HERE/name): raise ValueError('Analysis upload changed')
    c.write('ANALYSIS_SOURCE.json',dict(path=REMOTE+'/'+name,sha256=digest.strip()))
    print('Final CPU audit and three-seed comparisons staged')

def accept():
    for seed in (2025,2026):
        for arm in ('H0','H1'):
            stem=f'{seed}_{arm}'
            name=json.loads((HERE/f'{stem}_SUBMISSION_INTENT.json').read_text(encoding='utf-8'))['job']
            state=c.cli(['job','status',name,'--workspace',c.WORKSPACE],f'{stem}_STATUS.json')
            if state['status']!='SUCCEEDED': raise ValueError(stem+' not complete')
            workers=c.cli(['job','instances',name,'--workspace',c.WORKSPACE],f'{stem}_INSTANCES.json')['items']
            if len(workers)!=1 or workers[0]['status']!='instance_success': raise ValueError('GPU release unconfirmed')
    source=json.loads((HERE/'ANALYSIS_SOURCE.json').read_text(encoding='utf-8'))
    code=f"import hashlib,runpy; p={source['path']!r}; assert hashlib.sha256(open(p,'rb').read()).hexdigest()=={source['sha256']!r}; runpy.run_path(p,run_name='__main__')"
    result=json.loads(py(code,'FINAL_ANALYSIS_READBACK.json',timeout=600))
    result['all_four_gpu_released']=True
    c.write('FINAL_ANALYSIS.json',result)
    print(json.dumps(result['comparisons'],ensure_ascii=False))

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='submit': submit(int(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=='status': status()
    elif sys.argv[1]=='stage_analysis': stage_analysis()
    elif sys.argv[1]=='accept': accept()
