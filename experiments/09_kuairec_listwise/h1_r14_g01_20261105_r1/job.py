"""One authorized H100 allocation, gate then train and fixed-R20 test."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY=sys.executable
env=dict(os.environ)
for key in list(env):
    if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','NODE_RANK','MASTER_ADDR','MASTER_PORT'} or key.startswith(('TORCHELASTIC_','OMPI_','PMI_','PMIX_','MV2_')):
        env.pop(key)
env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
DIFFGRM_ENGINE_ROOT=str(ROOT/'code/external/DiffGRM'))
inputs=json.loads((ROOT/'INPUTS.json').read_text())
out=ROOT/'runs/R14-G01'
common=['--parent',inputs['parent'],'--dataset',inputs['dataset'],'--assets',inputs['assets'],
'--subset',str(ROOT/'validation_subset_8192.json'),'--r14-reference',inputs['r14_reference'],
'--output',str(out),'--device','cuda:0']
jobs=ROOT/'jobs'; jobs.mkdir(exist_ok=True)
stages=[]; started=time.time(); status='incomplete'
try:
    commands=[('gate',[PY,'-B','-u',str(ROOT/'gate.py'),'gpu'])]
    for stage in ('train','test'):
        commands.append((stage,[PY,'-B','-u','-m','scripts.listwise_diffgrm.recovery_r14_g01_run',stage]+common))
    commands.append(('restore',[PY,'-B','-u',str(ROOT/'verify_checkpoints.py')]))
    for stage,command in commands:
        with (jobs/(stage+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
        stages.append(dict(stage=stage,exit_code=result.returncode))
        if result.returncode: raise RuntimeError(f'{stage} failed; inspect log')
    status='training_validation_test_complete'
finally:
    receipt=dict(status=status,stages=stages,elapsed_seconds=time.time()-started)
    (jobs/'result.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
