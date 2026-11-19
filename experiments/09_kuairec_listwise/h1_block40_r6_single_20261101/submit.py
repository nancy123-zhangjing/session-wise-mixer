"""Submit the one authorized single-card E formal job once."""
import json,shlex,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1'
receipt=HERE/'SUBMISSION.json'
if receipt.exists():raise FileExistsError('Inspect existing job; do not resubmit')
d=json.loads((HERE/'DEPLOYMENT.json').read_text())
exe='/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
command=shlex.join([exe,REMOTE+'/job.py','--root',REMOTE,'--python',exe,
    '--self-sha',d['launcher_sha256'],'--gate',REMOTE+'/RUN_GATE.json','--gate-sha',d['gate_sha256']])
job='nancy-s09-e-single-1101-r1'
receipt.write_text(json.dumps(dict(job=job,unix=time.time(),gpus=1,max_seconds=72000,
    prior_gpu_seconds=3287,release_reserve_seconds=180,command=command),indent=2),encoding='utf-8')
cmd=['C:/Users/nancy/AppData/Roaming/uv/tools/inspire-skill/Scripts/python.exe','-X','utf8',
    str(ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/inspire_compat.py'),'--json','--account','sii-shared',
    'job','create','--name',job,'--workspace','分布式训练空间','--project','公共科研项目',
    '--group','开发区-H100-cuda12.8版本-183核','--quota','1,20,200',
    '--image','ngc-pytorch:25.02-cuda12.8.0-py3','--nodes','1','--shm-size','32',
    '--max-time','20','--no-auto-fault-tolerance','--no-enable-notification','--command',command]
r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
(HERE/'SUBMISSION_RESULT.json').write_text(json.dumps(dict(exit_code=r.returncode,result=rows),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(exit_code=r.returncode,result=rows),ensure_ascii=False))
raise SystemExit(r.returncode)
