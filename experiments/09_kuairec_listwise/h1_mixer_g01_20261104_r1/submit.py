"""Submit the one authorized single-card G01 formal job once."""
import json,shlex,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_mixer_g01_20261104_r1'
receipt=HERE/'SUBMISSION.json'
if receipt.exists():raise FileExistsError('Inspect existing job; do not resubmit')
exe='/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
command=shlex.join([exe,REMOTE+'/job.py','--root',REMOTE,'--python',exe])
job='nancy-s09-g01-1104-r1'
receipt.write_text(json.dumps(dict(job=job,unix=time.time(),gpus=1,
    forecast_hours=7.0,max_hours=12,prior_stage09_gpu_hours=10.897222222222222,
    command=command),indent=2),encoding='utf-8')
cmd=['C:/Users/nancy/AppData/Roaming/uv/tools/inspire-skill/Scripts/python.exe','-X','utf8',
    str(ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/inspire_compat.py'),'--json','--account','sii-shared',
    'job','create','--name',job,'--workspace','分布式训练空间','--project','公共科研项目',
    '--group','开发区-H100-cuda12.8版本-183核','--quota','1,20,200',
    '--image','ngc-pytorch:25.02-cuda12.8.0-py3','--nodes','1','--shm-size','32',
    '--max-time','12','--no-auto-fault-tolerance','--no-enable-notification','--command',command]
r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
(HERE/'SUBMISSION_RESULT.json').write_text(json.dumps(dict(exit_code=r.returncode,result=rows),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(exit_code=r.returncode,result=rows),ensure_ascii=False))
raise SystemExit(r.returncode)
