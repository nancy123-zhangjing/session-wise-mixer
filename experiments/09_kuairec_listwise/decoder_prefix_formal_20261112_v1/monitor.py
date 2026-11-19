"""Read-only snapshots of the four existing runs; never starts or restarts a job."""
import concurrent.futures
import json
from pathlib import Path
import runpy
import shlex
import time

HERE=Path(__file__).resolve().parent
ctl=runpy.run_path(str(HERE/'control.py'))
c=ctl['c']

REMOTE_CODE=r'''
import pathlib,json,time
p=pathlib.Path(REMOTE)
def tail(path,limit=65536):
 if not path.exists():return ''
 with path.open('rb') as f:
  f.seek(0,2);size=f.tell();f.seek(max(0,size-limit));return f.read().decode('utf-8',errors='replace')
def read(path):return json.loads(path.read_text())
result={'observed_unix':time.time(),'variants':{}}
for v in ('H0-A','H1-A','H0-B','H1-B'):
 root=p/v;row={'validations':[],'tests':[],'checkpoints':[]}
 for stage in ('warmup','train'):
  lines=tail(root/stage/'training.jsonl').splitlines();valid=[]
  for s in lines:
   try:valid.append(json.loads(s))
   except ValueError:pass
  if valid:row['training']={'stage':stage,'last':valid[-1],'mtime':(root/stage/'training.jsonl').stat().st_mtime}
  for f in sorted((root/stage).glob('validation_R*/summary.json')):
   data=read(f);data['path']=str(f);row['validations'].append(data)
  for f in sorted((root/stage/'checkpoints').glob('R*.json')):row['checkpoints'].append(read(f))
 for f in sorted(root.glob('test_*/summary.json')):
  data=read(f);data['path']=str(f);row['tests'].append(data)
 for name in ('R5_PAIR.json','TRAINING_COMPLETE.json','GPU_COMPLETE.json'):
  if (root/name).exists():row[name]=read(root/name)
 log=tail(p/(v+'.log'));events=[]
 for s in log.splitlines():
  try:
   d=json.loads(s)
   if isinstance(d,dict) and 'event' in d:events.append(d)
  except ValueError:pass
 row['events']=events[-3:];row['error_tail']=log[-2000:] if 'Traceback (most recent call last)' in log else None
 result['variants'][v]=row
print(json.dumps(result))
'''

def main():
 recovery_path=HERE/'RECOVERY_JOBS.json'
 recovery=json.loads(recovery_path.read_text(encoding='utf-8')) if recovery_path.exists() else {}
 def status(v):
  return v,c.cli(['job','status',recovery.get(v,'nancy-prefix-'+v.lower()+'-r20-1112-v1'),'--workspace',c.WORKSPACE],'STATUS_'+v+'.json',45)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:
  states=dict(pool.map(status,ctl['VARIANTS']))
 snapshot=json.loads(c.remote(c.PYTHON+' -B -c '+shlex.quote('REMOTE='+repr(ctl['REMOTE'])+'\n'+REMOTE_CODE),'MONITOR_READBACK.json',45))
 snapshot['platform']=states
 folder=HERE/'snapshots';folder.mkdir(exist_ok=True)
 name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'
 (folder/name).write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 c.write('LATEST.json',snapshot)
 for v,row in snapshot['variants'].items():
  print(json.dumps(dict(variant=v,status=states[v]['status'],training=row.get('training'),validations=[{'R':x['r_updates']/693,'T':x['t_updates'],'ndcg5':x['metrics']['list/ndcg@5'],'seconds':x['seconds']} for x in row['validations']],events=row['events'],error=row['error_tail']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
