"""Observe four existing final evaluation Jobs; never submit or restart."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import shlex
import time
import uuid

HERE = Path(__file__).absolute().parent
ARMS = ('J0', 'J1', 'H0', 'H1')
NAMES = {a: 'nancy-stage09-final-eval-' + a.lower() + '-20261027-' + ('r2' if a == 'H1' else 'r1') for a in ARMS}
DEADLINE = datetime(2026,10,28,12,0,tzinfo=timezone.utc).timestamp()
MARKER = 'STAGE09_FINAL_OBSERVATION '
READ = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
g=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
out={'remote_observed_at_utc':datetime.now(timezone.utc).isoformat(),'evaluations':{},'last_logged_progress':{}}
for arm in ('J0','J1','H0','H1'):
 for suffix in ('test_beam32','validation_subset_beam32','validation_subset_beam256'):
  name=arm+'_'+suffix+('_r2' if arm=='H1' else '');p=g/'evaluations'/name;s=p/'summary.json'
  row={'directory_exists':p.is_dir(),'summary_present':s.is_file()}
  if s.is_file():
   try:
    raw=s.read_bytes();v=json.loads(raw)
    row.update(summary_sha256=hashlib.sha256(raw).hexdigest(),split=v.get('split'),examples=v.get('examples'),beam=v.get('beam'),seconds=v.get('seconds'),r_updates=v.get('r_updates'),t_updates=v.get('t_updates'),metrics={k:v['metrics'][k] for k in ('list/ndcg@5','list/hr@5','list/recall@5','aggregate/ndcg@50','aggregate/coverage@50')})
   except Exception as e:row['read_exception_type']=type(e).__name__
  out['evaluations'][name]=row
 log=g/'logs'/('final_eval_'+arm.lower()+'_20261027_'+('r2' if arm=='H1' else 'r1')+'.log')
 if log.is_file():
  with log.open('rb') as f:
   f.seek(0,2);f.seek(max(0,f.tell()-65536));lines=f.read().decode('utf-8',errors='replace').splitlines()
  for line in reversed(lines):
   try:v=json.loads(line)
   except ValueError:continue
   if isinstance(v,dict) and v.get('event')=='evaluation_progress':
    out['last_logged_progress'][arm]=v;break
print('STAGE09_FINAL_OBSERVATION '+json.dumps(out,allow_nan=False))
'''


def observe():
    out=dict(schema='stage09_final_evaluation_observation_v1',observed_at_utc=datetime.now(timezone.utc).isoformat(),
             status='started',platform={},all_terminal=False)
    try:
        from inspire import InspireClient
        from inspire.services.job.job_status import TERMINAL_STATUSES, normalize_status
        started=time.monotonic()
        with InspireClient(account='sii-shared',allow_browser=False,timeout=12,
                           operation_timeout=65,catalog_disk_cache=False) as client:
            jobs={}
            for a in ARMS:
                client.operation_timeout=max(5,int(65-(time.monotonic()-started)))
                job=client.jobs.get(NAMES[a],workspace='分布式训练空间');jobs[a]=job
                out['platform'][a]=dict(name=job.name,status=job.status,raw_status=job.raw_status)
            out['all_terminal']=all(j.status.upper() in TERMINAL_STATUSES or normalize_status(j.raw_status) in TERMINAL_STATUSES for j in jobs.values())
            if out['all_terminal']:
                out['status']='all_platform_terminal';return out
            running=next((j for j in jobs.values() if j.status.upper()=='RUNNING'),None)
            if running is None:
                out['status']='none_running';return out
            client.operation_timeout=max(5,int(65-(time.monotonic()-started)))
            result=client.jobs.exec(running.ref,command='python -B -c '+shlex.quote(READ),timeout=35,max_output_bytes=65536)
            if not result.completed or result.returncode != 0 or result.truncated:raise RuntimeError('Read incomplete')
            rows=[json.loads(line[len(MARKER):]) for line in result.stdout.splitlines() if line.startswith(MARKER)]
            if len(rows)!=1 or len(rows[0]['evaluations'])!=12:raise ValueError('Observation identity differs')
            out.update(rows[0]);out['status']='observed'
    except Exception as error:
        out.update(status='observation_error',error_type=type(error).__name__,all_terminal=False)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    os.environ['NO_PROXY']='.sii.edu.cn,<institution-host>';logging.disable(logging.CRITICAL)
    while True:
        started=time.time()
        if started>=DEADLINE:
            print(json.dumps(dict(status='observation_deadline',goal_complete=False)),flush=True);return
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):record=observe()
        raw=json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'
        with (HERE/'OBSERVE_FINAL_EVAL_R2_HISTORY.jsonl').open('a',encoding='utf-8') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        temp=HERE/('.final_eval_observe_'+uuid.uuid4().hex+'.tmp');temp.write_text(raw,encoding='utf-8')
        os.replace(temp,HERE/'OBSERVE_FINAL_EVAL_R2_LATEST.json')
        print(json.dumps(dict(observed_at=record['observed_at_utc'],status=record['status'],platform=record['platform'],
            completed_summaries=sum(bool(v.get('summary_present') and not v.get('read_exception_type')) for v in record.get('evaluations',{}).values()),
            progress=record.get('last_logged_progress',{})),ensure_ascii=False),flush=True)
        if not args.watch or record['all_terminal']:return
        time.sleep(max(0,min(started+300,DEADLINE)-time.time()))


if __name__=='__main__':main()
