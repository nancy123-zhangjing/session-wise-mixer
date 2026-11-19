"""Observe the existing Stage09 Job; never submit, restart, or write remotely.

Default: one observation. --watch: every 300 seconds, stopping only on a live
platform terminal status or the fixed observation deadline. Summaries are
observations, not checkpoint/final-delivery acceptance.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import logging
import os
from pathlib import Path
import shlex
import time
import uuid

HERE = Path(__file__).resolve().parent
JOB = "nancy-stage09-four-arm-20261027-r1"
WORKSPACE = "分布式训练空间"
INTERVAL = 300
WATCH_DEADLINE = 1793201843 + 600
LATEST = HERE / "OBSERVE_STAGE09_LATEST.json"
HISTORY = HERE / "OBSERVE_STAGE09_OBSERVATIONS.jsonl"
MARKER = "STAGE09_COMPACT_JSON="

REMOTE_READ = r'''
from pathlib import Path
from datetime import datetime, timezone
import json, re, time
g=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')

def tail_json(path, event=None):
    if not path.is_file(): return None
    with path.open('rb') as f:
        f.seek(0,2); n=f.tell(); f.seek(max(0,n-262144))
        lines=f.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try: row=json.loads(line)
        except (ValueError,UnicodeError): continue
        if isinstance(row,dict) and (event is None or row.get('event')==event): return row
    return None

def pick(row, keys):
    return {k:row[k] for k in keys if k in row}

out={'remote_observed_at_utc':datetime.now(timezone.utc).isoformat(),'arms':{}}
for arm in ('J0','J1','H0','H1'):
    result={'training':None,'current_validation':None,'latest_complete_validation':None,
            'complete_file_present':False,'complete':None}
    out['arms'][arm]=result
    try:
        run=g/'runs'/arm
        training=tail_json(run/'training.jsonl')
        if training:
            result['training']=pick(training,('phase','phase_index','epoch','update','r_updates',
                                               't_updates','loss','gradient_norm'))
        directories=[]
        for path in run.glob('validation_R*'):
            match=re.fullmatch(r'validation_R(\d+)(?:_retry(\d+))?',path.name)
            if match and path.is_dir():
                directories.append((int(match[1]),int(match[2] or 0),path))
        directories.sort(reverse=True)
        for epoch,retry,path in directories:
            summary=path/'summary.json'
            if not summary.is_file(): continue
            try: data=json.loads(summary.read_text(encoding='utf-8'))
            except (ValueError,OSError): continue  # A summary can be mid-write.
            if data.get('examples')!=71657 or data.get('full_split') is not True: continue
            compact=pick(data,('split','examples','full_split','subset','diagnostic_prefix_limit',
                              'beam','inference_precision','architecture','seed','r_updates',
                              't_updates','seconds','catalog_size','metric_protocol'))
            compact.update(epoch=epoch,directory=path.name,
                data_manifest_sha256=data.get('dataset_binding',{}).get('manifest_sha256'),
                validation_sha256=data.get('dataset_binding',{}).get('files',{}).get('validation.parquet',{}).get('sha256'),
                metrics=pick(data.get('metrics',{}),('list/ndcg@5','list/hr@5','list/recall@5',
                    'list/completion_rate','list/invalid_rate','aggregate/ndcg@10',
                    'aggregate/ndcg@20','aggregate/ndcg@50','aggregate/coverage@10',
                    'aggregate/coverage@20','aggregate/coverage@50','aggregate/candidate_count')))
            result['latest_complete_validation']=compact
            break
        if directories and training:
            epoch,retry,path=directories[0]
            if (training.get('phase')=='R' and training.get('epoch')==epoch
                and training.get('update')==693 and not (path/'summary.json').is_file()):
                current={'epoch':epoch,'directory':path.name,'progress':None}
                log=g/'delivery/four_arm_supervisor_r1'/('train_'+arm+'.log')
                # Never relabel an old R08 progress line as a new validation.
                if log.is_file() and log.stat().st_mtime>=path.stat().st_mtime:
                    progress=tail_json(log,'evaluation_progress')
                    if (progress and progress.get('split')=='validation'
                        and 0<=progress.get('seconds',-1)<=time.time()-path.stat().st_mtime+120):
                        current['progress']=pick(progress,('processed','total','seconds'))
                result['current_validation']=current
        complete=run/'complete.json'
        result['complete_file_present']=complete.is_file()
        if complete.is_file():
            try:
                result['complete']=pick(json.loads(complete.read_text(encoding='utf-8')),
                    ('arm','r_updates','t_updates','best_epoch','best_score'))
            except (ValueError,OSError) as error:
                result['complete_read_exception_type']=type(error).__name__
    except Exception as error:
        result['exception_type']=type(error).__name__
print('STAGE09_COMPACT_JSON='+json.dumps(out,separators=(',',':'),allow_nan=False))
'''


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def observe(watch):
    record = {'schema':'stage09_compact_observation_v1', 'observed_at_utc':utc(),
              'job_name':JOB, 'status':'observation_started', 'platform_terminal':False}

    def budget():
        remaining = WATCH_DEADLINE - time.time() if watch else 60
        if remaining < 1:
            raise TimeoutError()
        return min(60, remaining)

    try:
        from inspire import InspireClient
        from inspire.services.job.job_status import TERMINAL_STATUSES, normalize_status
        with InspireClient(account='sii-shared', allow_browser=False, timeout=15,
                           operation_timeout=budget(), catalog_disk_cache=False) as client:
            job = client.jobs.get(JOB, workspace=WORKSPACE)
            record['platform'] = {k:getattr(job,k) for k in
                ('name','status','raw_status','project','created_at','finished_at')}
            terminal = job.status.upper() in TERMINAL_STATUSES or normalize_status(job.raw_status) in TERMINAL_STATUSES
            record['platform_terminal'] = terminal
            if terminal:
                record['status'] = 'platform_terminal'
                return record
            if job.status.upper() != 'RUNNING':
                record['status'] = 'platform_not_running'
                return record
            client.operation_timeout = budget()
            result = client.jobs.exec(job.ref, command='python -B -c '+shlex.quote(REMOTE_READ),
                                      timeout=min(40,budget()), max_output_bytes=65536)
            if not result.completed or result.returncode != 0 or result.truncated:
                raise RuntimeError()
            rows = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
            if len(rows) != 1:
                raise ValueError()
            remote = json.loads(rows[0])
            if set(remote.get('arms',{})) != {'J0','J1','H0','H1'}:
                raise ValueError()
            record.update(remote)
            record['status'] = 'observed'
    except Exception as error:
        record['status'] = 'observation_error'
        record['exception_type'] = type(error).__name__
        record['platform_terminal'] = False
    return record


def persist(record):
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
    # Append first, then atomically replace the compact latest snapshot.
    with HISTORY.open('a',encoding='utf-8') as stream:
        stream.write(encoded+'\n'); stream.flush(); os.fsync(stream.fileno())
    temporary = HERE / ('.observe_stage09_'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(encoded+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,LATEST)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true',help='Observe every 300s until platform terminal or fixed deadline.')
    args=parser.parse_args()
    logging.disable(logging.CRITICAL)
    while True:
        started=time.time()
        if args.watch and started>=WATCH_DEADLINE:
            record={'schema':'stage09_compact_observation_v1','observed_at_utc':utc(),
                    'job_name':JOB,'status':'observation_deadline_reached','platform_terminal':False}
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                record=observe(args.watch)
        persist(record)
        print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)
        if record['platform_terminal'] or record['status']=='observation_deadline_reached':
            return 0
        if not args.watch:
            return 1 if record['status']=='observation_error' else 0
        time.sleep(max(0,min(started+INTERVAL,WATCH_DEADLINE)-time.time()))


if __name__=='__main__':
    raise SystemExit(main())
