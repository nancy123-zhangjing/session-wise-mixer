"""Await the already-running download, then execute only authorized CPU acceptance.

No training, inference, remote resource creation, or automatic retry.
Final figure/document review and remote/resource closure remain separate.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PY='C:/Users/nancy/AppData/Local/Programs/Python/Python310/python.exe'
FRESH=ROOT/'outputs/stage09_block40_single_20261101_fresh_r1'
RUN=FRESH/'runs/H1-Block40-R6-R35'

def event(**row):
    row['time']=time.time()
    with (HERE/'LOCAL_ACCEPTANCE_PROGRESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download-pid',type=int,required=True);args=ap.parse_args()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,args.download_pid)
    if not handle:raise RuntimeError('Named download is not live; inspect before continuing')
    event(stage='wait_live_download',pid=args.download_pid)
    try:
        while True:
            status=kernel.WaitForSingleObject(handle,60000)
            if status==0:break
            if status!=258:raise OSError('Download process wait failed')
    finally:kernel.CloseHandle(handle)
    receipt=HERE/'delivery/RESUMED_DOWNLOAD_ACCEPTED.json'
    if not receipt.is_file() or json.loads(receipt.read_text())['status']!='passed_authenticated_prefix_resume_and_full_archive':
        raise RuntimeError('Download exited without authenticated full archive; no CPU acceptance started')
    steps=[
        ('extract',PY,'archive_delivery.py',['extract','--archive',HERE/'delivery/stage09_h1_block40_single_r1.tar.gz','--receipt',HERE/'delivery/ARCHIVE_RECEIPT.json','--destination',FRESH]),
        ('operational',PY,'audit_operational.py',['--extracted',FRESH,'--output',HERE/'OPERATIONAL_AUDIT.json']),
        ('metrics',PY,'analyze_e.py',['--run',RUN,'--output',HERE/'final_analysis']),
        ('cold_restore',ROOT/'.runtime/stage08_restore_env/Scripts/python.exe','cold_restore.py',['--extracted',FRESH,'--parent',ROOT/'outputs/stage09_restore/goal_20261027_r1/runs/H1/initial.pt','--dataset',ROOT/'outputs/kuairec_listwise_v1/run1','--assets',ROOT/'outputs/stage09_restore/goal_20261027_r1/assets/pse_v1','--output',HERE/'COLD_RESTORE.json']),
        ('process',PY,'process_e.py',['--run',RUN,'--cold-restore',HERE/'COLD_RESTORE.json','--output',HERE/'final_process']),
        ('temporal',PY,'temporal_e.py',['--analysis',HERE/'final_analysis','--output',HERE/'final_temporal']),
        ('report',PY,'report_e.py',['--output',HERE/'final_report'])]
    for name,interpreter,script,arguments in steps:
        log=HERE/f'ACCEPTANCE_{name}.log'
        event(stage=name,status='starting',log=str(log))
        with log.open('x',encoding='utf-8') as output:
            result=subprocess.run([str(interpreter),'-X','utf8',str(HERE/script),*map(str,arguments)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        event(stage=name,exit_code=result.returncode)
        if result.returncode:raise RuntimeError('Acceptance stopped at '+name+'; inspect its log, never rerun passed steps blindly')
    event(stage='local_pipeline_finished',status='manual_visual_and_Feishu_review_remote_sync_resource_closure_still_required')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        event(stage='pipeline_stopped',error=str(exc))
        raise
