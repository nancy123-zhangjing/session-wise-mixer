"""Explicit five-arm deployment worker; no platform submission on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]  # deployed as delivery/worker.py
DATA = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/inputs/dataset'
CONTENT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/assets/pse_v1/content_features.npy'
CF_PARENT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1/cf_prepared'
ARMS = ('AR-T0','AR-T1','AR-D0','AR-D1','CF-M0')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def state(arm, phase, **extra):
    path = ROOT/f'{arm}_STATE.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(arm=arm, phase=phase, unix=time.time(), pid=os.getpid(), **extra)), encoding='utf-8')
    temporary.replace(path)


def invoke(command):
    subprocess.run([sys.executable, '-u', '-B', *command], cwd=ROOT, check=True)


def parent(family):
    return ROOT/'parents_devicefix'/family


def prepare(family):
    target = parent(family)
    if (target/'manifest.json').exists():
        sys.path.insert(0,str(ROOT))
        from scripts.session_wise_ar.config import load
        from scripts.session_wise_ar.runtime import load_parent
        load_parent(load('AR-T0' if family=='tiger' else 'AR-D0'),target)
        return
    invoke(['-m','scripts.session_wise_ar.run','--arm','AR-T0' if family=='tiger' else 'AR-D0',
            'prepare','--dataset',DATA,'--content',CONTENT,'--output',str(target),'--device','cpu'])


def run(arm):
    target = ROOT/'runs'/arm
    target.mkdir(parents=True,exist_ok=True)
    try:
        if arm == 'CF-M0':
            common = ['--dataset',DATA,'--prepared',CF_PARENT,'--output',str(target)]
            for phase, receipt in (('pretrain',target/'pretrain/COMPLETE.json'),
                                   ('train',target/'train/TRAINING_COMPLETE.json'),
                                   ('test',target/'train/TEST_COMPLETE.json')):
                if receipt.exists(): continue
                state(arm,phase)
                resume = ['--resume'] if phase!='test' and (target/phase/'last.pt').exists() else []
                invoke(['-m','scripts.session_wise_ar.cf',phase,*common,*resume])
        else:
            family = 'tiger' if arm.startswith('AR-T') else 'diger'
            if not (parent(family)/'manifest.json').exists():
                raise RuntimeError('Common parent is missing; do not wait while holding a GPU')
            base = ['-m','scripts.session_wise_ar.run','--arm',arm]
            common = ['--dataset',DATA,'--content',CONTENT,'--prepared',str(parent(family)),'--device','cuda']
            for phase in ('pretrain','train','test'):
                if (target/phase/'complete.json').exists(): continue
                state(arm,phase)
                extra = []
                if phase in ('train','test'): extra += ['--pretrain',str(target/'pretrain')]
                if phase == 'test': extra += ['--training',str(target/'train')]
                elif (target/phase/'last.pt').exists(): extra += ['--resume']
                invoke([*base,phase,*common,'--output',str(target/phase),*extra])
        state(arm,'training_and_test_finished', acceptance='pending')
    except Exception as exc:
        state(arm,'failed',error=str(exc)); raise


def snapshot():
    result = {}
    for arm in ARMS:
        target = ROOT/'runs'/arm
        result[arm] = {'state':read(ROOT/f'{arm}_STATE.json')}
        for phase in ('pretrain','train'):
            progress = read(target/phase/'progress.json')
            if progress:
                result[arm][phase] = {k:v for k,v in progress.items()
                                     if k not in ('previous_books','records')}
                result[arm][phase]['last_record'] = (progress.get('records') or [None])[-1]
        result[arm]['test_complete'] = bool((target/'test/complete.json').exists() or (target/'train/TEST_COMPLETE.json').exists())
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=(*ARMS,'prepare-tiger','prepare-diger','snapshot'))
    args = p.parse_args()
    if args.action.startswith('prepare-'): prepare(args.action.split('-')[1])
    elif args.action=='snapshot': snapshot()
    else: run(args.action)
