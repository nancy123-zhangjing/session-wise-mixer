"""Versioned TIGER rerun worker: pretrain -> train -> test for AR-T0A / AR-T1A. No submission on import."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/inputs/dataset'
CONTENT = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/assets/pse_v1/content_features.npy'
PARENT = ROOT / 'parents' / 'tiger_adamw'


def state(arm, phase, **extra):
    path = ROOT / f'{arm}_STATE.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(arm=arm, phase=phase, unix=time.time(), pid=os.getpid(), **extra)), encoding='utf-8')
    tmp.replace(path)


def invoke(command):
    subprocess.run([sys.executable, '-u', '-B', *command], cwd=ROOT, check=True)


def prepare():
    if (PARENT / 'manifest.json').exists():
        return
    invoke(['-m', 'scripts.session_wise_ar.run', '--arm', 'AR-T0A', 'prepare', '--dataset', DATA,
            '--content', CONTENT, '--output', str(PARENT), '--device', 'cpu'])


def run(arm):
    assert arm in ('AR-T0A', 'AR-T1A')
    if not (PARENT / 'manifest.json').exists():
        raise RuntimeError('Shared parent missing; prepare on CPU first')
    target = ROOT / 'runs' / arm
    target.mkdir(parents=True, exist_ok=True)
    base = ['-m', 'scripts.session_wise_ar.run', '--arm', arm]
    common = ['--dataset', DATA, '--content', CONTENT, '--prepared', str(PARENT), '--device', 'cuda']
    try:
        for phase in ('pretrain', 'train', 'test'):
            if (target / phase / 'complete.json').exists():
                continue
            state(arm, phase)
            extra = []
            if phase in ('train', 'test'):
                extra += ['--pretrain', str(target / 'pretrain')]
            if phase == 'test':
                extra += ['--training', str(target / 'train')]
            elif (target / phase / 'last.pt').exists():
                extra += ['--resume']
            invoke([*base, phase, *common, '--output', str(target / phase), *extra])
        state(arm, 'training_and_test_finished')
    except Exception as exc:
        state(arm, 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    action = sys.argv[1]
    prepare() if action == 'prepare' else run(action)
