"""One screening group on one GPU: (own parent) -> train to R20 -> post-hoc evaluation.

Resumable: each stage is skipped when its receipt exists and resumes from last.pt.
Nothing here submits jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .variants import EVAL_EPOCHS, GUIDANCE, JOBS, SMOKE, STOP_AFTER, VARIANTS, build, own_parent

RUN = 'cf_repeat_v2/runs/CF-M0'
PARENT, PARENT_RQ = 'parents/cf', 'parents/cf_rq'


def invoke(module, *args):
    print(json.dumps({'invoke': module, 'args': list(map(str, args)), 'time': time.time()}), flush=True)
    subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)], check=True)


def locked_config(path, cfg):
    """Write once (race-free per-process temporary name); any later difference is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        tmp.replace(path)
    if json.loads(path.read_text(encoding='utf-8')) != cfg:
        raise ValueError(f'Screening config changed after locking: {path}')
    return path


def first_wave(wave):
    """The first-wave CF-M0 config, which must equal its prepared parent's config."""
    base = json.loads((wave/RUN/'config.json').read_text(encoding='utf-8'))
    meta = json.loads((wave/PARENT/'PREPARED.json').read_text(encoding='utf-8'))
    if meta['config'] != base:
        raise ValueError('First-wave run and parent configs differ')
    return base, meta


def fresh_output(path):
    """A run directory without last.pt cannot be resumed; keep it aside as a record."""
    if path.exists() and not (path/'last.pt').is_file():
        path.rename(path.with_name(f'{path.name}.incomplete.{int(time.time())}'))


def worker(name, a):
    smoke = a.smoke
    base, meta = first_wave(a.wave)
    cfg = build(name, base, smoke=smoke)
    root = a.root/('smoke' if smoke else 'runs')
    config = locked_config(root/'configs'/f'{name}.json', cfg)
    common = ('--runtime', a.runtime, '--dataset', meta['dataset_path'], '--device', 'cuda')
    head = ('--resolved-config', config)
    if own_parent(name):
        prepared, rq = root/'parents'/name/'prepared', root/'parents'/name/'rq'
        if not (prepared/'PREPARED.json').is_file():
            fresh_output(prepared)
            invoke('scripts.rqcd_mixer.run', *head, 'prepare', *common, '--content', meta['content_path'],
                   '--teacher', meta['teacher_path'], '--output', prepared)
        if not (rq/'COMPLETE.json').is_file():
            resume = ('--resume',) if (rq/'last.pt').is_file() else ()
            if not resume:
                fresh_output(rq)
            invoke('scripts.rqcd_mixer.run', *head, 'pretrain', *common, '--prepared', prepared, '--arm', 'B0',
                   '--output', rq, *resume)
    else:
        prepared, rq = a.wave/PARENT, a.wave/PARENT_RQ
    run = root/'train'/name
    stop = SMOKE['stop_after'] if smoke else STOP_AFTER
    if not (run/'STOPPED.json').is_file():
        resume = ('--resume',) if (run/'last.pt').is_file() else ()
        if not resume:
            fresh_output(run)
        invoke('scripts.rqcd_mixer.run', *head, 'train', *common, '--prepared', prepared, '--pretrained', rq,
               '--arm', 'B0', '--output', run, '--stop-after', stop, *resume)
    switches = cfg['screen']
    evaluation = ['--run-dir', run, '--prepared', prepared, '--dataset', meta['dataset_path'],
                  '--device', 'cuda', '--lists', '0']
    if smoke:
        evaluation += ['--output', root/'eval'/f'selftest_{name}', '--epochs', *SMOKE['eval_epochs'],
                       '--ks', *SMOKE['ks'], '--limit', SMOKE['limit']]
        guidance = SMOKE['guidance']
    else:
        evaluation += ['--output', root/'eval'/name, '--epochs', *EVAL_EPOCHS, '--ks', 16]
        guidance = GUIDANCE
    if 'guidance' in switches:
        evaluation += ['--guidance', *guidance]
    if 'history_head' in switches:
        evaluation.append('--history-head')
    invoke('scripts.cf_decision.run', *evaluation)
    record = {'name': name, 'smoke': smoke, 'config': str(config), 'prepared': str(prepared), 'pretrained': str(rq),
              'run': str(run), 'stop_after': stop, 'what': VARIANTS[name][2]}
    (root/'slots').mkdir(parents=True, exist_ok=True)
    (root/'slots'/f'{name}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=sorted(VARIANTS))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--wave', type=Path, required=True, help='first-wave L5O root (l5o_wave_20261117_v1)')
    p.add_argument('--runtime', type=Path, required=True, help='the locked CF runtime of the first wave')
    p.add_argument('--smoke', action='store_true', help='tiny budgets for the GPU smoke test (never results)')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    worker(a.name, a)
    print(json.dumps({'worker_complete': a.name, 'smoke': a.smoke}), flush=True)


if __name__ == '__main__':
    main()
