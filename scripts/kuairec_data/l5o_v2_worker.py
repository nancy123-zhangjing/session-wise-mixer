"""Protocol v2 worker: one (family, arm, seed) pipeline on one GPU.

Resumable and idempotent: every stage is skipped when its completion receipt
exists and resumed from last.pt otherwise. Arms of the same seed share one
parent (H warmup, TIGER prepare, CF prepare + RQ pretraining); the no-Mixer arm
builds it and the Mixer arm waits for its receipt. Nothing here submits jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .l5o_v2 import ARMS, ROOT, SEEDS, build

WAIT_SECONDS, WAIT_LIMIT = 60, 12 * 3600


def invoke(module, *args):
    print(json.dumps({'invoke': module, 'args': list(map(str, args))}), flush=True)
    subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)], check=True)


def wait_for(path, what):
    start = time.monotonic()
    while not Path(path).is_file():
        if time.monotonic() - start > WAIT_LIMIT:
            raise TimeoutError(f'Gave up waiting for {what}: {path}')
        time.sleep(WAIT_SECONDS)


def locked_config(root, name, cfg):
    """Write once; any later difference is an error (no silent protocol drift)."""
    path = root/'configs'/f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Both arms of a seed start at once and lock the same file: a per-process
        # temporary name keeps the atomic replace race-free (2026-11-18 H/CF bug).
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        tmp.replace(path)
    if json.loads(path.read_text(encoding='utf-8')) != cfg:
        raise ValueError(f'Protocol v2 config changed after locking: {path}')
    return path


def h_smoothed_node(arm_dir, warmup_dir):
    """Validation points of the whole trajectory (shared warmup R1-R5 + this arm)."""
    import torch
    from .selection import smoothed_choice
    progress = torch.load(arm_dir/'last.pt', map_location='cpu', weights_only=False)['progress']
    points = []
    for destination in progress['validations']:
        destination = Path(destination).resolve()
        summary = json.loads((destination/'summary.json').read_text(encoding='utf-8'))
        epoch = int(destination.name.split('_R')[1][:2])
        owner = warmup_dir if destination.parent == Path(warmup_dir).resolve() else arm_dir
        points.append(((epoch, owner/'checkpoints'/f'R{epoch:02d}.pt'), summary['metrics']['ev_ndcg@5']))
    epochs = [node[0] for node, _ in points]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f'Validation trajectory is not one point per R unit: {epochs}')
    (epoch, checkpoint), smoothed, raw, _ = smoothed_choice(points)
    return epoch, checkpoint, smoothed, raw


def h_worker(arm, seed, root, a):
    module = 'scripts.listwise_diffgrm.run'
    old_split = getattr(a, 'h_split', 'l5o') == 'old'
    cfg = locked_config(root, f'h_{seed}', build('h', arm, seed, receipt=a.receipt, dataset=a.dataset, assets=a.assets,
                                                  smoke_test=a.smoke, h_rhythm=getattr(a, 'h_rhythm', 'pass'),
                                                  h_split=getattr(a, 'h_split', 'l5o')))
    gpu = ('--device', 'cuda', '--precision', 'bf16')
    base = root/'h'/f'seed{seed}'
    warmup, output = base/'warmup', base/arm
    if arm == 'H0' and not (warmup/'complete.json').is_file():
        resume = ('--resume', warmup/'last.pt') if (warmup/'last.pt').is_file() else ()
        invoke(module, '--config', cfg, 'warmup', *gpu, '--output', warmup, *resume)
    wait_for(warmup/'complete.json', 'shared H warmup')
    if not (output/'complete.json').is_file():
        start = (('--resume', output/'last.pt') if (output/'last.pt').is_file()
                 else ('--shared-warmup', warmup/'last.pt'))
        invoke(module, '--config', cfg, 'train', *gpu, '--arm', arm, '--output', output, *start)
    if old_split:
        # Old-split control: validation every 3 R units with the old exposure NDCG@5, so the
        # accepted old comparison nodes are tested: validation best and the fixed R35 endpoint.
        for name, checkpoint in (('best', output/'best.pt'), ('final', output/'last.pt')):
            destination = output/'test'/name
            if not (destination/'summary.json').is_file():
                invoke(module, '--config', cfg, 'evaluate', *gpu, '--checkpoint', checkpoint,
                       '--split', 'test', '--output', destination)
        return
    epoch, smoothed_checkpoint, smoothed, raw = h_smoothed_node(output, warmup)
    (output/'SMOOTHED_SELECTION.json').write_text(json.dumps(dict(
        R=epoch, checkpoint=str(smoothed_checkpoint), smoothed_validation=smoothed, raw_validation=raw), indent=2))
    for name, checkpoint in (('best', output/'best.pt'), ('final', output/'last.pt'), ('smoothed', smoothed_checkpoint)):
        destination = output/'test'/name
        if not (destination/'summary.json').is_file():
            invoke(module, '--config', cfg, 'evaluate', *gpu, '--checkpoint', checkpoint,
                   '--split', 'test', '--output', destination)


def tiger_worker(arm, seed, root, a):
    module = 'scripts.session_wise_ar.run'
    cfg = locked_config(root, f'tiger_{arm}_{seed}', build('tiger', arm, seed, receipt=a.receipt,
                                                            tiger_batch=a.tiger_batch, smoke_test=a.smoke))
    base = ('--arm', f'AR-{arm}', '--resolved-config', cfg)
    common = ('--dataset', a.dataset, '--content', a.assets/'content_features.npy')
    parent = root/'tiger'/f'seed{seed}'/'parent'
    output = root/'tiger'/f'seed{seed}'/arm
    if arm == 'T0A' and not (parent/'manifest.json').is_file():
        # Same as the first wave: the shared parent is prepared on CPU.
        invoke(module, *base, 'prepare', *common, '--output', parent, '--device', 'cpu')
    wait_for(parent/'manifest.json', 'shared TIGER parent')
    for phase in ('pretrain', 'train', 'test'):
        target = output/phase
        if (target/'complete.json').is_file():
            continue
        dependencies = ('--pretrain', output/'pretrain') if phase != 'pretrain' else ()
        if phase == 'test':
            dependencies += ('--training', output/'train')
        elif (target/'last.pt').is_file():
            dependencies += ('--resume',)
        invoke(module, *base, phase, *common, '--prepared', parent, '--device', 'cuda',
               '--output', target, *dependencies)


def cf_worker(arm, seed, root, a):
    module = 'scripts.rqcd_mixer.run'
    cfg = locked_config(root, f'cf_{seed}', build('cf', arm, seed, receipt=a.receipt, smoke_test=a.smoke))
    base = ('--resolved-config', cfg)
    common = ('--runtime', a.cf_runtime, '--dataset', a.dataset, '--device', 'cuda')
    seed_root = root/'cf'/f'seed{seed}'
    parent, rq, output = seed_root/'parent', seed_root/'rq', seed_root/arm
    if arm == 'CF-M0':
        if not (parent/'PREPARED.json').is_file():
            invoke(module, *base, 'prepare', *common, '--content', a.assets/'content_features.npy',
                   '--teacher', Path(a.receipt).parent/'items.pt', '--output', parent)
        if not (rq/'COMPLETE.json').is_file():
            resume = ('--resume',) if (rq/'last.pt').is_file() else ()
            invoke(module, *base, 'pretrain', *common, '--prepared', parent, '--arm', 'B0', '--output', rq, *resume)
    wait_for(rq/'COMPLETE.json', 'shared CF RQ pretraining')
    shared = ('--prepared', parent, '--pretrained', rq, '--arm', 'B0' if arm == 'CF-M0' else 'B1')
    if not (output/'TRAINING_COMPLETE.json').is_file():
        resume = ('--resume',) if (output/'last.pt').is_file() else ()
        invoke(module, *base, 'train', *common, *shared, '--output', output, *resume)
    if not (output/'TEST_COMPLETE.json').is_file():
        invoke(module, *base, 'test', *common, *shared, '--run', output)


def sasrec_worker(arm, seed, root, a):
    cfg = locked_config(root, f'sasrec_{seed}', build('sasrec', arm, seed, smoke_test=a.smoke))
    output = root/'sasrec'/f'seed{seed}'
    if (output/'COMPLETE.json').is_file():
        return
    resume = ('--resume',) if (output/'last.pt').is_file() else ()
    # The SASRec trainer takes no micro-batch settings: the execution template
    # (BF16, fused AdamW, deterministic) is what the first-wave calibration used.
    runtime = ROOT/'configs/rqcd_mixer/runtime.example.json'
    invoke('scripts.rqcd_mixer.l5o_calibrate', '--dataset', a.dataset, '--output', output,
           '--config', cfg, '--runtime', runtime, '--device', 'cuda', *resume)


WORKERS = {'h': h_worker, 'tiger': tiger_worker, 'cf': cf_worker, 'sasrec': sasrec_worker}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family', choices=sorted(WORKERS))
    p.add_argument('arm')
    p.add_argument('seed', type=int, choices=SEEDS)
    for name in ('root', 'receipt', 'dataset', 'assets', 'cf-runtime'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--tiger-batch', type=int, default=256)
    p.add_argument('--smoke', action='store_true', help='tiny budgets for the GPU smoke test (never results)')
    p.add_argument('--h-rhythm', choices=['pass', 'old693'], default='pass',
                   help="H only: 'old693' keeps the old split's 693-update R unit (2026-11-19 control)")
    p.add_argument('--h-split', choices=['l5o', 'old'], default='l5o',
                   help="H only: 'old' runs the current code on the old split with the old evaluation (2026-11-19 step 1)")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.arm not in ARMS[a.family]:
        raise ValueError(f'{a.arm} is not an arm of {a.family}')
    a.root.mkdir(parents=True, exist_ok=True)
    WORKERS[a.family](a.arm, a.seed, a.root, a)
    print(json.dumps({'worker_complete': [a.family, a.arm, a.seed]}), flush=True)


if __name__ == '__main__':
    main()
