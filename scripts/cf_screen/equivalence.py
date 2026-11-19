"""CPU check: the B0 training and generation path is unchanged by the screening code.

Run once with PYTHONPATH pointing at the code before the screening change ("old") and
once at the new code ("new"); both print a digest of three R updates, one T update and
32 generated validation lists, all in FP32 on CPU from the first-wave CF parent. The two
digests must be identical. Only APIs that exist in both versions are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(tensors):
    h = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b'')
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', type=Path, required=True)
    p.add_argument('--mode', choices=['old', 'new'], required=True)
    a = p.parse_args()
    from scripts.rqcd_mixer.catalog import generate
    from scripts.rqcd_mixer.common import configure_runtime, read_state, seed_all, seed_for
    from scripts.rqcd_mixer.data import PackedSplit
    from scripts.rqcd_mixer.tokenizer import RQTokenizer
    from scripts.rqcd_mixer.training import JointTrainer
    run = a.wave/'cf_repeat_v2/runs/CF-M0'
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    if a.mode == 'new':
        cfg = {**cfg, 'screen': {'name': 'B0'}}
    runtime = json.loads((run/'runtime.json').read_text(encoding='utf-8'))
    runtime.update(precision='fp32', data_device='cpu')
    configure_runtime(runtime, 'cpu')
    meta = json.loads((a.wave/'parents/cf/PREPARED.json').read_text(encoding='utf-8'))
    asset = read_state(a.wave/'parents/cf_rq/catalog.pt')
    tokenizer = RQTokenizer(cfg, 'none')
    tokenizer.load_state_dict(read_state(a.wave/'parents/cf_rq/best_objective.pt')['tokenizer'])
    initial = read_state(a.wave/'parents/cf/initial.pt')
    trainer = JointTrainer(cfg, runtime, tokenizer, asset, initial['recommender'], {'check': 'equivalence'}, 'cpu')
    train = PackedSplit(meta['dataset_path'], 'train', cfg, device='cpu')
    # As run.py train(): dropout draws from the global generator seeded here.
    seed_all(seed_for(cfg['seed'], 'recommender_dropout'))
    losses = []
    for step in range(3):
        history, target = train.batch(torch.arange(step*64, step*64+64), 'cpu')
        losses.append(trainer.update(history, target, 'R')['losses'].tolist())
    history, target = train.batch(torch.arange(5000, 5064), 'cpu')
    losses.append(trainer.update(history, target, 'T')['losses'].tolist())
    validation = PackedSplit(meta['dataset_path'], 'validation', cfg, device='cpu')
    history, _ = validation.batch(torch.arange(32), 'cpu')
    noise = torch.from_numpy(np.load(a.wave/'parents/cf/validation_noise.npy', mmap_mode='r')[:32].copy())
    items, _ = generate(trainer.model, trainer.diffusion, trainer.catalog, history, noise, runtime, filter_history=True)
    print(json.dumps({'mode': a.mode, 'model': digest(trainer.model.state_dict()),
                      'tokenizer': digest(tokenizer.state_dict()),
                      'optimizer_r_steps': sorted({int(v['step']) for v in trainer.opt_r.state_dict()['state'].values()}),
                      'losses': losses, 'items': items.tolist()}))


if __name__ == '__main__':
    main()
