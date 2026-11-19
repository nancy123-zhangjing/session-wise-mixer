"""CPU check of every screening group through the real parent and trainer path (FP32).

For each group: load (or, for the E groups, prepare and briefly pretrain) its parent
through run.py's own checks, build the joint trainer, run two R updates and one T
update, round-trip the state, and generate with every rule the group is evaluated
with (A; C2; guidance 2 for H2; the history head for H1). Prints one JSON line per
group. Not a result: tiny batches, CPU, FP32 instead of the recorded BF16.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import resource
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def fp32_runtime_loader(original):
    def load(path, cfg, *, require_batches=True):
        runtime = original(path, cfg, require_batches=require_batches)
        return {**runtime, 'precision': 'fp32', 'data_device': 'cpu'}
    return load


def check(name, a, base, meta):
    from scripts.cf_decision.rules import LandingScorer, generate_mode, history_head_top_five
    from scripts.cf_screen.variants import build, own_parent
    from scripts.rqcd_mixer import run as cli
    from scripts.rqcd_mixer.catalog import evaluation_noise, generate
    from scripts.rqcd_mixer.data import PackedSplit, target_counts
    from scripts.rqcd_mixer.screen import generation_dim
    cfg = build(name, base, smoke=True)
    args = SimpleNamespace(runtime=a.runtime, dataset=Path(meta['dataset_path']), device='cpu', arm='B0',
                           prepared=a.wave/'parents/cf', pretrained=a.wave/'parents/cf_rq')
    if own_parent(name):
        root = a.scratch/name
        args.prepared, args.pretrained = root/'prepared', root/'rq'
        if not (args.prepared/'PREPARED.json').is_file():
            cli.prepare(SimpleNamespace(**vars(args), content=Path(meta['content_path']),
                                        teacher=Path(meta['teacher_path']), output=args.prepared), cfg)
        if not (args.pretrained/'COMPLETE.json').is_file():
            cli.pretrain(SimpleNamespace(**vars(args), output=args.pretrained, resume=False), cfg)
    runtime, environment = cli._runtime(args, cfg)
    trainer, prepared = cli._joint(args, cfg, runtime, environment)
    train = PackedSplit(args.dataset, 'train', cfg, device='cpu')
    trainer.configure_generation(target_counts(train, cfg['catalog_size']))
    fixed = {k: v.clone() for k, v in trainer.model.state_dict().items() if k.startswith('to_latent.')}
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    # State round trip (consistent clocks before any update): new parameters, buffers,
    # generators and teacher-space targets survive save/restore.
    state = trainer.state()
    twin, _ = cli._joint(args, cfg, runtime, environment)
    twin.restore(state)
    if any(not torch.equal(v, twin.model.state_dict()[k]) for k, v in before.items()):
        raise RuntimeError(f'{name}: restore does not reproduce the model')
    if trainer.gen_x_pad is not None and not torch.equal(trainer.gen_x_pad, twin.gen_x_pad):
        raise RuntimeError(f'{name}: restore does not rebuild the teacher-space targets')
    del twin, state
    # Small R and T updates (the notebook has a 16 GiB limit; the micro-batch loop is unchanged code).
    losses = {}
    for step in range(2):
        history, target = train.batch(torch.arange(step*64, step*64+64), 'cpu')
        losses[f'R{step}'] = dict(zip(trainer.loss_keys, trainer.update(history, target, 'R')['losses'].tolist()))
    after = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    moved = sorted(k for k in after if not torch.equal(before[k], after[k]))
    if not moved:
        raise RuntimeError(f'{name}: no parameter moved')
    history, target = train.batch(torch.arange(4000, 4032), 'cpu')
    losses['T'] = dict(zip(trainer.loss_keys, trainer.update(history, target, 'T')['losses'].tolist()))
    del train
    gc.collect()
    if not all(math.isfinite(v) for part in losses.values() for v in part.values()):
        raise RuntimeError(f'{name}: nonfinite loss {losses}')
    if any(not torch.equal(fixed[k], trainer.model.state_dict()[k]) for k in fixed):
        raise RuntimeError(f'{name}: the frozen RQ Encoder copy changed')
    validation = PackedSplit(args.dataset, 'validation', cfg, device='cpu')
    history, _ = validation.batch(torch.arange(16), 'cpu')
    noise = torch.from_numpy(evaluation_noise(cfg, validation.sample_ids[:16], 'validation'))
    rules = {}
    lists = {'A': generate(trainer.model, trainer.diffusion, trainer.catalog, history, noise, runtime,
                           filter_history=True)[0]}
    scorer = LandingScorer(trainer.catalog, cfg)
    two = torch.stack((noise, noise.flip(0)), 1)
    lists['C2'] = generate_mode(trainer.model, trainer.diffusion, trainer.catalog, scorer, history, two, runtime)[0]
    if 'guidance' in cfg['screen']:
        lists['A_w2'] = generate(trainer.model, trainer.diffusion, trainer.catalog, history, noise, runtime,
                                 filter_history=True, guidance=2.0)[0]
        lists['C2_w2'] = generate_mode(trainer.model, trainer.diffusion, trainer.catalog, scorer, history, two,
                                       runtime, guidance=2.0)[0]
    if 'history_head' in cfg['screen']:
        lists['HEAD'] = history_head_top_five(trainer.model, trainer.catalog, history, trainer.item_vectors,
                                              cfg['screen']['history_head']['temperature'], runtime)
    for rule, items in lists.items():
        for row, past in zip(items.tolist(), history.tolist()):
            if len(set(row)) != 5 or min(row) < 1 or set(row) & set(past):
                raise RuntimeError(f'{name} {rule}: invalid list {row}')
        rules[rule] = items[:2].tolist()
    return {'group': name, 'generation_dim': generation_dim(cfg), 'latent_dim': cfg['rq']['latent_dim'],
            'own_parent': own_parent(name), 'loss_keys': list(trainer.loss_keys), 'losses': losses,
            'moved_parameters': len(moved), 'frozen_parameters': len(fixed), 'rules': rules,
            'max_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True, help='where the E groups prepare/pretrain (CPU, tiny)')
    p.add_argument('--groups', nargs='+')
    a = p.parse_args()
    from scripts.cf_screen.variants import VARIANTS
    from scripts.cf_screen.worker import first_wave
    from scripts.rqcd_mixer import run as cli
    cli.load_runtime = fp32_runtime_loader(cli.load_runtime)
    base, meta = first_wave(a.wave)
    for name in a.groups or list(VARIANTS):
        torch.manual_seed(0)
        np.random.seed(0)
        print(json.dumps(check(name, a, base, meta)), flush=True)


if __name__ == '__main__':
    main()
