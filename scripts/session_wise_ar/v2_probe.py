"""Protocol v2 TIGER GPU gate (short test, not a scientific run).

Reads first-wave T0A artifacts (parent, tokenizer) read-only and writes only --output.
Gate (pre-registered in PROTOCOL.md section 4.2):
  1. kernel: on CUDA, the pinned memory-efficient SDPA path runs and matches eager
     attention (dropout off, strict FP32) for loss and gradients;
  2. batch: at the same number of training samples, batch 1024 (lr 0.02, warmup 2,500)
     reaches a mean loss over the last 20% of samples no more than 5% above batch 256
     (lr 0.01, warmup 10,000); otherwise the formal runs use batch 512;
  3. speed is recorded for TF32 eager b256, TF32 SDPA b256 and TF32 SDPA b1024.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from scripts.kuairec_data.l5o_v2 import build
from scripts.rqcd.data import PackedSplit
from scripts.rqcd_mixer.common import read_state, seed_all, seed_for, write_json
from scripts.rqcd_mixer.data import PermutationStream, load_content
from . import runtime
from .catalog import add_suffix
from .data import SessionTasks
from .model import ARRecommender
from .tokenizer import make_tokenizer


def variant(cfg, *, batch, attention, dropout=None):
    v = copy.deepcopy(cfg)
    scale = batch / 256
    v['training'].update(batch_size=batch, r_lr=0.01 * math.sqrt(scale), inverse_sqrt_warmup=round(10000 / scale))
    v['model']['attention'] = attention
    if dropout is not None:
        v['model']['dropout'] = dropout
    return v


def model_for(cfg, parent, device):
    seed_all(seed_for(cfg['seed'], cfg['family'], 'recommendation-updates'))
    rec = ARRecommender(cfg).to(device)
    rec.load_state_dict(parent['recommender'])
    return rec


def kernel_check(cfg, parent, sid, tasks, device):
    out = {}
    batch = tasks.batch(torch.arange(256), device)
    grads = {}
    for attention in ('eager', 'sdpa'):
        v = variant(cfg, batch=256, attention=attention, dropout=0.0)
        v['execution']['precision'] = 'fp32'
        rec = model_for(v, parent, device); rec.train()
        with runtime.train_kernels(v, device):
            loss = rec.hard_loss(batch, sid)
            loss.backward()
        out[attention + '_loss'] = loss.item()
        grads[attention] = torch.cat([p.grad.flatten() for p in rec.parameters()])
    diff = (grads['sdpa'] - grads['eager']).norm() / grads['eager'].norm()
    out['loss_rel_diff'] = abs(out['sdpa_loss'] - out['eager_loss']) / abs(out['eager_loss'])
    out['grad_rel_l2_diff'] = diff.item()
    out['pass'] = out['loss_rel_diff'] < 1e-4 and out['grad_rel_l2_diff'] < 1e-3
    return out


def train_run(cfg, parent, tok, sid, tasks, device, steps, warm=20):
    rec = model_for(cfg, parent, device); rec.train()
    per_epoch = math.ceil(len(tasks) / cfg['training']['batch_size'])
    optimizers, schedulers = runtime.recommendation_optimizers(rec, tok, cfg, per_epoch)
    stream = PermutationStream(len(tasks), seed_for(cfg['seed'], 'ar', 'recommendation-data'))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    losses, started = [], None
    for step in range(steps):
        if step == warm:
            torch.cuda.synchronize(); started = time.perf_counter()
        batch = tasks.batch(stream.take(cfg['training']['batch_size']), device)
        with runtime.train_kernels(cfg, device):
            with runtime.train_autocast(cfg, device):
                loss = rec.hard_loss(batch, sid)
            runtime.finite_step(loss, [rec], optimizers, schedulers, cfg['training']['clip_norm'])
        losses.append(loss.detach())
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / (steps - warm)
    return dict(batch=cfg['training']['batch_size'], attention=cfg['model']['attention'],
                precision=cfg['execution']['precision'], steps=steps, seconds_per_update=seconds,
                samples_per_second=cfg['training']['batch_size'] / seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                losses=torch.stack(losses).float().cpu().tolist())


def tail_mean(losses, fraction=0.2):
    k = max(1, round(len(losses) * fraction))
    return sum(losses[-k:]) / k


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--content', required=True)
    p.add_argument('--wave', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=1_024_000, help='training samples per arm of the batch comparison')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = build('tiger', 'T0A', 2024, receipt=a.receipt)
    device = runtime.configure_device('cuda')
    content = load_content(a.content, cfg, device)
    parent = read_state(a.wave/'parents/tiger/initial.pt')
    tok = make_tokenizer(cfg).to(device)
    tok.load_state_dict(read_state(a.wave/'runs/T0A/pretrain/selected.pt')['tokenizer'])
    tok.requires_grad_(False); tok.eval()
    with torch.no_grad():
        sid = add_suffix(tok.indices(content), cfg['rq']['codes'])
    tasks = SessionTasks(PackedSplit(a.dataset, 'train', cfg))
    result = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'config_seed': cfg['seed']}
    result['kernel'] = kernel_check(cfg, parent, sid, tasks, device)
    print(json.dumps({'kernel': result['kernel']}), flush=True)
    runs = []
    eager = variant(cfg, batch=256, attention='eager')
    runs.append(train_run(eager, parent, tok, sid, tasks, device, 300))
    for batch in (256, 1024):
        v = variant(cfg, batch=batch, attention='sdpa')
        runs.append(train_run(v, parent, tok, sid, tasks, device, a.samples // batch))
    for r in runs:
        print(json.dumps({k: v for k, v in r.items() if k != 'losses'}), flush=True)
    small, large = runs[1], runs[2]
    gap = (tail_mean(large['losses']) - tail_mean(small['losses'])) / tail_mean(small['losses'])
    result['batch'] = dict(samples=a.samples, tail_mean_b256=tail_mean(small['losses']),
                           tail_mean_b1024=tail_mean(large['losses']), relative_gap=gap, pass_=gap <= 0.05)
    result['speed'] = {f"{r['attention']}_b{r['batch']}": {k: r[k] for k in ('seconds_per_update', 'samples_per_second',
                                                                              'peak_allocated_gib')} for r in runs}
    result['selected_batch'] = 1024 if gap <= 0.05 else 512
    result['gate_pass'] = bool(result['kernel']['pass'])
    result['runs'] = runs
    write_json(a.output/'V2_PROBE.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
