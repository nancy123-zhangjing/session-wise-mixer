"""Short TIGER speed/numerics benchmark on existing L5O first-wave artifacts.

Not a scientific run: it reads the first-wave T0A parent, tokenizer and one trained
checkpoint (read-only), trains a few hundred updates from the identical parent state
under each precision variant, and reports seconds/update, peak memory, GPU
utilization and the loss difference to FP32. It also checks that validation beam
search returns identical lists at evaluation batch sizes 32/128/256 (FP32), and times
an optional BF16 evaluation. Everything is written only under --output.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import torch

from scripts.rqcd.data import PackedSplit
from scripts.rqcd_mixer.common import atomic_save, read_state, seed_all, seed_for, write_json
from scripts.rqcd_mixer.data import PermutationStream, load_content
from . import runtime
from .catalog import Catalog, add_suffix
from .config import load
from .data import SessionTasks, user_buckets
from .model import ARRecommender
from .tokenizer import make_tokenizer


class GpuSampler:
    def __init__(self, index=0, period=0.5):
        self.index, self.period, self.samples, self._stop = index, period, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', f'--id={self.index}', '--query-gpu=utilization.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:  # sampling is informational only
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._thread.join()

    def mean(self):
        return sum(self.samples)/len(self.samples) if self.samples else None


def set_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def train_variant(variant, cfg, parent, tok, sid, tasks, device, steps, warm):
    vcfg = copy.deepcopy(cfg)
    vcfg['execution']['precision'] = 'bf16' if variant == 'bf16' else 'fp32'
    set_tf32(variant == 'tf32')
    seed_all(seed_for(cfg['seed'], cfg['family'], 'recommendation-updates'))
    rec = ARRecommender(cfg).to(device)
    rec.load_state_dict(parent['recommender'])
    rec.train()
    per_epoch = math.ceil(len(tasks)/cfg['training']['batch_size'])
    optimizers, schedulers = runtime.recommendation_optimizers(rec, tok, cfg, per_epoch)
    stream = PermutationStream(len(tasks), seed_for(cfg['seed'], 'ar', 'recommendation-data'))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    losses, started = [], None
    with GpuSampler() as sampler:
        for step in range(steps):
            if step == warm:
                torch.cuda.synchronize(); started = time.perf_counter(); sampler.samples.clear()
            batch = tasks.batch(stream.take(cfg['training']['batch_size']), device)
            with runtime.train_autocast(vcfg, device):
                loss = rec.hard_loss(batch, sid)
            runtime.finite_step(loss, [rec], optimizers, schedulers, cfg['training']['clip_norm'])
            losses.append(loss.detach())
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
    losses = torch.stack(losses).float().cpu().tolist()
    probe = Path(cfg['_benchmark_output'])/f"checkpoint_probe_{cfg['_benchmark_tag']}_{variant}.pt"
    t0 = time.perf_counter()
    atomic_save(probe, {'recommender': rec.state_dict(), 'optimizers': [o.state_dict() for o in optimizers]})
    save_seconds = time.perf_counter()-t0
    size = probe.stat().st_size
    probe.unlink()
    set_tf32(False)
    return {'variant': variant, 'steps': steps, 'warm_steps_excluded': warm,
            'seconds_per_update': elapsed/(steps-warm), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'gpu_utilization_mean': sampler.mean(), 'losses': losses,
            'checkpoint_save_seconds': save_seconds, 'checkpoint_bytes': size}


@torch.inference_mode()
def eval_check(cfg, parent, checkpoint, dataset, device, samples, batch_sizes):
    state = read_state(checkpoint)
    rec = ARRecommender(cfg).to(device); rec.load_state_dict(state['recommender']); rec.eval()
    sid = state['sid'].to(device)
    catalog = Catalog(sid, cfg['rq']['codes'], parent['original_ids'])
    data = PackedSplit(dataset, 'validation', cfg)
    users = user_buckets(data.users)
    n = min(samples, len(data))
    out, reference = {}, None
    for name, size, bf16 in [(f'fp32_b{b}', b, False) for b in batch_sizes] + [(f'bf16_b{batch_sizes[-1]}', batch_sizes[-1], True)]:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        preds = []
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            for start in range(0, n, size):
                stop = min(start+size, n)
                history, _ = data.batch(torch.arange(start, stop), device)
                preds.append(catalog.generate(rec, history, users[start:stop].to(device), cfg['list_length'],
                                              cfg['evaluation']['beam'],
                                              filter_history=cfg['evaluation'].get('filter_history', False)))
        torch.cuda.synchronize()
        pred = torch.cat(preds)
        reference = pred if reference is None else reference
        out[name] = {'seconds_per_sample': (time.perf_counter()-t0)/n, 'batch_size': size, 'bf16': bf16,
                     'identical_lists_vs_fp32_b32': (pred == reference).all(1).float().mean().item(),
                     'identical_slots_vs_fp32_b32': (pred == reference).float().mean().item()}
    return {'checkpoint': str(checkpoint), 'samples': n, 'runs': out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', default='AR-T0A')
    p.add_argument('--l5o-calibration', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--content', required=True)
    p.add_argument('--wave', type=Path, required=True, help='first-wave root (read-only)')
    p.add_argument('--run', default='T0A')
    p.add_argument('--eval-checkpoint', default='step_0166284.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--variants', default='fp32,tf32,bf16')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--warm', type=int, default=50)
    p.add_argument('--eval-samples', type=int, default=512)
    p.add_argument('--skip-eval', action='store_true')
    a = p.parse_args()
    if a.output.resolve() == a.wave.resolve() or a.wave.resolve() in a.output.resolve().parents:
        raise ValueError('Benchmark output must be outside the first-wave directory')
    a.output.mkdir(parents=True, exist_ok=True)
    from scripts.kuairec_data.l5o_launch import wave_config
    cfg = wave_config(load(a.arm), a.l5o_calibration, 'tiger')
    device = runtime.configure_device('cuda')
    content = load_content(a.content, cfg, device)
    parent = read_state(a.wave/'parents/tiger/initial.pt')
    tok = make_tokenizer(cfg).to(device)
    tok.load_state_dict(read_state(a.wave/'runs'/a.run/'pretrain/selected.pt')['tokenizer'])
    tok.requires_grad_(False); tok.eval()
    with torch.no_grad():
        sid = add_suffix(tok.indices(content), cfg['rq']['codes'])
    tasks = SessionTasks(PackedSplit(a.dataset, 'train', cfg))
    bench_cfg = dict(cfg, _benchmark_output=str(a.output), _benchmark_tag=a.tag)
    result = {'tag': a.tag, 'arm': a.arm, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'batch_size': cfg['training']['batch_size'], 'history_max': cfg['history_max'], 'train': []}
    for variant in a.variants.split(','):
        row = train_variant(variant, bench_cfg, parent, tok, sid, tasks, device, a.steps, a.warm)
        result['train'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'losses'}), flush=True)
    base = next((r for r in result['train'] if r['variant'] == 'fp32'), None)
    if base is not None:
        for row in result['train']:
            x = torch.tensor(row['losses'][a.warm:]); y = torch.tensor(base['losses'][a.warm:])
            row['mean_loss_relative_diff_vs_fp32'] = ((x.mean()-y.mean())/y.mean()).item()
            row['max_abs_step_loss_diff_vs_fp32'] = (x-y).abs().max().item()
            row['speedup_vs_fp32'] = base['seconds_per_update']/row['seconds_per_update']
    if not a.skip_eval:
        result['eval'] = eval_check(cfg, parent, a.wave/'runs'/a.run/'train/checkpoints'/a.eval_checkpoint,
                                    a.dataset, device, a.eval_samples, (32, 128, 256))
        print(json.dumps(result['eval']), flush=True)
    write_json(a.output/f'RESULT_{a.tag}.json', result)


if __name__ == '__main__':
    main()
