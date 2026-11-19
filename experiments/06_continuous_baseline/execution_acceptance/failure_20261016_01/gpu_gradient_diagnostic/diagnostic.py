"""Bounded, read-only checkpoint diagnostic; not a campaign training result."""
from pathlib import Path
import contextlib
import dataclasses
import hashlib
import json
import math
import signal
import subprocess
import sys
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016')
CODE = BASE / 'code'
sys.path.insert(0, str(CODE))
from scripts.continuous.model import ContinuousRecommender, ModelConfig
from scripts.continuous.data import load_cached_split
from scripts.continuous.trainer import restore_random_state

MAX_STEPS_PER_VARIANT = 514
MAX_SECONDS = 240
OUTPUT = BASE / 'delivery/gpu_gradient_diagnostic_20261016_01'


def number(value):
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def gradient_statistics(model):
    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    return {
        'nonfinite_parameters': [name for name, g in grads if not torch.isfinite(g).all().item()],
        'norm_fp32': number(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g) for _, g in grads]))),
        'norm_fp64': number(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g.double()) for _, g in grads]))),
        'maximum_absolute_gradient': number(max(g.abs().max().item() for _, g in grads)),
    }


def main():
    started = time.time()
    signal.alarm(MAX_SECONDS)
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0)
    OUTPUT.mkdir(exist_ok=False)
    run = CODE / 'outputs/stage06_continuous/pipeline_20261016/campaign/runs/search__esasrec_17__seed2024/attempt_0'
    current = json.loads((run / 'CURRENT.json').read_text())
    path = run / current['last']['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == current['last']['sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['config']
    data = load_cached_split(BASE / 'inputs/cache', 'full_train_prefix_v1', 'train')
    histories, targets = data['histories'].cuda(), data['targets'].cuda()
    variants = [
        ('original_bf16_default', True, None),
        ('bf16_math', True, SDPBackend.MATH),
        ('fp32_math', False, SDPBackend.MATH),
    ]
    reports = []
    for name, bf16, backend in variants:
        model = ContinuousRecommender(ModelConfig(**{
            f.name: config[f.name] for f in dataclasses.fields(ModelConfig) if f.name in config
        })).cuda()
        model.load_state_dict(state['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                     betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
        optimizer.load_state_dict(state['optimizer'])
        model.train()
        restore_random_state(state['random_state'])
        order = torch.randperm(len(targets), device='cuda')
        report = {'variant': name, 'config_unchanged': True, 'checkpoint_epoch': state['epoch'],
                  'mode': 'diagnostic replay only; no checkpoint writes', 'steps': []}
        context = contextlib.nullcontext() if backend is None else sdpa_kernel([backend])
        with context:
            for index, start in enumerate(range(0, len(targets), config['batch_size'])):
                if index >= MAX_STEPS_PER_VARIANT or time.time() - started > MAX_SECONDS - 15:
                    report['bounded_stop'] = True
                    break
                batch = order[start:start + config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    loss = model.loss(histories[batch], targets[batch])
                step = {'step_in_epoch28': index + 1, 'loss': number(loss.detach())}
                if not torch.isfinite(loss):
                    step['failure'] = 'nonfinite_loss'
                    report['steps'].append(step)
                    break
                loss.backward()
                try:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'],
                                                         error_if_nonfinite=True)
                except RuntimeError as error:
                    step.update(failure=str(error), **gradient_statistics(model))
                    report['steps'].append(step)
                    break
                optimizer.step()
                step['gradient_norm_before_clip'] = number(norm)
                report['steps'].append(step)
        report['steps_completed'] = sum('failure' not in s for s in report['steps'])
        report['failed'] = any('failure' in s for s in report['steps'])
        report['elapsed_since_start_seconds'] = time.time() - started
        (OUTPUT / (name + '.json')).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'steps'}), flush=True)
        if report['failed']:
            print(json.dumps(report['steps'][-1]), flush=True)
        reports.append(report)
        del optimizer, model
        torch.cuda.empty_cache()
    proof = {'status': 'DIAGNOSTIC_COMPLETE', 'not_valid_training_run': True,
             'checkpoint_sha256': current['last']['sha256'], 'source_sha256': state['binding']['source_sha256'],
             'max_steps_per_variant': MAX_STEPS_PER_VARIANT, 'max_seconds': MAX_SECONDS,
             'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
             'inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                                    '--format=csv,noheader'], text=True).strip(),
             'variants': [{k: v for k, v in r.items() if k != 'steps'} for r in reports]}
    (OUTPUT / 'DIAGNOSTIC_COMPLETE.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
