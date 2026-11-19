"""Bounded precision calibration/stress; no validation, test or campaign writes."""
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
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016')
CODE = BASE / 'code'
sys.path.insert(0, str(CODE))
from scripts.continuous.model import ContinuousRecommender, ModelConfig
from scripts.continuous.data import load_cached_split
from scripts.continuous.trainer import seed_everything, restore_random_state

OUTPUT = BASE / 'delivery/gpu_precision_diagnostic_20261016_02'
MAX_SECONDS = 510
STRESS_EPOCHS = 35


def value(x):
    x = float(x)
    return x if math.isfinite(x) else repr(x)


def install_attention_fp32(model):
    for block in model.blocks:
        original = block.multi_head_attn.forward

        def forward(q, k, v, *args, _original=original, **kwargs):
            with torch.autocast('cuda', enabled=False), sdpa_kernel([SDPBackend.MATH]):
                return _original(q.float(), k.float(), v.float(), *args, **kwargs)

        block.multi_head_attn.forward = forward


def loss_for(model, histories, targets, policy, negatives=None):
    bf16 = policy != 'fp32_math'
    math_backend = policy in ('bf16_math', 'fp32_math')
    backend = sdpa_kernel([SDPBackend.MATH]) if math_backend else contextlib.nullcontext()
    with backend, torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        if policy != 'fp32_attention_scores':
            return model.loss(histories, targets, negative_ids=negatives)
        queries = model(histories)
        if negatives is None:
            negatives = torch.randint(1, model.config.n_items + 1,
                                      (len(targets), model.config.negatives), device='cuda')
        candidates = torch.cat((targets[:, None], negatives), dim=1)
        with torch.autocast('cuda', enabled=False):
            logits = torch.einsum('bd,bkd->bk', queries.float(), model.item_embedding(candidates).float())
            return F.cross_entropy(logits, torch.zeros_like(targets))


def create_model(config, state=None, policy='fp32_math'):
    model = ContinuousRecommender(ModelConfig(**{
        f.name: config[f.name] for f in dataclasses.fields(ModelConfig) if f.name in config
    })).cuda()
    if state is not None:
        model.load_state_dict(state['model'])
    if policy.startswith('fp32_attention'):
        install_attention_fp32(model)
    return model


def gradients(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def comparison(actual, reference):
    per_parameter = {}
    dot = difference = ref_square = actual_square = 0.0
    for name, g in actual.items():
        r = reference[name]
        d2 = (g.double() - r.double()).square().sum().item()
        r2 = r.double().square().sum().item()
        g2 = g.double().square().sum().item()
        product = (g.double() * r.double()).sum().item()
        dot += product
        difference += d2
        ref_square += r2
        actual_square += g2
        per_parameter[name] = {'relative_l2': math.sqrt(d2 / max(r2, 1e-30)),
                               'reference_norm': math.sqrt(r2), 'actual_norm': math.sqrt(g2)}
    return {'relative_l2': math.sqrt(difference / max(ref_square, 1e-30)),
            'cosine': dot / max(math.sqrt(actual_square * ref_square), 1e-30),
            'reference_norm': math.sqrt(ref_square), 'actual_norm': math.sqrt(actual_square),
            'per_parameter': per_parameter}


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
    policies = ['fp32_math', 'original_bf16_default', 'bf16_math',
                'fp32_attention', 'fp32_attention_scores']
    seed_everything(66271)
    batches = torch.randperm(len(targets), device='cuda')[:3 * config['batch_size']]
    negative_ids = torch.randint(1, config['n_items'] + 1,
                                (len(batches), config['negatives']), device='cuda')
    calibration = []
    for i in range(3):
        indices = batches[i * config['batch_size']:(i + 1) * config['batch_size']]
        negatives = negative_ids[i * config['batch_size']:(i + 1) * config['batch_size']]
        reference = None
        for policy in policies:
            model = create_model(config, state, policy)
            model.train()
            # Fixed weights/data/negatives, p=0: backend RNG differences cannot
            # masquerade as gradient error. Training stress restores real p=0.1.
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.dropout = 0.0
            loss = loss_for(model, histories[indices], targets[indices], policy, negatives)
            loss.backward()
            actual = gradients(model)
            if reference is None:
                reference = actual
                reference_loss = loss.detach().item()
            report = {'batch': i, 'policy': policy, 'loss': value(loss.detach()),
                      'reference_loss': reference_loss,
                      'all_gradients_finite': all(torch.isfinite(g).all().item() for g in actual.values()),
                      **comparison(actual, reference)}
            calibration.append(report)
            print(json.dumps({k: v for k, v in report.items() if k != 'per_parameter'}), flush=True)
            del actual, model
        del reference
    (OUTPUT / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    # Capture the actual operator chosen for the ORIGINAL train/dropout path.
    model = create_model(config, state, 'original_bf16_default')
    model.train()
    restore_random_state(state['random_state'])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        loss_for(model, histories[:256], targets[:256], 'original_bf16_default').backward()
    operations = sorted({e.key for e in prof.key_averages()
                         if any(s in e.key.lower() for s in ('attention', 'efficient', 'flash', 'softmax'))})
    (OUTPUT / 'original_train_operators.json').write_text(json.dumps(operations, indent=2))
    del model, prof

    eligible = [p for p in ('fp32_attention', 'fp32_attention_scores')
                if all(r['all_gradients_finite'] and r['relative_l2'] < 0.05 and r['cosine'] > 0.999
                       for r in calibration if r['policy'] == p)]
    chosen = eligible[0] if eligible else 'fp32_math'
    replay = {'policy': chosen, 'checkpoint_epoch': state['epoch'], 'steps': [], 'passed': False}
    model = create_model(config, state, chosen)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    model.train()
    restore_random_state(state['random_state'])
    order = torch.randperm(len(targets), device='cuda')
    for step, start in enumerate(range(0, len(targets), config['batch_size']), 1):
        batch = order[start:start + config['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, histories[batch], targets[batch], chosen)
        assert torch.isfinite(loss), 'nonfinite replay loss'
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        assert norm < 1e6, 'unreasonable replay gradient norm'
        optimizer.step()
        replay['steps'].append({'step': step, 'loss': value(loss.detach()), 'grad_norm': value(norm)})
    replay['passed'] = len(replay['steps']) == 514
    (OUTPUT / 'replay.json').write_text(json.dumps(replay, indent=2))
    print(json.dumps({'replay_policy': chosen, 'steps_completed': len(replay['steps']),
                      'passed': replay['passed']}), flush=True)
    del model, optimizer

    seed_everything(2024)
    model = create_model(config, policy=chosen)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    stress = {'policy': chosen, 'seed': 2024, 'config': config, 'epochs': [],
              'purpose': 'numerical stress only; no validation, selection or campaign checkpoint',
              'passed': False}
    for epoch in range(1, STRESS_EPOCHS + 1):
        epoch_started = time.time()
        model.train()
        order = torch.randperm(len(targets), device='cuda')
        max_norm = total_loss = 0.0
        count = updates = 0
        for start in range(0, len(targets), config['batch_size']):
            assert time.time() - started < MAX_SECONDS - 15, 'bounded diagnostic time limit'
            batch = order[start:start + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, histories[batch], targets[batch], chosen)
            assert torch.isfinite(loss), 'nonfinite stress loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
            assert norm < 1e6, 'unreasonable stress gradient norm'
            optimizer.step()
            max_norm = max(max_norm, norm.item())
            total_loss += loss.detach().item() * len(batch)
            count += len(batch)
            updates += 1
        row = {'epoch': epoch, 'updates': updates, 'samples': count,
               'train_loss': total_loss / count, 'max_grad_norm': max_norm,
               'seconds': time.time() - epoch_started}
        stress['epochs'].append(row)
        (OUTPUT / 'stress.json').write_text(json.dumps(stress, indent=2))
        print('STRESS:' + json.dumps(row), flush=True)
    stress['passed'] = len(stress['epochs']) == STRESS_EPOCHS
    (OUTPUT / 'stress.json').write_text(json.dumps(stress, indent=2))
    proof = {'status': 'DIAGNOSTIC_COMPLETE', 'not_valid_training_run': True,
             'checkpoint_sha256': current['last']['sha256'],
             'source_sha256': state['binding']['source_sha256'],
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'max_seconds': MAX_SECONDS, 'stress_epochs': STRESS_EPOCHS,
             'calibration_dropout_disabled': True, 'stress_dropout': config['dropout'],
             'chosen_policy': chosen, 'calibration_relative_l2_max': 0.05,
             'calibration_cosine_min': 0.999, 'gradient_norm_sanity_limit': 1e6,
             'replay_passed': replay['passed'], 'stress_passed': stress['passed'],
             'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
             'inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                                    '--format=csv,noheader'], text=True).strip(),
             'original_train_operators': operations,
             'elapsed_seconds': time.time() - started}
    (OUTPUT / 'DIAGNOSTIC_COMPLETE.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
