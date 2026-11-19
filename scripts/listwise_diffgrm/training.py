"""Alternating R/T updates and complete, portable update-boundary checkpoints.

This module has no platform API and never launches a job. The recommendation
network and the original JointCrossPSEAlignment remain separate modules so that
freezing the network in T does not detach its input/soft-label gradient paths.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from .execution import enabled, ValidatedTrainingBatch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG checkpoint requires CUDA for exact continuation')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def phase_plan(config, train_count):
    t = config['training']
    r_steps = int(t['r_updates_per_unit']) if t.get('r_updates_per_unit') else math.ceil(train_count / t['effective_batch_size'])
    result, epoch = [], 0
    for _ in range(t['warmup_epochs']):
        epoch += 1
        result.append(dict(kind='R', epoch=epoch, cycle=0, updates=r_steps,
                           validate=t.get('validate_every_r_unit', False) or epoch == t['warmup_epochs']))
    freeze = t.get('freeze_after_epoch')
    if freeze is not None and (type(freeze) is not int or not
                               t['warmup_epochs'] <= freeze <= t['warmup_epochs'] + t['cycles'] * t['r_epochs_per_cycle']):
        raise ValueError('freeze_after_epoch must be an R-epoch boundary inside the plan')
    for cycle in range(1, t['cycles'] + 1):
        # In-run freeze: every tokenizer block after R epoch `freeze` is skipped
        # (zero updates, no SID refresh); R units and the full R schedule continue.
        frozen = freeze is not None and epoch >= freeze
        result.append(dict(kind='T', epoch=epoch, cycle=cycle,
                           updates=0 if frozen else t['t_updates_per_cycle'], validate=False,
                           **(dict(frozen=True) if frozen else {})))
        for j in range(t['r_epochs_per_cycle']):
            epoch += 1
            result.append(dict(kind='R', epoch=epoch, cycle=cycle, updates=r_steps,
                               validate=t.get('validate_every_r_unit', False) or j == t['r_epochs_per_cycle'] - 1))
    return result


def nominal_t_updates(config):
    """T budget of the unfrozen plan; the annealing clock keeps this total under freeze,
    matching the accepted R20 freeze continuations (noise schedule of the full 400T)."""
    t = config['training']
    return t['cycles'] * t['t_updates_per_cycle']


def expected_update_counts(plan, progress):
    """Derive formal update clocks from the cursor, including an unfinished boundary.

    update == phase.updates is valid: all optimizer steps may have been saved
    before the pending SID refresh/validation advances the phase cursor.
    """
    if not isinstance(progress, dict):
        raise ValueError('Checkpoint progress must be a mapping')
    phase, update = progress.get('phase'), progress.get('update')
    if type(phase) is not int or type(update) is not int:
        raise ValueError('Checkpoint phase/update cursor must contain integers')
    if not 0 <= phase <= len(plan) or update < 0:
        raise ValueError('Checkpoint phase/update cursor is outside the plan')
    if phase == len(plan):
        if update != 0:
            raise ValueError('Completed checkpoint must have update cursor zero')
    elif update > plan[phase]['updates']:
        raise ValueError('Checkpoint update cursor exceeds its phase budget')
    counts = {'R': 0, 'T': 0}
    for row in plan[:phase]:
        counts[row['kind']] += row['updates']
    if phase < len(plan):
        counts[plan[phase]['kind']] += update
    return counts


def validate_optimizer_clock(state, updates, name):
    """Adam clocks must match real updates; absent frozen-side entries are allowed."""
    entries = state.get('state', {})
    if updates and not entries:
        raise ValueError(f'{name} optimizer has no state despite completed updates')
    if not updates and entries:
        raise ValueError(f'{name} optimizer has state before its first update')
    for parameter_state in entries.values():
        step = parameter_state.get('step')
        if step is None:
            raise ValueError(f'{name} optimizer state has no Adam step')
        value = float(step.item() if torch.is_tensor(step) else step)
        if value != updates:
            raise ValueError(f'{name} Adam step disagrees with cumulative updates')


class AlternatingTrainer:
    def __init__(self, model, alignment, config, train_count, *, precision='fp32'):
        if precision not in ('fp32', 'bf16'):
            raise ValueError('Supported precision: fp32, bf16')
        self.model, self.alignment, self.config = model, alignment, config
        self.precision = precision
        self.device = next(model.parameters()).device
        self.plan = phase_plan(config, train_count)
        t = config['training']
        common = dict(betas=tuple(t['adam_betas']), eps=t['adam_eps'])
        if config.get('execution', {}).get('fused_adamw', False) and self.device.type == 'cuda':
            common['fused'] = True
        self.r_optimizer = torch.optim.AdamW(model.parameters(), lr=t['decoder_lr'],
                                              weight_decay=t['decoder_weight_decay'], **common)
        # Preserve an identical parameter-group topology in both arms; disabled
        # attention weights receive no gradients and therefore no AdamW updates.
        base = alignment.mixer.base_codebooks
        mixer = [p for p in alignment.mixer.parameters() if p is not base]
        self.t_optimizer = torch.optim.AdamW([
            dict(params=[base], lr=t['codebook_lr'], group_name='codebook'),
            dict(params=mixer, lr=t['mixer_lr'], group_name='mixer')],
            weight_decay=t['tokenizer_weight_decay'], **common)
        total_r = sum(p['updates'] for p in self.plan if p['kind'] == 'R')
        warmup = t['decoder_warmup_updates']

        def factor(step):
            if step < warmup:
                return max(step, 1) / max(warmup, 1)
            progress = min(max((step - warmup) / max(total_r - warmup, 1), 0), 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.r_optimizer, factor)
        alignment.configure_diger_schedule(nominal_t_updates(config))
        self.r_updates = 0
        self.t_updates = 0

    def phase_mode(self, kind):
        if kind not in ('R', 'T'):
            raise ValueError('phase must be R or T')
        self.model.train(kind == 'R')
        self.alignment.train(kind == 'T')
        self.alignment.set_recommender_cache(False)
        for p in self.model.parameters():
            p.requires_grad_(kind == 'R')
        for name, p in self.alignment.named_parameters():
            p.requires_grad_(kind == 'T' and (
                name == 'mixer.base_codebooks' or self.alignment.mixer.mixer_enabled)
                and not (name == 'mixer.code_embedding'
                         and not self.alignment.mixer.code_embedding_enabled))

    def auxiliary(self, targets, *, defer_metrics=False):
        t = self.config['training']
        with torch.autocast(device_type=self.device.type, enabled=False):
            aux = self.alignment.tokenizer_objective(
                sequence_slots=None, target_item_ids=targets.to(self.device).reshape(-1),
                balance_weight=t['balance_weight'], anchor_weight=t['anchor_weight'],
                balance_temperature=t['balance_temperature'], sia_weight=0.0)
            loss = (t['reconstruction_weight'] * aux['reconstruction']
                    + t['balance_weight'] * aux['balance'] + t['anchor_weight'] * aux['anchor'])
        return loss, {k: aux[k].detach() if defer_metrics else float(aux[k].detach())
                      for k in ('reconstruction', 'balance', 'anchor')}

    def objective(self, batch, kind, *, include_aux=True, defer_metrics=False):
        ids = batch['history_item_ids'].to(self.device)
        targets = batch['target_item_ids'].to(self.device)
        mask = batch['history_mask'].to(self.device)
        if not isinstance(batch, ValidatedTrainingBatch) and (targets <= 0).any():
            raise ValueError('All target slots must contain real items')
        if kind == 'T':
            # Distances and Gumbel sampling remain float32 under outer BF16.
            with torch.autocast(device_type=self.device.type, enabled=False):
                hs, _ = self.alignment.diger_assignments(ids, stochastic=True)
                ts, _ = self.alignment.diger_assignments(targets, stochastic=True)
            hsid = hs.detach().argmax(-1).masked_fill(~mask.unsqueeze(-1), -1)
            tsid = ts.detach().argmax(-1)
        else:
            hs = ts = None
            hsid, tsid = self.alignment.hard_codes[ids], self.alignment.hard_codes[targets]
        amp = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
               if self.precision == 'bf16' else nullcontext())
        with amp:
            result = self.model.forward_loss(hsid, mask, tsid, history_st=hs, target_st=ts)
        loss = result['loss']
        metric = lambda value: value.detach() if defer_metrics else float(value.detach())
        values = {'recommendation': metric(loss)}
        if kind == 'T':
            t = self.config['training']
            loss = t['recommendation_weight'] * loss
            if include_aux:
                auxiliary, aux_values = self.auxiliary(targets, defer_metrics=defer_metrics)
                loss = loss + auxiliary
                values.update(aux_values)
        values['loss'] = metric(loss)
        return loss, values

    def update(self, microbatches, kind):
        """Exactly one optimizer update; correctly normalize a short final batch."""
        self.phase_mode(kind)
        batches = list(microbatches)
        count = sum(len(b['target_item_ids']) for b in batches)
        if not count:
            raise ValueError('Empty effective batch')
        self.r_optimizer.zero_grad(set_to_none=True)
        self.t_optimizer.zero_grad(set_to_none=True)
        optimizer = self.r_optimizer if kind == 'R' else self.t_optimizer
        result = {}
        deferred = enabled(self.config)
        metric_rows = []
        finite_losses = []
        for batch in batches:
            weight = len(batch['target_item_ids']) / count
            kwargs = {'defer_metrics': True} if deferred else {}
            loss, values = self.objective(batch, kind, include_aux=False, **kwargs)
            if deferred:
                finite_losses.append(loss.detach())
            elif not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {kind} loss')
            (loss * weight).backward()
            if deferred:
                metric_rows.append((weight, values))
            else:
                for key, value in values.items():
                    result[key] = result.get(key, 0.0) + weight * value
        if kind == 'T':
            # Aggregate unique targets over the effective list batch. Otherwise
            # balance/unique reconstruction change when microbatch is profiled.
            kwargs = {'defer_metrics': True} if deferred else {}
            auxiliary, aux_values = self.auxiliary(torch.cat([b['target_item_ids'] for b in batches]), **kwargs)
            if deferred:
                finite_losses.append(auxiliary.detach())
            elif not torch.isfinite(auxiliary):
                raise FloatingPointError('Non-finite T auxiliary loss')
            auxiliary.backward()
            if not deferred:
                result.update(aux_values)
                result['loss'] += float(auxiliary.detach())
        parameters = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
        if not any(p.grad is not None for p in parameters):
            raise RuntimeError(f'{kind} has no gradient path')
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config['training']['max_grad_norm'],
                                              error_if_nonfinite=not deferred)
        if deferred and not torch.isfinite(torch.stack(finite_losses + [norm.detach()])).all():
            raise FloatingPointError(f'Non-finite {kind} loss or gradient; optimizer was not advanced')
        optimizer.step()
        if kind == 'R':
            self.r_updates += 1
            self.scheduler.step()
        else:
            self.t_updates += 1
            self.alignment.advance_diger_step()
        if deferred:
            # Copy scalar values once; reproduce the original Python-double
            # weighted sum IN THE SAME ORDER instead of reducing on the GPU.
            scalars = [v for _, row in metric_rows for v in row.values()]
            if kind == 'T':
                scalars.extend(aux_values.values())
                scalars.append(auxiliary.detach())
            scalars.append(norm.detach())
            numbers = iter(torch.stack([v.to(dtype=torch.float64) for v in scalars]).cpu().tolist())
            for weight, row in metric_rows:
                for key in row:
                    result[key] = result.get(key, 0.0) + weight * next(numbers)
            if kind == 'T':
                result.update({key: next(numbers) for key in aux_values})
                result['loss'] += next(numbers)
            norm_value = next(numbers)
        else:
            norm_value = float(norm)
        result.update(lists=count, phase=kind, gradient_norm=norm_value,
                      r_updates=self.r_updates, t_updates=self.t_updates)
        return result

    @torch.no_grad()
    def refresh_catalog(self):
        self.alignment.eval()
        self.alignment.set_recommender_cache(False)
        return self.alignment.refresh_hard_codes()

    def checkpoint(self, binding, progress, arm):
        return cpu_copy(dict(format='kuairec_listwise_v1', binding=binding, arm=arm,
                             model=self.model.state_dict(), alignment=self.alignment.state_dict(),
                             r_optimizer=self.r_optimizer.state_dict(),
                             t_optimizer=self.t_optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                             r_updates=self.r_updates, t_updates=self.t_updates,
                             precision=self.precision, progress=progress, rng=rng_state()))

    def restore(self, state, binding, *, arm, fork=False, allow_preflight=False):
        if state.get('format') != 'kuairec_listwise_v1' or state['binding'] != binding:
            raise ValueError('Checkpoint code/config/data/assets binding mismatch')
        if state['precision'] != self.precision:
            raise ValueError('Checkpoint precision mismatch')
        progress = state.get('progress')
        expected_counts = expected_update_counts(self.plan, progress)
        preflight_only = bool(state.get('preflight_only', False) or progress.get('preflight_only', False))
        if preflight_only and (not allow_preflight or fork):
            raise ValueError('Preflight-only checkpoint cannot initialize or resume a formal run')
        for kind, key in (('R', 'r_updates'), ('T', 't_updates')):
            updates = state.get(key)
            if type(updates) is not int or updates < 0:
                raise ValueError(f'Checkpoint {key} must be a nonnegative integer')
            if not preflight_only and updates != expected_counts[kind]:
                raise ValueError(f'{kind} cumulative updates disagree with the phase/update cursor')
            validate_optimizer_clock(state[f'{kind.lower()}_optimizer'], updates, kind)
        if state['scheduler']['last_epoch'] != state['r_updates']:
            raise ValueError('R scheduler clock disagrees with optimizer count')
        if int(state['alignment']['diger_optimizer_step']) != state['t_updates']:
            raise ValueError('T annealing clock disagrees with optimizer count')
        total_t = nominal_t_updates(self.config)
        if int(state['alignment']['diger_total_optimizer_steps']) != total_t:
            raise ValueError('T annealing total differs from the bound phase plan')
        if fork:
            expected = self.config['training']['warmup_epochs']
            expected_updates = sum(p['updates'] for p in self.plan[:expected])
            if (state['arm'] != 'warmup' or state['t_updates'] != 0
                    or state['progress']['phase'] != expected or state['progress']['update'] != 0
                    or state['r_updates'] != expected_updates):
                raise ValueError('Fork requires completed shared warmup with no T updates')
        elif state['arm'] != arm:
            raise ValueError('Cannot resume another arm; use explicit shared warmup fork')
        self.model.load_state_dict(state['model'], strict=True)
        self.alignment.load_state_dict(state['alignment'], strict=True)
        self.r_optimizer.load_state_dict(state['r_optimizer'])
        self.t_optimizer.load_state_dict(state['t_optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.r_updates, self.t_updates = state['r_updates'], state['t_updates']
        if self.scheduler.last_epoch != self.r_updates:
            raise ValueError('R scheduler clock disagrees with optimizer count')
        if int(self.alignment.diger_optimizer_step) != self.t_updates:
            raise ValueError('T annealing clock disagrees with optimizer count')
        restore_rng(state['rng'])
        return copy.deepcopy(progress)
