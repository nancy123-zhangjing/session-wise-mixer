"""Audited H1 continuation, separate from the immutable Stage09 trainer."""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch

from .training import cpu_copy, expected_update_counts, restore_rng, validate_optimizer_clock

FORMAT = 'stage09_h1_recovery_v1'
PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'


def fingerprint(value):
    """Value hash independent of tensor storage IDs and torch.save containers."""
    digest = hashlib.sha256()

    def visit(obj):
        if torch.is_tensor(obj):
            item = obj.detach().cpu().contiguous()
            digest.update(f'tensor:{item.dtype}:{tuple(item.shape)}:'.encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, np.ndarray):
            digest.update(f'array:{obj.dtype}:{obj.shape}:'.encode())
            digest.update(obj.tobytes())
        elif isinstance(obj, dict):
            digest.update(b'dict{')
            for key in sorted(obj, key=repr):
                visit(key)
                visit(obj[key])
            digest.update(b'}')
        elif isinstance(obj, (tuple, list)):
            digest.update(type(obj).__name__.encode() + b'[')
            for item in obj:
                visit(item)
            digest.update(b']')
        else:
            digest.update(f'{type(obj).__name__}:{obj!r};'.encode())

    visit(value)
    return digest.hexdigest()


def verify_parent_binding(parent, current):
    """Allow explicit path relocation and new recovery modules, never old edits."""
    if set(parent) != set(current):
        raise ValueError('Unexpected binding schema change')
    old_cfg, new_cfg = copy.deepcopy(parent['config']), copy.deepcopy(current['config'])
    relocations = {}
    for key in ('dataset', 'assets'):
        old, new = old_cfg.pop(key), new_cfg.pop(key)
        if old != new:
            relocations[key] = dict(parent=old, execution=new)
    if old_cfg != new_cfg:
        raise ValueError('Scientific configuration differs from parent')
    for key in ('asset_manifest', 'data', 'engine'):
        if parent[key] != current[key]:
            raise ValueError(f'Parent {key} bytes do not match execution')
    for name, digest in parent['code'].items():
        if current['code'].get(name) != digest:
            raise ValueError(f'Original code bytes changed: {name}')
    additions = sorted(set(current['code']) - set(parent['code']))
    if any(not name.startswith('scripts/listwise_diffgrm/recovery') or not name.endswith('.py')
           for name in additions):
        raise ValueError('Unexpected extension outside recovery modules')
    return dict(relocations=relocations, extensions={name: current['code'][name] for name in additions})


class RecoveryController:
    """Preserve the full original phase plan and validate policy-specific clocks.

    A fresh controller must fork from the independently authenticated parent
    before restore(), so a resumed file cannot supply its own expected hashes.
    """

    def __init__(self, trainer, policy, parent_binding, current_binding,
                 parent_sha256, *, start_phase=25, stop_phase=33):
        if policy not in ('decay', 'freeze'):
            raise ValueError('Policy must be decay or freeze')
        if not 0 < start_phase < stop_phase <= len(trainer.plan):
            raise ValueError('Invalid continuation phase range')
        if trainer.plan[start_phase - 1]['kind'] != 'R' or trainer.plan[stop_phase - 1]['kind'] != 'R':
            raise ValueError('Fork and stop must be complete R boundaries')
        self.trainer, self.policy = trainer, policy
        self.parent_binding = copy.deepcopy(parent_binding)
        self.execution_binding = copy.deepcopy(current_binding)
        proof = verify_parent_binding(parent_binding, current_binding)
        self.protocol = dict(format=FORMAT, policy=policy, parent_sha256=parent_sha256,
                             start_phase=start_phase, stop_phase=stop_phase,
                             t_lr='peak_times_half_one_plus_cos_pi_completed_T_over_original_total',
                             source_proof=proof)
        self.start_phase, self.stop_phase = start_phase, stop_phase
        self.start_counts = expected_update_counts(trainer.plan, dict(phase=start_phase, update=0))
        self.total_t = sum(row['updates'] for row in trainer.plan if row['kind'] == 'T')
        self.peak_lrs = dict(codebook=trainer.config['training']['codebook_lr'],
                             mixer=trainer.config['training']['mixer_lr'])
        self.progress = None
        self.parent_hashes = None
        self.parent_optimizer_groups = None
        self.diagnostic_snapshot = None
        self.diagnostic_events = []

    def counts_at(self, progress):
        phase, update = progress.get('phase'), progress.get('update')
        if type(phase) is not int or not self.start_phase <= phase <= self.stop_phase:
            raise ValueError('Recovery cursor outside authorized phases')
        if phase == self.stop_phase and update != 0:
            raise ValueError('Stop cursor must have update=0')
        prepared = progress.get('boundary_prepared')
        if type(prepared) is not bool:
            raise ValueError('Boundary marker must be boolean')
        if progress.get('preflight_only'):
            raise ValueError('Preflight progress cannot be resumed formally')
        if prepared:
            if phase == self.stop_phase:
                raise ValueError('Completed recovery cannot have a pending boundary')
            row = self.trainer.plan[phase]
            required = 0 if self.policy == 'freeze' and row['kind'] == 'T' else row['updates']
            if update != required:
                raise ValueError('Prepared boundary has unfinished updates')
        counts = expected_update_counts(self.trainer.plan, progress)
        if self.policy == 'freeze':
            if phase < self.stop_phase and self.trainer.plan[phase]['kind'] == 'T' and update != 0:
                raise ValueError('Frozen T phase cannot contain optimizer updates')
            counts['T'] = self.start_counts['T']
        return counts

    def t_lrs(self, completed=None):
        u = self.trainer.t_updates if completed is None else completed
        factor = 0.5 * (1 + math.cos(math.pi * u / self.total_t))
        return {name: value * factor for name, value in self.peak_lrs.items()}

    def set_t_lrs(self):
        if self.policy == 'decay':
            values = self.t_lrs()
            for group in self.trainer.t_optimizer.param_groups:
                group['lr'] = values[group['group_name']]

    def fork(self, parent_state, *, exact_rng=True):
        if parent_state.get('arm') != 'H1' or parent_state.get('preflight_only'):
            raise ValueError('Requires a formal H1 parent')
        if parent_state['progress']['phase'] != self.start_phase or parent_state['progress']['update'] != 0:
            raise ValueError('Parent is not the locked completed R boundary')
        state = copy.deepcopy(parent_state)
        if not exact_rng:
            # Only a disclosed CPU restoration/inspection mode; never a GPU run.
            if self.trainer.device.type != 'cpu':
                raise ValueError('Non-exact RNG restoration is CPU inspection only')
            state['rng']['cuda'] = None
        self.trainer.restore(state, self.parent_binding, arm='H1')
        self.parent_hashes = {key: fingerprint(parent_state[key])
                              for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')}
        self.parent_optimizer_groups = {key: cpu_copy(parent_state[key]['param_groups'])
                                        for key in ('r_optimizer', 't_optimizer')}
        self.progress = dict(phase=self.start_phase, update=0, boundary_prepared=False)
        self.set_t_lrs()
        return self.checkpoint()

    def validate_clocks(self, state):
        counts = self.counts_at(state['progress'])
        for kind, key in (('R', 'r_updates'), ('T', 't_updates')):
            if state[key] != counts[kind]:
                raise ValueError(f'{kind} count disagrees with recovery cursor')
            validate_optimizer_clock(state[f'{kind.lower()}_optimizer'], counts[kind], kind)
        if state['scheduler']['last_epoch'] != counts['R']:
            raise ValueError('R scheduler was reset or advanced incorrectly')
        total_r = sum(row['updates'] for row in self.trainer.plan if row['kind'] == 'R')
        warmup = self.trainer.config['training']['decoder_warmup_updates']
        step = counts['R']
        factor = (max(step, 1) / max(warmup, 1) if step < warmup else
                  0.5 * (1 + math.cos(math.pi * min(max((step - warmup) / max(total_r - warmup, 1), 0), 1))))
        peak_r = self.trainer.config['training']['decoder_lr']
        expected_r = peak_r * factor
        if (state['scheduler']['base_lrs'] != [peak_r]
                or state['scheduler']['_step_count'] != step + 1
                or len(state['scheduler']['_last_lr']) != 1
                or not math.isclose(state['scheduler']['_last_lr'][0], expected_r, rel_tol=1e-12)
                or not math.isclose(state['r_optimizer']['param_groups'][0]['lr'], expected_r, rel_tol=1e-12)):
            raise ValueError('R learning rate differs from original full schedule')
        for key, original in self.parent_optimizer_groups.items():
            actual = state[key]['param_groups']
            strip_lr = lambda groups: [{k: v for k, v in group.items() if k != 'lr'} for group in groups]
            if strip_lr(actual) != strip_lr(original):
                raise ValueError(f'{key} topology or optimizer hyperparameters changed')
        if int(state['alignment']['diger_optimizer_step']) != counts['T']:
            raise ValueError('DIGER clock differs from actual T count')
        if int(state['alignment']['diger_total_optimizer_steps']) != self.total_t:
            raise ValueError('Original 400T noise schedule changed')
        if self.policy == 'freeze':
            for key in ('alignment', 't_optimizer'):
                if fingerprint(state[key]) != self.parent_hashes[key]:
                    raise ValueError(f'Frozen {key} changed from parent')
        else:
            expected = self.t_lrs(counts['T'])
            for group in state['t_optimizer']['param_groups']:
                if not math.isclose(group['lr'], expected[group['group_name']], rel_tol=1e-12):
                    raise ValueError('Restored T learning rate differs from completed-step clock')

    def checkpoint(self):
        if self.parent_hashes is None:
            raise RuntimeError('Authenticate and fork parent first')
        inner = self.trainer.checkpoint(self.execution_binding, self.progress, f'H1-{self.policy}')
        self.validate_clocks(inner)
        return dict(format=FORMAT, protocol=copy.deepcopy(self.protocol),
                    parent_binding=copy.deepcopy(self.parent_binding),
                    execution_binding=copy.deepcopy(self.execution_binding),
                    parent_hashes=copy.deepcopy(self.parent_hashes), inner=inner,
                    diagnostic_snapshot=cpu_copy(self.diagnostic_snapshot),
                    diagnostic_events=list(self.diagnostic_events))

    def restore(self, state, *, exact_rng=True, allow_preflight=False):
        if self.parent_hashes is None:
            raise RuntimeError('Authenticate and fork original parent before resume')
        if state.get('preflight_only') and not allow_preflight:
            raise ValueError('Preflight checkpoint cannot be resumed formally')
        for key, expected in (('format', FORMAT), ('protocol', self.protocol),
                              ('parent_binding', self.parent_binding),
                              ('execution_binding', self.execution_binding),
                              ('parent_hashes', self.parent_hashes)):
            if state.get(key) != expected:
                raise ValueError(f'Recovery {key} mismatch')
        inner = state['inner']
        if inner.get('format') != 'kuairec_listwise_v1':
            raise ValueError('Unsupported inner checkpoint format')
        if inner['binding'] != self.execution_binding or inner['arm'] != f'H1-{self.policy}':
            raise ValueError('Inner checkpoint identity mismatch')
        if inner['precision'] != self.trainer.precision or (inner.get('preflight_only') and not allow_preflight):
            raise ValueError('Precision mismatch or preflight state')
        self.validate_clocks(inner)
        t = self.trainer
        t.model.load_state_dict(inner['model'], strict=True)
        t.alignment.load_state_dict(inner['alignment'], strict=True)
        # Adam can retain CPU step tensors from load_state_dict even when the
        # parameters are on CUDA. Keep the caller's reusable snapshot immutable.
        t.r_optimizer.load_state_dict(cpu_copy(inner['r_optimizer']))
        t.t_optimizer.load_state_dict(cpu_copy(inner['t_optimizer']))
        t.scheduler.load_state_dict(copy.deepcopy(inner['scheduler']))
        t.r_updates, t.t_updates = inner['r_updates'], inner['t_updates']
        saved_rng = cpu_copy(inner['rng'])
        if not exact_rng:
            if t.device.type != 'cpu':
                raise ValueError('Non-exact RNG restoration is CPU inspection only')
            saved_rng['cuda'] = None
        restore_rng(saved_rng)
        self.progress = copy.deepcopy(inner['progress'])
        self.diagnostic_snapshot = cpu_copy(state.get('diagnostic_snapshot'))
        self.diagnostic_events = list(state.get('diagnostic_events', []))

    @property
    def done(self):
        return self.progress['phase'] == self.stop_phase

    @property
    def row(self):
        if self.done:
            raise RuntimeError('Authorized continuation already completed')
        return self.trainer.plan[self.progress['phase']]

    def step(self, microbatches, kind):
        row = self.row
        if kind != row['kind'] or self.progress['update'] >= row['updates']:
            raise ValueError('Update does not match phase cursor')
        if self.progress.get('boundary_prepared'):
            raise ValueError('Cannot update after boundary preparation')
        if kind == 'T' and self.policy == 'freeze':
            raise ValueError('Frozen branch must skip T optimizer entirely')
        used = {'R': self.trainer.r_optimizer.param_groups[0]['lr'],
                **{group['group_name']: group['lr'] for group in self.trainer.t_optimizer.param_groups}}
        values = self.trainer.update(microbatches, kind)
        self.progress['update'] += 1
        self.set_t_lrs()
        return dict(values, learning_rates_used=used, phase_index=self.progress['phase'])

    def prepare_boundary(self):
        row = self.row
        required = 0 if self.policy == 'freeze' and row['kind'] == 'T' else row['updates']
        if self.progress['update'] != required:
            raise ValueError('Phase has unfinished optimizer steps')
        if not self.progress.get('boundary_prepared'):
            if row['kind'] == 'T' and self.policy == 'decay':
                self.trainer.refresh_catalog()
            self.progress['boundary_prepared'] = True

    def complete_phase(self):
        if not self.progress.get('boundary_prepared'):
            raise ValueError('Prepare and persist boundary before completion')
        self.progress = dict(phase=self.progress['phase'] + 1, update=0, boundary_prepared=False)
