"""Explicit R5 fork: pair the original first 200 T batches across ten blocks.

The original model, loss, optimizer update and decoder are imported unchanged.
This module owns only the new schedule, paired data/RNG streams and envelope.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .recovery import fingerprint
from .recovery35 import Recovery35Controller
from .run import build_context, phase_indices
from .training import cpu_copy, expected_update_counts, restore_rng, rng_state, seed_all, sha256, validate_optimizer_clock

FORMAT = 'stage09_h1_spread200_v1'
NAME = 'H1-Spread200-R35'
BEST_EPOCHS = (20, 23, 26, 29, 32, 35)
VALIDATION_EPOCHS = (8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
STATE_KEYS = ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')


def verify_binding(parent, execution):
    if set(parent) != set(execution):
        raise ValueError('Binding schema changed')
    old, new = copy.deepcopy(parent['config']), copy.deepcopy(execution['config'])
    paths = {k: dict(parent=old.pop(k), execution=new.pop(k)) for k in ('dataset', 'assets')}
    old_block = old['training'].pop('t_updates_per_cycle')
    new_block = new['training'].pop('t_updates_per_cycle')
    if old != new or old_block != 2 * new_block:
        raise ValueError('Only halving T block size and explicit path relocation are allowed')
    for k in ('asset_manifest', 'data', 'engine'):
        if parent[k] != execution[k]:
            raise ValueError('Historical ' + k + ' changed')
    for name, digest in parent['code'].items():
        if execution['code'].get(name) != digest:
            raise ValueError('Historical source changed: ' + name)
    additions = sorted(set(execution['code']) - set(parent['code']))
    if any(not (name.startswith('scripts/listwise_diffgrm/recovery') or
                name.startswith('scripts/listwise_diffgrm/spread200')) or not name.endswith('.py')
           for name in additions):
        raise ValueError('Unexpected source extension')
    return dict(paths=paths, original_t_block=old_block, new_t_block=new_block,
                extensions={k: execution['code'][k] for k in additions})


def paired_indices(config, count, phase, row):
    """Use identical original first-half/second-half batches, never new draws."""
    if row['kind'] == 'R':
        return phase_indices(config, count, phase, row)
    block = (phase - config['training']['warmup_epochs']) // 4
    width = config['training']['t_updates_per_cycle']
    if phase != config['training']['warmup_epochs'] + 4 * block:
        raise ValueError('Unexpected T phase')
    old_phase = config['training']['warmup_epochs'] + 4 * (block // 2)
    old = phase_indices(config, count, old_phase, dict(kind='T', updates=2 * width))
    start = (block % 2) * width * config['training']['effective_batch_size']
    size = width * config['training']['effective_batch_size']
    return old[start:start + size]


def index_sha(indices):
    return hashlib.sha256(np.asarray(indices).astype('<i8', copy=False).tobytes()).hexdigest()


def reject_tainted(state):
    if state.get('preflight_only') or state.get('inspection_only'):
        raise ValueError('Preflight/inspection state cannot initialize formal training')
    if state.get('progress', {}).get('preflight_only'):
        raise ValueError('Preflight progress cannot initialize formal training')
    if 'inner' in state:
        reject_tainted(state['inner'])


class SpreadController:
    _validate_structures = Recovery35Controller._validate_structures
    _load = Recovery35Controller._load

    def __init__(self, trainer, parent_binding, execution_binding, parent_sha):
        self.branch = 'D'
        self.trainer = trainer
        self.parent_binding = copy.deepcopy(parent_binding)
        self.execution_binding = copy.deepcopy(execution_binding)
        proof = verify_binding(parent_binding, execution_binding)
        self.width = proof['new_t_block']
        self.old_width = proof['original_t_block']
        self.start_phase = trainer.config['training']['warmup_epochs']
        self.stop_phase = len(trainer.plan)
        if (trainer.config['architecture'] != 'H' or not trainer.alignment.mixer.mixer_enabled
                or [p['kind'] for p in trainer.plan] != ['R'] * self.start_phase + ['T', 'R', 'R', 'R'] * 10):
            raise ValueError('Only the ten-cycle H1 schedule is supported')
        self.noise_total = self.old_width * 10
        self.protocol = dict(format=FORMAT, name=NAME, parent_sha256=parent_sha,
            source_proof=proof, t_updates=self.width * 10, noise_total=self.noise_total,
            paired_t_batches='original_first_half_then_second_half',
            paired_t_rng='original_40T_stream_saved_across_inserted_R',
            original_r_phase_ids=True, best_epochs=list(BEST_EPOCHS))
        self.parent_hashes = None
        self.parent_optimizer_groups = None
        self.progress = None
        self.t_rng = None
        self.inspection_only = False

    @property
    def done(self):
        return self.progress is not None and self.progress['phase'] == self.stop_phase

    @property
    def row(self):
        if self.progress is None or self.done:
            raise RuntimeError('Controller is not active')
        return self.trainer.plan[self.progress['phase']]

    def validate_clocks(self, inner):
        progress = inner['progress']
        if not self.start_phase <= progress['phase'] <= self.stop_phase:
            raise ValueError('Phase outside the authorized schedule')
        counts = expected_update_counts(self.trainer.plan, progress)
        for kind in ('R', 'T'):
            key = kind.lower() + '_updates'
            if inner[key] != counts[kind]:
                raise ValueError(kind + ' counter differs from schedule')
            validate_optimizer_clock(inner[kind.lower() + '_optimizer'], counts[kind], kind)
        if inner['scheduler']['last_epoch'] != counts['R']:
            raise ValueError('Original R scheduler clock changed')
        if (int(inner['alignment']['diger_optimizer_step']) != counts['T'] or
                int(inner['alignment']['diger_total_optimizer_steps']) != self.noise_total):
            raise ValueError('Original 400T noise clock changed')
        if progress['phase'] < self.stop_phase:
            row = self.trainer.plan[progress['phase']]
            if progress.get('boundary_prepared') and progress['update'] != row['updates']:
                raise ValueError('Boundary prepared before completing updates')
        if self.parent_optimizer_groups:
            for key in ('r_optimizer', 't_optimizer'):
                def strip(groups):
                    return [{k: v for k, v in g.items() if k not in ('lr', 'initial_lr')}
                            for g in groups]
                if strip(inner[key]['param_groups']) != strip(self.parent_optimizer_groups[key]):
                    raise ValueError('Optimizer topology or hyperparameters changed')
            if [g['lr'] for g in inner['t_optimizer']['param_groups']] != [
                    g['lr'] for g in self.parent_optimizer_groups['t_optimizer']]:
                raise ValueError('T learning rate changed')

    def fork(self, parent, *, exact_rng=True):
        reject_tainted(parent)
        if (parent.get('format') != 'kuairec_listwise_v1' or parent['arm'] != 'H1'
                or parent['binding'] != self.parent_binding or parent['precision'] != self.trainer.precision
                or parent['progress']['phase'] != self.start_phase
                or parent['progress']['update'] != 0 or parent['t_updates'] != 0):
            raise ValueError('Requires authenticated original H1 R5/T0')
        self.parent_hashes = {k: fingerprint(parent[k]) for k in STATE_KEYS}
        self.parent_optimizer_groups = {k: cpu_copy(parent[k]['param_groups'])
                                        for k in ('r_optimizer', 't_optimizer')}
        self.progress = dict(phase=self.start_phase, update=0, boundary_prepared=False, rng_prepared=False)
        inner = cpu_copy(parent)
        inner['progress'] = copy.deepcopy(self.progress)
        self.validate_clocks(inner)
        self._load(inner, exact_rng=exact_rng)
        self.inspection_only = not exact_rng
        return self.checkpoint()

    def prepare_rng(self):
        if self.progress['rng_prepared']:
            return
        if self.progress['update']:
            raise ValueError('Cannot initialize RNG mid-phase')
        phase = self.progress['phase']
        if self.row['kind'] == 'R':
            seed_all(self.trainer.config['seed'] + 100003 * (phase + 1))
        elif self.trainer.t_updates % self.old_width == 0:
            old_phase = self.start_phase + 4 * (self.trainer.t_updates // self.old_width)
            seed_all(self.trainer.config['seed'] + 100003 * (old_phase + 1))
            self.t_rng = cpu_copy(rng_state())
        else:
            if self.t_rng is None:
                raise ValueError('Missing continuation of original T random stream')
            restore_rng(self.t_rng)
        self.progress['rng_prepared'] = True

    def step(self, batches, kind):
        if self.inspection_only or self.done or self.progress['boundary_prepared']:
            raise RuntimeError('State cannot advance')
        if kind != self.row['kind'] or self.progress['update'] >= self.row['updates']:
            raise ValueError('Update outside the current phase')
        self.prepare_rng()
        before = self.trainer.alignment.diger_schedule_state()
        lrs = dict(R=self.trainer.r_optimizer.param_groups[0]['lr'],
                   **{g['group_name']: g['lr'] for g in self.trainer.t_optimizer.param_groups})
        result = self.trainer.update(batches, kind)
        if kind == 'T':
            self.t_rng = cpu_copy(rng_state())
        self.progress['update'] += 1
        return dict(result, learning_rates_used=lrs, noise_used=before,
                    phase_index=self.progress['phase'], update=self.progress['update'])

    def prepare_boundary(self):
        if self.inspection_only or self.done or self.progress['update'] != self.row['updates']:
            raise RuntimeError('Incomplete/inspection phase cannot publish boundary')
        if not self.progress['boundary_prepared']:
            if self.row['kind'] == 'T':
                self.trainer.refresh_catalog()
            self.progress['boundary_prepared'] = True

    def complete_phase(self):
        if not self.progress['boundary_prepared']:
            raise RuntimeError('Publish boundary first')
        self.progress = dict(phase=self.progress['phase'] + 1, update=0,
                             boundary_prepared=False, rng_prepared=False)

    def checkpoint(self):
        if self.parent_hashes is None:
            raise RuntimeError('Authenticate parent first')
        inner = self.trainer.checkpoint(self.execution_binding, self.progress, NAME)
        self.validate_clocks(inner)
        return dict(format=FORMAT, protocol=copy.deepcopy(self.protocol),
            parent_binding=copy.deepcopy(self.parent_binding),
            execution_binding=copy.deepcopy(self.execution_binding),
            parent_hashes=copy.deepcopy(self.parent_hashes),
            inspection_only=self.inspection_only, t_rng=cpu_copy(self.t_rng), inner=inner)

    def restore(self, state, *, exact_rng=True):
        reject_tainted(state)
        for key, expected in (('format', FORMAT), ('protocol', self.protocol),
                ('parent_binding', self.parent_binding), ('execution_binding', self.execution_binding),
                ('parent_hashes', self.parent_hashes)):
            if state.get(key) != expected:
                raise ValueError('Spread checkpoint ' + key + ' mismatch')
        inner = state['inner']
        if (inner['binding'] != self.execution_binding or inner['arm'] != NAME
                or inner['precision'] != self.trainer.precision):
            raise ValueError('Spread inner identity changed')
        self.validate_clocks(inner)
        if inner['t_updates'] and state['t_rng'] is None:
            raise ValueError('Missing paired T RNG')
        self._load(inner, exact_rng=exact_rng)
        self.progress = copy.deepcopy(inner['progress'])
        self.t_rng = cpu_copy(state['t_rng'])
        if not exact_rng and self.t_rng is not None:
            self.t_rng['cuda'] = None
        self.inspection_only = not exact_rng


def build_spread(parent_path, expected_sha, device, dataset=None, assets=None, *, exact_rng=True):
    if sha256(parent_path) != expected_sha:
        raise ValueError('Authenticated parent bytes changed')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(parent['binding']['config'])
    if cfg['training']['t_updates_per_cycle'] != 40 or cfg['training']['cycles'] != 10:
        raise ValueError('Formal source is not the original 40T ten-cycle plan')
    cfg['training']['t_updates_per_cycle'] = 20
    if dataset:
        cfg['dataset'] = dataset
    if assets:
        cfg['assets'] = assets
    trainer, loaded_assets, data, current = build_context(cfg, device, parent['precision'], True)
    if len(data) != 709443 or loaded_assets.n_items != 10728:
        raise ValueError('Formal dataset/catalog dimensions changed')
    controller = SpreadController(trainer, parent['binding'], current, expected_sha)
    controller.fork(parent, exact_rng=exact_rng)
    return controller, loaded_assets, data
