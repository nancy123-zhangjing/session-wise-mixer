"""Authenticated A/B/C continuations on the immutable original 35-R plan.

This module never submits a job. The runner owns evaluation, immutable file
publication and the numerical replay audit; this controller owns state/clocks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .recovery import RecoveryController as LegacyController, fingerprint
from .recovery import PARENT_SHA256 as LEGACY_PARENT_SHA256
from .training import cpu_copy, expected_update_counts, restore_rng, sha256, validate_optimizer_clock

FORMAT = 'stage09_recovery35_v1'
BRANCHES = {
    'A': dict(name='H1-Freeze20-R35', arm='H1-freeze', mixer=True,
              parent_sha256='1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
              start_phase=33, freeze_phase=25, parent_r=18018, parent_t=200),
    'B': dict(name='H0-Freeze32-R35', arm='H0', mixer=False,
              parent_sha256='8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e',
              start_phase=41, freeze_phase=41, parent_r=22176, parent_t=360),
    'C': dict(name='H0-Freeze20-R35', arm='H0', mixer=False,
              parent_sha256='b1adf68c7c5b3f1818773c1ef34adaeac4df49be57a867c6b11bcaf38cdf9166',
              start_phase=5, freeze_phase=25, parent_r=3465, parent_t=0),
}
REFERENCE_R20_PREDICTION_SHA256 = '795a50ebe753376647aac8b53e87a2c254d860d8b76d2bed0ae95d79949b91ec'
LEGACY_EXTENSIONS = {
    'recovery.py': '979edd8b1aa8a21fa15c08fc6c3d7af3b98afd2d3caecb4799360a7b21efc56e',
    'recovery_diagnostics.py': '968559500b14153b42cba6019afd294c4c4449ce277d55f68cf87dff06f4b219',
    'recovery_evaluation.py': '2786b1824ab42553019410c0d7a9a7a167136c7156738c5a3550408877c7f06e',
    'recovery_preflight.py': '2a6ea4e88d999530b964b8ad6193ba3d28fdb33a702b1a21797d68a774349c96',
    'recovery_run.py': '193efb63f5f81e8c07b08d1e17cce5e7caa399f99ec99a59ecb8e014da07c7f0',
}
STATE_KEYS = ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')
_UNSET = object()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def verify_binding(parent, current):
    """Only paths, authenticated legacy extensions and recovery35*.py may differ."""
    if set(parent) != set(current):
        raise ValueError('Unexpected binding schema change')
    old, new = copy.deepcopy(parent['config']), copy.deepcopy(current['config'])
    relocations = {}
    for key in ('dataset', 'assets'):
        left, right = old.pop(key), new.pop(key)
        if left != right:
            relocations[key] = dict(parent=left, execution=right)
    if old != new:
        raise ValueError('Scientific configuration differs from parent')
    for key in ('asset_manifest', 'data', 'engine'):
        if parent[key] != current[key]:
            raise ValueError(f'Parent {key} differs from execution')
    for name, digest in parent['code'].items():
        if current['code'].get(name) != digest:
            raise ValueError(f'Original source changed: {name}')
    additions = sorted(set(current['code']) - set(parent['code']))
    for name in additions:
        relative = Path(name)
        if relative.parent.as_posix() != 'scripts/listwise_diffgrm':
            raise ValueError(f'Unexpected source extension: {name}')
        if relative.name in LEGACY_EXTENSIONS:
            if current['code'][name] != LEGACY_EXTENSIONS[relative.name]:
                raise ValueError(f'Legacy extension bytes changed: {name}')
        elif not (relative.name.startswith('recovery35') and relative.suffix == '.py'):
            raise ValueError(f'Unexpected source extension: {name}')
        if not _hex(current['code'][name]):
            raise ValueError('Invalid source SHA256')
    return dict(relocations=relocations, extensions={name: current['code'][name] for name in additions})


def _reject_tainted(state):
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must be a mapping')
    if state.get('preflight_only') or state.get('inspection_only'):
        raise ValueError('Preflight/inspection checkpoint cannot initialize a formal run')
    if isinstance(state.get('progress'), dict) and state['progress'].get('preflight_only'):
        raise ValueError('Preflight progress cannot initialize a formal run')
    if 'inner' in state:
        _reject_tainted(state['inner'])


class Recovery35Controller:
    """A new envelope: authenticate source before every restore, never re-warmup."""

    def __init__(self, trainer, branch, parent_binding, current_binding, parent_sha256):
        if branch not in BRANCHES:
            raise ValueError('Branch must be A, B or C')
        self.trainer, self.branch = trainer, branch
        self.spec = copy.deepcopy(BRANCHES[branch])
        if parent_sha256 != self.spec['parent_sha256']:
            raise ValueError('Parent SHA is not the locked branch source')
        if len(trainer.plan) != 45 or trainer.plan[-1]['epoch'] != 35:
            raise ValueError('Continuation requires the original 45-phase/35R plan')
        if [r['kind'] for r in trainer.plan] != ['R'] * 5 + ['T', 'R', 'R', 'R'] * 10:
            raise ValueError('Original phase numbering changed')
        if trainer.config['architecture'] != 'H' or bool(trainer.alignment.mixer.mixer_enabled) != self.spec['mixer']:
            raise ValueError('Branch architecture or Mixer flag mismatch')
        self.parent_binding, self.execution_binding = copy.deepcopy(parent_binding), copy.deepcopy(current_binding)
        self.start_phase, self.stop_phase = self.spec['start_phase'], 45
        self.freeze_phase = self.spec['freeze_phase']
        self.freeze_t = expected_update_counts(trainer.plan, dict(phase=self.freeze_phase, update=0))['T']
        self.total_t = sum(row['updates'] for row in trainer.plan if row['kind'] == 'T')
        self.protocol = dict(format=FORMAT, branch=branch, name=self.spec['name'],
                             parent_sha256=parent_sha256, start_phase=self.start_phase,
                             freeze_phase=self.freeze_phase, stop_phase=self.stop_phase,
                             original_r_schedule=True, frozen_t_clock=self.freeze_t,
                             source_proof=verify_binding(parent_binding, current_binding))
        self.progress = self.parent_hashes = self.parent_optimizer_groups = None
        self.freeze_hashes = self.replay_gate = self.replay_gate_state_sha256 = None
        self.inspection_only = False

    @property
    def protocol_sha256(self):
        return hashlib.sha256(json.dumps(self.protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @property
    def frozen(self):
        return self.progress is not None and self.progress['phase'] >= self.freeze_phase

    @property
    def needs_replay_gate(self):
        return self.branch == 'C' and self.frozen and self.replay_gate is None

    @property
    def done(self):
        return self.progress is not None and self.progress['phase'] == self.stop_phase

    @property
    def row(self):
        if self.progress is None or self.done:
            raise RuntimeError('Controller is not active or continuation completed')
        return self.trainer.plan[self.progress['phase']]

    @property
    def skip_t(self):
        return not self.done and self.frozen and self.row['kind'] == 'T'

    def counts_at(self, progress):
        phase, update = progress.get('phase'), progress.get('update')
        if (type(phase) is not int or not self.start_phase <= phase <= self.stop_phase
                or type(progress.get('boundary_prepared')) is not bool or progress.get('preflight_only')):
            raise ValueError('Invalid recovery35 progress')
        counts = expected_update_counts(self.trainer.plan, progress)
        frozen = phase >= self.freeze_phase
        if frozen:
            if phase < self.stop_phase and self.trainer.plan[phase]['kind'] == 'T' and update != 0:
                raise ValueError('Frozen T phase must have zero updates')
            counts['T'] = self.freeze_t
        if progress['boundary_prepared']:
            if phase == self.stop_phase:
                raise ValueError('Completed continuation has no pending boundary')
            row = self.trainer.plan[phase]
            if update != (0 if frozen and row['kind'] == 'T' else row['updates']):
                raise ValueError('Prepared boundary has incomplete optimizer updates')
        return counts

    def _check_scheduler(self, inner, step):
        cfg = self.trainer.config['training']
        total = sum(r['updates'] for r in self.trainer.plan if r['kind'] == 'R')
        warmup, peak = cfg['decoder_warmup_updates'], cfg['decoder_lr']
        ratio = (max(step, 1) / max(warmup, 1) if step < warmup else
                 .5 * (1 + math.cos(math.pi * min(max((step - warmup) / max(total - warmup, 1), 0), 1))))
        expected = peak * ratio
        scheduler = inner['scheduler']
        if (scheduler['last_epoch'] != step or scheduler['_step_count'] != step + 1
                or scheduler['base_lrs'] != [peak] or len(scheduler['_last_lr']) != 1
                or len(inner['r_optimizer']['param_groups']) != 1
                or not math.isclose(scheduler['_last_lr'][0], expected, rel_tol=1e-12, abs_tol=1e-18)
                or not math.isclose(inner['r_optimizer']['param_groups'][0]['lr'], expected, rel_tol=1e-12, abs_tol=1e-18)):
            raise ValueError('R scheduler differs from the original full-35R schedule')

    def validate_clocks(self, inner, *, freeze_hashes=_UNSET):
        counts = self.counts_at(inner['progress'])
        for kind, key in (('R', 'r_updates'), ('T', 't_updates')):
            if type(inner[key]) is not int or inner[key] != counts[kind]:
                raise ValueError(f'{kind} count disagrees with recovery35 cursor')
            validate_optimizer_clock(inner[f'{kind.lower()}_optimizer'], counts[kind], kind)
        self._check_scheduler(inner, counts['R'])
        for name, original in self.parent_optimizer_groups.items():
            strip = lambda groups: [{k: v for k, v in g.items() if k != 'lr'} for g in groups]
            if strip(inner[name]['param_groups']) != strip(original):
                raise ValueError(f'{name} topology or hyperparameters changed')
        for actual, original in zip(inner['t_optimizer']['param_groups'], self.parent_optimizer_groups['t_optimizer']):
            if actual['lr'] != original['lr']:
                raise ValueError('Original T learning rate changed')
        if (int(inner['alignment']['diger_optimizer_step']) != counts['T']
                or int(inner['alignment']['diger_total_optimizer_steps']) != self.total_t):
            raise ValueError('DIGER noise clock or original total changed')
        anchor = self.freeze_hashes if freeze_hashes is _UNSET else freeze_hashes
        if inner['progress']['phase'] >= self.freeze_phase:
            if not isinstance(anchor, dict) or set(anchor) != {'alignment', 't_optimizer'}:
                raise ValueError('Frozen state has no authenticated freeze anchor')
            for key in anchor:
                if fingerprint(inner[key]) != anchor[key]:
                    raise ValueError(f'Frozen {key} changed')
            if self.branch in ('A', 'B') and anchor != {k: self.parent_hashes[k] for k in anchor}:
                raise ValueError('Freeze anchor differs from authenticated parent')
        elif freeze_hashes is not _UNSET and anchor is not None:
            raise ValueError('Replay cannot have a freeze anchor before R20')

    def _validate_structures(self, inner):
        for key, module in (('model', self.trainer.model), ('alignment', self.trainer.alignment)):
            expected = module.state_dict()
            if set(inner[key]) != set(expected):
                raise ValueError(f'{key} state keys changed')
            for name, value in inner[key].items():
                if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                    raise ValueError(f'{key}.{name} shape/dtype changed')
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f'{key}.{name} is not finite')
        for key, optimizer in (('r_optimizer', self.trainer.r_optimizer), ('t_optimizer', self.trainer.t_optimizer)):
            saved = inner[key]
            if len(saved['param_groups']) != len(optimizer.param_groups):
                raise ValueError('Optimizer group count changed')
            ids = set()
            for old, active in zip(saved['param_groups'], optimizer.param_groups):
                if len(old['params']) != len(active['params']):
                    raise ValueError('Optimizer parameter mapping changed')
                for index, parameter in zip(old['params'], active['params']):
                    if index in ids:
                        raise ValueError('Duplicate optimizer parameter mapping')
                    ids.add(index)
                    for name, value in saved['state'].get(index, {}).items():
                        if name != 'step' and (value.shape != parameter.shape or not torch.isfinite(value).all()):
                            raise ValueError('Optimizer moment shape/finite check failed')
            if not set(saved['state']).issubset(ids):
                raise ValueError('Unmapped optimizer state')

    def _load(self, inner, *, exact_rng):
        if not exact_rng and self.trainer.device.type != 'cpu':
            raise ValueError('Non-exact RNG restoration is CPU inspection only')
        saved_rng = cpu_copy(inner['rng'])
        if not exact_rng:
            saved_rng['cuda'] = None
        if saved_rng['cuda'] is not None and not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG requires CUDA; request explicit CPU inspection')
        self._validate_structures(inner)
        t = self.trainer
        t.model.load_state_dict(inner['model'], strict=True)
        t.alignment.load_state_dict(inner['alignment'], strict=True)
        # load_state_dict can alias Adam step/moment storage: clone reusable snapshots.
        t.r_optimizer.load_state_dict(cpu_copy(inner['r_optimizer']))
        t.t_optimizer.load_state_dict(cpu_copy(inner['t_optimizer']))
        t.scheduler.load_state_dict(copy.deepcopy(inner['scheduler']))
        t.r_updates, t.t_updates = inner['r_updates'], inner['t_updates']
        restore_rng(saved_rng)

    def fork(self, parent_state, *, exact_rng=True):
        _reject_tainted(parent_state)
        if (parent_state.get('format') != 'kuairec_listwise_v1' or parent_state['arm'] != self.spec['arm']
                or parent_state['binding'] != self.parent_binding
                or parent_state['precision'] != self.trainer.precision
                or parent_state['progress'].get('phase') != self.start_phase
                or parent_state['progress'].get('update') != 0
                or parent_state['progress'].get('boundary_prepared', False)):
            raise ValueError('Parent is not the authenticated branch R boundary')
        self.parent_hashes = {key: fingerprint(parent_state[key]) for key in STATE_KEYS}
        self.parent_optimizer_groups = {key: cpu_copy(parent_state[key]['param_groups'])
                                        for key in ('r_optimizer', 't_optimizer')}
        self.progress = dict(phase=self.start_phase, update=0, boundary_prepared=False)
        inner = cpu_copy(parent_state)
        inner['progress'] = copy.deepcopy(self.progress)
        if self.branch in ('A', 'B'):
            self.freeze_hashes = {key: self.parent_hashes[key] for key in ('alignment', 't_optimizer')}
        self.validate_clocks(inner)
        self._load(inner, exact_rng=exact_rng)
        self.inspection_only = not exact_rng
        return self.checkpoint()

    def checkpoint(self):
        if self.parent_hashes is None:
            raise RuntimeError('Authenticate and fork the original source first')
        inner = self.trainer.checkpoint(self.execution_binding, self.progress, self.spec['name'])
        self.validate_clocks(inner)
        return dict(format=FORMAT, protocol=copy.deepcopy(self.protocol),
                    parent_binding=copy.deepcopy(self.parent_binding), execution_binding=copy.deepcopy(self.execution_binding),
                    parent_hashes=copy.deepcopy(self.parent_hashes), freeze_hashes=copy.deepcopy(self.freeze_hashes),
                    replay_gate=copy.deepcopy(self.replay_gate), replay_gate_state_sha256=self.replay_gate_state_sha256,
                    inspection_only=self.inspection_only, inner=inner)

    def _validate_gate(self, receipt, state_sha):
        if (not isinstance(receipt, dict) or receipt.get('status') != 'passed' or receipt.get('branch') != 'C'
                or receipt.get('checkpoint_state_sha256') != state_sha
                or receipt.get('protocol_sha256') != self.protocol_sha256
                or receipt.get('reference_prediction_sha256') != REFERENCE_R20_PREDICTION_SHA256):
            raise ValueError('Replay gate does not bind this protocol/R20 state/reference')
        for key in ('checkpoint_state_sha256', 'validation_sha256', 'reference_prediction_sha256',
                    'candidate_prediction_sha256', 'protocol_sha256'):
            if not _hex(receipt.get(key)):
                raise ValueError(f'Invalid replay gate digest: {key}')

    def accept_replay_gate(self, receipt):
        if (self.branch != 'C' or self.progress != dict(phase=25, update=0, boundary_prepared=False)
                or self.replay_gate is not None or self.inspection_only):
            raise ValueError('Replay gate can only be accepted once at the completed R20 boundary')
        digest = fingerprint(self.checkpoint()['inner'])
        self._validate_gate(receipt, digest)
        self.replay_gate_state_sha256, self.replay_gate = digest, copy.deepcopy(receipt)

    def restore(self, state, *, exact_rng=True, allow_preflight=False):
        if self.parent_hashes is None:
            raise RuntimeError('Authenticate and fork the original source before resume')
        # The compatibility keyword is deliberately not a formal-run bypass.
        _reject_tainted(state)
        for key, expected in (('format', FORMAT), ('protocol', self.protocol),
                              ('parent_binding', self.parent_binding), ('execution_binding', self.execution_binding),
                              ('parent_hashes', self.parent_hashes)):
            if state.get(key) != expected:
                raise ValueError(f'Recovery35 {key} mismatch')
        inner = state['inner']
        if (inner.get('format') != 'kuairec_listwise_v1' or inner['binding'] != self.execution_binding
                or inner['arm'] != self.spec['name'] or inner['precision'] != self.trainer.precision):
            raise ValueError('Inner checkpoint identity/precision mismatch')
        anchor, gate, gate_sha = state.get('freeze_hashes'), state.get('replay_gate'), state.get('replay_gate_state_sha256')
        self.validate_clocks(inner, freeze_hashes=anchor)
        phase = inner['progress']['phase']
        if self.branch == 'C':
            if gate is not None:
                if phase < 25:
                    raise ValueError('Replay gate predates R20')
                self._validate_gate(gate, gate_sha)
                if (inner['progress'] == dict(phase=25, update=0, boundary_prepared=False)
                        and fingerprint(inner) != gate_sha):
                    raise ValueError('Accepted R20 gate belongs to a different current state')
            elif phase >= 25 and inner['progress'] != dict(phase=25, update=0, boundary_prepared=False):
                raise ValueError('Replay passed R20 without an accepted numerical gate')
            elif gate_sha is not None:
                raise ValueError('Gate digest without gate receipt')
        elif gate is not None or gate_sha is not None:
            raise ValueError('Non-replay branch contains a replay gate')
        self._load(inner, exact_rng=exact_rng)
        self.progress = copy.deepcopy(inner['progress'])
        self.freeze_hashes, self.replay_gate = copy.deepcopy(anchor), copy.deepcopy(gate)
        self.replay_gate_state_sha256, self.inspection_only = gate_sha, not exact_rng

    def _require_active(self):
        if self.inspection_only:
            raise RuntimeError('CPU inspection cannot advance training')
        if self.needs_replay_gate:
            raise RuntimeError('R20 replay gate must pass before any frozen continuation')

    def step(self, microbatches, kind):
        self._require_active()
        row = self.row
        if (kind != row['kind'] or self.progress['boundary_prepared']
                or self.progress['update'] >= row['updates'] or (kind == 'T' and self.frozen)):
            raise ValueError('Update conflicts with phase, prepared boundary or frozen T')
        used = {'R': self.trainer.r_optimizer.param_groups[0]['lr'],
                **{g['group_name']: g['lr'] for g in self.trainer.t_optimizer.param_groups}}
        result = self.trainer.update(microbatches, kind)
        self.progress['update'] += 1
        return dict(result, learning_rates_used=used, phase_index=self.progress['phase'])

    def prepare_boundary(self):
        self._require_active()
        row = self.row
        if self.progress['update'] != (0 if self.skip_t else row['updates']):
            raise ValueError('Cannot prepare unfinished phase')
        if not self.progress['boundary_prepared']:
            if row['kind'] == 'T' and not self.frozen:
                self.trainer.refresh_catalog()
            self.progress['boundary_prepared'] = True

    def complete_phase(self):
        self._require_active()
        if not self.progress['boundary_prepared']:
            raise ValueError('Prepare and persist the boundary before completing it')
        self.progress = dict(phase=self.progress['phase'] + 1, update=0, boundary_prepared=False)
        if self.branch == 'C' and self.progress['phase'] == 25:
            self.freeze_hashes = dict(alignment=fingerprint(self.trainer.alignment.state_dict()),
                                      t_optimizer=fingerprint(self.trainer.t_optimizer.state_dict()))


def build_recovery35(parent_path, branch, device, dataset=None, assets=None, *, ancestor_path=None, inspection=False):
    """Hash files before loading. A additionally authenticates the old R20 fork.

    ``inspection=True`` is disclosed CPU-only loading, never an exact CUDA
    continuation, and its controller cannot train or initialize a formal resume.
    """
    if branch not in BRANCHES:
        raise ValueError('Branch must be A, B or C')
    spec = BRANCHES[branch]
    parent_path = Path(parent_path)
    if sha256(parent_path) != spec['parent_sha256']:
        raise ValueError('Parent checkpoint file SHA mismatch')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    _reject_tainted(parent)
    inner = parent['inner'] if branch == 'A' else parent
    config = copy.deepcopy(inner['binding']['config'])
    if dataset is not None:
        config['dataset'] = str(dataset)
    if assets is not None:
        config['assets'] = str(assets)
    from .run import build_context
    trainer, loaded_assets, data, current = build_context(config, device, inner['precision'], spec['mixer'])
    if (inner['r_updates'], inner['t_updates'], inner['progress']['phase'], inner['progress']['update']) != (
            spec['parent_r'], spec['parent_t'], spec['start_phase'], 0):
        raise ValueError('Source is not the exact formal branch boundary')
    if inner['precision'] != 'bf16' or len(data) != 709443 or sum(r['updates'] for r in trainer.plan if r['kind'] == 'R') != 24255:
        raise ValueError('Formal training precision/data/update budget changed')
    if branch == 'A':
        if ancestor_path is None or sha256(ancestor_path) != LEGACY_PARENT_SHA256:
            raise ValueError('A requires the authenticated original H1 R20 ancestor')
        ancestor = torch.load(ancestor_path, map_location='cpu', weights_only=False)
        _reject_tainted(ancestor)
        # Validate the old envelope against its OWN sealed execution binding;
        # current recovery35 extensions are checked separately, never retrofitted.
        verify_binding(parent['execution_binding'], current)
        legacy = LegacyController(trainer, 'freeze', ancestor['binding'],
                                  parent['execution_binding'], LEGACY_PARENT_SHA256)
        legacy.fork(ancestor, exact_rng=not inspection)
        legacy.restore(parent, exact_rng=not inspection)
    controller = Recovery35Controller(trainer, branch, inner['binding'], current, spec['parent_sha256'])
    controller.fork(inner, exact_rng=not inspection)
    # Source bytes may not change while the builder is importing/loading them.
    if sha256(parent_path) != spec['parent_sha256'] or (branch == 'A' and sha256(ancestor_path) != LEGACY_PARENT_SHA256):
        raise ValueError('Authenticated source file changed during restore')
    return controller, loaded_assets, data


# Small compatibility alias for callers that name their controller generically.
RecoveryController = Recovery35Controller
