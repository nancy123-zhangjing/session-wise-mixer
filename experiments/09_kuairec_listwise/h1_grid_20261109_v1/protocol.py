"""Explicit R5 fork, added-parameter Adam clocks, and R20 Tokenizer freeze."""
import copy
import hashlib

import torch

from grid import attach_grid
from scripts.listwise_diffgrm.training import restore_rng, validate_optimizer_clock

FORMAT = 'stage09_history_grid_freeze_v1'
PARENT_SHA256 = '991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1'
GRID_PREFIX = 'backbone.item_local_attention.'


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(obj):
        if torch.is_tensor(obj):
            obj = obj.detach().cpu().contiguous()
            digest.update(str((obj.dtype, tuple(obj.shape))).encode())
            digest.update(obj.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                visit(key); visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            digest.update(type(obj).__name__.encode())
            for item in obj: visit(item)
        else:
            digest.update(repr(obj).encode())
        digest.update(b';')
    visit(value)
    return digest.hexdigest()


def skipped(row):
    return row['kind'] == 'T' and row['epoch'] >= 20


def counts_at(plan, progress):
    phase, update = progress['phase'], progress['update']
    if (type(phase) is not int or type(update) is not int or not 5 <= phase <= len(plan)
            or update < 0 or (phase == len(plan) and update != 0)
            or (phase < len(plan) and update > plan[phase]['updates'])):
        raise ValueError('Invalid grid phase/update cursor')
    if phase < len(plan) and skipped(plan[phase]) and update:
        raise ValueError('Frozen T phase cannot contain updates')
    counts = {'R': 0, 'T': 0}
    for row in plan[:phase]:
        if not skipped(row): counts[row['kind']] += row['updates']
    if phase < len(plan): counts[plan[phase]['kind']] += update
    return counts


def verify_binding(parent, current):
    """Permit asset/data path relocation only; all bytes/protocol remain bound."""
    old, new = copy.deepcopy(parent), copy.deepcopy(current)
    for key in ('dataset', 'assets'):
        old['config'].pop(key); new['config'].pop(key)
    if old != new:
        raise ValueError('Original code, engine, data, assets or scientific config changed')


class GridController:
    def __init__(self, trainer, parent, binding, spec, *, exact_rng=True):
        verify_binding(parent['binding'], binding)
        self.trainer, self.binding, self.spec = trainer, binding, copy.deepcopy(spec)
        self.start_r = sum(p['updates'] for p in trainer.plan[:5])
        if (parent['arm'] != 'H1' or parent['progress']['phase'] != 5
                or parent['progress']['update'] != 0 or parent['t_updates'] != 0
                or parent['r_updates'] != self.start_r):
            raise ValueError('Requires original completed H1 R5/T0')
        state = copy.deepcopy(parent)
        if not exact_rng:
            if trainer.device.type != 'cpu':
                raise ValueError('Non-exact CUDA RNG mode is only for CPU verification')
            state['rng']['cuda'] = None
        trainer.restore(state, parent['binding'], arm='H1')
        self.inherited_adam_ids = set(parent['r_optimizer']['state'])
        self.grid = attach_grid(trainer, spec['grid'])
        names = {id(p): name for name, p in trainer.model.named_parameters()}
        self.optimizer_names = [names[id(p)] for p in trainer.r_optimizer.param_groups[0]['params']]
        self.progress = copy.deepcopy(parent['progress'])
        self.freeze_hash = None

    def frozen_state(self):
        t = self.trainer
        return fingerprint(dict(alignment=t.alignment.state_dict(), optimizer=t.t_optimizer.state_dict()))

    def assert_frozen(self):
        if self.freeze_hash is not None and self.frozen_state() != self.freeze_hash:
            raise ValueError('Frozen Tokenizer, SID, T optimizer or T annealing clock changed')

    def update(self, batches, kind):
        row = self.trainer.plan[self.progress['phase']]
        if kind != row['kind'] or skipped(row) or self.progress['update'] >= row['updates']:
            raise ValueError('Update is not authorized at this phase cursor')
        result = self.trainer.update(batches, kind)
        self.progress['update'] += 1
        return result

    def finish_phase(self):
        row = self.trainer.plan[self.progress['phase']]
        required = 0 if skipped(row) else row['updates']
        if self.progress['update'] != required:
            raise ValueError('Cannot advance unfinished phase')
        if row['kind'] == 'R' and row['epoch'] == 20:
            self.freeze_hash = self.frozen_state()
        self.progress['phase'] += 1
        self.progress['update'] = 0
        self.assert_frozen()

    def checkpoint(self):
        self.assert_frozen()
        state = self.trainer.checkpoint(self.binding, copy.deepcopy(self.progress), self.spec['arm'])
        state.update(grid_format=FORMAT, experiment_spec=copy.deepcopy(self.spec),
                     r_optimizer_names=self.optimizer_names[:], freeze_hash=self.freeze_hash)
        return state

    def restore(self, state):
        t = self.trainer
        if (state.get('grid_format') != FORMAT or state['experiment_spec'] != self.spec
                or state['arm'] != self.spec['arm'] or state['binding'] != self.binding
                or state['precision'] != t.precision
                or state['r_optimizer_names'] != self.optimizer_names):
            raise ValueError('Grid checkpoint config/source/parameter mapping mismatch')
        counts = counts_at(t.plan, state['progress'])
        if (state['r_updates'], state['t_updates']) != (counts['R'], counts['T']):
            raise ValueError('Optimizer counters disagree with progress')
        r = state['r_optimizer']
        if len(r['param_groups']) != 1 or r['param_groups'][0]['params'] != list(range(len(self.optimizer_names))):
            raise ValueError('R optimizer parameter ordering changed')
        if not self.inherited_adam_ids.issubset(r['state']):
            raise ValueError('Inherited Adam moments missing')
        for index, entry in r['state'].items():
            if not 0 <= index < len(self.optimizer_names):
                raise ValueError('Unknown optimizer parameter')
            expected = counts['R'] - self.start_r if self.optimizer_names[index].startswith(GRID_PREFIX) else counts['R']
            if expected == 0 or float(entry['step']) != expected:
                raise ValueError('Inherited/new Adam clock mismatch')
        grid_indices = {i for i, name in enumerate(self.optimizer_names) if name.startswith(GRID_PREFIX)}
        if counts['R'] > self.start_r and not grid_indices.issubset(r['state']):
            raise ValueError('Grid Adam moments missing after R training')
        validate_optimizer_clock(state['t_optimizer'], counts['T'], 'T')
        total_t = sum(row['updates'] for row in t.plan if row['kind'] == 'T')
        if (state['scheduler']['last_epoch'] != counts['R']
                or int(state['alignment']['diger_optimizer_step']) != counts['T']
                or int(state['alignment']['diger_total_optimizer_steps']) != total_t):
            raise ValueError('Learning-rate or DIGER schedule mismatch')
        freeze_phase = next(i + 1 for i, row in enumerate(t.plan) if row['kind'] == 'R' and row['epoch'] == 20)
        if (state['progress']['phase'] >= freeze_phase) != (state['freeze_hash'] is not None):
            raise ValueError('Freeze anchor missing or premature')
        if state['freeze_hash'] is not None and fingerprint(dict(alignment=state['alignment'], optimizer=state['t_optimizer'])) != state['freeze_hash']:
            raise ValueError('Saved frozen state changed')
        t.model.load_state_dict(state['model'], strict=True)
        t.alignment.load_state_dict(state['alignment'], strict=True)
        t.r_optimizer.load_state_dict(r)
        t.t_optimizer.load_state_dict(state['t_optimizer'])
        t.scheduler.load_state_dict(state['scheduler'])
        t.r_updates, t.t_updates = counts['R'], counts['T']
        self.progress = copy.deepcopy(state['progress'])
        self.freeze_hash = state['freeze_hash']
        restore_rng(state['rng'])
