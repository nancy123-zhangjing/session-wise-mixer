"""Fail-closed engineering replay alarms, not equivalence/old-weight recovery.

The reference bytes and phase-order table were authenticated before execution.
Thresholds are fixed engineering drift alarms, not theoretical floating bounds.
No retries, optimizer updates, GPU calls or source log modifications occur here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .run import phase_indices
from .training import sha256

REFERENCE_LOG = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/runs/H0/training.jsonl'
REFERENCE_LOG_SHA256 = 'cacc662296711898aa105d6aab25042e45c2d4583273ab3dcfc945d4ad848dcd'
INPUT_AUDIT_SHA256 = 'e18583b5c649b0dc56dd9c9f4707b4ee089e75c421b7cb1143fa6230c68c1075'
TRAIN_COUNT = 709443
# (kind, epoch, updates, phase_seed, order_count, SHA256 of little-endian int64).
# Extracted from INPUT_AUDIT_COVERAGE.json / source_replay_training_log.phases.
PHASES = {
    5: ('T', 5, 40, 602042, 40960, '9594224be94e29d212be9e114979addfea99f44d5f9d373b57fd7809c29fe390'),
    6: ('R', 6, 693, 702045, 709443, 'c87db1dc3fffb226b32d1229d0f12fe0773180db9b1645a2db331100436d8dd3'),
    7: ('R', 7, 693, 802048, 709443, '7de9558e6f8591cc94a98b81216c44b5a74fec1ef648f72aa47faa0602396249'),
    8: ('R', 8, 693, 902051, 709443, '7b5155666303f6f14a548a6799ea83794df4ee5bcd7488f4f61343d116fbf20b'),
    9: ('T', 8, 40, 1002054, 40960, 'a4aa19f1ec1b9af575d8f9b59e7b8067c6951e101497d5881d4a4e9e6ba91c76'),
    10: ('R', 9, 693, 1102057, 709443, '570938722d9c126ca44f036c1dae091ae3b82b4f9fcc3894b35fbd3d4c648b64'),
    11: ('R', 10, 693, 1202060, 709443, 'fc22d77d9716ddbffae90bea93e14ba0abc3c7727d41e074f0dc88eff482821b'),
    12: ('R', 11, 693, 1302063, 709443, 'f1b7787b54fdc231c8e662b843e21e1e54521eb10cfb79addf90f5cd1a95f634'),
    13: ('T', 11, 40, 1402066, 40960, '48ef25e8c978d65ce343d7406744970586356d024cb9a83ceaa98728706d9547'),
    14: ('R', 12, 693, 1502069, 709443, 'b61fd0686e69f8be1358b573bde111994fec161d1563eb67724770112f40ff22'),
    15: ('R', 13, 693, 1602072, 709443, 'ea723242610aee3f983ad15540456b5832f5db94bcb4154f3df0991fb6702cb3'),
    16: ('R', 14, 693, 1702075, 709443, 'f8f3b7845fa72b569b72ef33079f6c6f6c06f77deb420e1ec57f252851296cb5'),
    17: ('T', 14, 40, 1802078, 40960, '3d864180d543e679ab183fb2e71087f63bac504f95b41797503680b5937d2709'),
    18: ('R', 15, 693, 1902081, 709443, '947cb0201821a5b9fcc91855ae30635bb9e97a7f86f3ece9fdef70b1f3333ea8'),
    19: ('R', 16, 693, 2002084, 709443, '74548f0807408bae4e3aa5cca8fe778865ba8b24d5894967fb5e48b41a34e2d0'),
    20: ('R', 17, 693, 2102087, 709443, '4061f79be9800702f82b0b0cb2b84c4226b9e90e45d56bc3a17b05fa00cf0b88'),
    21: ('T', 17, 40, 2202090, 40960, 'c6400c6064ca6bb71312b3f46b600789c51e098790e4d0fd781e0ddfd593b7e1'),
    22: ('R', 18, 693, 2302093, 709443, '1a464e3267f6e265aabc9bd9300cf2c0244409876a917c322d49ac2d415e7608'),
    23: ('R', 19, 693, 2402096, 709443, '0ac7317ad224d23d7569c2c7108eb56c300b63660840d0c41f583e822ec9858f'),
    24: ('R', 20, 693, 2502099, 709443, '26bda2df7fc00948d9ba2ff4d7a8cb45d9fe3bf6401056b5ff84fa48cadaec68'),
}
RULE = dict(format='recovery35_update_trajectory_rule_v1',
            loss_atol=1e-6, loss_rtol=1e-4, gradient_norm_atol=1e-6, gradient_norm_rtol=1e-3,
            comparison='abs(candidate-reference) <= atol + rtol*abs(reference)',
            integer_fields='exact', duplicates='scientific content exact; ignore only seconds/attempt',
            interpretation='engineering drift alarm; not theoretical error bound, equivalence or bitwise recovery')
INTEGER_FIELDS = ('phase_index', 'epoch', 'update', 'lists', 'r_updates', 't_updates')


class ReplayTrajectoryError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__('Replay trajectory requires review; retained evidence: ' + result['report_path'])


def _write_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise FileExistsError(f'Immutable replay evidence differs: {path}')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(encoded)


def _save_report(output, phase, result, *, category='trajectory'):
    # Same log bytes can fail for different live cursors; preserve both audits.
    token = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = Path(output) / category / f'phase{phase:02d}' / (token + '.json')
    _write_once(path, result)
    return dict(result, report_path=str(path), report_sha256=sha256(path))


def _context(controller, phase):
    if controller.branch != 'C' or type(phase) is not int or phase not in PHASES:
        raise ValueError('Trajectory audit only permits C replay phases 5..24')
    kind, epoch, updates, seed, count, digest = PHASES[phase]
    row = controller.trainer.plan[phase]
    if (row['kind'], row['epoch'], row['updates']) != (kind, epoch, updates):
        raise ValueError('Original phase plan differs from locked input audit')
    config = controller.trainer.config
    if config['seed'] != 2024 or config['seed'] + 100003 * (phase + 1) != seed:
        raise ValueError('Original phase seed differs')
    return row, dict(phase=phase, kind=kind, epoch=epoch, updates=updates, phase_seed=seed,
                    data_order_count=count, data_order_sha256_int64_le=digest)


def _order_digest(indices):
    array = np.asarray(indices)
    if array.ndim != 1 or array.dtype.kind not in 'iu' or (array < 0).any() or (array >= TRAIN_COUNT).any():
        raise ValueError('Data order must be in-range integral train-row indices')
    return len(array), hashlib.sha256(array.astype('<i8', copy=False).tobytes()).hexdigest()


def record_phase_order(controller, output, phase, indices):
    """Persist the ACTUAL array used by the runner, before its first update."""
    result = dict(format='recovery35_observed_order_v1', branch='C', phase=phase,
                  input_audit_sha256=INPUT_AUDIT_SHA256, status='requires_review')
    try:
        _, expected = _context(controller, phase)
        if controller.progress['phase'] != phase:
            raise ValueError('Observed data order belongs to another current phase')
        count, digest = _order_digest(indices)
        result.update(expected=expected, observed_count=count, observed_sha256_int64_le=digest)
        if (count, digest) != (expected['data_order_count'], expected['data_order_sha256_int64_le']):
            raise ValueError('Observed data order differs from locked input audit')
        result['status'] = 'passed'
        path = Path(output) / 'phase_order' / f'phase{phase:02d}.json'
        _write_once(path, result)
        return dict(result, report_path=str(path), report_sha256=sha256(path))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise ReplayTrajectoryError(_save_report(output, phase, result, category='order_failures')) from exc


def _read_phase(path, phase, *, candidate):
    seen, ordered, duplicates, physical = {}, [], 0, 0
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            def bad_constant(value): raise ValueError('Nonfinite JSON constant: ' + value)
            row = json.loads(line, parse_constant=bad_constant)
            if not isinstance(row, dict) or type(row.get('phase_index')) is not int:
                raise ValueError(f'Malformed training row at line {line_number}')
            if row['phase_index'] != phase:
                continue
            for key in INTEGER_FIELDS:
                if type(row.get(key)) is not int:
                    raise ValueError(f'Non-integral {key} at line {line_number}')
            if candidate and row.get('branch') != 'C':
                raise ValueError('Candidate training row is not branch C')
            key = (phase, row['r_updates'], row['t_updates'])
            scientific = {k: v for k, v in row.items() if k not in ('attempt', 'seconds')}
            physical += 1
            if key in seen:
                if scientific != seen[key]:
                    raise ValueError(f'Conflicting duplicate update at line {line_number}: {key}')
                duplicates += 1
            else:
                seen[key] = scientific
                ordered.append(row)
    return ordered, dict(physical_rows=physical, unique_updates=len(ordered), identical_duplicates=duplicates)


def audit_phase(controller, output, phase, *, reference_log=REFERENCE_LOG):
    """Check a completed/prepared phase, save diagnostics, raise on ANY mismatch."""
    candidate = Path(output) / 'training.jsonl'
    result = dict(format='recovery35_update_trajectory_audit_v1', branch='C', phase=phase,
                  status='requires_review', rule=RULE, input_audit_sha256=INPUT_AUDIT_SHA256,
                  reference_log=str(reference_log), reference_log_sha256=REFERENCE_LOG_SHA256)
    try:
        if sha256(reference_log) != REFERENCE_LOG_SHA256:
            raise ValueError('Reference training log bytes differ from authenticated original')
        result['candidate_log_sha256'] = sha256(candidate)
        result['candidate_log_bytes_at_audit'] = candidate.stat().st_size
        result['controller_counts'] = dict(r_updates=controller.trainer.r_updates, t_updates=controller.trainer.t_updates)
        result['progress'] = dict(controller.progress)
        row, expected = _context(controller, phase)
        if (controller.progress['phase'] != phase or not controller.progress['boundary_prepared']
                or controller.progress['update'] != row['updates']):
            raise ValueError('Audit requires the completed, prepared replay phase before advancing')
        order_path = Path(output) / 'phase_order' / f'phase{phase:02d}.json'
        observed = json.loads(order_path.read_text(encoding='utf-8'))
        derived_count, derived_sha = _order_digest(phase_indices(controller.trainer.config, TRAIN_COUNT, phase, row))
        if (observed.get('status') != 'passed' or observed.get('branch') != 'C'
                or observed.get('phase') != phase or observed.get('expected') != expected
                or observed.get('input_audit_sha256') != INPUT_AUDIT_SHA256
                or observed.get('observed_count') != derived_count
                or (observed.get('observed_count'), observed.get('observed_sha256_int64_le')) !=
                   (expected['data_order_count'], expected['data_order_sha256_int64_le'])
                or (derived_count, derived_sha) != (expected['data_order_count'], expected['data_order_sha256_int64_le'])):
            raise ValueError('Observed/derived phase order or pinned audit binding differs')
        result['phase_order'] = dict(path=str(order_path), sha256=sha256(order_path), **expected)
        old, old_counts = _read_phase(reference_log, phase, candidate=False)
        new, new_counts = _read_phase(candidate, phase, candidate=True)
        result.update(reference_counts=old_counts, candidate_counts=new_counts)
        if len(old) != row['updates'] or len(new) != row['updates']:
            raise ValueError('Phase has missing or extra unique updates')
        metrics = ['loss', 'recommendation', 'gradient_norm']
        if row['kind'] == 'T': metrics += ['reconstruction', 'balance', 'anchor']
        errors = {key: [] for key in metrics}; failed_updates, examples = set(), []
        for update, (left, right) in enumerate(zip(old, new), 1):
            if left['update'] != update or right['update'] != update:
                raise ValueError('Unique update order differs from original sequence')
            for key in INTEGER_FIELDS + ('phase',):
                if left[key] != right[key]:
                    raise ValueError(f'Exact clock/count mismatch at update {update}: {key}')
            if left['epoch'] != expected['epoch'] or left['phase'] != expected['kind']:
                raise ValueError('Reference row differs from authenticated phase metadata')
            for key in metrics:
                a, b = left.get(key), right.get(key)
                if (type(a) not in (int, float) or type(b) not in (int, float)
                        or not math.isfinite(a) or not math.isfinite(b)):
                    raise ValueError(f'Missing/nonfinite {key} at update {update}')
                delta = abs(b - a)
                atol, rtol = (1e-6, 1e-3) if key == 'gradient_norm' else (1e-6, 1e-4)
                bound = atol + rtol * abs(a)
                errors[key].append(delta)
                if delta > bound:
                    failed_updates.add(update)
                    if len(examples) < 20:
                        examples.append(dict(update=update, metric=key, reference=a, candidate=b,
                                             absolute_delta=delta, allowed_delta=bound))
        if (controller.trainer.r_updates, controller.trainer.t_updates) != (old[-1]['r_updates'], old[-1]['t_updates']):
            raise ValueError('Live controller counters differ from completed phase log')
        result.update(metrics={key: dict(compared=len(values), max_absolute_delta=max(values),
                        mean_absolute_delta=math.fsum(values) / len(values)) for key, values in errors.items()},
                      failed_updates=len(failed_updates), failure_examples=examples,
                      status='passed' if not failed_updates else 'requires_review')
        if sha256(reference_log) != REFERENCE_LOG_SHA256 or sha256(candidate) != result['candidate_log_sha256']:
            raise ValueError('Training log changed during trajectory audit')
        saved = _save_report(output, phase, result)
        if failed_updates:
            raise ReplayTrajectoryError(saved)
        return saved
    except ReplayTrajectoryError:
        raise
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise ReplayTrajectoryError(_save_report(output, phase, result)) from exc
