"""Immutable boundary checkpoints and pre-registered H0 replay checks."""
from __future__ import annotations

import gzip
import itertools
import json
import math
from pathlib import Path

import torch

from .recovery import fingerprint
from .recovery_evaluation import write_json
from .training import atomic_save, sha256


REPLAY_RULE = dict(format='freeze35_replay_gate_v1', examples=71657,
                   max_ndcg5_absolute_delta=1e-6, minimum_top_list_agreement=.999,
                   identity_agreement=1.0, seed=2024,
                   interpretation='engineering drift alarm, not equivalence or old-weight recovery')
VALIDATION_EPOCHS = {'A': (29, 32, 35), 'B': (35,),
                     'C': (8, 11, 14, 17, 20, 23, 26, 29, 32, 35)}
BEST_EPOCHS = (20, 23, 26, 29, 32, 35)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise FileExistsError(f'Immutable JSON already differs: {path}')
    else:
        write_json(path, value)


def save_boundary(controller, output, name):
    """Existing snapshots must match by value; never overwrite old weights."""
    root = Path(output)
    path = root / name
    if path.parent != root or not name.endswith('.pt'):
        raise ValueError('Checkpoint name must be a flat .pt file')
    state = controller.checkpoint()
    value_hash = fingerprint(state)
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        if fingerprint(existing) != value_hash:
            raise FileExistsError(f'Immutable checkpoint differs: {path}')
    else:
        atomic_save(path, state)
    inner = state['inner']
    receipt = dict(path=name, sha256=sha256(path), bytes=path.stat().st_size,
                   state_sha256=value_hash, r_updates=inner['r_updates'],
                   t_updates=inner['t_updates'], progress=inner['progress'],
                   fields={k: k in inner for k in ('model', 'alignment', 'r_optimizer',
                                                  't_optimizer', 'scheduler', 'rng', 'binding')})
    if not all(receipt['fields'].values()):
        raise ValueError('Incomplete checkpoint state')
    write_once(root / 'checkpoint_receipts' / (name + '.json'), receipt)
    return receipt


def reference_record(manifest, branch, epoch, *, split='validation', local=False):
    record = dict(manifest[split][branch][str(epoch)])
    if local:
        record.update(record.get('local', {}))
    for kind in ('summary', 'predictions'):
        path = Path(record[kind])
        if not path.is_file() or sha256(path) != record[kind + '_sha256']:
            raise ValueError(f'Changed reference {branch}/{epoch}/{kind}')
    summary = json.loads(Path(record['summary']).read_text(encoding='utf-8'))
    if (summary['split'] != split or not summary['full_split'] or summary['beam'] != 32
            or summary['inference_precision'] != 'fp32' or summary['seed'] != 2024
            or summary['architecture'] != 'H'
            or summary['examples'] != (71657 if split == 'validation' else 53194)):
        raise ValueError('Reference protocol mismatch')
    if record.get('checkpoint'):
        if sha256(record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Reference checkpoint changed')
    return record, summary


def compare_replay_predictions(reference, candidate, *, rule=REPLAY_RULE):
    """Recompute float64 NDCG from IDs; require exact identity/order first."""
    if rule != REPLAY_RULE:
        raise ValueError('Pre-registered replay rule changed')
    n, same = 0, 0
    reference_values, candidate_values = [], []
    ideal = sum(1 / math.log2(i + 2) for i in range(5))
    sentinel = object()
    with gzip.open(reference, 'rt', encoding='utf-8') as left, gzip.open(candidate, 'rt', encoding='utf-8') as right:
        for a, b in itertools.zip_longest(left, right, fillvalue=sentinel):
            if a is sentinel or b is sentinel:
                raise ValueError('Replay evaluation sample counts differ')
            a, b = json.loads(a), json.loads(b)
            for key in ('sample_id', 'dataset_index', 'user_id', 'targets', 'cold_target'):
                if a[key] != b[key]:
                    raise ValueError(f'Replay sample identity mismatch at {n}: {key}')
            if a['dataset_index'] != n or len(set(a['targets'])) != 5:
                raise ValueError('Replay evaluation is not the full ordered five-target split')
            values = []
            for row in (a, b):
                top = row['predictions'][0]
                if len(top) != 5:
                    raise ValueError('Top list is not five items')
                seen, dcg = set(), 0.0
                for position, item in enumerate(top):
                    if item > 0 and item not in seen and item in row['targets']:
                        dcg += 1 / math.log2(position + 2)
                    seen.add(item)
                values.append(dcg / ideal)
            reference_values.append(values[0])
            candidate_values.append(values[1])
            same += a['predictions'][0] == b['predictions'][0]
            n += 1
    if n != rule['examples']:
        raise ValueError('Replay validation population changed')
    old, new = math.fsum(reference_values) / n, math.fsum(candidate_values) / n
    passed = abs(new - old) <= rule['max_ndcg5_absolute_delta'] and same / n >= rule['minimum_top_list_agreement']
    return dict(status='passed' if passed else 'requires_review', rule=rule,
                reference_prediction_sha256=sha256(reference), candidate_prediction_sha256=sha256(candidate),
                examples=n, identical_top_lists=same, changed_top_lists=n - same,
                top_list_agreement=same / n, reference_ndcg5=old, candidate_ndcg5=new,
                ndcg5_absolute_delta=abs(new - old), exact_prediction_replay=(same == n))


def select_best(records):
    """Records already authenticated against the common validation grid."""
    if set(map(int, records)) != set(BEST_EPOCHS):
        raise ValueError('Incomplete shared best-selection grid')
    epoch = min(BEST_EPOCHS, key=lambda e: (-records[str(e)]['ndcg5'], e))
    return dict(records[str(epoch)], epoch=epoch)


def audit_checkpoints(controller, output):
    root = Path(output)
    expected = ['initial.pt']
    for phase in range(controller.start_phase, controller.stop_phase):
        row = controller.trainer.plan[phase]
        # Ask the controller for the actual post-phase clocks, including skipped T.
        clocks = controller.counts_at(dict(phase=phase + 1, update=0, boundary_prepared=False))
        expected.append(f'phase{phase:02d}_{row["kind"]}_R{row["epoch"]:02d}_T{clocks["T"]:03d}.pt')
    if controller.branch == 'C':
        expected.append('R20_pre_freeze.pt')
    records = {}
    for name in expected:
        receipt = json.loads((root / 'checkpoint_receipts' / (name + '.json')).read_text(encoding='utf-8'))
        path = root / name
        if sha256(path) != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
            raise ValueError(f'Missing/changed checkpoint: {name}')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if fingerprint(saved) != receipt['state_sha256']:
            raise ValueError(f'Checkpoint content fingerprint mismatch: {name}')
        controller.validate_clocks(saved['inner'])
        records[name] = receipt
    return dict(status='passed', expected=len(expected), present=len(records), records=records,
                actual_model_restore='separate required acceptance; hashes and clocks are not restore proof')
