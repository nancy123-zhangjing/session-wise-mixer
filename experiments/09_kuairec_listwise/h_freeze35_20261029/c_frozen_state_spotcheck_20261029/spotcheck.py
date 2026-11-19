"""Bounded, stdout-only CPU inspection of three immutable C checkpoints."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

signal.alarm(90)
started = time.monotonic()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert len(EXECUTED_SOURCE_SHA256) == 64
import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_r1')
RUN = ROOT / 'runs/H0-Freeze20-R35'
SPECS = {
    20: ('R20_pre_freeze.pt', 25, 13860),
    21: ('phase26_R_R21_T200.pt', 27, 14553),
    22: ('phase27_R_R22_T200.pt', 28, 15246),
}
R20_SHA = '686388225489d3d264fd1a25edd0bf69ba2171358b2845f54aed9fa7a5775e2b'
R20_RECEIPT_SHA = 'f5a84849fba80130ae22ba33b1957124d01955f42657291f461d38bf31eb9763'
GATE_SHA = '04740499c307f251ab0ea4986fb42a4b1d1f5083d45524e24637c5e89bca233c'
SOURCE_SHAS = {
    'recovery.py': '979edd8b1aa8a21fa15c08fc6c3d7af3b98afd2d3caecb4799360a7b21efc56e',
    'recovery35.py': '500d272dc1f7bbc9e2eb920433ded137c80359e1dd4ac5d63283738ac65256a7',
    'training.py': '2da7cc318fc35c576a5144489c5ea2db43e4e534a3664c041ebcd5bf141cddd8',
    'model.py': '55badb671bec5f71cc355f3bb6cc1076f2f88ed0819c61546f042896b3676855',
}
inputs = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path, expected=None):
    path = path.resolve()
    assert path.is_relative_to(ROOT) and path.name != 'last.pt'
    digest = sha(path)
    assert expected is None or digest == expected, f'SHA mismatch: {path.name}'
    assert str(path) not in inputs or inputs[str(path)] == digest
    inputs[str(path)] = digest
    return digest


for name, expected in SOURCE_SHAS.items():
    bind(ROOT / 'code/scripts/listwise_diffgrm' / name, expected)
fp_path = ROOT / 'code/scripts/listwise_diffgrm/recovery.py'
source = fp_path.read_text(encoding='utf-8')
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
namespace = dict(hashlib=hashlib, torch=torch, np=np)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(fp_path), 'exec'), namespace)
fingerprint = namespace['fingerprint']


def exact_tensor(a, b):
    assert a.device.type == b.device.type == 'cpu'
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def exact_tree(a, b):
    counts = dict(tensors=0, scalars=0, containers=0)
    differing = []

    def visit(x, y, path):
        if torch.is_tensor(x) or torch.is_tensor(y):
            counts['tensors'] += 1
            if not (torch.is_tensor(x) and torch.is_tensor(y) and exact_tensor(x, y)):
                differing.append(path)
        elif type(x) is not type(y):
            differing.append(path)
        elif isinstance(x, dict):
            counts['containers'] += 1
            if set(x) != set(y):
                differing.append(path + ':keys')
            else:
                for k in x:
                    visit(x[k], y[k], path + '/' + str(k))
        elif isinstance(x, (list, tuple)):
            counts['containers'] += 1
            if len(x) != len(y):
                differing.append(path + ':length')
            else:
                for i, (xx, yy) in enumerate(zip(x, y)):
                    visit(xx, yy, path + '/' + str(i))
        else:
            counts['scalars'] += 1
            if x != y:
                differing.append(path)
    visit(a, b, '')
    return dict(equal=not differing, counts=counts, differing_paths=differing)


def tensor_record(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    return dict(shape=list(a.shape), dtype=str(a.dtype), numel=a.numel(),
                exact_bytes_equal=exact_tensor(a, b), r20_fingerprint=fingerprint(a), later_fingerprint=fingerprint(b))


gate_path = RUN / 'replay_gate.json'
bind(gate_path, GATE_SHA)
gate = json.loads(gate_path.read_text(encoding='utf-8'))
assert gate['status'] == 'passed' and gate['branch'] == 'C'
states, receipts, checkpoint_evidence = {}, {}, {}
for epoch, (name, phase, updates) in SPECS.items():
    receipt_path = RUN / 'checkpoint_receipts' / (name + '.json')
    receipt_sha = bind(receipt_path, R20_RECEIPT_SHA if epoch == 20 else None)
    receipt = receipts[epoch] = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['path'] == name and receipt['r_updates'] == updates and receipt['t_updates'] == 200
    assert receipt['progress'] == dict(phase=phase, update=0, boundary_prepared=False)
    assert all(receipt['fields'][k] is True for k in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'binding'))
    path = RUN / name
    assert 0 < path.stat().st_size == receipt['bytes'] < 100 * 1024 * 1024
    digest = bind(path, receipt['sha256'])
    if epoch == 20:
        assert digest == R20_SHA
    envelope = states[epoch] = torch.load(path, map_location='cpu', weights_only=False)
    assert fingerprint(envelope) == receipt['state_sha256']
    assert envelope['format'] == 'stage09_recovery35_v1' and envelope['protocol']['branch'] == 'C'
    assert envelope['inspection_only'] is False and not envelope.get('preflight_only')
    inner = envelope['inner']
    assert inner['format'] == 'kuairec_listwise_v1' and inner['arm'] == 'H0-Freeze20-R35'
    assert not inner.get('preflight_only') and inner['progress'] == receipt['progress']
    assert (inner['r_updates'], inner['t_updates']) == (updates, 200)
    for file, source_sha in SOURCE_SHAS.items():
        assert inner['binding']['code']['scripts/listwise_diffgrm/' + file] == source_sha
    assert int(inner['alignment']['diger_optimizer_step']) == 200
    assert int(inner['alignment']['diger_total_optimizer_steps']) == 400
    checkpoint_evidence[str(epoch)] = dict(path=str(path), sha256=digest, bytes=receipt['bytes'],
        receipt_path=str(receipt_path), receipt_sha256=receipt_sha, state_fingerprint=receipt['state_sha256'],
        progress=inner['progress'], r_updates=updates, t_updates=200,
        r_optimizer_steps=sorted({float(v['step']) for v in inner['r_optimizer']['state'].values()}),
        t_optimizer_steps=sorted({float(v['step']) for v in inner['t_optimizer']['state'].values()}))
    assert checkpoint_evidence[str(epoch)]['r_optimizer_steps'] == [float(updates)]
    assert checkpoint_evidence[str(epoch)]['t_optimizer_steps'] == [200.0]

base = states[20]['inner']
assert states[20]['replay_gate'] is None
assert gate['checkpoint_state_sha256'] == fingerprint(base)
comparisons = {}
for epoch in (21, 22):
    envelope, later = states[epoch], states[epoch]['inner']
    assert envelope['protocol'] == states[20]['protocol'] and later['binding'] == base['binding']
    assert envelope['replay_gate'] == gate and envelope['replay_gate_state_sha256'] == fingerprint(base)
    assert set(base['alignment']) == set(later['alignment']) and set(base['model']) == set(later['model'])
    alignment = {k: tensor_record(base['alignment'][k], later['alignment'][k]) for k in base['alignment']}
    assert all(v['exact_bytes_equal'] for v in alignment.values())
    optimizer = exact_tree(base['t_optimizer'], later['t_optimizer'])
    assert optimizer['equal']
    anchors = {}
    for key in ('alignment', 't_optimizer'):
        original, actual = fingerprint(base[key]), fingerprint(later[key])
        assert original == actual == envelope['freeze_hashes'][key] == states[20]['freeze_hashes'][key]
        anchors[key] = dict(r20=original, later=actual, r20_saved_freeze=states[20]['freeze_hashes'][key], later_saved_freeze=envelope['freeze_hashes'][key], equal=True)
    model = {k: tensor_record(base['model'][k], later['model'][k]) for k in base['model']}
    changed = [k for k, v in model.items() if not v['exact_bytes_equal']]
    assert changed and not model['backbone.embedding.weight']['exact_bytes_equal']
    sid_changes = int((base['alignment']['hard_codes'][1:] != later['alignment']['hard_codes'][1:]).any(dim=1).sum())
    assert sid_changes == 0
    comparisons[str(epoch)] = dict(alignment_tensor_count=len(alignment), alignment_all_exact_bytes_equal=True,
        alignment_tensors=alignment, t_optimizer=optimizer, freeze_fingerprints=anchors,
        hard_codes_shape=list(base['alignment']['hard_codes'].shape), changed_catalog_sid_items=sid_changes,
        r_updates_delta=later['r_updates'] - base['r_updates'], t_updates_delta=0,
        diger_optimizer_step=200, diger_total_optimizer_steps=400,
        model_state_tensor_count=len(model), changed_model_tensor_count=len(changed), changed_model_keys=changed,
        model_tensors=model, recommender_sid_embedding_key='backbone.embedding.weight',
        recommender_sid_embedding_changed=True, r_optimizer_changed=fingerprint(base['r_optimizer']) != fingerprint(later['r_optimizer']))

assert all(sha(Path(path)) == digest for path, digest in inputs.items())
assert not torch.cuda.is_initialized()
result = dict(format='freeze35_c_frozen_state_spotcheck_v1', status='passed',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), seconds=time.monotonic() - started,
    executed_source_sha256=EXECUTED_SOURCE_SHA256, root=str(ROOT), checkpoints=checkpoint_evidence,
    comparisons_to_replayed_R20_pre_freeze=comparisons, input_file_sha256_before=inputs,
    all_input_file_sha256_after_equal=True, source_fingerprint_method='Bound frozen recovery.py FunctionDef AST; no training/controller imports',
    cpu_threads=torch.get_num_threads(), python=sys.version, torch=torch.__version__, numpy=np.__version__,
    constraints=dict(stdout_only=True, remote_files_written=False, last_checkpoint_read=False,
        model_instantiated=False, load_state_dict_called=False, optimizer_restored=False,
        forward_or_inference=False, gpu_initialized=False, training=False),
    limits=['Three immutable saved-state endpoints only, not an independent replay of every intervening optimizer update.',
        'C R20 is the gate-accepted replay state, not an unavailable historical H0 R20 checkpoint.',
        'This is raw CPU state comparison, not real model restore or final delivery acceptance.',
        'Recommender backbone.embedding.weight remains trainable; it is distinct from the frozen Tokenizer base codebooks.',
        'Persistent=False initial_codebooks and transient recommender effective-codebook cache are not checkpoint tensors; no new encoding/forward was run.'])
signal.alarm(0)
print('C_FROZEN_SPOTCHECK_RESULT=' + json.dumps(result, ensure_ascii=False, allow_nan=False))
