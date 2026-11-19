"""Unchanged CPU payload validators from the prior Stage07 verified backup.
Only selected functions are carried forward; no old campaign-specific main.
"""
from __future__ import annotations
import hashlib, json, math, random, re
from pathlib import Path
BOUNDARY = 'diffgrm_joint_phase_boundary_v1'
ORIGINAL_VALIDATOR_SHA256 = 'e653c2c89c5dc0288f68cb0292677490e57bbe22ac1302ff323f072fa0efde5c'

def need(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): digest.update(block)
    return digest.hexdigest()

def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key); result[key] = value
    return result

def read(path):
    def invalid(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)

def count(value, label):
    need(type(value) is int and value >= 0, 'Invalid nonnegative counter: ' + label)
    return value

def tensor_digest(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def tensor_inventory(value, label):
    import torch
    tensors = elements = 0
    if torch.is_tensor(value):
        need(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'Nonfinite/non-CPU tensor: ' + label)
        return 1, value.numel()
    if isinstance(value, dict):
        children = ((label + '/' + str(key), item) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        children = ((label + '/' + str(index), item) for index, item in enumerate(value))
    else:
        if isinstance(value, float): need(math.isfinite(value), 'Nonfinite scalar: ' + label)
        return 0, 0
    for name, item in children:
        n, size = tensor_inventory(item, name); tensors += n; elements += size
    return tensors, elements

def inspect_boundary(payload, best, *, terminal):
    import numpy as np
    import torch
    fields = {'format','model','optimizer','scheduler','scaler','rng','progress','joint_runtime',
              'configuration_fingerprint','best_checkpoint_bytes','best_checkpoint_sha256','diagnostic_offsets'}
    need(isinstance(payload, dict) and fields <= payload.keys() and payload['format'] == BOUNDARY, 'Incomplete joint boundary')
    need(re.fullmatch('[0-9a-f]{64}', payload['configuration_fingerprint']), 'Malformed configuration fingerprint')
    state, p, runtime = payload['model'], payload['progress'], payload['joint_runtime']
    need(isinstance(state, dict) and state and all(torch.is_tensor(value) for value in state.values()), 'Invalid model state dictionary')
    progress_keys = {'next_epoch','best_epoch','best_val_score','eval_count','no_improve_count','cumulative_optimizer_updates',
        'decoder_epochs_completed','fit_elapsed_seconds','training_complete','stop_reason','epoch_timing_records','output_paths','joint_progress'}
    need(progress_keys <= p.keys() and type(p['training_complete']) is bool and (not terminal or p['training_complete']), 'Missing/nonterminal committed progress')
    for key in ('next_epoch','best_epoch','eval_count','no_improve_count','cumulative_optimizer_updates','decoder_epochs_completed'): count(p[key], key)
    need(p['next_epoch'] >= p['best_epoch'] and p['next_epoch'] == len(p['epoch_timing_records'])
         and math.isfinite(p['best_val_score']) and math.isfinite(p['fit_elapsed_seconds']) and p['fit_elapsed_seconds'] >= 0, 'Malformed epoch/timing progress')
    jp = p['joint_progress']
    need({'boundary_phase','last_committed_phase','next_phase','next_phase_index','tokenizer_epochs_completed',
        'tokenizer_optimizer_updates','sid_refresh_count','last_refresh_stats','initial_validation_complete'} <= jp.keys(), 'Incomplete alternating progress')
    for key in ('tokenizer_epochs_completed','tokenizer_optimizer_updates','sid_refresh_count'): count(jp[key], key)
    need(jp['boundary_phase'] in ('recommender','tokenizer') and jp['next_phase'] in ('recommender','tokenizer')
         and jp['last_committed_phase'] == jp['boundary_phase'] and jp['initial_validation_complete'] is True, 'Malformed phase boundary')
    runtime_keys = {'version','phase','optimizer_groups','initial_codebooks','recommender_codebook_cache','requires_grad',
        'module_training','alignment_settings','diger_schedule','hard_codes_sha256','generate_w_decoding_graph','mapping_restore'}
    need(runtime_keys <= runtime.keys() and runtime['version'] == 1 and runtime['phase'] == jp['boundary_phase']
         and runtime['requires_grad'] and runtime['module_training'] and all(type(v) is bool for v in (*runtime['requires_grad'].values(), *runtime['module_training'].values())), 'Incomplete joint runtime')
    need('joint_alignment.hard_codes' in state and tensor_digest(state['joint_alignment.hard_codes'][1:]) == runtime['hard_codes_sha256'], 'Committed SID/runtime mismatch')
    need(int(state['joint_alignment.diger_optimizer_step']) == runtime['diger_schedule']['optimizer_step'] == jp['tokenizer_optimizer_updates']
         and int(state['joint_alignment.diger_total_optimizer_steps']) == runtime['diger_schedule']['total_optimizer_steps'], 'DIGER runtime clock mismatch')
    optimizer, scheduler = payload['optimizer'], payload['scheduler']
    groups, layout = optimizer['param_groups'], runtime['optimizer_groups']
    need(len(groups) == len(layout) == 2 and [row['group_name'] for row in layout] == ['recommender','joint_codebook'], 'Wrong alternating optimizer groups')
    steps = {'recommender':p['cumulative_optimizer_updates'], 'joint_codebook':jp['tokenizer_optimizer_updates']}
    identifiers, covered = [], []
    for group, definition in zip(groups, layout):
        names, ids = definition['parameters'], group['params']; identifiers.extend(ids)
        need(group['group_name'] == definition['group_name'] and len(names) == len(ids), 'Optimizer named layout mismatch')
        for name, identifier in zip(names, ids):
            entry = optimizer['state'].get(identifier); updates = steps[group['group_name']]
            if name.startswith('joint_sequence_adapter.') or updates == 0:
                need(entry is None, 'Inactive optimizer parameter has state: ' + name); continue
            need(name in state and entry is not None and float(entry['step']) == updates, 'Missing/wrong Adam state clock: ' + name)
            need(all(torch.is_tensor(entry[key]) and entry[key].shape == state[name].shape for key in ('exp_avg','exp_avg_sq')), 'Adam moment shape mismatch: ' + name)
            covered.append(name)
    need(len(identifiers) == len(set(identifiers)) and set(optimizer['state']) <= set(identifiers)
         and len(covered) == len(optimizer['state']) > 0, 'Duplicate/foreign/incomplete Adam state')
    need(scheduler['last_epoch'] == steps['recommender'] and len(scheduler['_last_lr']) == len(groups), 'Scheduler/recommender clock mismatch')
    rng = payload['rng']
    need({'python','numpy','torch_cpu','torch_cuda_all','loader_counters','loader_generators'} <= rng.keys(), 'Incomplete RNG state')
    random.Random().setstate(rng['python']); np.random.RandomState().set_state(rng['numpy'])
    need(isinstance(rng['torch_cuda_all'], list) and len(rng['torch_cuda_all']) == 1, 'Missing single-H100 CUDA RNG bytes')
    for value in [rng['torch_cpu'], *rng['torch_cuda_all'], *rng['loader_generators'].values()]:
        need(torch.is_tensor(value) and value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0, 'Malformed serialized RNG bytes')
    need(all(type(value) is int for value in rng['loader_counters'].values()), 'Malformed loader counters')
    need(len(payload['diagnostic_offsets']) == 5 and all(type(v) is int and v >= 0 for v in payload['diagnostic_offsets'].values()), 'Incomplete diagnostic offsets')
    embedded = payload['best_checkpoint_bytes']
    need(isinstance(embedded, bytes) and hashlib.sha256(embedded).hexdigest() == payload['best_checkpoint_sha256'] == sha(best), 'Embedded best/disk SHA mismatch')
    return {'next_epoch':p['next_epoch'], 'best_epoch':p['best_epoch'], 'training_complete':p['training_complete'],
        'recommender_optimizer_updates':steps['recommender'], 'tokenizer_optimizer_updates':steps['joint_codebook'],
        'optimizer_state_count':len(optimizer['state']), 'sid_refresh_count':jp['sid_refresh_count'],
        'phase':runtime['phase'], 'embedded_best_sha256':payload['best_checkpoint_sha256'], 'scheduler_last_epoch':scheduler['last_epoch']}
