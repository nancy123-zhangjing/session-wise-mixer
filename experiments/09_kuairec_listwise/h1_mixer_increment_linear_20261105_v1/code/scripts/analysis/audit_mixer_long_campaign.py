"""Independent CPU audit/export of the stage05 long four-arm study.

Reads fixed-protocol records and optionally complete checkpoints. It never
trains, tests a model, allocates a GPU, or declares the Goal delivered.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
CAMPAIGN = 'goal_20261020_long_mixer_v1'
CODE_COMMIT = '69682698c15ba796b98e6329c950061a2f68a05b'
ENGINE_COMMIT = '1c879252f6395d6e64462511823c3929b9c42c00'
ARMS = ('no_mixer_frozen', 'no_mixer_joint', 'mixer_frozen', 'mixer_joint')
METRICS = ('hit_rate@5', 'hit_rate@10', 'ndcg@5', 'ndcg@10', 'weighted_score')
VAL_EPOCHS = (0, 5, 10, 15, 20, 25, 30, 35)
INITIAL_SID_SHA256 = 'ee955d3b79e69e4c9c96fce6b7f4f8f46aad17eeedd0f55d04298078b24967ea'
ITEMS = 12101


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def number(value):
    need(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value), 'Invalid finite observation')
    return float(value)


def count(value, expected, name):
    need(type(value) is int and value == expected, 'Wrong ' + name)


def same(values):
    return len({json.dumps(value, sort_keys=True, allow_nan=False) for value in values}) <= 1


def within(root, relative):
    root = Path(root).resolve()
    name = Path(relative)
    path = (root / name).resolve()
    need(not name.is_absolute() and '..' not in name.parts and path.is_relative_to(root), 'Path escaped owned root')
    return path


def metric_values(value):
    result = {key: number(value[key]) for key in METRICS}
    need(all(0 <= v <= 1 for v in result.values()), 'Out-of-range recommendation metric')
    need(abs(result['weighted_score'] - .8 * result['ndcg@10'] - .2 * result['hit_rate@10']) <= 1e-7,
         'Weighted score is not .8*NDCG10 + .2*HR10')
    for k in (5, 10):
        if 'recall@' + str(k) in value:
            need(abs(number(value['recall@' + str(k)]) - result['hit_rate@' + str(k)]) <= 1e-12, 'HR/recall alias differs')
    return result


def lr_factor(step):
    if step < 500:
        return max(step, 1) / 500
    return .5 * (1 + math.cos(math.pi * (step - 500) / 3140))


def plan(arm):
    need(arm in ARMS, 'Unknown experimental arm')
    result = [('recommender', epoch, 1652, 104) for epoch in range(1, 6)]
    for cycle in range(10):
        if arm.endswith('_joint'):
            result.append(('tokenizer', cycle, 96, 6))
        result.extend(('recommender', 6 + cycle * 3 + offset, 1652, 104) for offset in range(3))
    return result


def bounded_integer(value, low, high, name):
    need(type(value) is int and low <= value <= high, 'Invalid ' + name)
    return value


def sid_observation(stats, changes):
    """Check every recorded SID field without trusting a self-consistent rate alone."""
    need(set(stats['prefixes']) == {'1', '2', '3', '4'} and len(stats['positions']) == 4,
         'Wrong SID layout')
    collisions = [stats['prefixes'][str(i)] for i in range(1, 5)]
    need(stats['full_sid'] == collisions[-1], 'Full SID and four-digit prefix statistics differ')
    for value in collisions:
        unique = bounded_integer(value['unique_semantic_ids'], 1, ITEMS, 'unique SID count')
        collided = bounded_integer(value['collided_items'], 0, ITEMS, 'collided-item count')
        groups = bounded_integer(value['collision_groups'], 0, min(unique, ITEMS // 2), 'collision-group count')
        largest = bounded_integer(value['largest_group'], 1, ITEMS, 'largest SID group')
        need(unique - groups + collided == ITEMS and abs(number(value['collision_rate']) - collided / ITEMS) < 1e-12,
             'SID collision counts/rate do not cover the catalogue')
        need((groups == 0 and collided == 0 and largest == 1)
             or (groups > 0 and 2 * groups <= collided and math.ceil(collided / groups) <= largest
                 <= collided - 2 * (groups - 1)), 'Impossible SID collision groups')
    for previous, following in zip(collisions, collisions[1:]):
        need(previous['unique_semantic_ids'] <= following['unique_semantic_ids']
             and previous['collided_items'] >= following['collided_items']
             and previous['largest_group'] >= following['largest_group'], 'SID prefix statistics are inconsistent')
    for index, value in enumerate(stats['positions']):
        count(value['position'], index, 'SID position')
        count(value['codebook_size'], 256, 'SID codebook size')
        used = bounded_integer(value['used_codes'], 1, 256, 'used codeword count')
        need(abs(number(value['utilization_rate']) - used / 256) < 1e-12, 'Codebook utilization differs')
    need(stats['positions'][0]['used_codes'] == collisions[0]['unique_semantic_ids'], 'First-position/prefix count differs')
    items = bounded_integer(changes['changed_items'], 0, ITEMS, 'changed-SID item count')
    digits = bounded_integer(changes['changed_digits'], 0, ITEMS * 4, 'changed-SID digit count')
    need(items <= digits <= items * 4
         and abs(number(changes['item_change_rate']) - items / ITEMS) < 1e-12
         and abs(number(changes['digit_change_rate']) - digits / (ITEMS * 4)) < 1e-12,
         'SID change counts/rates differ')
    rates = [number(value) for value in changes['per_position_change_rate']]
    need(len(rates) == 4 and all(0 <= value <= 1 and abs(value * ITEMS - round(value * ITEMS)) < 1e-7 for value in rates)
         and abs(sum(rates) - digits / ITEMS) < 1e-12 and max(rates) <= items / ITEMS + 1e-12,
         'Per-position SID changes differ')


def sid_from_codes(codes, initial_codes):
    """Independent catalogue recomputation from the actual saved integer tensors."""
    import numpy as np
    codes, initial_codes = np.asarray(codes), np.asarray(initial_codes)
    need(codes.shape == initial_codes.shape == (ITEMS, 4)
         and np.issubdtype(codes.dtype, np.integer) and np.issubdtype(initial_codes.dtype, np.integer)
         and bool(((codes >= 0) & (codes < 256)).all())
         and bool(((initial_codes >= 0) & (initial_codes < 256)).all()), 'Wrong actual checkpoint SID catalogue')
    def collision(values):
        _, sizes = np.unique(values, axis=0, return_counts=True)
        collided = int(sizes[sizes > 1].sum())
        return dict(unique_semantic_ids=int(len(sizes)), collided_items=collided, collision_rate=collided / ITEMS,
                    collision_groups=int((sizes > 1).sum()), largest_group=int(sizes.max()))
    stats = dict(full_sid=collision(codes), prefixes={str(i): collision(codes[:, :i]) for i in range(1, 5)},
                 positions=[dict(position=i,codebook_size=256,used_codes=int(len(np.unique(codes[:, i]))),
                                 utilization_rate=len(np.unique(codes[:, i])) / 256) for i in range(4)])
    changed = codes != initial_codes
    changes = dict(changed_items=int(changed.any(-1).sum()),item_change_rate=float(changed.any(-1).mean()),
                   changed_digits=int(changed.sum()),digit_change_rate=float(changed.mean()),
                   per_position_change_rate=changed.mean(0).tolist())
    sid_observation(stats, changes)
    return dict(SID_sha256=hashlib.sha256(codes.copy(order='C').tobytes()).hexdigest(),statistics=stats,changes=changes)


def selected_progress(progress):
    """The best checkpoint must contain the exact committed prefix, including epoch zero."""
    epoch = progress['best_decoder_epoch']
    rows = [r for r in progress['training'] if (r['logical_index'] <= epoch if r['phase'] == 'recommender'
            else 5 + 3 * r['logical_index'] < epoch)]
    updates = sum(r['optimizer_updates'] for r in rows if r['phase'] == 'tokenizer')
    return dict(progress,training=rows,validation=[v for v in progress['validation'] if v['decoder_epoch'] <= epoch],
                next_stage=len(rows),decoder_epochs=epoch,decoder_updates=epoch * 104,
                tokenizer_updates=updates,noisy_calls=updates * 32)


def audit_progress(p, arm, *, complete):
    rows, expected = p['training'], plan(arm)
    need(0 < len(rows) <= len(expected), 'No committed full phase or excess phase count')
    count(p['next_stage'], len(rows), 'next-stage cursor')
    if complete:
        count(len(rows), len(expected), 'complete phase budget (short smoke is not a long run)')
    r_count = t_updates = noisy = 0
    previous_sid = None
    for row, (kind, index, batches, updates) in zip(rows, expected):
        need(row['phase'] == kind, 'Wrong alternating phase order')
        count(row['logical_index'], index, 'logical phase index')
        count(row['batches'], batches, 'full microbatch count')
        count(row['optimizer_updates'], updates, 'actual phase optimizer updates')
        count(row['accumulation'], 16, 'gradient accumulation')
        count(row['final_window_batches'], 4 if kind == 'recommender' else 16, 'partial-window normalization')
        need(row['frozen_side_unchanged'] is True, 'Inactive parameter side changed')
        audit = row['parameter_audit']
        need(audit['all_expected_gradients_finite'] is True and audit['all_expected_Adam_states_present'] is True
             and audit['expected_parameter_count'] > 0 and audit['actual_changed_parameters'], 'Missing real gradient/update/Adam evidence')
        inactive = 'tokenizer' if kind == 'recommender' else 'recommender'
        need(number(row['last_gradient_sides'][inactive]) == 0 and number(row['last_gradient_sides'][kind]) > 0,
             'Invalid gradient routing')
        need(row['loss_means'] and all(math.isfinite(number(v)) for v in row['loss_means'].values()), 'Missing/nonfinite loss')
        stats, changes = row['formal_SID_statistics'], row['formal_SID_change_since_initial']
        if arm.endswith('_frozen'):
            need(changes['changed_items'] == 0, 'Frozen catalogue SID changed')
        sid_observation(stats, changes)
        sid_sha = row['formal_SID_sha256']
        need(isinstance(sid_sha, str) and len(sid_sha) == 64 and all(c in '0123456789abcdef' for c in sid_sha),
             'Invalid recorded SID digest')
        if kind == 'recommender':
            need(sid_sha == (previous_sid[0] if previous_sid else INITIAL_SID_SHA256), 'Recommender phase changed the SID digest')
            if previous_sid:
                need((stats, changes) == previous_sid[1:], 'Recommender phase changed the SID observations')
        previous_sid = (sid_sha, stats, changes)
        if kind == 'recommender':
            r_count += 1
            need(abs(number(row['lr_start']) - .001 * lr_factor((index - 1) * 104)) < 1e-12
                 and abs(number(row['lr_end']) - .001 * lr_factor(index * 104)) < 1e-12, 'Recommender LR clock drift/reset')
            need(all(row[k] == 0 for k in ('gumbel_calls', 'noise_applied_calls', 'assignment_training_enabled_calls')),
                 'Recommender was assigned stochastic Tokenizer noise')
        else:
            t_updates += 6
            noisy += 192
            for key in ('gumbel_calls', 'noise_applied_calls', 'assignment_training_enabled_calls'):
                count(row[key], 192, 'actual stochastic assignment calls')
            for state, step in ((row['tokenizer_schedule_start'], index * 6), (row['tokenizer_schedule_end'], index * 6 + 6)):
                count(state['optimizer_step'], step, 'cumulative Tokenizer step')
                count(state['total_optimizer_steps'], 60, 'cumulative annealing horizon')
                expected_noise = .2 + .4 * (1 + math.cos(math.pi * step / 60))
                need(state['softmax_temperature'] == 1 and abs(number(state['gumbel_noise_scale']) - expected_noise) < 1e-12,
                     'Gumbel schedule reset/sharpened in a cycle')
            need(row['lr_start'] == row['lr_end'] == 1e-5, 'Base codebook learning rate differs')
            expected_mean_noise = sum(.2 + .4 * (1 + math.cos(math.pi * step / 60)) for step in range(index * 6,index * 6 + 6)) / 6
            need(abs(number(row['loss_means']['noise_scale']) - expected_mean_noise) < 1e-7
                 and row['loss_means']['softmax_temperature'] == 1, 'Actual forward noise/temperature differs from cumulative schedule')
            feedback = row['feedback_gradient_audit']
            raw, weighted = number(feedback['raw_direct_gradient_norm']), number(feedback['weighted_direct_gradient_norm'])
            need(feedback['recommendation_weight'] == 1 and raw > 0 and weighted > 0
                 and math.isclose(weighted, raw, rel_tol=1e-4, abs_tol=1e-7),
                 'No direct recommendation feedback gradient')
            loss = row['loss_means']
            reconstructed = loss['recommendation_loss'] + loss['reconstruction_loss'] + .01 * (loss['balance_loss'] + loss['anchor_loss'])
            need(abs(loss['total_loss'] - reconstructed) <= max(1e-5, abs(reconstructed) * 1e-5), 'Loss weights/extra objective differ')
    count(p['decoder_epochs'], r_count, 'recommender epoch counter')
    count(p['decoder_updates'], r_count * 104, 'cumulative recommender updates')
    count(p['tokenizer_updates'], t_updates, 'cumulative Tokenizer updates')
    count(p['noisy_calls'], noisy, 'cumulative real noise calls')
    if complete:
        count(r_count, 35, 'fixed recommender budget')
        count(t_updates, 60 if arm.endswith('_joint') else 0, 'full Tokenizer budget')
    validations = p['validation']
    need([v['decoder_epoch'] for v in validations] == [e for e in VAL_EPOCHS if e <= r_count], 'Missing/extra validation node')
    for v in validations:
        need(v['arm'] == arm and v['split'] == v['actual_dataset_split'] == 'val' and v['beam_profile_not_test'] is False,
             'Training selection used non-validation observations')
        metric_values(v['metrics'])
    best = max(validations, key=lambda v: v['metrics']['weighted_score'])
    count(p['best_decoder_epoch'], best['decoder_epoch'], 'independently selected best validation epoch')
    need(abs(number(p['best_score']) - best['metrics']['weighted_score']) <= 1e-12, 'Selected best score differs')
    return dict(arm=arm, complete=complete, recommender_epochs=r_count, recommender_updates=r_count * 104,
                tokenizer_updates=t_updates, noisy_calls=noisy, best_epoch=best['decoder_epoch'],
                best_tokenizer_updates=sum(r['optimizer_updates'] for r in rows if r['phase'] == 'tokenizer'
                    and 5 + 3 * r['logical_index'] < best['decoder_epoch']),
                best_validation=metric_values(best['metrics']), final_validation=metric_values(validations[-1]['metrics']))


def paired_controls(progress, *, complete):
    need(set(progress) == set(ARMS), 'Exactly four arms are required for paired controls')
    need(same([p['validation'][0]['metrics'] for p in progress.values()]), 'Initial four-arm validation differs')
    rec = {a: {r['logical_index']:r for r in p['training'] if r['phase'] == 'recommender'} for a,p in progress.items()}
    common = sorted(set.intersection(*(set(v) for v in rec.values())))
    keys = ('logical_index','batches','optimizer_updates','phase_seed','data_seed','data_order_sha256','lr_start','lr_end')
    need(all(same([{k:r[i][k] for k in keys} for r in rec.values()]) for i in common), 'Unpaired recommender data/RNG/updates/LR')
    need(all(same([r[i]['loss_means'] for r in rec.values()]) for i in common if i <= 5), 'Shared warmup losses differ')
    frozen = [rec[a] for a in (ARMS[0], ARMS[2])]
    frozen_common = sorted(set(frozen[0]) & set(frozen[1]))
    need(all(same([r[i]['loss_means'] for r in frozen]) for i in frozen_common), 'Frozen loss trajectories differ')
    fv = [{v['decoder_epoch']:v['metrics'] for v in progress[a]['validation']} for a in (ARMS[0],ARMS[2])]
    need(all(fv[0][i] == fv[1][i] for i in set(fv[0]) & set(fv[1])), 'Frozen validation trajectories differ')
    ts = [{r['logical_index']:r for r in progress[a]['training'] if r['phase'] == 'tokenizer'} for a in (ARMS[1],ARMS[3])]
    cycles = sorted(set(ts[0]) & set(ts[1]))
    t_keys = keys + ('gumbel_stream_seed','gumbel_calls','noise_applied_calls','assignment_training_enabled_calls',
                      'tokenizer_schedule_start','tokenizer_schedule_end')
    need(all(same([{k:r[i][k] for k in t_keys} for r in ts]) for i in cycles), 'Unpaired Tokenizer data/Gumbel/schedule/updates')
    if complete:
        need(common == list(range(1,36)) and frozen_common == list(range(1,36)) and cycles == list(range(10)), 'Incomplete four-arm controls')
    return dict(common_recommender_epochs=common, frozen_loss_epochs=frozen_common, common_tokenizer_cycles=cycles)


def tensor_hash(state):
    import torch
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        need(torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all()), 'Nonfinite/non-tensor model state')
        t = tensor.detach().cpu().contiguous()
        value.update(name.encode() + b'\0' + str(t.dtype).encode() + b'\0' + json.dumps(list(t.shape)).encode() + b'\0')
        value.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def checkpoint(path, arm, identity, protocol, *, terminal, initial_codes, expected_progress):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    need(state['format'] == 'stage05_long_phase_boundary_v1' and state['protocol_id'] == protocol['protocol_id']
         and state['arm'] == arm and state['mode'] == 'formal' and state['identity'] == identity, 'Wrong checkpoint identity/format')
    need(state['E0_array_sha256'] == protocol['canonical_E0_array_sha256'], 'Missing/wrong fixed anchor identity')
    p = state['progress']
    need(p == expected_progress, 'Checkpoint progress differs from the committed training/selection record')
    if terminal:
        audit_progress(p, arm, complete=True)
    elif p['decoder_epochs']:
        audit_progress(p, arm, complete=False)
    else:
        need(p['next_stage'] == p['decoder_updates'] == p['tokenizer_updates'] == p['noisy_calls'] == 0
             and not p['training'] and [v['decoder_epoch'] for v in p['validation']] == [0], 'Invalid initial best checkpoint')
        metric_values(p['validation'][0]['metrics'])
    need(state['scheduler']['last_epoch'] == p['decoder_updates'], 'Checkpoint scheduler counter differs')
    for name in ('python','numpy','torch_cpu','torch_cuda'):
        need(name in state['rng'] and state['rng'][name] is not None, 'Missing Python/NumPy/CPU/CUDA RNG')
    model_sha = tensor_hash(state['model'])
    codes = state['model']['joint_alignment.hard_codes'][1:].contiguous().numpy()
    observed_sid = sid_from_codes(codes, initial_codes)
    if p['training']:
        final_row = p['training'][-1]
        need(observed_sid['SID_sha256'] == final_row['formal_SID_sha256']
             and observed_sid['statistics'] == final_row['formal_SID_statistics']
             and observed_sid['changes'] == final_row['formal_SID_change_since_initial'],
             'Actual checkpoint SID differs from the last committed phase hash/statistics/changes')
    else:
        need(observed_sid['SID_sha256'] == protocol['canonical_SID_array_sha256']
             and observed_sid['changes']['changed_items'] == 0, 'Initial best checkpoint changed the shared SID')
    need(int(state['model']['joint_alignment.diger_optimizer_step']) == p['tokenizer_updates']
         and int(state['model']['joint_alignment.diger_total_optimizer_steps']) == 60, 'Checkpoint annealing buffers differ')
    optimizer_counts = {}
    for kind, expected_updates in (('recommender',p['decoder_updates']), ('tokenizer',p['tokenizer_updates'])):
        opt = state['optimizers'][kind]
        if kind == 'tokenizer' and arm.endswith('_frozen'):
            need(opt is None, 'Frozen arm has a Tokenizer optimizer')
            continue
        need(opt is not None and opt['param_groups'], 'Missing complete optimizer groups')
        if kind == 'tokenizer':
            names = [g['group_name'] for g in opt['param_groups']]
            need(names == (['joint_codebook','joint_mixer'] if arm == ARMS[3] else ['joint_codebook']), 'Unexpected Tokenizer parameter groups')
        else:
            need(len(opt['param_groups']) == 1, 'Unexpected recommender parameter groups')
        for group in opt['param_groups']:
            need(tuple(group['betas']) == (.9,.999) and group['eps'] == 1e-8, 'Adam controls differ')
            wd = 0 if kind == 'recommender' else 1e-4
            need(group['weight_decay'] == wd, 'Optimizer weight decay differs')
            expected_lr = .001 * lr_factor(p['decoder_updates']) if kind == 'recommender' else (1e-5 if group['group_name'] == 'joint_codebook' else 1e-4)
            need(abs(group['lr'] - expected_lr) < 1e-12, 'Saved optimizer LR differs')
        need(bool(opt['state']) == (expected_updates > 0), 'Adam state absent despite committed updates')
        for value in opt['state'].values():
            need(int(value['step']) == expected_updates and bool(torch.isfinite(value['exp_avg']).all())
                 and bool(torch.isfinite(value['exp_avg_sq']).all()), 'Incomplete/nonfinite/wrong-step Adam moments')
        if expected_updates:
            counts = [r['parameter_audit']['expected_parameter_count'] for r in p['training'] if r['phase'] == kind]
            need(same(counts) and len(opt['state']) == counts[-1], 'Some expected parameters lack complete Adam state')
        optimizer_counts[kind] = len(opt['state'])
    return dict(model_tensor_sha256=model_sha, SID_sha256=observed_sid['SID_sha256'],
                recommender_epochs=p['decoder_epochs'], tokenizer_updates=p['tokenizer_updates'],
                optimizer_state_counts=optimizer_counts,actual_SID_statistics=observed_sid['statistics'],
                actual_SID_change_since_initial=observed_sid['changes'])


def source_and_inputs(goal, protocol, identity):
    manifest = read(goal / 'delivery/INPUTS.json')
    need(manifest['protocol'] == protocol and manifest['counts'] == dict(users=22363,items=12101,interactions=198502,train=105668,val=22363,test=22363),
         'Input protocol/data counts differ')
    need(identity['inputs_sha256'] == sha(goal / 'delivery/INPUTS.json')
         and identity['protocol_sha256'] == sha(goal / 'code/configs/05_crosspse_joint/long/mixer_long_seed2024.json'), 'Input/config identity differs')
    for record in manifest['records']:
        path = within(goal / 'inputs', record['path'])
        need(path.stat().st_size == record['size_bytes'] and sha(path) == record['sha256'], 'Input byte stream differs: ' + record['path'])
    for directory, fields, expected in (('code','root_sources',CODE_COMMIT), ('engine','engine_sources',ENGINE_COMMIT)):
        for name, digest in identity[fields].items():
            need(sha(within(goal / directory, name)) == digest, 'Execution source differs: ' + name)
        if (goal / directory / '.git').exists():
            actual = subprocess.check_output(['git','-C',str(goal / directory),'rev-parse','HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git','-C',str(goal / directory),'status','--porcelain'], text=True).strip()
            need(actual == expected and not dirty, 'Execution checkout changed')


def shared_initial(goal, protocol, identity, *, include_codes=False):
    import numpy as np
    import torch
    common = torch.load(goal / 'formal/common/initial.pt',map_location='cpu',weights_only=False)
    need(common['identity'] == identity and common['offline_recommender_loaded_all_tensors'] is True
         and common['E0_file_sha256'] == protocol['canonical_E0_file_sha256'], 'Shared start identity differs')
    state = common['model']
    initial_sha = tensor_hash(state)
    e0 = torch.from_numpy(np.load(goal / 'inputs/canonical/E0.npy',allow_pickle=False))
    need(torch.equal(state['joint_alignment.mixer.base_codebooks'],e0), 'Initial base books differ from canonical effective E0')
    for name in ('weight','bias'):
        need(bool((state['joint_alignment.mixer.attention.out_proj.' + name] == 0).all()), 'New Mixer output was not zero')
    need(float(state['joint_alignment.mixer.residual_scale']) == 1, 'Initial Mixer residual scale differs')
    codes = state['joint_alignment.hard_codes'][1:].contiguous()
    need(hashlib.sha256(codes.numpy().tobytes()).hexdigest() == protocol['canonical_SID_array_sha256'], 'Shared official SID differs')
    offline = torch.load(goal / 'inputs' / protocol['initial_decoder'],map_location='cpu',weights_only=False)
    need(all(name in state and torch.equal(value,state[name]) for name,value in offline.items()),
         'Shared start omitted or changed an original offline recommender tensor')
    gate = read(goal / 'smoke/environment_smoke_all.json')
    need(gate['initial_model_tensor_sha256'] == initial_sha, 'Formal common start differs from accepted H100 start')
    return (initial_sha, codes.numpy().copy()) if include_codes else initial_sha


def audit_goal(goal, *, tests=True, tensors=True):
    goal = Path(goal).resolve()
    need(goal.name == CAMPAIGN, 'Wrong campaign root')
    protocol = read(goal / 'code/configs/05_crosspse_joint/long/mixer_long_seed2024.json')
    fixed = dict(campaign_id=CAMPAIGN, seed=2024, decoder_epochs=35, tokenizer_total_updates=60,
                 decoder_total_updates=3640, mixer_scope='full', mixer_heads=4, mixer_dropout=0,
                 mixer_residual_scale=1, mixer_learn_residual_scale=False, mixer_zero_output=True,
                 noise_start=1, noise_end=.2, noise_decay='cosine', softmax_temperature=1, sia_weight=0,
                 recipe_id='cross_joint_v2',initial_decoder='external/DiffGRM/saved/AmazonReviews2014_Aug-28-2026_19-01-02/pytorch_model.bin',
                 decoder_warmup_epochs=5,cycles=10,decoder_epochs_per_cycle=3,decoder_batches_per_epoch=1652,
                 tokenizer_batches_per_phase=96,gradient_accumulation_steps=16,train_batch_size=64,eval_batch_size=8,
                 decoder_lr=.001,decoder_warmup_updates=500,decoder_weight_decay=0,codebook_lr=1e-5,mixer_lr=1e-4,
                 tokenizer_weight_decay=1e-4,adam_betas=[.9,.999],adam_eps=1e-8,recommendation_weight=1,reconstruction_weight=1,
                 balance_weight=.01,anchor_weight=.01,assignment_temperature=.1,balance_temperature=.1,
                 label_smoothing=.2,max_grad_norm=1)
    need(all(protocol[k] == v for k,v in fixed.items()) and protocol['arms'] == list(ARMS), 'Fixed experimental definition differs')
    identity = read(goal / 'formal/identity.json')
    need(identity['campaign_id'] == CAMPAIGN and identity['execution_backend'] == 'cuda', 'CPU/smoke substituted for formal CUDA')
    source_and_inputs(goal, protocol, identity)
    initial_sha, initial_codes = shared_initial(goal,protocol,identity,include_codes=True) if tensors else (None,None)
    summaries, progress, done = {}, {}, {}
    for arm in ARMS:
        directory = goal / 'formal' / arm
        result = done[arm] = read(directory / 'TRAIN_COMPLETE.json')
        need(result['status'] == 'trained_not_tested' and result['arm'] == arm and result['identity'] == identity,
             'Four full formal arms are required; smoke or partial progress is insufficient')
        p = progress[arm] = result['progress']
        need(read(directory / 'progress.json') == p, 'Final progress sidecar differs')
        summaries[arm] = audit_progress(p, arm, complete=True)
        need(result['selected_decoder_epoch'] == p['best_decoder_epoch'], 'Selection summary differs')
        for filename, field in (('last.pt','last_sha256'),('best.pt','best_sha256')):
            need(sha(directory / filename) == result[field], 'Checkpoint byte hash differs: ' + arm + '/' + filename)
        env = read(goal / 'formal' / ('environment_train_' + arm + '.json'))
        need(env['code_commit'] == CODE_COMMIT and env['engine_commit'] == ENGINE_COMMIT and env['device'] == 'cuda:0'
             and 'H100' in env['gpu'], 'Actual execution environment differs')
        if tensors:
            need(env['initial_model_tensor_sha256'] == initial_sha, 'An arm did not use the verified common start')
        resolved = read(goal / 'formal' / ('resolved_config_train_' + arm + '.json'))
        expected_config = dict(train_batch_size=64,eval_batch_size=8,gradient_accumulation_steps=16,label_smoothing=.2,
             max_grad_norm=1,n_digit=4,codebook_size=256,encoder_n_layer=1,decoder_n_layer=4,n_head=4,n_embd=256,n_inner=1024,
             dropout=.1,train_sliding=True,min_hist_len=2,max_hist_len=50,evaluation_level='sid',eval_expand_sid_to_items=False,
             joint_gradient_mode='diger_st',joint_diger_recommendation_weight=1,joint_recommender_sia_weight=0,
             joint_tokenizer_sia_weight=0,joint_diger_assignment_temperature=.1,joint_balance_temperature=.1,
             joint_balance_weight=.01,joint_anchor_weight=.01,crosspse_attention_scope='full',crosspse_heads=4,crosspse_dropout=0)
        need(all(resolved.get(k) == v for k,v in expected_config.items()), 'Actual parsed architecture/data/loss controls differ')
        beams = resolved['vectorized_beam_search']
        need(resolved['beam_search_modes'] == ['confidence'] and beams['val'] == dict(beam_act=32,beam_max=32)
             and beams['test'] == dict(beam_act=256,beam_max=256), 'Actual validation/test decoding protocol differs')
        if tensors:
            last = checkpoint(directory / 'last.pt',arm,identity,protocol,terminal=True,
                              initial_codes=initial_codes,expected_progress=p)
            best = checkpoint(directory / 'best.pt',arm,identity,protocol,terminal=False,
                              initial_codes=initial_codes,expected_progress=selected_progress(p))
            need(last['model_tensor_sha256'] == result['final_model_tensor_sha256'] and last['SID_sha256'] == result['final_SID_sha256'],
                 'Final model/SID does not match complete checkpoint')
            need(best['recommender_epochs'] == p['best_decoder_epoch']
                 and best['tokenizer_updates'] == summaries[arm]['best_tokenizer_updates'], 'Best checkpoint belongs to another stage')
            summaries[arm]['checkpoint_audit'] = dict(last=last,best=best)
    controls = paired_controls(progress, complete=True)
    need(done[ARMS[0]]['final_model_tensor_sha256'] == done[ARMS[2]]['final_model_tensor_sha256'], 'Frozen final complete models differ')
    values = {arm:summaries[arm]['final_validation'] for arm in ARMS}
    increment = {key:(values[ARMS[3]][key]-values[ARMS[2]][key])-(values[ARMS[1]][key]-values[ARMS[0]][key]) for key in METRICS}
    if tests:
        sealed = goal / 'formal/SELECTION_SEALED.json'
        choices = read(sealed)
        need(choices['identity'] == identity and set(choices['arms']) == set(ARMS), 'Missing four-arm sealed selection')
        for arm in ARMS:
            directory = goal / 'formal' / arm
            choice = choices['arms'][arm]
            need(choice == {k:done[arm][k] for k in ('best_sha256','selected_decoder_epoch','last_sha256')}, 'Sealed selection differs')
            started_path = directory / 'TEST_STARTED.json'
            started, result = read(started_path), read(directory / 'TEST_RESULT.json')
            need(started['arm'] == result['arm'] == arm and started['identity'] == identity
                 and started['actual_split'] == result['actual_dataset_split'] == result['split'] == 'test'
                 and result['beam_profile_not_test'] is False, 'Wrong/missing actual test split')
            need(started['selection_sha256'] == result['selection_sha256'] == sha(sealed)
                 and started['best_sha256'] == result['best_sha256'] == done[arm]['best_sha256']
                 and result['started_sha256'] == sha(started_path), 'Test/source/selection receipts differ')
            need(result['decoder_epoch'] == done[arm]['selected_decoder_epoch']
                 and result['selected_tokenizer_updates'] == summaries[arm]['best_tokenizer_updates'], 'Test checkpoint stage differs')
            summaries[arm]['selected_test'] = metric_values(result['metrics'])
        computed = read(goal / 'formal/COMPUTE_COMPLETE.json')
        need(computed['status'] == 'four_arm_training_and_testing_complete_not_delivery' and computed['identity'] == identity
             and computed['selection_sha256'] == sha(sealed), 'Compute completion receipt missing/different')
        need(computed['tests'] == {a:read(goal / 'formal' / a / 'TEST_RESULT.json') for a in ARMS}, 'Computed test summaries differ from actual per-arm test records')
        control_path = goal / 'formal/CONTROL_PASS.json'
        control = read(control_path)
        need(computed['controls_sha256'] == sha(control_path) and control['status'] == 'passed' and control['formal'] is True,
             'Formal control receipt is missing/stale/smoke')
        need(all(metric_values(control['fixed_final_validation'][a]) == values[a] for a in ARMS)
             and control['Mixer_increment'] == increment, 'Independent fixed-budget comparison differs from recorded controls')
    return dict(status='FORMAL_ARTIFACT_AUDIT_PASS_NOT_DELIVERY', campaign_id=CAMPAIGN,
                utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(Path(__file__)), identity=identity, summaries=summaries, controls=controls,
                initial_model_tensor_sha256=initial_sha,
                source_receipt_sha256={arm:{name:sha(goal / 'formal' / arm / name)
                    for name in ('TRAIN_COMPLETE.json','progress.json')} for arm in ARMS},
                fixed_epoch35_validation=values, Mixer_increment=increment, tests_checked=tests, checkpoints_checked=tensors,
                resource_release_verified=False, full_backup_restore_verified=False, goal_complete=False)


def export_tables(goal, output, result):
    def table(name, fields, rows):
        with (output / name).open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    validations, phases = [], []
    for arm in ARMS:
        p = read(goal / 'formal' / arm / 'TRAIN_COMPLETE.json')['progress']
        validations.extend(dict(arm=arm,recommender_epoch=v['decoder_epoch'],**metric_values(v['metrics'])) for v in p['validation'])
        r = t = 0
        for index,row in enumerate(p['training']):
            if row['phase'] == 'recommender': r += 1
            else: t += 6
            phases.append(dict(arm=arm,stage=index+1,phase=row['phase'],logical_index=row['logical_index'],recommender_epoch=r,
                 tokenizer_updates=t,batches=row['batches'],optimizer_updates=row['optimizer_updates'],
                 seconds=row['elapsed_seconds'],lr_start=row['lr_start'],lr_end=row['lr_end'],
                 loss=row['loss_means']['total_loss'],recommendation_loss=row['loss_means'].get('recommendation_loss'),
                 reconstruction_loss=row['loss_means'].get('reconstruction_loss'),balance_loss=row['loss_means'].get('balance_loss'),
                 anchor_loss=row['loss_means'].get('anchor_loss'),noise_calls=row['noise_applied_calls'],
                 noise_start=row['tokenizer_schedule_start']['gumbel_noise_scale'],noise_end=row['tokenizer_schedule_end']['gumbel_noise_scale'],
                 changed_SID_items=row['formal_SID_change_since_initial']['changed_items'],
                 collided_items=row['formal_SID_statistics']['full_sid']['collided_items'],
                 collision_rate=row['formal_SID_statistics']['full_sid']['collision_rate'],data_order_sha256=row['data_order_sha256']))
    table('validation.csv',['arm','recommender_epoch',*METRICS],validations)
    table('phases.csv',list(phases[0]),phases)
    table('fixed_epoch35_validation.csv',['arm',*METRICS],[dict(arm=a,**result['fixed_epoch35_validation'][a]) for a in ARMS])
    if result['tests_checked']:
        table('best_selected_test.csv',['arm','selected_recommender_epoch','selected_tokenizer_updates',*METRICS],
              [dict(arm=a,selected_recommender_epoch=result['summaries'][a]['best_epoch'],
                    selected_tokenizer_updates=result['summaries'][a]['best_tokenizer_updates'],**result['summaries'][a]['selected_test']) for a in ARMS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-only',action='store_true',help='Still requires all four complete 35R arms; no delivery claim')
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a new audit output directory')
    args.output.mkdir(parents=True)
    try:
        result = audit_goal(args.goal_root,tests=not args.training_only,tensors=True)
        write(args.output / 'AUDIT.json',result)
        export_tables(args.goal_root,args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as error:
        write(args.output / 'AUDIT_FAILED.json',dict(status='not_accepted',error=type(error).__name__,message=str(error),goal_complete=False))
        raise


if __name__ == '__main__':
    main()
