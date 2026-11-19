"""Read actual local joint backup checkpoints on CPU; never restore or train.

Input is the new directory produced by pack_item_local_joint_delivery.py verify.
PACKAGE.json anchors its manifest; all archived .pt/.bin files are loaded and
checked, including the fourteen mandatory formal/smoke files. This is neither
the remote paired experiment audit nor an end-to-end relocated resume test.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys

CAMPAIGN = 'stage07_item_local_alternating_joint_seed2024_20261020_r1'
ROOT_COMMIT = '842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1'
ENGINE_COMMIT = '1c879252f6395d6e64462511823c3929b9c42c00'
FORMAT = 'stage07_item_local_joint_delivery_v1'
BOUNDARY = 'diffgrm_joint_phase_boundary_v1'
NECESSARY = ('training_state/last.pt', 'training_state/best_model.bin',
    'smoke/training_state/last.pt', 'smoke/training_state/best_model.bin',
    'smoke/smoke_tokenizer_boundary.pt', 'smoke/recovery_probe/source_last.pt', 'smoke/recovery_probe/best_model.bin')

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

def verify(backup_root, package_dir):
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    import torch
    root = Path(backup_root).resolve(strict=True); package_path = Path(package_dir)/'PACKAGE.json'
    package, byte_receipt, manifest = read(package_path), read(root/'BYTE_VERIFICATION.json'), read(root/'MANIFEST.json')
    need(package['format'] == manifest['format'] == FORMAT and package['campaign_id'] == manifest['campaign_id'] == CAMPAIGN
         and byte_receipt['status'] == 'BYTE_RESTORATION_VERIFIED' and byte_receipt['archive_sha256'] == package['archive_sha256']
         and sha(root/'MANIFEST.json') == package['manifest_sha256'], 'Backup does not match its verified package/manifest')
    need(manifest['source_identity']['code']['commit'] == ROOT_COMMIT and manifest['source_identity']['engine']['commit'] == ENGINE_COMMIT, 'Wrong frozen training sources')
    entries = manifest['files']; records = {row['member']:row for row in entries}
    need(len(records) == len(entries) == package['files'] == byte_receipt['files']
         and len({name.casefold() for name in records}) == len(records), 'Duplicate/incomplete manifest file inventory')
    def checked(member):
        need(member in records, 'Required backup member absent: '+member)
        name = PurePosixPath(member)
        need(not name.is_absolute() and '..' not in name.parts and '\\' not in member and ':' not in member, 'Unsafe backup member')
        path = root.joinpath(*name.parts); value = records[member]
        need(path.resolve().is_relative_to(root) and path.is_file() and not any(p.is_symlink() for p in (path,*path.parents))
             and path.stat().st_size == value['size_bytes'] and sha(path) == value['sha256'], 'Backup bytes differ: '+member)
        return path
    required = {f'runs/stage07_item_local_joint_{variant}_seed2024_20261020_r1/'+relative
        for variant in ('control','item_local') for relative in NECESSARY}
    checkpoint_members = {name for name in records if PurePosixPath(name).suffix in ('.pt','.bin')}
    need(required <= checkpoint_members and len(checkpoint_members) >= 14, 'Missing necessary fourteen checkpoint files')
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file() and path.suffix in ('.pt','.bin')}
    need(actual == checkpoint_members, 'Unlisted or missing local checkpoint files')
    reports, formal = [], {}
    for member in sorted(checkpoint_members):
        path = checked(member); payload = torch.load(path, map_location='cpu', weights_only=False)
        tensors, elements = tensor_inventory(payload, member); need(tensors > 0, 'Checkpoint has no actual tensors: '+member)
        report = {'member':member, 'sha256':sha(path), 'size_bytes':path.stat().st_size,
                  'tensor_count':tensors, 'tensor_elements':elements, 'all_tensors_finite':True}
        if member in required and member.endswith('.pt'):
            work = path.parents[1] if path.name == 'last.pt' else path.parent
            if path.name == 'source_last.pt': best = path.parent/'best_model.bin'
            elif path.name == 'smoke_tokenizer_boundary.pt': best = work/'recovery_probe/best_model.bin'
            else: best = path.parent/'best_model.bin'
            report['joint_boundary'] = inspect_boundary(payload, checked(best.relative_to(root).as_posix()), terminal=path.name == 'last.pt')
            if path.name == 'last.pt' and work.name != 'smoke': formal[work.name] = (payload,report)
        elif member in required:
            need(isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()), 'Best checkpoint is not an actual model state')
        reports.append(report); del payload
    need(len(formal) == 2, 'Missing two formal terminal states')
    arms = {}
    for variant in ('control','item_local'):
        run_id = f'stage07_item_local_joint_{variant}_seed2024_20261020_r1'; prefix = 'runs/'+run_id+'/'
        state, last_report = formal[run_id]
        result_path = checked(prefix+'results.json'); result = read(result_path)
        reload = read(checked(prefix+'checkpoint_reload.json')); complete = read(checked(prefix+'COMPLETE.json'))
        best_path = checked(prefix+'training_state/best_model.bin'); best = torch.load(best_path, map_location='cpu', weights_only=False)
        sid = tensor_digest(best['joint_alignment.hard_codes'][1:]); best_sha = sha(best_path)
        need(result['campaign_id'] == complete['campaign_id'] == CAMPAIGN and result['variant'] == complete['variant'] == variant
             and result['run_id'] == complete['run_id'] == run_id and complete['mode'] == 'formal'
             and result['status'] == complete['status'] == 'formal_completed', 'Formal result identity/status differs')
        need(result['fit']['training_state_sha256'] == last_report['sha256'] and result['fit']['completed_epochs'] == state['progress']['next_epoch']
             and result['fit']['best_epoch'] == state['progress']['best_epoch'] and result['fit']['joint_progress'] == state['progress']['joint_progress'], 'Formal result/terminal progress mismatch')
        need(result['checkpoint_sha256'] == reload['checkpoint_sha256'] == complete['checkpoint_sha256'] == best_sha
             and complete['result_file'] == 'results.json' and complete['result_sha256'] == sha(result_path)
             and reload['strict_all_tensor_match'] is True and reload['mapping_synchronized_without_requantization'] is True
             and reload['best_epoch'] == state['progress']['best_epoch']
             and result['selected_SID_sha256'] == reload['matching_committed_SID_sha256'] == sid, 'Formal selected checkpoint/reload/SID SHA mismatch')
        need(set(best) == set(state['model']), 'Best/terminal model key sets differ')
        arms[variant] = {**last_report['joint_boundary'], 'best_checkpoint_sha256':best_sha, 'selected_SID_sha256':sid,
                        'results_sha256':sha(result_path), 'checkpoint_reload_sha256':sha(checked(prefix+'checkpoint_reload.json'))}
        del best
    return {'status':'LOCAL_CPU_CHECKPOINT_READBACK_PASS', 'campaign_id':CAMPAIGN, 'verified_utc':datetime.now(timezone.utc).isoformat(),
        'backup_root':str(root), 'package_receipt_sha256':sha(package_path), 'archive_sha256':package['archive_sha256'],
        'manifest_sha256':sha(root/'MANIFEST.json'), 'byte_verification_receipt_sha256':sha(root/'BYTE_VERIFICATION.json'),
        'source_identity':manifest['source_identity'], 'checkpoints':reports, 'checkpoint_count':len(reports), 'formal_arms':arms,
        'load_method':"torch.load(map_location='cpu', weights_only=False)", 'all_actual_tensors_finite':True,
        'strict_model_reload_performed':False, 'remote_pair_audit_performed':False, 'full_relocated_resume_exercised':False,
        'GPU_or_platform_used':False, 'goal_complete':False,
        'scope':'Local archived checkpoint bytes and serialized-state consistency only; no model construction, restore, update, comparison audit or GPU-release proof.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-root', type=Path, required=True); parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    need(not os.path.lexists(args.output), 'Choose a new readback receipt; existing evidence is preserved')
    try: result, code = verify(args.backup_root,args.package_dir), 0
    except Exception as error:
        result, code = {'status':'LOCAL_CPU_CHECKPOINT_READBACK_FAILED','error_type':type(error).__name__,'error':str(error),
            'remote_pair_audit_performed':False,'full_relocated_resume_exercised':False,'GPU_or_platform_used':False,'goal_complete':False}, 1
    result['tool_sha256'] = sha(__file__)
    with args.output.open('x',encoding='utf-8') as handle: json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write('\n')
    print(json.dumps({'status':result['status'],'receipt':str(args.output),'sha256':sha(args.output)},ensure_ascii=False)); return code

if __name__ == '__main__': raise SystemExit(main())
