"""CPU-only, read-only Stage09 warmup / branch-initial acceptance.

Print receipts to stdout; the caller may save them. No platform calls or model
forward passes. Only trusted checkpoints under the owned Goal root are loaded.
"""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def owned(relative):
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            'Unsafe relative path')
    path = ROOT / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()),
            'Path is a symlink or escapes the owned Goal root')
    return path


def source_check(manifest):
    helper = owned('delivery/formal_warmup_precheck.py')
    require(sha(helper) == manifest['warmup_precheck_sha256'], 'Existing source-check helper changed')
    spec = importlib.util.spec_from_file_location('_stage09_source_precheck', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_path = owned('delivery/FORMAL_WARMUP_BINDING.json')
    require(sha(base_path) == manifest['warmup_binding_sha256'], 'Existing binding manifest changed')
    base = json.loads(base_path.read_text(encoding='utf-8'))
    for relative, expected in base['frozen_repositories'].items():
        repo = owned(relative)
        require(Path(module.git(repo, 'rev-parse', '--show-toplevel')).resolve() == repo.resolve(),
                'Unexpected Git worktree root')
        require(module.git(repo, 'rev-parse', 'HEAD') == expected, 'Frozen HEAD differs: ' + relative)
        require(not module.git(repo, 'status', '--porcelain=v1', '--untracked-files=all'),
                'Frozen source tree is dirty: ' + relative)
    actual = set()
    for directory in ('code/scripts/listwise_diffgrm', 'code/external/DiffGRM/genrec',
                      'code/external/DiffGRM/genrec/models/DIFF_GRM'):
        actual.update(p.relative_to(ROOT).as_posix() for p in owned(directory).glob('*.py'))
    require(actual == set(base['source_binding_paths']), 'Runtime source-file set changed')
    for relative, expected in base['file_sha256'].items():
        require(sha(owned(relative)) == expected, 'Frozen source/metadata SHA differs: ' + relative)
    engine = os.environ.get('DIFFGRM_ENGINE_ROOT')
    require(not engine or Path(engine).resolve() == owned('code/external/DiffGRM').resolve(),
            'Environment overrides the frozen engine')


def cpu_libraries():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'Acceptance must not initialize CUDA')
    return torch, np


def fingerprint(value):
    torch, np = cpu_libraries()
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
                    'Nonfinite checkpoint tensor')
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            require(not isinstance(item, float) or math.isfinite(item), 'Nonfinite checkpoint scalar')
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def load_checkpoint(path):
    torch, _ = cpu_libraries()
    before = sha(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(sha(path) == before, 'Checkpoint changed while being read')
    return state, dict(path=str(path), bytes=path.stat().st_size, sha256=before)


def validate_state(state, expected_binding, *, arm, initial=False):
    require(state['format'] == 'kuairec_listwise_v1' and state['arm'] == arm,
            'Wrong checkpoint format or arm')
    require(state['binding'] == expected_binding and state['precision'] == 'bf16',
            'Complete code/config/data/assets binding or precision differs')
    progress = state['progress']
    require(not state.get('preflight_only') and not progress.get('preflight_only'),
            'Preflight checkpoint is forbidden')
    steps, phase = (0, 0) if initial else (3465, 5)
    require((state['r_updates'], state['t_updates'], progress['phase'], progress['update'])
            == (steps, 0, phase, 0), 'Warmup update counts or phase cursor differ')
    require(state['scheduler']['last_epoch'] == steps, 'Scheduler clock differs')
    require(int(state['alignment']['diger_optimizer_step']) == 0
            and int(state['alignment']['diger_total_optimizer_steps']) == 400,
            'Tokenizer schedule differs')
    require(not state['t_optimizer']['state'], 'Tokenizer optimizer is not empty before T')
    r_states = state['r_optimizer']['state']
    require(bool(r_states) == bool(steps), 'R optimizer state presence differs')
    require(all(float(v['step']) == steps for v in r_states.values()), 'R Adam clock differs')
    require(set(state['rng']) == {'python', 'numpy', 'torch', 'cuda'}, 'RNG snapshot incomplete')
    require(state['rng']['cuda'] is not None and len(state['rng']['cuda']) == 1,
            'Expected exactly one saved CUDA RNG stream')
    fingerprint(state)


def warmup_check(architecture, manifest):
    directory = owned('runs/warmup_' + architecture)
    require(not (directory / '.running').exists(), 'Warmup runner lock remains; inspect its Job first')
    expected = manifest['expected_bindings'][architecture]
    last, last_file = load_checkpoint(directory / 'last.pt')
    best, best_file = load_checkpoint(directory / 'best.pt')
    initial, initial_file = load_checkpoint(directory / 'initial.pt')
    validate_state(last, expected, arm='warmup')
    validate_state(best, expected, arm='warmup')
    validate_state(initial, expected, arm='warmup', initial=True)
    require(fingerprint(last) == fingerprint(best), 'The sole R5 best differs from last')
    require(fingerprint(last['alignment']) == fingerprint(initial['alignment']),
            'R-only warmup changed tokenizer/E0/SID state')
    complete_path = directory / 'complete.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    progress = last['progress']
    require(complete == dict(arm='warmup', r_updates=3465, t_updates=0,
                            best_epoch=5, best_score=progress['best_score']),
            'Warmup completion receipt differs')
    require(progress['best_epoch'] == 5 and len(progress['validations']) == 1,
            'Expected one accepted R5 validation')
    validation = Path(progress['validations'][0])
    require(validation.resolve().parent == directory.resolve()
            and re.fullmatch(r'validation_R05(?:_retry[1-9][0-9]*)?', validation.name),
            'Validation provenance is outside this warmup')
    summary_path = validation / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fields = dict(split='validation', examples=71657, full_split=True, subset=None,
                  diagnostic_prefix_limit=None, beam=32, architecture=architecture,
                  seed=2024, r_updates=3465, t_updates=0, inference_precision='fp32',
                  catalog_size=10728, dataset_binding=expected['data'])
    require(all(summary.get(k) == v for k, v in fields.items()), 'Full R5 validation identity differs')
    protocol = dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
                    target_relevance='unordered_set_of_five_future_distinct_items',
                    aggregation='list_score_order_then_item_position_first_occurrence')
    require(summary['metric_protocol'] == protocol, 'R5 metric protocol differs')
    score = summary['metrics']['list/ndcg@5']
    require(math.isfinite(score) and 0 <= score <= 1 and score == progress['best_score'],
            'R5 selection score differs from saved best')
    prediction_path = validation / 'per_sample.jsonl.gz'
    ids, count, score_sum = set(), 0, 0.0
    with gzip.open(prediction_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            require(row['dataset_index'] == count and row['sample_id'] not in ids,
                    'R5 emitted rows are duplicated, skipped or out of order')
            ids.add(row['sample_id'])
            value = row['metrics']['list/ndcg@5']
            require(math.isfinite(value) and 0 <= value <= 1, 'Invalid per-row NDCG')
            score_sum += value; count += 1
    require(count == 71657 and abs(score_sum / count - score) <= 1e-12,
            'Incomplete R5 rows or mean differs from summary')
    receipt = dict(architecture=architecture, last=last_file, best=best_file, initial=initial_file,
        complete=dict(path=str(complete_path), sha256=sha(complete_path)),
        accepted_validation=dict(path=str(validation), examples=count, beam=32,
            summary_sha256=sha(summary_path), predictions_sha256=sha(prediction_path),
            ndcg_at_5=score, row_indices_and_unique_ids_complete=True),
        state_fingerprints={k: fingerprint(last[k]) for k in
            ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'progress')},
        source_binding_complete=True, tokenizer_unchanged_from_initial=True,
        scope='Checkpoint and complete emitted R5 output audit, not independent label/metric recomputation; platform terminal/release must be confirmed separately')
    return last, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['warmup', 'ready', 'pair'])
    parser.add_argument('--architecture', choices=['J', 'H'], required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--arm', choices=['J0', 'J1', 'H0', 'H1'])
    parser.add_argument('--accepted-warmup-sha256')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.manifest_sha256):
        parser.error('Manifest SHA must contain 64 lowercase hex characters')
    if args.mode == 'ready' and (not args.arm or args.arm[0] != args.architecture
            or not re.fullmatch('[0-9a-f]{64}', args.accepted_warmup_sha256 or '')):
        parser.error('ready requires a matching --arm and the CPU-accepted warmup SHA')
    if args.mode != 'ready' and args.arm is not None:
        parser.error('--arm is only valid in ready mode')
    path = owned('delivery/FORMAL_BRANCH_BINDING.json')
    require(sha(path) == args.manifest_sha256, 'Branch binding manifest changed')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    require(manifest['root'] == str(ROOT), 'Wrong Goal root')
    source_check(manifest)
    if args.mode == 'ready':
        output = owned('runs/' + args.arm)
        require(not output.exists(), 'Branch output already exists; use reviewed recovery')
        warmup = owned('runs/warmup_' + args.architecture)
        require(not (warmup / '.running').exists() and (warmup / 'complete.json').is_file(),
                'Warmup is not complete or still locked')
        require(sha(warmup / 'last.pt') == args.accepted_warmup_sha256,
                'Warmup checkpoint differs from CPU-accepted snapshot')
        result = dict(arm=args.arm, accepted_warmup_sha256=args.accepted_warmup_sha256,
                      output_absent=str(output), checkpoint_loaded=False)
    else:
        state, result = warmup_check(args.architecture, manifest)
        if args.mode == 'pair':
            common = {k: v for k, v in state.items() if k != 'arm'}
            common_sha = fingerprint(common)
            pair = {}
            for arm in (args.architecture + '0', args.architecture + '1'):
                initial, evidence = load_checkpoint(owned('runs/' + arm + '/initial.pt'))
                validate_state(initial, manifest['expected_bindings'][args.architecture], arm=arm)
                require(fingerprint({k: v for k, v in initial.items() if k != 'arm'}) == common_sha,
                        'Branch initial state differs from accepted warmup: ' + arm)
                pair[arm] = evidence
            result.update(pair_initials=pair, exact_state_equal_except_arm=True,
                shared_state_sha256=common_sha,
                mixer_switch='0 disabled; 1 enabled by runner arm suffix, not a differing saved tensor')
    receipt = dict(status='passed', mode=args.mode, architecture=args.architecture,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=args.manifest_sha256, gpu_used=False)
    receipt.update(result)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
