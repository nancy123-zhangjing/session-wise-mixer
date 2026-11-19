"""Read-only CPU acceptance and four-best identity snapshot after Stage09 training.

Run ``python -B formal_final_acceptance.py`` only after all four branch jobs
finish. JSON goes to stdout; the caller may save it as the freeze receipt.
No checkpoint is changed/copied, no test split is read, and no GPU is used.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
HELPER_SHA = '2c255470cdd58b2c13dd50fb130aa750f1609306720081ab0e3cd33050cb2831'
BINDING_SHA = '871e480d2a77d7e22bfed091d02065f0cdc42eb1db42cf20d43519634959e80a'
SOURCE = '1c423662d9a25a926328ee6466c30c0f2857b4f6'
EPOCHS = tuple(range(5, 36, 3))
METRIC = 'list/ndcg@5'
PROTOCOL = dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
    target_relevance='unordered_set_of_five_future_distinct_items',
    aggregation='list_score_order_then_item_position_first_occurrence')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def clocks(epoch):
    require(type(epoch) is int and epoch in EPOCHS, 'Epoch is not an accepted full-validation boundary')
    cycle = (epoch - 5) // 3
    return dict(r_updates=epoch * 693, t_updates=cycle * 40, phase=5 + cycle * 4, update=0)


def select_best(scores):
    require(len(scores) == len(EPOCHS) and all(type(s) in (int, float) and math.isfinite(s) and 0 <= s <= 1 for s in scores),
            'Selection needs eleven finite validation scores')
    # Python max returns the first equal maximum: the earlier epoch wins ties.
    return max(range(len(scores)), key=lambda i: scores[i])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def helper_and_binding():
    helper_path = ROOT / 'delivery/formal_branch_acceptance.py'
    binding_path = ROOT / 'delivery/FORMAL_BRANCH_BINDING.json'
    require(not helper_path.is_symlink() and helper_path.resolve().is_relative_to(ROOT.resolve()),
            'Branch helper escapes owned root')
    require(sha(helper_path) == HELPER_SHA and sha(binding_path) == BINDING_SHA,
            'Pinned branch helper or binding changed')
    spec = importlib.util.spec_from_file_location('_stage09_final_branch_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    manifest = read_json(helper.owned('delivery/FORMAL_BRANCH_BINDING.json'))
    require(manifest['root'] == str(ROOT) and manifest['execution_source'] == SOURCE,
            'Unexpected Goal root or frozen source')
    helper.source_check(manifest)
    return helper, manifest


def validate_summary(summary, *, architecture, epoch, binding):
    expected = clocks(epoch)
    fields = dict(split='validation', examples=71657, full_split=True, subset=None,
        diagnostic_prefix_limit=None, beam=32, architecture=architecture, seed=2024,
        r_updates=expected['r_updates'], t_updates=expected['t_updates'],
        inference_precision='fp32', catalog_size=10728, dataset_binding=binding['data'])
    require(all(summary.get(key) == value for key, value in fields.items()),
            f'R{epoch} full-validation identity differs')
    require(summary['metric_protocol'] == PROTOCOL, 'Validation metric protocol differs')
    score = summary['metrics'][METRIC]
    require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            'Invalid selection score')
    return score


def validation_receipt(helper, path, *, architecture, epoch, binding, cache):
    path = Path(path)
    expected_parent = helper.owned('runs/warmup_' + architecture if epoch == 5 else 'runs/' + binding['arm'])
    require(path.is_absolute() and path.resolve().parent == expected_parent.resolve() and
            re.fullmatch(fr'validation_R{epoch:02d}(?:_retry[1-9][0-9]*)?', path.name),
            'Accepted validation is outside its correct warmup/branch directory')
    path = helper.owned(path.relative_to(ROOT).as_posix())
    key = str(path)
    if key in cache:
        return cache[key]
    summary_path, predictions_path = path / 'summary.json', path / 'per_sample.jsonl.gz'
    summary = read_json(summary_path)
    score = validate_summary(summary, architecture=architecture, epoch=epoch, binding=binding['expected'])
    summary_sha = helper.sha(summary_path)
    prediction_sha = helper.sha(predictions_path)
    count, total, ids = 0, 0.0, set()
    with gzip.open(predictions_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = row['sample_id']
            require(type(row['dataset_index']) is int and row['dataset_index'] == count and
                    isinstance(sample_id, str) and sample_id and sample_id not in ids,
                    'Validation rows are skipped, duplicated, or out of order')
            ids.add(sample_id)
            value = row['metrics'][METRIC]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    'Invalid per-row NDCG')
            total += value
            count += 1
    require(count == 71657 and abs(total / count - score) <= 1e-12,
            'Validation is incomplete or saved NDCG mean differs')
    require(helper.sha(summary_path) == summary_sha and helper.sha(predictions_path) == prediction_sha,
            'Validation changed during acceptance')
    result = dict(path=str(path), epoch=epoch, examples=count, beam=32, ndcg_at_5=score,
                  summary_sha256=summary_sha, predictions_sha256=prediction_sha,
                  emitted_indices_and_unique_ids_complete=True)
    cache[key] = result
    return result


def state_receipt(helper, state, *, expected, arm, epoch, reference_slots, reference_state):
    torch, _ = helper.cpu_libraries()
    require(state['format'] == 'kuairec_listwise_v1' and state['arm'] == arm and
            state['binding'] == expected and state['precision'] == 'bf16',
            'Checkpoint format/arm/full binding/training precision differs')
    progress, clock = state['progress'], clocks(epoch)
    require(not state.get('preflight_only') and not progress.get('preflight_only'), 'Preflight state is forbidden')
    for key in ('r_updates', 't_updates'):
        require(type(state[key]) is int and state[key] == clock[key], f'{arm}: wrong {key}')
    for key in ('phase', 'update'):
        require(type(progress[key]) is int and progress[key] == clock[key], f'{arm}: wrong {key}')
    require(state['scheduler']['last_epoch'] == clock['r_updates'], 'R scheduler clock differs')
    alignment = state['alignment']
    require(int(alignment['diger_optimizer_step']) == clock['t_updates'] and
            int(alignment['diger_total_optimizer_steps']) == 400, 'Tokenizer annealing clock differs')
    for kind in ('r', 't'):
        steps = clock[kind + '_updates']
        entries = state[kind + '_optimizer']['state']
        require(bool(entries) == bool(steps), 'Optimizer state presence differs')
        require(all(float(value['step']) == steps for value in entries.values()), 'Adam update clock differs')
    require(set(state['rng']) == {'python', 'numpy', 'torch', 'cuda'} and
            state['rng']['cuda'] is not None and len(state['rng']['cuda']) == 1,
            'Single-GPU RNG state is incomplete')
    for field in ('model', 'alignment'):
        require(set(state[field]) == set(reference_state[field]), f'{field} state keys differ from shared initialization')
        for key, initial_value in reference_state[field].items():
            value = state[field][key]
            require(torch.is_tensor(value) and value.shape == initial_value.shape and value.dtype == initial_value.dtype,
                    f'{field}.{key} shape/dtype differs from shared initialization')
    floating = [value for value in state['model'].values() if torch.is_tensor(value) and value.is_floating_point()]
    require(floating and all(value.dtype == torch.float32 for value in floating),
            'Stored model floating tensors must remain FP32 despite BF16 training autocast')
    sid = alignment['hard_codes']
    require(torch.is_tensor(sid) and sid.dtype == torch.long and tuple(sid.shape) == (10729, 4) and
            bool((sid[0] == -1).all()) and bool(((sid[1:] >= 0) & (sid[1:] < 256)).all()),
            'Checkpoint SID shape/range/PAD row differs')
    require(torch.equal(alignment['item_slots'], reference_slots), 'Frozen item slots changed')
    require(tuple(alignment['mixer.base_codebooks'].shape) == (4, 256, 64), 'Codebook layout differs')
    return dict(epoch=epoch, **clock, stored_model_floating_dtype='torch.float32',
        checkpoint_precision='bf16', evaluation_restore_precision='bf16', inference_precision='fp32',
        state_fingerprint=helper.fingerprint(state),
        sid_snapshot=dict(field='alignment.hard_codes', shape=list(sid.shape), pad_row=-1,
                          real_code_range=[0, 255], fingerprint=helper.fingerprint(sid)),
        alignment_fingerprint=helper.fingerprint(alignment))


def common_assets(helper, manifest):
    torch, np = helper.cpu_libraries()
    root = helper.owned('assets/pse_v1')
    asset_manifest = read_json(root / 'manifest.json')
    require(helper.sha(root / 'manifest.json') == manifest['expected_bindings']['J']['asset_manifest'] ==
            manifest['expected_bindings']['H']['asset_manifest'], 'Shared asset manifest differs')
    arrays = {}
    for name in ('semantic_ids.npy', 'opq_residual_embeddings.npy', 'base_pse_codebooks.npy'):
        path = root / name
        entry = asset_manifest['artifacts'][name]
        require(path.stat().st_size == entry['bytes'] and helper.sha(path) == entry['sha256'],
                'Shared asset identity differs: ' + name)
        arrays[name] = torch.from_numpy(np.load(path, allow_pickle=False))
    require(tuple(arrays['semantic_ids.npy'].shape) == (10728, 4) and
            tuple(arrays['opq_residual_embeddings.npy'].shape) == (10728, 256) and
            tuple(arrays['base_pse_codebooks.npy'].shape) == (4, 256, 64), 'Shared asset dimensions differ')
    slots = torch.cat((torch.zeros(1, 4, 64), arrays['opq_residual_embeddings.npy'].reshape(10728, 4, 64)))
    return slots, arrays['semantic_ids.npy'], arrays['base_pse_codebooks.npy']


def main():
    helper, manifest = helper_and_binding()
    torch, _ = helper.cpu_libraries()
    slots, sid0, books0 = common_assets(helper, manifest)
    cache, warmups, warmup_receipts = {}, {}, {}
    for architecture in ('J', 'H'):
        state, receipt = helper.warmup_check(architecture, manifest)
        require(torch.equal(state['alignment']['hard_codes'][1:], sid0) and
                torch.equal(state['alignment']['item_slots'], slots) and
                torch.equal(state['alignment']['mixer.base_codebooks'], books0),
                'Warmup SID0/E0/item-slot snapshot differs from common assets')
        warmups[architecture], warmup_receipts[architecture] = state, receipt
        item = dict(receipt['accepted_validation'], epoch=5)
        cache[item['path']] = item  # Shared R5 was already fully scanned by warmup_check.
    arms = {}
    for arm in ('J0', 'J1', 'H0', 'H1'):
        architecture, directory = arm[0], helper.owned('runs/' + arm)
        require(not (directory / '.running').exists(), 'Branch lock remains: ' + arm)
        expected = manifest['expected_bindings'][architecture]
        last, last_file = helper.load_checkpoint(directory / 'last.pt')
        best, best_file = helper.load_checkpoint(directory / 'best.pt')
        final_state = state_receipt(helper, last, expected=expected, arm=arm, epoch=35,
                                    reference_slots=slots, reference_state=warmups[architecture])
        progress = last['progress']
        require(isinstance(progress['validations'], list) and len(progress['validations']) == 11,
                'Expected exactly eleven accepted validations per branch')
        validations = [validation_receipt(helper, path, architecture=architecture, epoch=epoch,
                        binding=dict(arm=arm, expected=expected), cache=cache)
                       for epoch, path in zip(EPOCHS, progress['validations'])]
        index = select_best([row['ndcg_at_5'] for row in validations])
        epoch, score = EPOCHS[index], validations[index]['ndcg_at_5']
        require(progress['best_epoch'] == epoch and progress['best_score'] == score,
                'Final selected best differs from earliest maximum of full validation NDCG@5')
        complete_path = directory / 'complete.json'
        require(read_json(complete_path) == dict(arm=arm, r_updates=24255, t_updates=400,
                                               best_epoch=epoch, best_score=score),
                'Final completion receipt differs')
        best_state = state_receipt(helper, best, expected=expected, arm=arm, epoch=epoch,
                                   reference_slots=slots, reference_state=warmups[architecture])
        require(best['progress']['best_epoch'] == epoch and best['progress']['best_score'] == score and
                best['progress']['validations'] == progress['validations'][:index + 1],
                'Best checkpoint cursor/accepted-validation prefix differs')
        initial, initial_file = helper.load_checkpoint(directory / 'initial.pt')
        helper.validate_state(initial, expected, arm=arm)
        common = warmups[architecture]
        without_arm = lambda value: {key: item for key, item in value.items() if key != 'arm'}
        require(helper.fingerprint(without_arm(initial)) == helper.fingerprint(without_arm(common)),
                'Branch initial no longer matches its accepted shared warmup')
        if epoch == 5:
            require(helper.fingerprint(without_arm(best)) == helper.fingerprint(without_arm(common)),
                    'R5 best no longer matches shared warmup, including SID snapshot')
        if epoch == 35:
            require(best_state['state_fingerprint'] == final_state['state_fingerprint'],
                    'R35 best differs from last, including SID snapshot')
        arms[arm] = dict(best=best_file, last=last_file, initial=initial_file,
            complete=dict(path=str(complete_path), sha256=helper.sha(complete_path)),
            best_state=best_state, final_state=final_state, accepted_validations=validations,
            selection=dict(metric=METRIC, split='validation', beam=32, tie_break='earlier',
                           best_epoch=epoch, best_t_updates=clocks(epoch)['t_updates'], score=score,
                           chosen_validation=validations[index]['path']),
            shared_initial_exact=True)
        del initial, last, best
    require(len(cache) == 42, 'Expected 42 distinct full validations, not 44 duplicated warmups')
    for row in arms.values():
        for kind in ('best', 'last', 'initial'):
            require(helper.sha(Path(row[kind]['path'])) == row[kind]['sha256'],
                    'Checkpoint changed during final acceptance')
    receipt = dict(schema='stage09_four_arm_final_selection_v1', status='passed',
        checked_at_utc=datetime.now(timezone.utc).isoformat(), execution_source=SOURCE,
        branch_binding_sha256=BINDING_SHA, branch_helper_sha256=HELPER_SHA,
        script_sha256=sha(Path(__file__)), root=str(ROOT), arms=arms,
        shared_warmups=warmup_receipts, distinct_full_validations=len(cache),
        four_best_sha256={arm: row['best']['sha256'] for arm, row in arms.items()},
        four_best_bytes={arm: row['best']['bytes'] for arm, row in arms.items()},
        test_used_for_selection=False, gpu_used=False, checkpoint_writes=False,
        freeze_scope='Identity snapshot only; this script does not set filesystem locks or change permissions.',
        next_evaluation=dict(checkpoint_restore_precision='bf16', actual_inference_precision='fp32',
                             rule='Recheck each frozen best SHA before test and both fixed-subset beam runs.'),
        limitations=[
            'Platform job terminal/release status and GPU budget require separate platform evidence.',
            'Saved NDCG values/counts were checked for internal consistency; independent item-level metric and source-label replay is separate.',
            'Old validation summaries contain no SID/model hash. Intermediate best provenance is linked by clocks, binding and accepted-validation prefix, not an independent historical SID receipt.',
            'Each checkpoint carries its own SID fingerprint. No CPU nearest-centroid equality gate is imposed; representative CPU model restoration is a separate delivery check.',
            'This receipt does not evaluate test data or establish recommendation quality/cross-seed stability.'
        ])
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
