"""Inference-only compatibility of updated hard SIDs with a fixed recommender.

Reuse the feedback pilot's initial/post_tokenizer snapshots. Freeze the complete
initial recommender, replace only joint_alignment state, and score train/val in
one deterministic fully-masked context. This is not training loss replay, beam
recommendation evaluation, or an intrinsic tokenizer-quality measurement.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('frozen', 'cross_recon', 'cross_rec')
SPLITS = ('train', 'val')
TOKENIZER_PREFIX = 'joint_alignment.'


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recommender_state(state):
    """Everything except the tokenizer, including the SID embedding and heads."""
    return {name: value for name, value in state.items()
            if not name.startswith(TOKENIZER_PREFIX)}


def graft_tokenizer_state(initial, candidate, equal=None):
    """Refuse changed recommender tensors before replacing tokenizer state only.

    The equality predicate is injectable so isolation can be tested without
    importing Torch. Production uses torch.equal, plus byte-state hash guards.
    """
    if set(initial) != set(candidate):
        raise ValueError('Snapshot state keys differ from complete initial model')
    tokenizer_keys = [name for name in initial if name.startswith(TOKENIZER_PREFIX)]
    if not tokenizer_keys or not recommender_state(initial):
        raise ValueError('Both complete recommender and tokenizer state required')
    equal = equal or (lambda left, right: left == right)
    for name, value in recommender_state(initial).items():
        if not bool(equal(value, candidate[name])):
            raise ValueError(f'Post-tokenizer recommender tensor changed: {name}')
    result = dict(initial)
    result.update({name: candidate[name] for name in tokenizer_keys})
    return result


def verify_diagnostic_config(config):
    """Preserve the resolved original repo-v6 controls, not common defaults.

    The historical repo-v6 layer overrides common label_smoothing=0.1 with
    0.2. This diagnostic scores that actual original configuration unchanged.
    """
    actual = {key: config.get(key) for key in
              ('label_smoothing', 'train_batch_size', 'eval_batch_size')}
    if actual != {'label_smoothing': 0.2, 'train_batch_size': 64, 'eval_batch_size': 8}:
        raise ValueError('Retain resolved original label_smoothing=0.2, '
                         f'train_batch_size=64, eval_batch_size=8; actual={actual!r}')
    return float(actual['label_smoothing'])


def comparison_pairs(actual, reference, fp32_per_digit, original_dtype_per_digit):
    """Compare only like-precision scoring paths; never equate unlike inputs."""
    if any(set(actual) != set(values) for values in
           (reference, fp32_per_digit, original_dtype_per_digit)):
        raise ValueError('Diagnostic comparison keys must match')
    return {'explicit_fp32': (actual, fp32_per_digit),
            'original_dtype': (original_dtype_per_digit, reference)}


def diagnostic_receipt(actual, reference, fp32_per_digit, original_dtype_per_digit, dtypes):
    """Preserve precision differences, with strict guards within each path."""
    comparison_pairs(actual, reference, fp32_per_digit, original_dtype_per_digit)
    for values in (actual, reference, fp32_per_digit, original_dtype_per_digit):
        if any(len(vector) != 4 for vector in values.values()):
            raise ValueError('Diagnostic comparisons require four positions')
    def differences(left, right):
        return {key: max(abs(float(a) - float(b)) for a, b in zip(left[key], right[key]))
                for key in actual}
    return {'actual': actual, 'reference': reference,
            'fp32_per_digit': fp32_per_digit,
            'original_dtype_per_digit': original_dtype_per_digit, 'dtypes': dtypes,
            'guard_rtol': 1e-5, 'guard_atol': 1e-5,
            'guard_paths': ['explicit_fp32', 'original_dtype'],
            'cross_precision_difference_is_not_a_failure': True,
            'max_absolute_differences': {
                'flat_fp32_vs_per_digit_fp32': differences(actual, fp32_per_digit),
                'native_per_digit_vs_original': differences(original_dtype_per_digit, reference),
                'explicit_fp32_vs_original_dtype': differences(actual, original_dtype_per_digit),
                'flat_fp32_vs_original': differences(actual, reference)}}


def run(args):
    shared = load_module('_hard_compat_shared', Path(__file__).with_name('evaluate_feedback_paired.py'))
    root = args.project_root.resolve()
    raw_run_dir = root / args.run_dir if not args.run_dir.is_absolute() else args.run_dir
    run_dir = raw_run_dir.resolve()
    if run_dir.parent != root / 'outputs/pilots/stage05_feedback' or raw_run_dir.is_symlink():
        raise ValueError('Input must be a completed stage05_feedback pilot')
    output = shared.validate_output(root, args.output, args.output_base)
    original = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    complete = json.loads((run_dir / 'COMPLETE.json').read_text(encoding='utf-8'))
    if (complete['summary_sha256'] != shared.sha256(run_dir / 'summary.json')
            or original['mode'] != 'pilot' or original['status'] != 'complete'
            or not original['all_protocol_arms_complete'] or complete['arms'] != list(ARMS)
            or not original['no_test_evaluation']):
        raise ValueError('Completed three-arm formal input receipt required')
    protocol = json.loads((run_dir / 'protocol.json').read_text(encoding='utf-8'))
    provenance = json.loads((run_dir / 'provenance.json').read_text(encoding='utf-8'))
    for relative, expected in provenance['source']['source_sha256'].items():
        if shared.sha256(root / relative) != expected:
            raise ValueError(f'Historical runtime source changed: {relative}')
    for asset in provenance['assets']:
        if shared.sha256(root / asset['path']) != asset['sha256']:
            raise ValueError(f'Bound input changed: {asset["path"]}')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Only a single GPU/process is supported')
    # Set before importing Pipeline/datasets; all mutable caches are job-local.
    args.work_handle = tempfile.TemporaryDirectory(prefix='nancy_feedback_hard_compat_')
    work = Path(args.work_handle.name)
    os.environ['HF_DATASETS_CACHE'] = str(work / 'hf_datasets_cache')
    import numpy as np
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 GPU is required for the compatibility diagnostic')
    if torch.__version__ != provenance['torch'] or np.__version__ != provenance['numpy']:
        raise RuntimeError('Use the original Torch and NumPy versions')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    sys.path.insert(0, str(root / 'external/DiffGRM'))
    sys.path.insert(0, str(root))
    runtime = shared.load_module('_hard_compat_runtime', root / 'scripts/train/run_diger_noise_pilot.py')
    pilot_helper = shared.load_module('_hard_compat_protocol', root / 'scripts/checks/diger_feedback_pilot_protocol.py')
    runtime.torch, runtime.np = torch, np
    from genrec.pipeline import Pipeline
    from genrec.utils import get_config
    from datasets import disable_caching
    disable_caching()
    arm_rows = {row['arm']: row for row in original['arms']}
    if list(arm_rows) != list(ARMS):
        raise ValueError('Unexpected arm order')
    inputs = {str(run_dir / name): shared.sha256(run_dir / name)
              for name in ('summary.json', 'COMPLETE.json', 'protocol.json', 'provenance.json')}
    snapshots = {}
    official = {}
    initial_state = None
    initial_codes = None
    initial_hash = None
    frozen_recommender_hash = None
    for arm in ARMS:
        snapshots[arm] = {}
        for label in ('initial', 'post_tokenizer'):
            matching = [row for row in arm_rows[arm]['snapshots'] if row['label'] == label]
            if len(matching) != 1:
                raise ValueError(f'Exactly one {arm}/{label} snapshot required')
            row = matching[0]
            if row['recommender_epochs'] != 0 or row['recommender_updates'] != 0:
                raise ValueError('Compatibility inputs must precede Decoder recovery')
            expected_updates = 0 if label == 'initial' or arm == 'frozen' else 24
            if row['tokenizer_updates'] != expected_updates:
                raise ValueError('Unexpected tokenizer update budget')
            snapshot = run_dir / arm / 'snapshots' / label
            for name, expected in row['files'].items():
                path = snapshot / name
                if shared.sha256(path) != expected:
                    raise ValueError(f'Snapshot hash differs: {path}')
                inputs[str(path)] = expected
            state = torch.load(snapshot / 'model.pt', map_location='cpu', weights_only=False)
            if runtime.tensor_state_hash(state) != row['model_tensor_state_sha256']:
                raise ValueError('Snapshot model tensor hash differs')
            codes = np.load(snapshot / 'semantic_ids.npy', allow_pickle=False)
            if codes.shape != (12101, 4) or not np.issubdtype(codes.dtype, np.integer):
                raise ValueError('Expected 12101 four-digit integer catalog SIDs')
            if np.any(codes < 0) or np.any(codes >= 256):
                raise ValueError('Snapshot SID outside the original 256-codebook range')
            if not np.array_equal(state['joint_alignment.hard_codes'][1:].numpy(), codes):
                raise ValueError('Snapshot hard codes differ from official SID file')
            if label == 'initial':
                if initial_state is None:
                    initial_state = state
                    initial_codes = codes.tolist()
                    initial_hash = runtime.tensor_state_hash(state)
                    frozen_recommender_hash = runtime.tensor_state_hash(recommender_state(state))
                elif (runtime.tensor_state_hash(state) != initial_hash
                      or codes.tolist() != initial_codes):
                    raise ValueError('Arms do not share the exact complete initial state/SIDs')
            else:
                graft_tokenizer_state(initial_state, state, equal=torch.equal)
                if runtime.tensor_state_hash(recommender_state(state)) != frozen_recommender_hash:
                    raise ValueError('Post-tokenizer recommender bytes differ from initial')
                if arm == 'frozen' and runtime.tensor_state_hash(state) != initial_hash:
                    raise ValueError('Frozen post-tokenizer model must equal initial completely')
                official[arm] = codes.tolist()
            snapshots[arm][label] = row
            if state is not initial_state:
                del state
    output.mkdir(parents=True, exist_ok=False)
    args.owned_output = output
    shared.write_json(output / 'catalog_ids.json', {'initial': initial_codes, 'arms': official})
    started = time.perf_counter()
    base = get_config('DIFF_GRM', 'AmazonReviews2014', [str(root / path) for path in protocol['config_layers']], {})
    base = runtime.absolute_engine_paths(base, root)
    base.update({'rand_seed': protocol['seed'], 'mixed_precision': 'no',
                 'joint_recommender_checkpoint': None, 'joint_crosspse_enabled': True,
                 'joint_gradient_mode': 'diger_st', 'joint_recommender_sia_weight': 0.0,
                 'joint_tokenizer_sia_weight': 0.0, 'joint_diger_recommendation_weight': 1.0,
                 'joint_diger_assignment_temperature': protocol['assignment_temperature'],
                 'joint_diger_softmax_temperature_start': protocol['softmax_temperature'],
                 'joint_diger_softmax_temperature_end': protocol['softmax_temperature'],
                 'joint_diger_gumbel_noise_start': 0.0, 'joint_diger_gumbel_noise_end': 0.0,
                 'joint_diger_uncertainty_decay': 'cosine', 'force_regenerate_opq': False,
                 'run_id': 'feedback_hard_compatibility', 'num_proc': 1})
    runtime.verify_controlled_settings(base, protocol)
    smoothing = verify_diagnostic_config(base)
    base = runtime.configure_paths(base, output / 'diagnostics')
    cache, cache_receipts = runtime.shadow_dataset_cache(base['cache_dir'], work, base['category'])
    base['cache_dir'] = str(cache)
    with runtime.working_directory(work):
        # Pipeline's default Trainer creates optimizers. An inert explicit
        # placeholder bypasses that constructor: inference uses the model only.
        pipeline = Pipeline('DIFF_GRM', 'AmazonReviews2014', config_dict=base, trainer=object())
    if pipeline.accelerator.num_processes != 1:
        raise RuntimeError('Expected one Accelerator process')
    model = pipeline.model.to(device)
    if model.joint_alignment.tokenizer_variant != 'cross_pse':
        raise ValueError('All arms must retain the original Cross-PSE module')
    expected_counts = {'train': 128 if args.smoke else 105668, 'val': 16 if args.smoke else 22363}
    loaders = {}
    for split, dataset_size in (('train', 105668), ('val', 22363)):
        if len(pipeline.tokenized_datasets[split]) != dataset_size:
            raise ValueError(f'Unexpected {split} sample count')
        loaders[split] = torch.utils.data.DataLoader(
            pipeline.tokenized_datasets[split], batch_size=64 if split == 'train' else 8,
            shuffle=False, collate_fn=pipeline.tokenizer.collate_fn[split], num_workers=0,
            generator=torch.Generator().manual_seed(pilot_helper.phase_seed(protocol['seed'], 'evaluation', 0)))
    summary = {'analysis_protocol': 'feedback_fixed_recommender_hard_compatibility_v1',
               'status': 'running', 'inference_only': True, 'no_optimizer': True,
               'no_backward': True, 'no_generation': True, 'no_test_evaluation': True,
               'mask_context': 'all four target positions masked; zero decoder input; use_cache=False',
               'score_precision': 'explicit FP32 logits input to cross_entropy; BF16 model forward',
               'historical_loss_exact_replay': False,
               'score_guard': 'FP32 flatten vs FP32 per-digit; original-dtype per-digit vs original full-mask diagnostic; unlike input precisions only recorded',
               'anchored_label_diagnostic': 'same logits scored against common initial target SID; not official evaluation',
               'interpretation': 'instantaneous compatibility with the fixed initial recommender, not intrinsic encoding quality',
               'input_run': str(run_dir), 'checkpoint_label': 'post_tokenizer',
               'fixed_recommender': 'complete initial non-joint_alignment state, including encoder/decoder/SID embeddings/heads',
               'initial_model_tensor_sha256': initial_hash,
               'fixed_recommender_tensor_sha256': frozen_recommender_hash,
               'smoke_not_for_research': args.smoke, 'label_smoothing': smoothing,
               'batch_sizes': {'train': 64, 'val': 8}, 'expected_counts': expected_counts,
               'executed_script_sha256': shared.sha256(__file__),
               'executed_analysis_sha256': shared.sha256(Path(__file__).with_name('feedback_hard_compat_analysis.py')),
               'executed_code_git_head': subprocess.check_output(
                   ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip(),
               'source': pilot_helper.source_receipt(root), 'inputs': inputs,
               'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'numpy': np.__version__,
               'isolated_cache': cache_receipts, 'arms': {}}
    records_by_split = {split: {} for split in SPLITS}
    initial_tensor = torch.tensor(initial_codes, dtype=torch.long, device=device)
    try:
        for arm in ARMS:
            arm_dir = output / arm
            arm_dir.mkdir()
            candidate = torch.load(run_dir / arm / 'snapshots/post_tokenizer/model.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(graft_tokenizer_state(initial_state, candidate, equal=torch.equal), strict=True)
            model.sync_joint_tokenizer_mapping()
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
                raise RuntimeError('All parameters and model modes must be frozen')
            if model.joint_alignment.hard_codes[1:].cpu().tolist() != official[arm]:
                raise ValueError('Loaded model hard codes differ from official snapshot')
            before = runtime.tensor_state_hash(model.state_dict())
            if before != snapshots[arm]['post_tokenizer']['model_tensor_state_sha256']:
                raise ValueError('Grafted full state differs from the guarded post-tokenizer snapshot')
            if runtime.tensor_state_hash(recommender_state(model.state_dict())) != frozen_recommender_hash:
                raise RuntimeError('Fixed recommender state changed before inference')
            arm_summary = {'post_tokenizer_tensor_sha256': before, 'splits': {}}
            del candidate
            for split in SPLITS:
                rows = []
                rng_before = runtime.rng_state_hash(device)
                split_started = time.perf_counter()
                diagnostic_check = None
                with gzip.open(arm_dir / f'{split}.records.jsonl.gz', 'xt', encoding='utf-8') as destination:
                    with torch.inference_mode():
                        for batch_index, batch in enumerate(loaders[split]):
                            if args.smoke and batch_index >= 2:
                                break
                            batch_rng = runtime.rng_state_hash(device)
                            prepared = model.prepare_joint_batch(batch, training=False, differentiable=False)
                            labels = prepared['labels']
                            targets = prepared['target_item_id']
                            initial_labels = initial_tensor[targets - 1]
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                hidden = model(prepared, return_loss=False).hidden_states
                                logits = model.forward_decoder_only({
                                    'decoder_input_ids': torch.zeros_like(labels), 'encoder_hidden': hidden,
                                    'mask_positions': torch.ones_like(labels, dtype=torch.float32)},
                                    return_loss=False, digit=None, use_cache=False).logits
                                flat = logits.float().reshape(-1, 256)
                                own_nll = F.cross_entropy(flat, labels.reshape(-1), reduction='none').reshape(-1, 4)
                                own_loss = F.cross_entropy(flat, labels.reshape(-1), reduction='none',
                                                          label_smoothing=smoothing).reshape(-1, 4)
                                anchor_nll = F.cross_entropy(flat, initial_labels.reshape(-1), reduction='none').reshape(-1, 4)
                                anchor_loss = F.cross_entropy(flat, initial_labels.reshape(-1), reduction='none',
                                                             label_smoothing=smoothing).reshape(-1, 4)
                                predicted = logits.argmax(dim=-1)
                                correct = predicted.eq(labels).long()
                                if batch_index == 0:
                                    # Output FP32 does not prove the original CE
                                    # used the same input-precision path. Verify
                                    # explicit FP32 and original dtype separately.
                                    digit_nll = [F.cross_entropy(logits[:, digit, :], labels[:, digit], reduction='none')
                                                 for digit in range(4)]
                                    digit_loss = [F.cross_entropy(logits[:, digit, :], labels[:, digit], reduction='none',
                                                                 label_smoothing=smoothing) for digit in range(4)]
                                    same_logits = {'position_nll_sums': torch.stack([value.sum() for value in digit_nll]),
                                                   'position_loss_sums': torch.stack([value.sum() for value in digit_loss]),
                                                   'position_correct_counts': correct.sum(0)}
                                    digit_fp32_nll = [F.cross_entropy(logits[:, digit, :].float(), labels[:, digit], reduction='none')
                                                       for digit in range(4)]
                                    digit_fp32_loss = [F.cross_entropy(logits[:, digit, :].float(), labels[:, digit], reduction='none',
                                                                     label_smoothing=smoothing) for digit in range(4)]
                                    same_fp32 = {'position_nll_sums': torch.stack([value.sum() for value in digit_fp32_nll]),
                                                 'position_loss_sums': torch.stack([value.sum() for value in digit_fp32_loss]),
                                                 'position_correct_counts': correct.sum(0)}
                                    reference = model.full_mask_diagnostics(hidden, labels)
                                    comparisons = {'position_nll_sums': own_nll.sum(0),
                                                   'position_loss_sums': own_loss.sum(0),
                                                   'position_correct_counts': correct.sum(0)}
                                    # The original diagnostic also returns counts
                                    # and view counts. Compare only scored fields.
                                    reference = {key: reference[key] for key in comparisons}
                                    comparison_receipt = diagnostic_receipt(
                                        {key: value.detach().cpu().tolist() for key, value in comparisons.items()},
                                        {key: reference[key].detach().cpu().tolist() for key in comparisons},
                                        {key: value.detach().cpu().tolist() for key, value in same_fp32.items()},
                                        {key: value.detach().cpu().tolist() for key, value in same_logits.items()},
                                        {'logits': str(logits.dtype), 'flattened_nll': str(own_nll.dtype),
                                         'flattened_loss': str(own_loss.dtype),
                                         'per_digit_nll': [str(value.dtype) for value in digit_nll],
                                         'per_digit_loss': [str(value.dtype) for value in digit_loss],
                                         'per_digit_fp32_nll': [str(value.dtype) for value in digit_fp32_nll],
                                         'per_digit_fp32_loss': [str(value.dtype) for value in digit_fp32_loss],
                                         'original_sums': {key: str(reference[key].dtype) for key in comparisons}})
                                    comparison_receipt.update(arm=arm, split=split, label_smoothing=smoothing,
                                        logits_shape=list(logits.shape), global_autocast_enabled=torch.is_autocast_enabled('cuda'),
                                        training_modules=[name for name, module in model.named_modules() if module.training])
                                    comparison_path = arm_dir / f'{split}.full_mask_check.json'
                                    shared.write_json(comparison_path, comparison_receipt)
                                    print('[HARD COMPAT FIRST BATCH] ' + json.dumps(comparison_receipt, sort_keys=True), flush=True)
                                    for guard, (left, right) in comparison_pairs(
                                            comparisons, reference, same_fp32, same_logits).items():
                                        for key, actual in left.items():
                                            if not torch.allclose(actual.float(), right[key].float(), rtol=1e-5, atol=1e-5):
                                                raise RuntimeError(f'Same-precision full-mask diagnostic mismatch: {guard}/{key}; details={comparison_path}')
                                    diagnostic_check = {key: float((actual.float() - reference[key].float()).abs().max())
                                                        for key, actual in comparisons.items()}
                            if runtime.rng_state_hash(device) != batch_rng:
                                raise RuntimeError('Inference consumed a global RNG stream')
                            vectors = {name: value.cpu().tolist() for name, value in (
                                ('target_sid', labels), ('predicted_sid', predicted), ('position_nll', own_nll),
                                ('position_loss', own_loss), ('position_correct', correct),
                                ('initial_label_position_nll', anchor_nll), ('initial_label_position_loss', anchor_loss))}
                            history = batch['history_item_ids'].cpu().tolist()
                            target_items = targets.cpu().tolist()
                            for index, target in enumerate(target_items):
                                target = int(target)
                                row = {'sample_index': len(rows), 'target_item_id': target,
                                       'history_item_ids': [int(item) for item in history[index] if int(item) != 0],
                                       **{key: values[index] for key, values in vectors.items()}}
                                if row['target_sid'] != official[arm][target - 1]:
                                    raise ValueError('Prepared target label differs from official hard SID')
                                if any(not math.isfinite(float(value)) for key in (
                                        'position_nll', 'position_loss', 'initial_label_position_nll',
                                        'initial_label_position_loss') for value in row[key]):
                                    raise FloatingPointError('Non-finite per-position score')
                                destination.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                                rows.append(row)
                            if batch_index % 200 == 0:
                                print(f'[HARD COMPAT] arm={arm} split={split} batch={batch_index + 1}/{len(loaders[split])} examples={len(rows)}', flush=True)
                if len(rows) != expected_counts[split]:
                    raise ValueError(f'Incomplete {arm}/{split} capture: {len(rows)}')
                if runtime.rng_state_hash(device) != rng_before:
                    raise RuntimeError('Data loading/inference changed a global RNG stream')
                records_by_split[split][arm] = rows
                arm_summary['splits'][split] = {'examples': len(rows), 'rng_unchanged': True,
                    'full_mask_reference_max_absolute_difference': diagnostic_check,
                    'records_sha256': shared.sha256(arm_dir / f'{split}.records.jsonl.gz'),
                    'elapsed_seconds': time.perf_counter() - split_started}
            after = runtime.tensor_state_hash(model.state_dict())
            if after != before or runtime.tensor_state_hash(recommender_state(model.state_dict())) != frozen_recommender_hash:
                raise RuntimeError('Model parameters/buffers changed during inference')
            arm_summary.update(model_unchanged=True, fixed_recommender_unchanged=True)
            summary['arms'][arm] = arm_summary
            shared.write_json(arm_dir / 'receipt.json', arm_summary)
            torch.cuda.empty_cache()
        analysis_helper = shared.load_module('_hard_compat_analysis', Path(__file__).with_name('feedback_hard_compat_analysis.py'))
        result = analysis_helper.analyze_records(records_by_split, initial_codes, official, expected_counts, label_smoothing=smoothing)
        shared.write_json(output / 'compatibility_summary.json', result)
        report = analysis_helper.markdown_report(result)
        if args.smoke:
            report = '# SMOKE ONLY: not research results\n\n' + report
        with (output / 'README.md').open('x', encoding='utf-8') as handle:
            handle.write(report)
    finally:
        pipeline.accelerator.end_training()
    for path, expected in inputs.items():
        if shared.sha256(path) != expected:
            raise RuntimeError(f'Read-only snapshot input changed: {path}')
    for relative, expected in provenance['source']['source_sha256'].items():
        if shared.sha256(root / relative) != expected:
            raise RuntimeError(f'Historical runtime changed during inference: {relative}')
    for asset in provenance['assets']:
        if shared.sha256(root / asset['path']) != asset['sha256']:
            raise RuntimeError('Historical asset changed during inference')
    summary.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                   inputs_unchanged=True, all_fixed_recommender_states_unchanged=True,
                   all_global_rng_states_unchanged=True)
    shared.write_json(output / 'summary.json', summary)
    shared.write_json(output / 'COMPLETE.json', {'summary_sha256': shared.sha256(output / 'summary.json'),
                      'compatibility_summary_sha256': shared.sha256(output / 'compatibility_summary.json'), 'smoke': args.smoke})
    print(f'[HARD COMPAT DONE] {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--output-base', type=Path, help='Explicitly approved own SSD analysis parent only')
    parser.add_argument('--smoke', action='store_true', help='First two batches of each split, not research results')
    args = parser.parse_args(argv)
    args.owned_output, args.work_handle = None, None
    try:
        run(args)
    except Exception as exc:
        if args.owned_output is not None:
            shared = load_module('_hard_compat_failure', Path(__file__).with_name('evaluate_feedback_paired.py'))
            shared.write_json(args.owned_output / 'FAILED.json', {'error': type(exc).__name__, 'message': str(exc), 'complete': False})
        raise
    finally:
        if args.work_handle is not None:
            args.work_handle.cleanup()


if __name__ == '__main__':
    main()
