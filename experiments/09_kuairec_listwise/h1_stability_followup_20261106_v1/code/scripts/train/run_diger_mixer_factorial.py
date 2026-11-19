"""Strict shared-start Mixer x tokenizer-update factorial pilot.

Four NEW arms share effective codebooks, official SID, and the complete initial
recommender. Joint arms receive one 24-update tokenizer block, then every arm
gets three equal Decoder recovery epochs. This is not historical exact resume,
not a full multi-cycle joint-training experiment, and never selects on test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('no_mixer_frozen', 'no_mixer_joint', 'mixer_frozen', 'mixer_joint')
DEFAULT_CONFIG = ROOT / 'configs/05_crosspse_joint/pilots/diger_mixer_factorial_seed2024.json'
GLOBAL_TRAIN_PARENT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training/outputs/pilots/stage05_mixer_factorial')
HDD_PROJECT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recommender_state(state):
    return {name: value for name, value in state.items()
            if not name.startswith('joint_alignment.')}


def validate_output(root, candidate, output_base=None):
    root = Path(root).resolve()
    parent = root / 'outputs/pilots/stage05_mixer_factorial'
    if output_base is not None:
        if root != HDD_PROJECT or Path(output_base).resolve() != GLOBAL_TRAIN_PARENT:
            raise ValueError('Explicit alternate must be the approved own GLOBAL TRAIN parent, not project storage or analysis')
        parent = GLOBAL_TRAIN_PARENT
    raw = Path(candidate) if Path(candidate).is_absolute() else root / candidate
    resolved = raw.resolve()
    if resolved.parent != parent or raw.is_symlink():
        raise ValueError(f'Use one new direct child of {parent}')
    for path in (parent, *parent.parents):
        if path.is_symlink():
            raise ValueError('Output parent must not be a symlink')
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def verify_resolved_controls(config):
    expected = {'label_smoothing': 0.2, 'train_batch_size': 64,
                'eval_batch_size': 8, 'gradient_accumulation_steps': 16}
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f'Retain resolved historical execution controls: expected={expected!r}, actual={actual!r}')


def configure_variant(model, settings):
    enabled = settings.get('mixer_enabled')
    if type(enabled) is not bool:
        raise ValueError('mixer_enabled must be explicitly Boolean')
    variant = 'cross_pse' if enabled else 'refined_pse'
    if settings.get('tokenizer_variant') != variant:
        raise ValueError('tokenizer_variant must agree with the Mixer switch')
    model.joint_alignment.mixer.mixer_enabled = enabled
    model.joint_alignment.tokenizer_variant = variant
    model.config['crosspse_mixer_enabled'] = enabled


def rebase_common_mixer(model, effective_codebooks, init_seed, mixer_class=None):
    """Preserve effective geometry while replacing the old learned Mixer.

    initial_codebooks is nonpersistent in the engine. Constructing the new
    module from E0 also binds its anchor to E0; loading an old state alone would
    not establish that anchor. Fresh out-projection zeroing is not a zero gate.
    """
    import torch
    if mixer_class is None:
        from genrec.models.DIFF_GRM.cross_pse import CrossPSECodebookMixer
        mixer_class = CrossPSECodebookMixer
    if not torch.is_tensor(effective_codebooks) or effective_codebooks.shape != (4, 256, 64):
        raise ValueError('Shared effective codebooks must be a [4,256,64] Tensor')
    base = effective_codebooks.detach().cpu().float().clone()
    if not bool(torch.isfinite(base).all()):
        raise FloatingPointError('Nonfinite shared codebooks')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(init_seed))
        mixer = mixer_class(base, n_heads=4, dropout=0.0, attention_scope='full',
                            residual_scale_init=1.0, learn_residual_scale=False,
                            zero_init_attention_output=True, train_base_codebooks=True,
                            mixer_enabled=True)
    model.joint_alignment.mixer = mixer.to(model.joint_alignment.item_slots.device)
    model.joint_alignment.set_recommender_cache(False)
    return model.joint_alignment.mixer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--output-base', type=Path, help='Explicit own GLOBAL TRAIN parent only')
    parser.add_argument('--arm', choices=('all', *ARMS), default='all')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true', help='Only with --smoke; not a GPU acceptance result')
    return parser.parse_args(argv)


def run(args):
    sys.dont_write_bytecode = True
    root = args.project_root.resolve()
    output = validate_output(root, args.output, args.output_base)
    helper_path = Path(__file__).resolve().parents[1] / 'checks/diger_mixer_factorial_protocol.py'
    helper = load_module('_executed_mixer_factorial_protocol', helper_path)
    config_path = args.config.resolve() if args.config.is_absolute() else (root / args.config).resolve()
    protocol = helper.load_protocol(config_path)
    available = list(helper.arm_names(protocol))
    if available != list(ARMS):
        raise ValueError('Protocol must retain all four ordered factorial arms')
    selected = available if args.arm == 'all' else [args.arm]
    if args.allow_cpu and not args.smoke:
        raise ValueError('--allow-cpu requires explicit --smoke')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use one process and one GPU, not torchrun')
    asset_receipts = helper.verify_assets(root, protocol)
    common_receipt = helper.verify_common_snapshot(root, protocol)
    os.environ.setdefault('GIT_OPTIONAL_LOCKS', '0')
    args.work_handle = tempfile.TemporaryDirectory(prefix='nancy_mixer_factorial_')
    work = Path(args.work_handle.name)
    # Set before importing datasets/Pipeline. The HDD stays read-only.
    os.environ['HF_DATASETS_CACHE'] = str(work / 'hf_datasets_cache')
    import numpy as np
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not (args.smoke and args.allow_cpu):
        raise RuntimeError('Formal factorial pilot requires GPU')
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU support required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(root / 'external/DiffGRM'))
    sys.path.insert(0, str(root))
    runtime = load_module('_factorial_matched_runtime', root / 'scripts/train/run_diger_noise_pilot.py')
    runtime.torch, runtime.np = torch, np
    runtime.phase_seed, runtime.loss_divisor = helper.phase_seed, helper.loss_divisor
    from genrec.pipeline import Pipeline
    from genrec.utils import get_config
    from genrec.models.DIFF_GRM.trainer import DIFF_GRMTrainer
    from datasets import disable_caching
    disable_caching()
    snapshot_dir = Path(common_receipt['snapshot_dir'])
    original_state = torch.load(snapshot_dir / 'model.pt', map_location='cpu', weights_only=False)
    if runtime.tensor_state_hash(original_state) != common_receipt['model_tensor_state_sha256']:
        raise ValueError('Bound bootstrap complete model tensor hash mismatch')
    initial_codes = np.load(snapshot_dir / 'semantic_ids.npy', allow_pickle=False)
    effective = np.load(snapshot_dir / 'crossed_codebooks.npy', allow_pickle=False)
    if (initial_codes.shape != (12101, 4) or effective.shape != (4, 256, 64)
            or effective.dtype != np.float32 or not np.issubdtype(initial_codes.dtype, np.integer)):
        raise ValueError('Unexpected shared catalog/codebook geometry')
    if np.any(initial_codes < 0) or np.any(initial_codes >= 256):
        raise ValueError('Official common SID outside the original 256-codebook range')
    if not np.array_equal(original_state['joint_alignment.hard_codes'][1:].numpy(), initial_codes):
        raise ValueError('Bound bootstrap hard codes differ from official SID')
    if (int(original_state['joint_alignment.diger_optimizer_step']) != 0
            or int(original_state['joint_alignment.diger_total_optimizer_steps']) != 60):
        raise ValueError('Bootstrap must be before any tokenizer updates with the original schedule denominator')
    base = get_config('DIFF_GRM', 'AmazonReviews2014', [str(root / p) for p in protocol['config_layers']], {})
    base = runtime.absolute_engine_paths(base, root)
    base.update({'rand_seed': protocol['seed'], 'mixed_precision': 'no',
                 'joint_recommender_checkpoint': None, 'joint_crosspse_enabled': True,
                 'joint_gradient_mode': 'diger_st', 'crosspse_mixer_enabled': True,
                 'joint_recommender_sia_weight': 0.0, 'joint_tokenizer_sia_weight': 0.0,
                 'joint_diger_recommendation_weight': 1.0,
                 'joint_diger_assignment_temperature': protocol['assignment_temperature'],
                 'joint_diger_softmax_temperature_start': protocol['softmax_temperature'],
                 'joint_diger_softmax_temperature_end': protocol['softmax_temperature'],
                 'joint_diger_gumbel_noise_start': 0.0, 'joint_diger_gumbel_noise_end': 0.0,
                 'joint_diger_uncertainty_decay': 'cosine', 'force_regenerate_opq': False,
                 'run_id': protocol['protocol_id'], 'num_proc': 1})
    runtime.verify_controlled_settings(base, protocol)
    verify_resolved_controls(base)
    if base.get('evaluation_level') != 'sid' or base.get('eval_expand_sid_to_items', False):
        raise ValueError('Retain historical SID-level evaluation, not strict item-level')
    if args.smoke:
        base['train_batch_size'] = int(protocol['smoke'].get('train_batch_size', 2))
        base['eval_batch_size'] = int(protocol['smoke'].get('eval_batch_size', 2))
        base['max_eval_batches'] = int(protocol['smoke']['eval_batches'])
    else:
        base['max_eval_batches'] = None
    output.mkdir(parents=True, exist_ok=False)
    args.owned_output = output
    runtime.write_json(output / 'protocol.json', protocol)
    started = time.perf_counter()
    base = runtime.configure_paths(base, output / 'common')
    shadow_cache, cache_receipts = runtime.shadow_dataset_cache(base['cache_dir'], work, base['category'])
    base['cache_dir'] = str(shadow_cache)
    with runtime.working_directory(work):
        pipeline = Pipeline('DIFF_GRM', 'AmazonReviews2014', config_dict=base, trainer=object())
    if pipeline.accelerator.num_processes != 1:
        raise RuntimeError('Only one Accelerator process supported')
    model = pipeline.model.to(device)
    model.load_state_dict(original_state, strict=True)
    model.eval()
    original_rec_hash = runtime.tensor_state_hash(recommender_state(original_state))
    expected_books = torch.from_numpy(effective).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        loaded_books, _ = model.joint_alignment.mixer.crossed_codebooks(return_attention=False)
    if not torch.equal(loaded_books, expected_books):
        raise RuntimeError('Execution-backend bootstrap effective codebooks differ from bound snapshot')
    rebase_common_mixer(model, expected_books, protocol['mixer_init_seed'])
    model.eval()
    model.sync_joint_tokenizer_mapping()
    common_state = runtime.cpu_state(model)
    common_hash = runtime.tensor_state_hash(common_state)
    if runtime.tensor_state_hash(recommender_state(common_state)) != original_rec_hash:
        raise RuntimeError('Rebasing changed the complete initial recommender')
    if not torch.equal(model.joint_alignment.mixer.initial_codebooks, expected_books):
        raise RuntimeError('Nonpersistent reconstruction anchor must equal shared E0')
    bootstrap_proofs = {}
    for arm in available:
        settings = helper.arm_settings(protocol, arm, smoke=args.smoke)
        model.load_state_dict(common_state, strict=True)
        configure_variant(model, settings)
        model.eval()
        with torch.no_grad():
            books, _ = model.joint_alignment.mixer.crossed_codebooks(return_attention=False)
        if not torch.equal(books, expected_books):
            raise RuntimeError(f'{arm}: initial effective codebooks not equal to shared E0')
        # Refresh is permitted in this training pilot, but it must prove that
        # module rebasing did not alter the deployed starting assignment.
        model.refresh_joint_codes()
        if (not np.array_equal(model.joint_alignment.hard_codes[1:].cpu().numpy(), initial_codes)
                or runtime.tensor_state_hash(model.state_dict()) != common_hash):
            raise RuntimeError(f'{arm}: initial nearest SID/full state differs from common state')
        bootstrap_proofs[arm] = {'mixer_enabled': settings['mixer_enabled'],
            'tokenizer_variant': settings['tokenizer_variant'], 'complete_tensor_state_sha256': common_hash,
            'recommender_tensor_state_sha256': original_rec_hash, 'effective_codebooks_equal': True,
            'official_sid_and_fresh_nearest_equal': True, 'anchor_equal_E0': True}
    # Store one common complete initial snapshot, not four duplicate copies.
    model.load_state_dict(common_state, strict=True)
    configure_variant(model, helper.arm_settings(protocol, 'mixer_frozen', smoke=args.smoke))
    model.eval()
    model.sync_joint_tokenizer_mapping()
    sequence_path = shadow_cache / 'AmazonReviews2014' / base['category'] / 'processed/all_item_seqs.json'
    sequences = json.loads(sequence_path.read_text(encoding='utf-8'))
    for split, size in (('train', 105668), ('val', 22363)):
        if len(pipeline.tokenized_datasets[split]) != size:
            raise ValueError(f'Unexpected {split} dataset size')
    val_loader = torch.utils.data.DataLoader(pipeline.tokenized_datasets['val'],
        batch_size=base['eval_batch_size'], shuffle=False, collate_fn=pipeline.tokenizer.collate_fn['val'],
        generator=torch.Generator().manual_seed(helper.phase_seed(protocol['seed'], 'evaluation', 0)), num_workers=0)
    val_loader = pipeline.accelerator.prepare_data_loader(val_loader)
    runtime.set_phase_seed(helper.phase_seed(protocol['seed'], 'evaluation', 0))
    common_snapshot = runtime.snapshot(pipeline, output / 'common', 'initial',
        {'recommender': None, 'tokenizer': None}, {'recommender_epochs': 0, 'recommender_updates': 0,
        'tokenizer_updates': 0}, initial_codes, sequences, device)
    initialization = {'bootstrap_snapshot': common_receipt, 'initial_controls': bootstrap_proofs,
        'initial_model_tensor_sha256': common_hash, 'initial_recommender_tensor_sha256': original_rec_hash,
        'E0_array_sha256': hashlib.sha256(effective.tobytes()).hexdigest(),
        'module_constructor': {'n_heads': 4, 'dropout': 0.0, 'attention_scope': 'full',
            'residual_scale_init': 1.0, 'learn_residual_scale': False,
            'zero_init_attention_output': True, 'train_base_codebooks': True},
        'mixer_init_seed': protocol['mixer_init_seed'], 'anchor_nonpersistent': True,
        'resume_recipe': 'Reconstruct the Mixer from common/snapshots/initial/crossed_codebooks.npy (E0) with these constructor controls; configure the arm; strict-load complete model.pt plus optimizer/RNG state.',
        'not_historical_exact_resume': True,
        'interpretation': 'Common learned effective encoding rebased into no-Mixer/Mixer variants; not Original PSE nor the historical Refined/Cross comparison.'}
    runtime.write_json(output / 'common_initialization.json', initialization)
    provenance = {'mode': 'smoke' if args.smoke else 'pilot', 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'source': helper.source_receipt(root), 'assets': asset_receipts,
        'common_snapshot': common_receipt, 'executed_runner': str(Path(__file__).resolve()),
        'runner_sha256': runtime.file_hash(__file__), 'executed_helper': str(helper_path),
        'executed_helper_sha256': runtime.file_hash(helper_path), 'executed_config': str(config_path),
        'executed_config_sha256': runtime.file_hash(config_path),
        'executed_code_git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'fresh_optimizer_pilot_not_historical_exact_resume': True,
        'smoke_not_for_metric_comparison': args.smoke, 'shadow_cache': cache_receipts,
        'raw_links_not_standalone_backup': True, 'tokenizer_forward_precision': 'float32_tf32_disabled',
        'decoder_forward_precision': 'bf16' if device.type == 'cuda' else 'float32',
        'label_smoothing': pipeline.config['label_smoothing'], 'no_test_evaluation': True,
        'recovery_rng': 'common logical Decoder/data seeds; common validation stage seeds and private DataLoader generator'}
    runtime.write_json(output / 'provenance.json', provenance)
    results = []
    frozen_final = {}
    try:
        for arm in selected:
            arm_dir = output / arm
            arm_dir.mkdir()
            settings = helper.arm_settings(protocol, arm, smoke=args.smoke)
            model.load_state_dict(common_state, strict=True)
            configure_variant(model, settings)
            model.sync_joint_tokenizer_mapping()
            if runtime.tensor_state_hash(model.state_dict()) != common_hash:
                raise RuntimeError('Arm complete initial tensor state differs')
            runtime.configure_arm(pipeline, settings)
            pipeline.config.update(runtime.configure_paths(pipeline.config, output, arm))
            pipeline.config['run_id'] = protocol['protocol_id'] + '_' + arm
            with runtime.working_directory(work):
                pipeline.trainer = DIFF_GRMTrainer(pipeline.config, model, pipeline.tokenizer)
            model.set_joint_phase('recommender')
            rec_optimizer = torch.optim.AdamW(list(model.joint_recommender_parameters()),
                lr=protocol['decoder_lr'], weight_decay=protocol['decoder_weight_decay'])
            groups = []
            if settings['tokenizer_batches']:
                for name, parameters in model.joint_tokenizer_parameter_groups():
                    if parameters:
                        groups.append({'params': parameters,
                            'lr': base['joint_codebook_lr'] if name == 'joint_codebook' else base['joint_mixer_lr'],
                            'weight_decay': base['joint_weight_decay'], 'group_name': name})
            if settings['tokenizer_batches'] and not groups:
                raise RuntimeError('Joint arm has no active tokenizer parameter groups')
            tok_optimizer = torch.optim.AdamW(groups) if groups else None
            optimizers = {'recommender': rec_optimizer, 'tokenizer': tok_optimizer}
            markers = {'recommender_epochs': 0, 'recommender_updates': 0, 'tokenizer_updates': 0}
            result = {'arm': arm, 'settings': settings, 'shared_initial_snapshot': 'common/snapshots/initial',
                      'initial_model_tensor_sha256': common_hash, 'training': [], 'snapshots': [],
                      'validation': [], 'tokenizer_probes': [], 'optimizer_groups': [
                          {'group_name': group['group_name'], 'lr': group['lr'],
                           'parameter_tensors': len(group['params']),
                           'parameter_elements': sum(parameter.numel() for parameter in group['params'])}
                          for group in groups]}
            runtime.write_json(arm_dir / 'settings.json', settings)
            runtime.write_json(arm_dir / 'resolved_pipeline_config.json', {
                key: str(value) if key in ('device', 'accelerator') else value for key, value in pipeline.config.items()})
            runtime.set_phase_seed(helper.phase_seed(protocol['seed'], 'evaluation', 0))
            result['validation'].append(runtime.validate_snapshot(pipeline, val_loader, 'initial', 0, device))
            if tok_optimizer is not None:
                probes = settings.get('tokenizer_probe_updates', [])
                def observe_update(updates, batches):
                    if updates in probes:
                        result['tokenizer_probes'].append(runtime.tokenizer_probe(
                            pipeline, arm_dir, updates, batches, initial_codes, sequences, device))
                result['training'].append(runtime.train_phase(pipeline, tok_optimizer, 'tokenizer',
                    settings['tokenizer_batches'], 0, protocol, device, base['train_batch_size'],
                    protocol['gradient_accumulation_steps'],
                    gumbel_seed=helper.phase_seed(protocol['gumbel_seed'], 'gumbel', 0),
                    feedback_weight=1.0, update_callback=observe_update if probes else None))
                markers['tokenizer_updates'] = result['training'][-1]['optimizer_updates']
                if [row['optimizer_updates'] for row in result['tokenizer_probes']] != probes:
                    raise RuntimeError('Joint tokenizer diagnostic probe budget incomplete')
                model.refresh_joint_codes()
            if runtime.tensor_state_hash(recommender_state(model.state_dict())) != original_rec_hash:
                raise RuntimeError('Tokenizer block changed recommender parameters/buffers')
            interaction_keys = [name for name in common_state
                if name.startswith('joint_alignment.mixer.')
                and name != 'joint_alignment.mixer.base_codebooks']
            current_state = model.state_dict()
            changed_interaction = [name for name in interaction_keys
                if not torch.equal(current_state[name].detach().cpu(), common_state[name])]
            result['tokenizer_update_integrity'] = {
                'changed_interaction_tensor_names': changed_interaction,
                'base_codebooks_changed': not torch.equal(
                    current_state['joint_alignment.mixer.base_codebooks'].detach().cpu(),
                    common_state['joint_alignment.mixer.base_codebooks']),
                'complete_recommender_unchanged': True}
            if not settings['mixer_enabled'] and changed_interaction:
                raise RuntimeError('Disabled no-Mixer interaction tensors changed')
            if settings['mixer_enabled'] and settings['tokenizer_batches'] and not changed_interaction:
                raise RuntimeError('Mixer Joint arm did not update any interaction tensor')
            runtime.set_phase_seed(helper.phase_seed(protocol['seed'], 'evaluation', 1))
            result['snapshots'].append(runtime.snapshot(pipeline, arm_dir, 'post_tokenizer',
                optimizers, markers.copy(), initial_codes, sequences, device))
            if tok_optimizer is not None:
                result['validation'].append(runtime.validate_snapshot(pipeline, val_loader, 'post_tokenizer', 1, device))
            else:
                result['validation'].append({'label': 'post_tokenizer',
                    'metrics': dict(result['validation'][0]['metrics']), 'unchanged_control_reuses_initial_evaluation': True})
            fixed_tokenizer_hash = runtime.tensor_state_hash(model.joint_alignment.state_dict())
            for epoch in range(1, int(settings['decoder_epochs']) + 1):
                row = runtime.train_phase(pipeline, rec_optimizer, 'recommender',
                    settings['decoder_batch_limit'], epoch, protocol, device,
                    base['train_batch_size'], protocol['gradient_accumulation_steps'])
                result['training'].append(row)
                markers['recommender_epochs'] = epoch
                markers['recommender_updates'] += row['optimizer_updates']
                if runtime.tensor_state_hash(model.joint_alignment.state_dict()) != fixed_tokenizer_hash:
                    raise RuntimeError('Tokenizer changed during the equal Decoder recovery')
                label = f'decoder_{epoch}'
                runtime.set_phase_seed(helper.phase_seed(protocol['seed'], 'evaluation', epoch + 1))
                if epoch == settings['decoder_epochs']:
                    result['snapshots'].append(runtime.snapshot(pipeline, arm_dir, label,
                        optimizers, markers.copy(), initial_codes, sequences, device))
                result['validation'].append(runtime.validate_snapshot(pipeline, val_loader, label, epoch + 1, device))
            result['final_snapshot'] = f'snapshots/decoder_{settings["decoder_epochs"]}'
            result['final_model_tensor_sha256'] = runtime.tensor_state_hash(model.state_dict())
            if not settings['tokenizer_batches']:
                frozen_final[arm] = result
            runtime.write_json(arm_dir / 'metrics.json', result)
            results.append(result)
            del rec_optimizer, tok_optimizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        # Shared data order and equal logical update budget, never wall-clock
        # epoch labels, are the controls for the factorial comparison.
        controls = ('logical_index', 'batches', 'optimizer_updates', 'phase_seed', 'data_seed', 'data_order_sha256')
        recovery = [[row for row in result['training'] if row['phase'] == 'recommender'] for result in results]
        if not args.smoke and any(len(rows) != 3 or any(
                row['batches'] != 1652 or row['optimizer_updates'] != 104 for row in rows) for rows in recovery):
            raise RuntimeError('Formal recovery must complete 3 x 1652 batches and 3 x 104 updates')
        for rows in recovery[1:]:
            if len(rows) != len(recovery[0]) or any(
                    any(left[key] != right[key] for key in controls) for left, right in zip(recovery[0], rows)):
                raise RuntimeError('Four-arm Decoder data/order/update controls differ')
        joint_rows = [row for result in results for row in result['training'] if row['phase'] == 'tokenizer']
        if len(joint_rows) == 2 and any(joint_rows[0][key] != joint_rows[1][key] for key in controls):
            raise RuntimeError('Joint arms have unequal tokenizer data/update controls')
        frozen_gate = None
        if set(frozen_final) == {'no_mixer_frozen', 'mixer_frozen'}:
            left, right = frozen_final['no_mixer_frozen'], frozen_final['mixer_frozen']
            if left['final_model_tensor_sha256'] != right['final_model_tensor_sha256']:
                raise RuntimeError('Frozen negative-control final complete tensors differ')
            for first, second in zip(left['validation'], right['validation']):
                if first['label'] != second['label'] or first['metrics'] != second['metrics']:
                    raise RuntimeError('Frozen negative-control validation metrics differ')
            frozen_gate = {'complete_final_tensor_hash_equal': True, 'all_validation_metrics_equal': True,
                           'all_recovery_data_order_and_budget_equal': True}
        post_assets = helper.verify_assets(root, protocol)
        post_common = helper.verify_common_snapshot(root, protocol)
        if post_assets != asset_receipts or post_common != common_receipt:
            raise RuntimeError('Bound historical source inputs changed')
        runtime.write_json(output / 'post_input_verification.json',
                           {'verified': True, 'assets': post_assets, 'common_snapshot': post_common})
        summary = {'protocol_id': protocol['protocol_id'], 'mode': 'smoke' if args.smoke else 'pilot',
            'status': 'complete', 'arms': results, 'common_initial_snapshot': common_snapshot,
            'common_initialization': initialization, 'frozen_negative_control': frozen_gate,
            'elapsed_seconds': time.perf_counter() - started, 'all_protocol_arms_complete': selected == available,
            'no_test_evaluation': True, 'no_historical_artifact_changes': True,
            'single_tokenizer_block_then_frozen_recovery': True, 'not_full_multicycle_joint_training': True}
        runtime.write_json(output / 'summary.json', summary)
        runtime.write_json(output / 'COMPLETE.json', {'summary_sha256': runtime.file_hash(output / 'summary.json'),
            'arms': selected, 'all_protocol_arms_complete': selected == available})
        print(f'[FACTORIAL DONE] {output}', flush=True)
        return summary
    finally:
        pipeline.accelerator.end_training()


def main(argv=None):
    args = parse_args(argv)
    args.owned_output, args.work_handle = None, None
    try:
        run(args)
    except Exception as exc:
        if args.owned_output is not None:
            runtime = load_module('_factorial_failure_io', args.project_root / 'scripts/train/run_diger_noise_pilot.py')
            runtime.write_json(args.owned_output / 'FAILED.json',
                               {'error': type(exc).__name__, 'message': str(exc), 'complete': False})
        raise
    finally:
        if args.work_handle is not None:
            args.work_handle.cleanup()


if __name__ == '__main__':
    main()
