"""Disposable R20 recovery acceptance/timing inside one authorized H100 process.

No platform API is used. The supervisor enforces a maximum 480-second lifetime.
Run one independent process per policy/GPU. Formal training must fork R20 again;
the marked acceptance checkpoints cannot initialize a formal continuation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

from .preflight import (GateFailure, StageLog, check_decoder_updates, check_generation,
                        changed_parameters, exact_equal, generate_target_free,
                        measure, parameter_snapshot, require, write_json)
from .recovery import PARENT_SHA256, fingerprint
from .recovery_diagnostics import diagnostic_context, diagnose_catalog
from .runtime import ROOT
from .training import atomic_save, seed_all


def mark_preflight(state):
    """Mark a newly produced state in place; leave all scientific binding intact."""
    state['preflight_only'] = True
    if 'inner' in state:
        state['inner']['preflight_only'] = True
    return state


def fixed_validation_rows(sample_ids, count):
    """Same unsalted UTF8-SHA order as the recovery process diagnostic design."""
    if type(count) is not int or not 0 < count <= len(sample_ids):
        raise ValueError('Validation sample count is outside the available split')
    if any(not isinstance(item, str) or not item for item in sample_ids):
        raise ValueError('Validation sample IDs must be exact nonempty strings')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Validation sample IDs must be unique')
    ordered = sorted((hashlib.sha256(value.encode('utf-8')).hexdigest(), value, index)
                     for index, value in enumerate(sample_ids))[:count]
    return [dict(index=index, sample_id=value, sha256=digest) for digest, value, index in ordered]


def gradient_groups(trainer, kind):
    """Require nonzero finite gradients in every expected active functional block."""
    if kind == 'R':
        module = trainer.model
        prefixes = dict(sid_embedding=('backbone.embedding.',),
                        history_projection=('backbone.item_mlp.',),
                        history_position=('backbone.pos_emb_enc.',),
                        mask_embedding=('backbone.mask_emb_table.',),
                        list_position=('item_position_embedding.',))
        for i in range(trainer.config['model']['encoder_n_layer']):
            prefixes[f'encoder_{i}'] = (f'backbone.encoder_blocks.{i}.',)
        for i in range(trainer.config['model']['decoder_n_layer']):
            for component in ('self_attn', 'cross_attn', 'mlp'):
                prefixes[f'decoder_{i}_{component}'] = (f'backbone.decoder_blocks.{i}.{component}.',)
        frozen = trainer.alignment
    elif kind == 'T':
        module = trainer.alignment
        prefixes = dict(base_codebooks=('mixer.base_codebooks',),
                        book_tags=('mixer.book_embedding',), code_tags=('mixer.code_embedding',),
                        mixer_norm=('mixer.input_norm.',),
                        mixer_attention_input=('mixer.attention.in_proj_',),
                        mixer_attention_output=('mixer.attention.out_proj.',))
        frozen = trainer.model
    else:
        raise ValueError('Gradient gate phase must be R or T')
    require(all(p.grad is None for p in frozen.parameters()), f'{kind}: frozen side received gradients')
    result = {}
    for label, names in prefixes.items():
        matching = [(name, p) for name, p in module.named_parameters() if name.startswith(names)]
        require(matching, f'{kind}: required module {label} is absent')
        gradients = [(name, p.grad) for name, p in matching if p.grad is not None]
        require(gradients, f'{kind}: module {label} has no gradient')
        require(all(bool(torch.isfinite(value).all()) for _, value in gradients),
                f'{kind}: module {label} has nonfinite gradients')
        l1 = sum(float(value.detach().double().abs().sum()) for _, value in gradients)
        require(l1 > 0, f'{kind}: module {label} has zero gradient')
        result[label] = dict(gradient_l1=l1, parameters_with_gradient=[name for name, _ in gradients])
    return result


def _initial_proof(controller):
    trainer = controller.trainer
    initial = controller.checkpoint()
    inner = initial['inner']
    for key in ('model', 'alignment', 'r_optimizer', 'scheduler'):
        require(fingerprint(inner[key]) == controller.parent_hashes[key], f'Fork changed parent {key}')
    expected_t = copy.deepcopy(inner['t_optimizer'])
    if controller.policy == 'decay':
        for group in expected_t['param_groups']:
            group['lr'] = controller.peak_lrs[group['group_name']]
    require(fingerprint(expected_t) == controller.parent_hashes['t_optimizer'],
            'Fork changed T optimizer moments or non-LR settings')
    require((trainer.r_updates, trainer.t_updates) == (13860, 200), 'Not the locked R20 clocks')
    expected = dict(codebook=5e-6, mixer=5e-5) if controller.policy == 'decay' else controller.peak_lrs
    for group in trainer.t_optimizer.param_groups:
        require(math.isclose(group['lr'], expected[group['group_name']], rel_tol=1e-12),
                'First planned T LR differs from the recovery policy')
    return initial, dict(parent_hashes=controller.parent_hashes,
        fork_hashes={key: fingerprint(inner[key]) for key in
                     ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler')},
        hard_sid_sha256=fingerprint(trainer.alignment.hard_codes),
        shared_model_and_alignment_exact=True, optimizer_moments_exact=True,
        r_updates=trainer.r_updates, t_updates=trainer.t_updates,
        initial_t_lrs={group['group_name']: group['lr'] for group in trainer.t_optimizer.param_groups})


def _rejects_formal_restore(callback):
    try:
        callback()
    except ValueError as exc:
        require('preflight' in str(exc).lower(), 'Checkpoint rejected for an unrelated reason')
        return str(exc)
    raise GateFailure('Disposable preflight checkpoint was accepted as a formal resume')


def _raw_state(controller, arm, r_phase):
    trainer = controller.trainer
    progress = dict(phase=r_phase, update=trainer.r_updates - controller.start_counts['R'],
                    preflight_only=True)
    return mark_preflight(trainer.checkpoint(controller.execution_binding, progress, arm))


def replay_next_update(controller, batches, kind, directory, r_phase):
    """Real disk save/load plus exact next stochastic update, with marked files."""
    trainer = controller.trainer
    arm = f'H1-{controller.policy}-preflight'
    path = Path(directory) / f'preflight_before_replay_{kind}.pt'
    if kind == 'T':
        saved = mark_preflight(controller.checkpoint())
        atomic_save(path, saved)
        rejection = _rejects_formal_restore(lambda: controller.restore(saved))
        expected_values = controller.step(batches, 'T')
        expected = mark_preflight(controller.checkpoint())
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        controller.restore(loaded, allow_preflight=True)
        exact_equal(saved, mark_preflight(controller.checkpoint()), 'T/reloaded_state')
        actual_values = controller.step(batches, 'T')
        actual = mark_preflight(controller.checkpoint())
    else:
        saved = _raw_state(controller, arm, r_phase)
        atomic_save(path, saved)
        rejection = _rejects_formal_restore(lambda: trainer.restore(
            saved, controller.execution_binding, arm=arm))
        expected_values = trainer.update(batches, 'R')
        expected = _raw_state(controller, arm, r_phase)
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        trainer.restore(loaded, controller.execution_binding, arm=arm, allow_preflight=True)
        exact_equal(saved, _raw_state(controller, arm, r_phase), 'R/reloaded_state')
        actual_values = trainer.update(batches, 'R')
        actual = _raw_state(controller, arm, r_phase)
    exact_equal(expected_values, actual_values, f'{kind}/next_update_values')
    exact_equal(expected, actual, f'{kind}/next_update_full_state')
    return dict(phase=kind, checkpoint=str(path), preflight_only=True,
                formal_restore_rejection=rejection, exact_loaded_state=True,
                exact_next_update=True, r_updates=trainer.r_updates, t_updates=trainer.t_updates)


def _profile_phase(controller, data, kind, repeats, log, directory):
    from .run import microbatches, phase_indices
    trainer = controller.trainer
    phase = controller.start_phase if kind == 'T' else controller.start_phase + 1
    require(trainer.plan[phase]['kind'] == kind, 'Unexpected original phase index')
    indices = phase_indices(trainer.config, len(data), phase, trainer.plan[phase])
    effective = trainer.config['training']['effective_batch_size']
    micro = trainer.config['training']['microbatch_size']
    require(len(indices) >= effective * (repeats + 1), 'Insufficient rows for profile and replay batches')
    seed_all(trainer.config['seed'] + 100003 * (phase + 1))
    records = []
    for index in range(repeats):
        selected = indices[index * effective:(index + 1) * effective]
        started = time.perf_counter()
        batches = list(microbatches(data, selected, micro))
        preparation = time.perf_counter() - started
        frozen = trainer.model if kind == 'T' else trainer.alignment
        frozen_hash = fingerprint(frozen.state_dict())
        frozen_optimizer = trainer.r_optimizer if kind == 'T' else trainer.t_optimizer
        frozen_optimizer_hash = fingerprint(frozen_optimizer.state_dict())
        prior_active = parameter_snapshot(trainer.alignment if kind == 'T' else trainer.model)
        sids_hash = fingerprint(trainer.alignment.hard_codes)
        used_lrs = {'R': trainer.r_optimizer.param_groups[0]['lr'],
                    **{group['group_name']: group['lr'] for group in trainer.t_optimizer.param_groups}}
        step = (lambda: controller.step(batches, 'T')) if kind == 'T' else (lambda: trainer.update(batches, 'R'))
        with log.stage(f'profile_{kind}_{index + 1}'):
            values, timing = measure(step, trainer.device)
        require(fingerprint(frozen.state_dict()) == frozen_hash, f'{kind}: frozen module changed')
        require(fingerprint(frozen_optimizer.state_dict()) == frozen_optimizer_hash,
                f'{kind}: inactive optimizer changed')
        require(fingerprint(trainer.alignment.hard_codes) == sids_hash,
                'Short profile update published a premature hard-SID refresh')
        gradients = gradient_groups(trainer, kind)
        changed = changed_parameters(prior_active, trainer.alignment if kind == 'T' else trainer.model)
        require(changed, f'{kind}: active parameters did not update')
        if kind == 'R':
            check_decoder_updates(changed, layers=trainer.config['model']['decoder_n_layer'])
        else:
            require('mixer.base_codebooks' in changed and any(name.startswith('mixer.attention.') for name in changed),
                    'T profile did not change both the base codebooks and Mixer')
        record = dict(timing, iteration=index + 1, data_prepare_seconds=preparation,
            end_to_end_seconds=preparation + timing['seconds'], effective_lists=effective,
            microbatch=micro, accumulation=len(batches), original_phase_index=phase,
            selected_indices_sha256=fingerprint(selected), first_sample_ids=batches[0]['sample_id'][:4],
            learning_rates_used=used_lrs, values=values, gradients=gradients,
            frozen_module_and_optimizer_unchanged=True, hard_sids_unchanged=True,
            changed_parameters=changed)
        records.append(record)
        log.event('profile_update', kind=kind, iteration=index + 1,
                  seconds=timing['seconds'], learning_rates_used=used_lrs)
    selected = indices[repeats * effective:(repeats + 1) * effective]
    batches = list(microbatches(data, selected, micro))
    with log.stage(f'replay_next_{kind}'):
        replay = replay_next_update(controller, batches, kind, directory, controller.start_phase + 1)
    return dict(records=records, replay=replay,
        median_update_seconds=statistics.median(row['seconds'] for row in records),
        median_end_to_end_seconds=statistics.median(row['end_to_end_seconds'] for row in records),
        timing_scope='H2D/forward/backward/optimizer included; data preparation reported separately; verification excluded',
        scientific_scope=('First real decay T updates from the inherited R20 state and original T phase seed'
                          if kind == 'T' else 'Disposable R20 fixed-SID weights, original R21/phase26 data; '
                          'does not represent completion of the preceding formal 40T block'))


def worker(args):
    if os.environ.get('LISTWISE_RECOVERY_PREFLIGHT_SUPERVISED') != '1':
        raise RuntimeError('Start this entry through its bounded supervisor')
    from .data import KuaiRecListDataset
    from .recovery_run import build_recovery
    from .run import make_catalog, microbatches
    directory = Path(args.output)
    log = StageLog(directory, timeout_seconds=180)
    require(torch.cuda.is_available(), 'Real recovery preflight requires CUDA')
    require(torch.cuda.device_count() == 1, 'Expose exactly one GPU per worker with CUDA_VISIBLE_DEVICES')
    require('h100' in torch.cuda.get_device_name(0).lower(), 'This acceptance profile requires an H100')
    require(torch.cuda.is_bf16_supported(), 'BF16 must be supported')
    torch.cuda.set_device(0)
    report = dict(format='stage09_h1_recovery_preflight_v1', status='running', policy=args.policy,
        preflight_only=True, formal_training_started=False, parent_sha256=PARENT_SHA256,
        gpu=torch.cuda.get_device_name(0), visible_gpu_count=1, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, max_seconds=args.max_seconds, profile={},
        profile_updates=args.profile_updates, validation_samples=args.validation_samples,
        note='All acceptance updates are discarded; formal continuation forks the authenticated R20 checkpoint again.')
    write_json(directory / 'report.json', report)
    with log.stage('authenticated_R20_fork'):
        controller, assets, data = build_recovery(args.parent, args.policy, 'cuda:0',
            dataset=args.dataset, assets=args.assets, inspection=False)
        trainer = controller.trainer
        cfg = trainer.config
        require(trainer.precision == 'bf16', 'Recovery preflight must retain BF16 training')
        require((cfg['architecture'], cfg['training']['effective_batch_size'],
                 cfg['training']['microbatch_size'], cfg['evaluation']['batch_size']) == ('H', 1024, 256, 32),
                'Recovery preflight must preserve H / effective1024 / micro256 / evaluation32')
        require((assets.n_items, len(data)) == (10728, 709443), 'Production catalog/train row count differs')
        require(controller.protocol['parent_sha256'] == PARENT_SHA256, 'Unrecognized parent checkpoint')
        require(torch.are_deterministic_algorithms_enabled() and
                not torch.is_deterministic_algorithms_warn_only_enabled(), 'Strict determinism is required')
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                'TF32 must remain disabled')
        initial, report['initial'] = _initial_proof(controller)
        report['source_binding'] = controller.execution_binding
        report['backend'] = dict(cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            deterministic_algorithms=True, cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark, cuda_matmul_tf32=False, cudnn_tf32=False)
    if args.policy == 'decay':
        report['profile']['T'] = _profile_phase(controller, data, 'T', args.profile_updates, log, directory)
        write_json(directory / 'report.json', report)
        controller.restore(initial)
    report['profile']['R'] = _profile_phase(controller, data, 'R', args.profile_updates, log, directory)
    write_json(directory / 'report.json', report)
    controller.restore(initial)
    with log.stage('full_catalog_read_only_diagnostic'):
        before = fingerprint(trainer.checkpoint(controller.execution_binding, controller.progress, 'H1-preflight'))
        (metrics, snapshot), timing = measure(lambda: diagnose_catalog(trainer), trainer.device)
        after = fingerprint(trainer.checkpoint(controller.execution_binding, controller.progress, 'H1-preflight'))
        require(before == after, 'Catalog diagnostic changed checkpoint/RNG state')
        require(torch.equal(snapshot['hard_codes'], trainer.alignment.hard_codes[1:].cpu()),
                'R20 restored SIDs differ from deterministic current codebook assignments')
        report['catalog_diagnostic'] = dict(timing, metrics=metrics, full_state_unchanged=True,
            hard_sids_match_parent=True, effective_codebooks_sha256=fingerprint(snapshot['effective_codebooks']))
    with log.stage('load_fixed_validation_subset'):
        validation = KuaiRecListDataset(cfg['dataset'], 'validation', assets=assets)
        rows = fixed_validation_rows(validation.table['sample_id'].to_pylist(), args.validation_samples)
        selected = [row['index'] for row in rows]
        subset = dict(split='validation', selection='sha256_utf8_sample_id_then_sample_id',
            population_count=len(validation), count=len(rows), rows=rows,
            rows_sha256=fingerprint(rows), purpose='throughput_and_restoration_only_not_quality_selection')
        write_json(directory / 'validation_subset.json', subset)
        report['validation_subset'] = {key: value for key, value in subset.items() if key != 'rows'}
    decode_records, best_lists, first_signature = [], [], None
    with diagnostic_context(trainer):
        with log.stage('parent_R20_catalog_for_decode'):
            catalog, catalog_timing = measure(lambda: make_catalog(trainer.alignment, assets), trainer.device)
        for index in range(0, len(selected), 32):
            started = time.perf_counter()
            batch = next(microbatches(validation, selected[index:index + 32], 32))
            preparation = time.perf_counter() - started
            with log.stage(f'beam32_validation_{index // 32 + 1}'):
                output, timing = measure(lambda: generate_target_free(trainer, batch, catalog, 32), trainer.device)
            checked = check_generation(output, assets.n_items, len(batch['sample_id']))
            signature = dict(items=output.item_ids.detach().cpu(), scores=output.scores.detach().cpu(),
                             mask=output.complete_mask.detach().cpu())
            if first_signature is None:
                first_signature = fingerprint(signature)
            best_lists.extend(checked['best_lists'])
            decode_records.append(dict(timing, data_prepare_seconds=preparation,
                end_to_end_seconds=timing['seconds'] + preparation, batch_size=len(batch['sample_id']),
                beam=32, precision='fp32', sample_ids=batch['sample_id'], output_sha256=fingerprint(signature)))
    controller.restore(initial)
    with diagnostic_context(trainer), log.stage('R20_reset_output_replay'):
        catalog = make_catalog(trainer.alignment, assets)
        batch = next(microbatches(validation, selected[:32], 32))
        output = generate_target_free(trainer, batch, catalog, 32)
        signature = dict(items=output.item_ids.detach().cpu(), scores=output.scores.detach().cpu(),
                         mask=output.complete_mask.detach().cpu())
        require(fingerprint(signature) == first_signature, 'R20 reset changed deterministic FP32 decoded output')
    report['decode_profile'] = dict(beam=32, precision='fp32', records=decode_records,
        catalog_timing=catalog_timing, examples=len(selected), best_lists=best_lists,
        total_generation_seconds=sum(row['seconds'] for row in decode_records),
        total_end_to_end_seconds=sum(row['end_to_end_seconds'] for row in decode_records) + catalog_timing['seconds'],
        median_batch_seconds=statistics.median(row['seconds'] for row in decode_records),
        reset_R20_output_exact=True, first_batch_output_sha256=first_signature,
        cache_policy='new catalog at restored R20, cold first batch then distinct fixed requests')
    controller.restore(initial)
    exact_equal(initial, controller.checkpoint(), 'discard_all_preflight_updates')
    report.update(status='passed', finished_at_parent_state=True,
                  formal_resume_from_preflight_forbidden=True,
                  worker_seconds=time.monotonic() - log.started)
    write_json(directory / 'report.json', report)
    log.event('preflight_complete', policy=args.policy)


def supervise(args):
    """Reserve shutdown time inside max_seconds; timeout is failure, never pass."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to reuse preflight output: {output}')
    started = time.monotonic()
    output.mkdir(parents=True)
    args.output = str(output)
    write_json(output / 'request.json', vars(args))
    environment = os.environ.copy()
    environment.update(LISTWISE_RECOVERY_PREFLIGHT_SUPERVISED='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    command = [sys.executable, '-u', '-m', 'scripts.listwise_diffgrm.recovery_preflight', '--worker']
    for key, value in vars(args).items():
        if key != 'worker' and value is not None:
            command.extend(['--' + key.replace('_', '-'), str(value)])
    reason = None
    with (output / 'worker.log').open('wb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= args.max_seconds - 12:
                    reason = 'overall_preflight_deadline'
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    elapsed = time.monotonic() - started
    result = dict(status='timeout' if reason else ('passed' if process.returncode == 0 else 'failed'),
        reason=reason, returncode=process.returncode, policy=args.policy, preflight_only=True,
        wall_seconds=elapsed, max_seconds=args.max_seconds, allocated_gpu_seconds_during_script=elapsed,
        accounting_scope='single-visible-GPU process lifetime only; platform allocation/release billing is external')
    write_json(output / 'supervisor.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--policy', required=True, choices=('decay', 'freeze'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--assets')
    parser.add_argument('--max-seconds', type=float, default=480)
    parser.add_argument('--profile-updates', type=int, default=3)
    parser.add_argument('--validation-samples', type=int, default=128, choices=(128, 256))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 60 <= args.max_seconds <= 480 or not 3 <= args.profile_updates <= 5:
        parser.error('max-seconds must be 60..480 and profile-updates must be 3..5')
    if args.worker:
        try:
            worker(args)
        except BaseException as exc:
            write_json(Path(args.output) / 'failure.json',
                       dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), preflight_only=True))
            raise
        return 0
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
