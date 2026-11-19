"""Bounded real-data H100 acceptance and timing; never submit a platform job.

Run only inside an already authorized single-visible-GPU allocation. Explicit
remaining GPU seconds include time spent by earlier encoding/preparation work.
The supervisor enforces both an overall deadline and per-stage deadlines. All
updates are disposable acceptance/profile updates, not formal experiment runs.
CPU unit tests exercise helpers; the production entry rejects CPU/toy assets.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
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
from typing import Any

import numpy as np
import torch

from .runtime import ROOT, read_config
from .training import atomic_save, cpu_copy


class GateFailure(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise GateFailure(message)


def budget_seconds(remaining_gpu_seconds, allocated_gpus, requested_seconds, reserve_seconds=30):
    """Conservative wall allowance; caller must subtract previous GPU use first."""
    if not 0 < remaining_gpu_seconds <= 7200 or allocated_gpus < 1:
        raise ValueError('Remaining budget must be in (0,7200] GPU seconds; allocated GPUs positive')
    available = remaining_gpu_seconds / allocated_gpus - reserve_seconds
    if requested_seconds <= 0 or requested_seconds > available:
        raise ValueError(f'Max seconds must be positive and at most {available:g}, after shutdown reserve')
    return float(requested_seconds)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    # A Windows reader/scanner may briefly hold the old status file. Keep the
    # atomic replacement, with a bounded retry rather than deleting evidence.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


class StageLog:
    def __init__(self, directory, timeout_seconds=600):
        self.directory = Path(directory)
        self.timeout_seconds = float(timeout_seconds)
        self.started = time.monotonic()

    def event(self, name, **values):
        row = dict(event=name, monotonic=time.monotonic(), elapsed=time.monotonic() - self.started, **values)
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    @contextmanager
    def stage(self, name, *, timeout=None, **values):
        started = time.monotonic()
        seconds = self.timeout_seconds if timeout is None else float(timeout)
        write_json(self.directory / 'active_stage.json',
                   dict(name=name, started=started, timeout_seconds=seconds, active=True))
        self.event('stage_start', stage=name, timeout_seconds=seconds, **values)
        try:
            yield
        except BaseException as exc:
            self.event('stage_failed', stage=name, seconds=time.monotonic() - started,
                       error=f'{type(exc).__name__}: {exc}')
            raise
        else:
            self.event('stage_pass', stage=name, seconds=time.monotonic() - started)
        finally:
            write_json(self.directory / 'active_stage.json',
                       dict(name=name, started=started, timeout_seconds=seconds, active=False))


def exact_equal(left: Any, right: Any, path='state'):
    """Fail on any checkpoint/RNG difference, including tensors and NumPy RNG."""
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()), f'{path} tensor differs')
    elif isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray) and np.array_equal(left, right), f'{path} array differs')
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), f'{path} keys differ')
        for key in left:
            exact_equal(left[key], right[key], f'{path}/{key}')
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), f'{path} sequence differs')
        for index, value in enumerate(left):
            exact_equal(value, right[index], f'{path}/{index}')
    else:
        require(left == right, f'{path} differs: {left!r} vs {right!r}')


def fingerprint(value: Any):
    """Stable initial-state identity, without torch.save container metadata."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def parameter_snapshot(module):
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def changed_parameters(before, module):
    result = []
    current = dict(module.named_parameters())
    require(before.keys() == current.keys(), 'Parameter topology changed during an update')
    for name, parameter in current.items():
        require(torch.isfinite(parameter).all(), f'Nonfinite parameter {name}')
        if not torch.equal(before[name], parameter.detach().cpu()):
            result.append(name)
    return result


def check_decoder_updates(changed, layers=4):
    for index in range(layers):
        prefix = f'backbone.decoder_blocks.{index}.'
        for suffix in ('self_attn.qkv.weight', 'cross_attn.qkv.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'):
            require(prefix + suffix in changed, f'Actual decoder weight did not update: {prefix + suffix}')


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def measure(function, device):
    device = torch.device(device)
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = function()
    sync(device)
    seconds = time.perf_counter() - started
    memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else {}
    return result, dict(seconds=seconds, **memory)


def optimizer_steps(optimizer):
    return sorted(int(value['step'].item() if torch.is_tensor(value['step']) else value['step'])
                  for value in optimizer.state.values() if 'step' in value)


def recommendation_gradient_gate(trainer, batch):
    trainer.phase_mode('T')
    trainer.r_optimizer.zero_grad(set_to_none=True)
    trainer.t_optimizer.zero_grad(set_to_none=True)
    loss, values = trainer.objective(batch, 'T', include_aux=False)
    require(torch.isfinite(loss), 'Recommendation-only T loss is nonfinite')
    loss.backward()
    base = trainer.alignment.mixer.base_codebooks.grad
    require(base is not None and torch.isfinite(base).all() and base.abs().sum() > 0,
            'Recommendation loss does not reach PSE codebooks')
    require(all(p.grad is None for p in trainer.model.parameters()), 'Frozen recommendation parameters received T gradients')
    result = dict(loss=float(loss.detach()), base_gradient_l1=float(base.abs().sum()))
    if trainer.alignment.mixer.mixer_enabled:
        output = trainer.alignment.mixer.attention.out_proj.weight.grad
        require(output is not None and torch.isfinite(output).all() and output.abs().sum() > 0,
                'Recommendation loss does not reach Full Mixer output projection')
        result['mixer_output_gradient_l1'] = float(output.abs().sum())
    trainer.r_optimizer.zero_grad(set_to_none=True)
    trainer.t_optimizer.zero_grad(set_to_none=True)
    return result


def generation_inputs(trainer, batch):
    """The only accepted batch fields are visible history; never read targets."""
    history_ids = batch['history_item_ids'].to(trainer.device)
    history_mask = batch['history_mask'].to(trainer.device)
    return trainer.alignment.hard_codes[history_ids], history_mask


def generate_target_free(trainer, batch, catalog, beam):
    from .decoding import generate_lists
    history, mask = generation_inputs(trainer, batch)
    require(all(parameter.dtype == torch.float32 for parameter in trainer.model.parameters()),
            'FP32 inference requires FP32 master parameters')
    with torch.no_grad(), torch.autocast(device_type=trainer.device.type, enabled=False):
        return generate_lists(trainer.model, history, mask, catalog,
                              route=trainer.config['architecture'], list_length=5, beam_size=beam)


def check_generation(output, n_items, expected_batch):
    require(output.item_ids.shape[0] == expected_batch and output.item_ids.shape[-1] == 5,
            'Generation did not return five item slots per list')
    require(output.complete_mask[:, 0].all(), 'A best generated list is incomplete')
    complete_items = output.item_ids[output.complete_mask]
    require(((complete_items > 0) & (complete_items <= n_items)).all(), 'Generation contains illegal item IDs')
    require((complete_items.sort(dim=-1).values[:, 1:] != complete_items.sort(dim=-1).values[:, :-1]).all(),
            'Generation contains duplicate items inside a list')
    require(torch.isfinite(output.scores[output.complete_mask]).all(), 'Complete list score is nonfinite')
    return dict(examples=expected_batch, complete_lists=int(output.complete_mask.sum()),
                best_lists=output.best_item_ids.detach().cpu().tolist(), counters=output.counters)


def check_catalog_refresh(trainer):
    old = trainer.alignment.hard_codes.detach().cpu().clone()
    result = trainer.refresh_catalog()
    alignment = trainer.alignment
    with torch.no_grad(), torch.autocast(device_type=trainer.device.type, enabled=False):
        books, _ = alignment.mixer.crossed_codebooks(False)
        require(alignment.hard_codes[0].eq(-1).all(), 'Catalog refresh overwrote PAD')
        for start in range(1, len(alignment.item_slots), 1024):
            expected = alignment.mixer.squared_distances(alignment.item_slots[start:start + 1024].float(),
                                                        books.float()).argmin(-1)
            require(torch.equal(expected, alignment.hard_codes[start:start + 1024]),
                    'Refreshed catalog differs from deterministic current-codebook nearest assignments')
    result['changed_item_count'] = int(old[1:].ne(alignment.hard_codes[1:].cpu()).any(-1).sum())
    # Tiny updates can preserve all hard SIDs; that is not a failed refresh.
    result['verified_against_current_codebooks'] = True
    return result


def replay_gate(trainer, batch, directory, binding, arm, kind):
    progress = dict(phase=0, update=0, preflight_only=True)
    path = Path(directory) / f'{arm}_before_replay_{kind}.pt'
    atomic_save(path, trainer.checkpoint(binding, progress, arm))
    expected_values = trainer.update([batch], kind)
    expected = trainer.checkpoint(binding, progress, arm)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    restored_progress = trainer.restore(saved, binding, arm=arm, allow_preflight=True)
    require(restored_progress == progress, 'Checkpoint progress was not restored')
    # Verify loading before executing any new stochastic operation. If this
    # passes but a later update differs, report an execution-reproducibility
    # failure; do not assert that checkpoint loading itself was defective.
    exact_equal(saved, trainer.checkpoint(binding, progress, arm), f'{kind}/restored_before_update')
    actual_values = trainer.update([batch], kind)
    actual = trainer.checkpoint(binding, progress, arm)
    try:
        exact_equal(expected_values, actual_values, f'{kind}/values')
        exact_equal(expected, actual, f'{kind}/checkpoint')
    except GateFailure as exc:
        raise GateFailure('Restored input state is exact but update replay differs; '
                          'investigate kernel determinism or unsaved execution state, '
                          'without relaxing numerical acceptance: ' + str(exc)) from exc
    return dict(phase=kind, exact_loaded_state=True, exact_next_update=True, checkpoint=str(path),
                r_updates=trainer.r_updates, t_updates=trainer.t_updates)


def arm_gate(trainer, batch, directory, binding, arm, log):
    report = dict(arm=arm, precision=trainer.precision, small_batch_lists=len(batch['target_item_ids']),
                  formal_training=False)
    with log.stage(arm + '/R_parameter_partition'):
        before_r, before_t = parameter_snapshot(trainer.model), parameter_snapshot(trainer.alignment)
        r_clock, t_clock = trainer.r_updates, trainer.t_updates
        values = trainer.update([batch], 'R')
        changed_r = changed_parameters(before_r, trainer.model)
        require(not changed_parameters(before_t, trainer.alignment), 'R changed tokenizer parameters')
        check_decoder_updates(changed_r, len(trainer.model.backbone.decoder_blocks))
        require(trainer.r_updates == r_clock + 1 and trainer.t_updates == t_clock, 'R clock partition failed')
        require(all(step == trainer.r_updates for step in optimizer_steps(trainer.r_optimizer)), 'R Adam steps disagree')
        report['R'] = dict(values=values, changed_parameters=changed_r,
                           tokenizer_unchanged=True, optimizer_steps=optimizer_steps(trainer.r_optimizer))
        log.event('gate_result', arm=arm, section='R', result=report['R'])
    with log.stage(arm + '/recommendation_only_gradient'):
        report['recommendation_gradient'] = recommendation_gradient_gate(trainer, batch)
        log.event('gate_result', arm=arm, section='recommendation_gradient', result=report['recommendation_gradient'])
    with log.stage(arm + '/T_parameter_partition'):
        before_r, before_t = parameter_snapshot(trainer.model), parameter_snapshot(trainer.alignment)
        r_clock, t_clock = trainer.r_updates, trainer.t_updates
        values = trainer.update([batch], 'T')
        require(trainer.r_updates == r_clock and trainer.t_updates == t_clock + 1, 'T clock partition failed')
        require(not changed_parameters(before_r, trainer.model), 'T changed frozen recommendation parameters')
        changed_t = changed_parameters(before_t, trainer.alignment)
        require('mixer.base_codebooks' in changed_t, 'T did not update PSE codebooks')
        if trainer.alignment.mixer.mixer_enabled:
            require('mixer.attention.out_proj.weight' in changed_t, 'T did not update Full Mixer output projection')
            # The zero-output initialization intentionally blocks Q/K/V on the
            # first step. Verify it receives and applies feedback after opening.
            before_mixer = parameter_snapshot(trainer.alignment)
            trainer.update([batch], 'T')
            require('mixer.attention.in_proj_weight' in changed_parameters(before_mixer, trainer.alignment),
                    'Mixer Q/K/V did not update after its output projection opened')
        else:
            require(changed_t == ['mixer.base_codebooks'], 'Disabled Mixer parameters changed')
        require(not changed_parameters(before_r, trainer.model), 'Second T update changed recommendation parameters')
        require(all(step == trainer.t_updates for step in optimizer_steps(trainer.t_optimizer)), 'T Adam steps disagree')
        report['T'] = dict(values=values, changed_parameters_first_step=changed_t,
                           recommender_unchanged=True, optimizer_steps=optimizer_steps(trainer.t_optimizer))
        log.event('gate_result', arm=arm, section='T', result=report['T'])
    with log.stage(arm + '/checkpoint_replay_R'):
        report['replay_R'] = replay_gate(trainer, batch, directory, binding, arm, 'R')
    with log.stage(arm + '/checkpoint_replay_T'):
        report['replay_T'] = replay_gate(trainer, batch, directory, binding, arm, 'T')
    with log.stage(arm + '/SID_refresh'):
        report['catalog_refresh'] = check_catalog_refresh(trainer)
    return report


def fixed_indices(count, sample_count, seed):
    if sample_count < 1 or sample_count > count:
        raise ValueError('Fixed sample must be positive and no larger than the actual split')
    return np.random.default_rng(seed).choice(count, size=sample_count, replace=False)


def profile_updates(trainer, data, indices, log, arm, repeats, microbatch):
    from .run import microbatches
    effective = int(trainer.config['training']['effective_batch_size'])
    require(effective == 1024, 'Formal throughput profile must use 1024 lists per optimizer update')
    require(effective % microbatch == 0, 'Profile microbatch must divide 1024')
    result = {}
    for kind in ('R', 'T'):
        records = []
        for iteration in range(repeats):
            selected = indices[iteration * effective:(iteration + 1) * effective]
            require(len(selected) == effective, 'Profile must not silently time a short batch')
            prepare_started = time.perf_counter()
            batches = list(microbatches(data, selected, microbatch))
            prepare_seconds = time.perf_counter() - prepare_started
            with log.stage(f'{arm}/profile_{kind}/{iteration}', effective_lists=effective,
                           microbatch=microbatch, accumulation=len(batches)):
                values, timing = measure(lambda: trainer.update(batches, kind), trainer.device)
                timing.update(effective_lists=effective, microbatch=microbatch, accumulation=len(batches),
                              data_prepare_seconds=prepare_seconds,
                              end_to_end_seconds=prepare_seconds + timing['seconds'],
                              lists_per_second=effective / timing['seconds'], values=values)
                records.append(timing)
                log.event('profile_update', arm=arm, phase=kind, iteration=iteration, **timing)
        median = statistics.median(row['seconds'] for row in records)
        result[kind] = dict(records=records, median_update_seconds=median,
                            median_lists_per_second=effective / median,
                            median_end_to_end_seconds=statistics.median(row['end_to_end_seconds'] for row in records),
                            timing_scope='update includes H2D/forward/backward/step; end_to_end also includes Arrow/collation')
    return result


def initial_signature(trainer):
    with torch.no_grad():
        books, _ = trainer.alignment.mixer.crossed_codebooks(False)
    return dict(model=fingerprint(trainer.model.state_dict()), effective_codebooks=fingerprint(books),
                hard_sids=fingerprint(trainer.alignment.hard_codes),
                r_optimizer=fingerprint(trainer.r_optimizer.state_dict()),
                t_optimizer=fingerprint(trainer.t_optimizer.state_dict()),
                scheduler=fingerprint(trainer.scheduler.state_dict()))


def worker(args):
    if os.environ.get('LISTWISE_PREFLIGHT_SUPERVISED') != '1':
        raise RuntimeError('The worker must be started by the deadline supervisor')
    from .run import build_context, make_catalog, microbatches, resolved
    from .data import KuaiRecListDataset
    directory = Path(args.output)
    log = StageLog(directory, args.stage_timeout)
    require(torch.cuda.is_available(), 'The real-data acceptance entry requires CUDA')
    require(torch.cuda.device_count() == 1, 'Expose exactly one authorized GPU with CUDA_VISIBLE_DEVICES')
    require('h100' in torch.cuda.get_device_name(0).lower(), 'This gate is specifically for H100')
    require(torch.cuda.is_bf16_supported(), 'CUDA device must support BF16')
    require(args.allocated_gpus >= torch.cuda.device_count(), 'Allocation accounting understates visible GPUs')
    torch.cuda.set_device(0)
    cfg = read_config(resolved(args.config))
    cfg = copy.deepcopy(cfg)
    cfg['training']['microbatch_size'] = args.microbatch
    require(cfg['training'].get('deterministic_algorithms') is True,
            'Formal configuration must explicitly enable strict deterministic algorithms')
    require(cfg['training']['effective_batch_size'] == 1024, 'Profile requires the locked effective batch 1024')
    require((cfg['history_max'], cfg['list_length'], cfg['model']['n_embd'],
             cfg['model']['encoder_n_layer'], cfg['model']['decoder_n_layer'],
             cfg['model']['codebook_size'], cfg['model']['n_head'],
             cfg['model']['n_inner']) == (50, 5, 256, 1, 4, 256, 4, 1024),
            'Formal H100 gate must use the full locked model')
    summary = dict(status='running', purpose='disposable_real_data_preflight_not_formal_training',
                   gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   allocated_gpus=args.allocated_gpus, remaining_gpu_seconds_at_launch=args.remaining_gpu_seconds,
                   effective_batch=1024, microbatch=args.microbatch, gate_batch=args.gate_batch,
                   eval_batch=args.eval_batch, decode_batches=args.decode_batches,
                   profile_updates=args.profile_updates, arms={}, common_start={})
    write_json(directory / 'report.json', summary)
    starts = {}
    for arm in ('J0', 'J1', 'H0', 'H1'):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg['architecture'] = arm[0]
        with log.stage(arm + '/load_real_assets_and_split'):
            trainer, assets, data, binding = build_context(arm_cfg, 'cuda:0', 'bf16', arm.endswith('1'))
            require(not assets.manifest.get('toy', True), 'Toy PSE assets cannot pass the real-data H100 gate')
            require(tuple(assets.centroids.shape) == (4, 256, 64), 'Real PSE must have 4x256x64 centers')
            require(assets.manifest['dataset_binding'] == data.binding, 'Formal dataset/asset identity mismatch')
            require(len(data) >= 1024 * args.profile_updates, 'Not enough real rows for distinct profile batches')
            validation = KuaiRecListDataset(resolved(cfg['dataset']), 'validation', assets, verify=False)
            initial = initial_signature(trainer)
            if arm.endswith('0'):
                starts[arm[0]] = initial
            else:
                exact_equal(starts[arm[0]], initial, arm[0] + '/common_start')
                summary['common_start'][arm[0]] = dict(equal=True, hashes=initial,
                    scope='same random initialization and E0; full 5R shared warmup is not run here')
            train_indices = fixed_indices(len(data), max(args.gate_batch, 1024 * args.profile_updates), cfg['seed'] + 1701)
            val_indices = fixed_indices(len(validation), args.eval_batch * args.decode_batches, cfg['seed'] + 1702)
            gate_batch = next(microbatches(data, train_indices[:args.gate_batch], args.gate_batch))
            eval_batch = next(microbatches(validation, val_indices[:args.eval_batch], args.eval_batch))
            summary['data'] = dict(train_rows=len(data), validation_rows=len(validation), catalog_items=assets.n_items,
                                   binding=binding, fixed_training_sample_ids=gate_batch['sample_id'],
                                   fixed_validation_indices=val_indices.tolist(),
                                   fixed_validation_sample_ids=[validation[int(index)]['sample_id'] for index in val_indices])
        # Exact replay and throughput use the same strict runtime policy as
        # the formal four arms. Unsupported deterministic kernels fail loudly.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        report = arm_gate(trainer, gate_batch, directory, binding, arm, log)
        report['binding'] = binding
        report['replay_deterministic_algorithms'] = True
        report['replay_backend_settings'] = dict(
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32)
        report['status'] = 'training_gate_passed_profile_pending'
        summary['arms'][arm] = report
        write_json(directory / 'report.json', summary)
        with log.stage(arm + '/target_free_five_item_generation'):
            catalog = make_catalog(trainer.alignment, assets)
            generated = generate_target_free(trainer, eval_batch, catalog, beam=2)
            report['generation_gate'] = check_generation(generated, assets.n_items, args.eval_batch)
        require(torch.are_deterministic_algorithms_enabled(), 'Profile must keep strict deterministic algorithms')
        require(not torch.is_deterministic_algorithms_warn_only_enabled(), 'Warn-only determinism is not an acceptance gate')
        report['profile_backend_settings'] = dict(deterministic_algorithms=True,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            scope='same strict deterministic policy for replay, throughput profile, and formal four-arm execution')
        report['profile'] = profile_updates(trainer, data, train_indices, log, arm,
                                            args.profile_updates, args.microbatch)
        with log.stage(arm + '/profile_catalog_refresh'):
            trainer.refresh_catalog()
            catalog, catalog_timing = measure(lambda: make_catalog(trainer.alignment, assets), trainer.device)
            report['catalog_timing'] = catalog_timing
        report['decode_profile'] = {}
        for beam in (32, 256):
            overall_started = time.perf_counter()
            with log.stage(f'{arm}/FP32_beam{beam}/fresh_catalog', timeout=args.decode_timeout):
                # Each beam starts from an independent empty legality cache.
                # Subsequent batches have different request IDs; only natural
                # overlap in partial SID patterns can create warm-cache hits.
                catalog, fresh_catalog_timing = measure(lambda: make_catalog(trainer.alignment, assets), trainer.device)
            records = []
            report['decode_profile'][str(beam)] = dict(status='partial', records=records,
                planned_examples=len(val_indices), measured_examples=0, fresh_catalog_timing=fresh_catalog_timing,
                cache_policy='new SIDCatalog per beam; first batch cold, later distinct requests may warm partial-pattern caches')
            write_json(directory / 'report.json', summary)
            for iteration in range(args.decode_batches):
                selected = val_indices[iteration * args.eval_batch:(iteration + 1) * args.eval_batch]
                prepare_started = time.perf_counter()
                decode_batch = next(microbatches(validation, selected, args.eval_batch))
                prepare_seconds = time.perf_counter() - prepare_started
                with log.stage(f'{arm}/FP32_beam{beam}/{iteration}', timeout=args.decode_timeout,
                               beam=beam, lists=args.eval_batch):
                    generated, timing = measure(lambda: generate_target_free(trainer, decode_batch, catalog, beam), trainer.device)
                    checked = check_generation(generated, assets.n_items, args.eval_batch)
                    timing.update(batch_size=args.eval_batch, beam=beam, precision='fp32',
                                  dataset_indices=selected.tolist(), sample_ids=decode_batch['sample_id'],
                                  cache_state='cold' if iteration == 0 else 'naturally_warmed_by_distinct_requests',
                                  data_prepare_seconds=prepare_seconds,
                                  end_to_end_seconds=prepare_seconds + timing['seconds'],
                                  amortized_seconds_per_request=timing['seconds'] / args.eval_batch,
                                  counters=checked['counters'])
                    records.append(timing)
                    log.event('profile_decode', arm=arm, iteration=iteration, **timing)
                    report['decode_profile'][str(beam)]['measured_examples'] += args.eval_batch
                    write_json(directory / 'report.json', summary)
            seconds = statistics.median(record['seconds'] for record in records)
            overall_seconds = time.perf_counter() - overall_started
            report['decode_profile'][str(beam)].update(status='complete', overall_end_to_end_seconds=overall_seconds,
                overall_requests_per_second=len(val_indices) / overall_seconds,
                median_decode_batch_seconds=seconds,
                estimated_full_validation_seconds=len(validation) / len(val_indices) * overall_seconds,
                estimate_caveat='at least 16 distinct fixed validation requests, not full validation; includes catalog build/collation/generation/checks/logging but excludes full metric/prediction serialization')
            write_json(directory / 'report.json', summary)
        report['status'] = 'passed'
        summary['arms'][arm] = report
        write_json(directory / 'report.json', summary)
        log.event('arm_pass', arm=arm)
        del generated, trainer, assets, data, validation, catalog
        gc.collect()
        torch.cuda.empty_cache()
    summary['status'] = 'passed'
    summary['formal_training_started'] = False
    write_json(directory / 'report.json', summary)
    log.event('preflight_pass', arms=list(summary['arms']))


def deadline_reason(started, allowance, stage, now=None):
    now = time.monotonic() if now is None else now
    if now - started >= allowance:
        return 'overall_budget_deadline'
    if stage and stage.get('active') and now - stage['started'] >= stage['timeout_seconds']:
        return 'stage_timeout:' + stage['name']
    return None


def supervise(args):
    allowance = budget_seconds(args.remaining_gpu_seconds, args.allocated_gpus, args.max_seconds)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to reuse preflight output: {output}')
    output.mkdir(parents=True)
    args.output = str(output)
    write_json(output / 'request.json', vars(args))
    environment = os.environ.copy()
    environment.update(LISTWISE_PREFLIGHT_SUPERVISED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', '-m', 'scripts.listwise_diffgrm.preflight', '--worker']
    for key, value in vars(args).items():
        if key != 'worker':
            command += ['--' + key.replace('_', '-'), str(value)]
    started = time.monotonic()
    reason = None
    with (output / 'worker.log').open('wb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        last_heartbeat = 0.0
        try:
            while process.poll() is None:
                stage = None
                try:
                    stage = json.loads((output / 'active_stage.json').read_text(encoding='utf-8'))
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                now = time.monotonic()
                reason = deadline_reason(started, allowance, stage, now)
                if reason:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break
                if now - last_heartbeat >= 30:
                    print(json.dumps(dict(event='supervisor_heartbeat', elapsed=now - started,
                                          max_seconds=allowance, stage=stage)), flush=True)
                    last_heartbeat = now
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    elapsed = time.monotonic() - started
    result = dict(status='timeout' if reason else ('passed' if process.returncode == 0 else 'failed'),
                  reason=reason, returncode=process.returncode, wall_seconds=elapsed,
                  allocated_gpu_seconds_during_script=elapsed * args.allocated_gpus,
                  budget_scope='caller-subtracted remaining budget; platform allocation billing is external',
                  output=str(output))
    write_json(output / 'supervisor.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/09_kuairec_listwise/base.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--remaining-gpu-seconds', type=float, required=True,
                        help='Remaining from the entire authorized 2 GPU hours after prior work')
    parser.add_argument('--allocated-gpus', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--stage-timeout', type=float, default=600)
    parser.add_argument('--decode-timeout', type=float, default=300)
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--gate-batch', type=int, default=8)
    parser.add_argument('--profile-updates', type=int, default=2)
    parser.add_argument('--eval-batch', type=int, default=4)
    parser.add_argument('--decode-batches', type=int, default=4,
                        help='Distinct fixed validation batches per beam (at least four and sixteen total requests)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (min(args.stage_timeout, args.decode_timeout, args.microbatch, args.gate_batch,
            args.profile_updates, args.eval_batch, args.decode_batches) <= 0
            or 1024 % args.microbatch):
        parser.error('Sizes/timeouts must be positive and microbatch must divide 1024')
    if args.decode_batches < 4 or args.decode_batches * args.eval_batch < 16:
        parser.error('Decode profile needs at least four distinct batches and sixteen validation requests')
    budget_seconds(args.remaining_gpu_seconds, args.allocated_gpus, args.max_seconds)
    if args.worker:
        try:
            worker(args)
        except BaseException as exc:
            write_json(Path(args.output) / 'failure.json',
                       dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
            raise
        return 0
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
