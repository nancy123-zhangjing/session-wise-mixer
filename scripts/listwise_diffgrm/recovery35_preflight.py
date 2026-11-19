"""Disposable, deadline-supervised A/B/C acceptance on one allocated H100.

No platform API is used. A/B skip only their frozen T boundary; C really runs
the original 40-T block before R6. Every persisted state is a preflight bundle,
rejected by the formal controller. The supervisor caps all three branches,
including loading and verification, at 1620 seconds (0.45 GPU hours).
"""
from __future__ import annotations

import argparse
import csv
import gc
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

from .preflight import (GateFailure, StageLog, changed_parameters, check_decoder_updates,
    check_generation, exact_equal, generate_target_free, measure, parameter_snapshot,
    require, write_json)
from .recovery import fingerprint
from .recovery35 import BRANCHES, build_recovery35
from .recovery_diagnostics import diagnostic_context
from .recovery_preflight import fixed_validation_rows, gradient_groups
from .runtime import ROOT
from .training import atomic_save, seed_all, sha256


BUNDLE_FORMAT = 'stage09_recovery35_disposable_preflight_bundle_v1'
MAX_SECONDS = 1620
SHUTDOWN_RESERVE = 12


def require_formal_rejection(controller, bundle):
    for override in (False, True):
        try:
            controller.restore(bundle, allow_preflight=override)
        except ValueError as error:
            require('preflight' in str(error).lower(), 'Bundle rejected for an unrelated reason')
        else:
            raise GateFailure('Disposable bundle was accepted by formal restore')


def save_bundle(controller, path):
    """Only the outer envelope goes to disk; no loadable formal state is published."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    snapshot = controller.checkpoint()
    materialize_seconds = time.perf_counter() - started
    payload_digest, payload_timing = measure(lambda: fingerprint(snapshot), controller.trainer.device)
    bundle = dict(format=BUNDLE_FORMAT, preflight_only=True, formal_restore_forbidden=True,
                  branch=controller.branch, snapshot_sha256=payload_digest, snapshot=snapshot)
    require_formal_rejection(controller, bundle)
    def publish():
        atomic_save(path, bundle)
        # Include a file fsync, not just a page-cache write, in the I/O profile.
        with path.open('rb+') as stream:
            os.fsync(stream.fileno())
    _, timing = measure(publish, controller.trainer.device)
    digest, hash_timing = measure(lambda: sha256(path), controller.trainer.device)
    return snapshot, dict(path=str(path), bytes=path.stat().st_size, sha256=digest,
        seconds=timing['seconds'], state_materialization_seconds=materialize_seconds,
        payload_fingerprint_seconds=payload_timing['seconds'], file_sha256_seconds=hash_timing['seconds'],
        preflight_only=True, formal_restore_rejected=True, fsync_completed=True)


def load_bundle(controller, path):
    bundle, timing = measure(lambda: torch.load(path, map_location='cpu', weights_only=False),
                             controller.trainer.device)
    if (bundle.get('format') != BUNDLE_FORMAT or bundle.get('preflight_only') is not True
            or bundle.get('formal_restore_forbidden') is not True or bundle.get('branch') != controller.branch):
        raise ValueError('Not this branch\'s disposable preflight bundle')
    require_formal_rejection(controller, bundle)
    snapshot = bundle['snapshot']
    digest, hash_timing = measure(lambda: fingerprint(snapshot), controller.trainer.device)
    require(digest == bundle['snapshot_sha256'], 'Saved bundle payload digest differs')
    timing['payload_fingerprint_seconds'] = hash_timing['seconds']
    return snapshot, timing


def h0_t_gradients(trainer):
    require(not trainer.alignment.mixer.mixer_enabled, 'C T acceptance must keep Mixer disabled')
    require(all(p.grad is None for p in trainer.model.parameters()), 'T reached frozen recommender gradients')
    base = trainer.alignment.mixer.base_codebooks.grad
    require(base is not None and torch.isfinite(base).all() and base.abs().sum() > 0,
            'H0 base codebook has no finite nonzero T gradient')
    require(all(p.grad is None for n, p in trainer.alignment.named_parameters()
                if n != 'mixer.base_codebooks'), 'Disabled H0 Mixer received a gradient')
    return dict(base_codebooks=dict(gradient_l1=float(base.double().abs().sum())),
                disabled_mixer_has_no_gradient=True)


def checked_step(controller, batches, kind):
    """Use controller.step, checking inactive state and real Decoder weight updates."""
    trainer = controller.trainer
    frozen = trainer.alignment if kind == 'R' else trainer.model
    inactive_optimizer = trainer.t_optimizer if kind == 'R' else trainer.r_optimizer
    active = trainer.model if kind == 'R' else trainer.alignment
    frozen_hash, opt_hash = fingerprint(frozen.state_dict()), fingerprint(inactive_optimizer.state_dict())
    hard_hash = fingerprint(trainer.alignment.hard_codes)
    before = parameter_snapshot(active)
    phase, update = controller.progress['phase'], controller.progress['update']
    counts = (trainer.r_updates, trainer.t_updates)
    values, timing = measure(lambda: controller.step(batches, kind), trainer.device)
    require(fingerprint(frozen.state_dict()) == frozen_hash, f'{kind}: frozen module changed')
    require(fingerprint(inactive_optimizer.state_dict()) == opt_hash, f'{kind}: inactive optimizer changed')
    require(fingerprint(trainer.alignment.hard_codes) == hard_hash, 'Premature SID publication inside an update')
    require((trainer.r_updates, trainer.t_updates) ==
            (counts[0] + (kind == 'R'), counts[1] + (kind == 'T')), 'Incorrect update clocks')
    require(controller.progress['phase'] == phase and controller.progress['update'] == update + 1,
            'Controller did not preserve original phase/update cursor')
    gradients = gradient_groups(trainer, 'R') if kind == 'R' else h0_t_gradients(trainer)
    changed = changed_parameters(before, active)
    require(changed, 'Active parameters did not change')
    if kind == 'R':
        check_decoder_updates(changed, layers=trainer.config['model']['decoder_n_layer'])
    else:
        require(changed == ['mixer.base_codebooks'], 'H0 T update changed something beyond base codebooks')
    return dict(**timing, kind=kind, original_phase_index=phase, update=update + 1,
                values=values, gradients=gradients, changed_parameters=changed,
                frozen_module_and_optimizer_unchanged=True, hard_sids_unchanged=True)


def disk_replay_next(controller, batches, kind, directory):
    """Mutate once, strictly restore the disk payload, then repeat the same update."""
    path = Path(directory) / f'before_replay_{kind}.preflight.bundle.pt'
    saved, io = save_bundle(controller, path)
    saved_hash = fingerprint(saved)
    expected_record = checked_step(controller, batches, kind)
    expected = controller.checkpoint()
    loaded, load_timing = load_bundle(controller, path)
    exact_equal(saved, loaded, 'saved_disk_payload')
    _, restore_timing = measure(lambda: controller.restore(loaded), controller.trainer.device)
    exact_equal(loaded, controller.checkpoint(), 'strict_restored_state')
    actual_record = checked_step(controller, batches, kind)
    exact_equal(expected_record['values'], actual_record['values'], f'{kind}_next_update_values')
    exact_equal(expected, controller.checkpoint(), f'{kind}_next_update_full_state')
    require(fingerprint(saved) == saved_hash and fingerprint(loaded) == saved_hash,
            'A restored Adam optimizer mutated the reusable input snapshot')
    return actual_record, dict(kind=kind, save=io, load=load_timing, restore=restore_timing,
        exact_loaded_state=True, exact_next_update=True, reusable_snapshot_unchanged=True,
        extra_discarded_update_seconds=expected_record['seconds'], extra_discarded_updates=1,
        formal_restore_rejected=True)


def skip_frozen_boundary(controller):
    require(controller.skip_t, 'Expected a real frozen T boundary')
    before = controller.checkpoint()
    controller.prepare_boundary()
    controller.complete_phase()
    after = controller.checkpoint()
    for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'r_updates', 't_updates'):
        exact_equal(before['inner'][key], after['inner'][key], 'skip_T/' + key)
    require(controller.row['kind'] == 'R', 'Skipping T did not advance to its original R phase')
    return dict(from_phase=before['inner']['progress']['phase'], to_phase=controller.progress['phase'],
                r_updates=controller.trainer.r_updates, t_updates=controller.trainer.t_updates,
                exact_non_cursor_state_unchanged=True)


def phase_batches(controller, data, indices):
    from .run import microbatches
    cfg = controller.trainer.config['training']
    begin = controller.progress['update'] * cfg['effective_batch_size']
    selected = indices[begin:begin + cfg['effective_batch_size']]
    require(len(selected) == cfg['effective_batch_size'], 'Short throughput profiling batch')
    started = time.perf_counter()
    batches = list(microbatches(data, selected, cfg['microbatch_size']))
    return batches, dict(data_prepare_seconds=time.perf_counter() - started,
        selected_indices_sha256=fingerprint(selected), effective_lists=len(selected),
        microbatch=cfg['microbatch_size'], accumulation=len(batches),
        first_sample_ids=batches[0]['sample_id'][:4])


def start_phase(controller, data):
    from .run import phase_indices
    require(controller.progress['update'] == 0 and not controller.progress['boundary_prepared'],
            'Profiling phase must start at the original boundary')
    phase = controller.progress['phase']
    indices = phase_indices(controller.trainer.config, len(data), phase, controller.row)
    seed_all(controller.trainer.config['seed'] + 100003 * (phase + 1))
    return indices


def summarize_profile(records):
    warm = records[1:] or records
    return dict(records=records, updates=len(records),
        median_update_seconds=statistics.median(r['seconds'] for r in records),
        median_end_to_end_seconds=statistics.median(r['seconds'] + r['data_prepare_seconds'] for r in records),
        warm_median_update_seconds=statistics.median(r['seconds'] for r in warm),
        warm_records=len(warm), timing_scope='Update includes H2D, forward, backward and optimizer; '
            'data preparation is separate. State/gradient auditing is excluded from throughput, included in overall wall time.')


def record_update(controller, directory, row):
    """Retain original logger fields once per logical update, including C T1..40."""
    values = dict(row['values'], branch=controller.branch, preflight_only=True,
                  epoch=controller.trainer.plan[row['original_phase_index']]['epoch'],
                  phase_index=row['original_phase_index'], update=row['update'], seconds=row['seconds'])
    with (Path(directory) / 'training.preflight.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(values, allow_nan=False) + '\n')
        stream.flush()


def profile_training(controller, data, directory, repeats, log):
    report = {}
    if controller.branch in ('A', 'B'):
        report['frozen_T_boundary'] = skip_frozen_boundary(controller)
    else:
        require(controller.progress['phase'] == 5 and controller.row['kind'] == 'T'
                and controller.row['updates'] == 40 and controller.trainer.t_updates == 0,
                'C must execute its unchanged original 40-T block')
        initial_r = controller.trainer.r_updates
        indices, records = start_phase(controller, data), []
        while controller.progress['update'] < 40:
            batches, prep = phase_batches(controller, data, indices)
            with log.stage(f'C/T{controller.progress["update"] + 1}'):
                if controller.progress['update'] == repeats:
                    row, report['T_disk_replay'] = disk_replay_next(controller, batches, 'T', directory)
                else:
                    row = checked_step(controller, batches, 'T')
            row.update(prep)
            records.append(row)
            record_update(controller, directory, row)
        report['T'] = summarize_profile(records)
        with log.stage('C/original_T40_catalog_boundary'):
            _, refresh_timing = measure(controller.prepare_boundary, controller.trainer.device)
            controller.complete_phase()
            require((controller.progress['phase'], controller.trainer.r_updates, controller.trainer.t_updates)
                    == (6, initial_r, 40), 'C did not reach the real R6/T40 boundary')
            _, report['T40_boundary_checkpoint'] = save_bundle(controller, Path(directory) / 'C_T40.preflight.bundle.pt')
            report['T40_refresh'] = refresh_timing
    indices, records = start_phase(controller, data), []
    for _ in range(repeats):
        batches, prep = phase_batches(controller, data, indices)
        with log.stage(f'{controller.branch}/R_profile{controller.progress["update"] + 1}'):
            row = checked_step(controller, batches, 'R')
        row.update(prep)
        records.append(row)
        record_update(controller, directory, row)
    batches, prep = phase_batches(controller, data, indices)
    with log.stage(f'{controller.branch}/R_disk_replay'):
        replay_row, report['R_disk_replay'] = disk_replay_next(controller, batches, 'R', directory)
    replay_row.update(prep)
    record_update(controller, directory, replay_row)
    report['R'] = summarize_profile(records)
    report['R_disk_replay']['actual_update'] = replay_row
    report['short_endpoint'] = dict(phase=controller.progress['phase'], update=controller.progress['update'],
        r_updates=controller.trainer.r_updates, t_updates=controller.trainer.t_updates)
    return report


def diagnostic_signature(trainer):
    modules = [*trainer.model.modules(), *trainer.alignment.modules()]
    return dict(modes=[m.training for m in modules],
        requires_grad=[p.requires_grad for root in (trainer.model, trainer.alignment) for p in root.parameters()],
        gradients=fingerprint([p.grad for root in (trainer.model, trainer.alignment) for p in root.parameters()]))


def profile_decode(controller, assets, count, directory, log):
    from .data import KuaiRecListDataset
    from .run import make_catalog, microbatches
    trainer = controller.trainer
    validation = KuaiRecListDataset(trainer.config['dataset'], 'validation', assets=assets)
    rows = fixed_validation_rows(validation.table.column('sample_id').to_pylist(), count)
    write_json(Path(directory) / 'validation_rows.json', dict(preflight_only=True,
        selection='sha256_utf8_sample_id_then_sample_id', count=count,
        validation_sha256=sha256(Path(trainer.config['dataset']) / 'validation.parquet'), rows=rows))
    indices = [r['index'] for r in rows]
    before, modes = controller.checkpoint(), diagnostic_signature(trainer)
    records, first = [], None
    with diagnostic_context(trainer), log.stage(f'{controller.branch}/FP32_beam32_profile'):
        catalog, catalog_time = measure(lambda: make_catalog(trainer.alignment, assets), trainer.device)
        for offset in range(0, count, 32):
            started = time.perf_counter()
            batch = next(microbatches(validation, indices[offset:offset + 32], 32))
            prepare = time.perf_counter() - started
            output, timing = measure(lambda: generate_target_free(trainer, batch, catalog, 32), trainer.device)
            checked = check_generation(output, assets.n_items, len(batch['sample_id']))
            signature = fingerprint(dict(items=output.item_ids, scores=output.scores,
                                         complete=output.complete_mask, beam=output.beam_mask))
            if first is None:
                first = signature
            records.append(dict(**timing, data_prepare_seconds=prepare,
                end_to_end_seconds=timing['seconds'] + prepare, batch_size=32, beam=32,
                inference_precision='fp32', sample_ids=batch['sample_id'], output_sha256=signature,
                complete_lists=checked['complete_lists'], counters=checked['counters']))
    exact_equal(before, controller.checkpoint(), 'decode_state_unchanged')
    exact_equal(modes, diagnostic_signature(trainer), 'decode_modes_gradients_unchanged')
    with diagnostic_context(trainer), log.stage(f'{controller.branch}/FP32_exact_output_repeat'):
        catalog = make_catalog(trainer.alignment, assets)
        batch = next(microbatches(validation, indices[:32], 32))
        output = generate_target_free(trainer, batch, catalog, 32)
        require(fingerprint(dict(items=output.item_ids, scores=output.scores,
                                 complete=output.complete_mask, beam=output.beam_mask)) == first,
                'Repeated FP32 decode differs')
    exact_equal(before, controller.checkpoint(), 'repeat_decode_state_unchanged')
    exact_equal(modes, diagnostic_signature(trainer), 'repeat_decode_modes_unchanged')
    return dict(examples=count, batch_size=32, beam=32, inference_precision='fp32',
        records=records, catalog_timing=catalog_time,
        median_batch_seconds=statistics.median(r['seconds'] for r in records),
        median_end_to_end_seconds=statistics.median(r['end_to_end_seconds'] for r in records),
        exact_output_repeat=True, full_training_state_rng_modes_gradients_unchanged=True,
        quality_metric=False, scope='Fixed small validation requests for throughput only; not full validation or model selection.')


def branch_worker(args, branch, log):
    directory = Path(args.output) / branch
    directory.mkdir(exist_ok=False)
    parent = getattr(args, 'parent_' + branch.lower())
    with log.stage(f'{branch}/authenticate_and_restore'):
        (controller, assets, data), build_time = measure(lambda: build_recovery35(
            parent, branch, 'cuda:0', args.dataset, args.assets, ancestor_path=args.ancestor), 'cuda:0')
    trainer = controller.trainer
    cfg = trainer.config
    require(trainer.precision == 'bf16' and not controller.inspection_only, 'Real BF16 continuation required')
    require((cfg['architecture'], cfg['training']['effective_batch_size'], cfg['training']['microbatch_size'],
             cfg['evaluation']['batch_size'], cfg['model']['encoder_n_layer'], cfg['model']['decoder_n_layer'])
            == ('H', 1024, 256, 32, 1, 4), 'Formal model/batch dimensions changed')
    require((assets.n_items, len(data)) == (10728, 709443) and not assets.manifest.get('toy', False),
            'Real fixed dataset/assets required')
    require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
            'Strict deterministic algorithms and disabled TF32 required')
    initial, initial_io = save_bundle(controller, directory / 'initial.preflight.bundle.pt')
    for key in controller.parent_hashes:
        require(fingerprint(initial['inner'][key]) == controller.parent_hashes[key], 'Fork changed parent ' + key)
    report = dict(format='stage09_recovery35_branch_preflight_v1', status='running', branch=branch,
        name=BRANCHES[branch]['name'], preflight_only=True, formal_training_started=False,
        parent_sha256=sha256(parent), build_timing=build_time, initial_checkpoint=initial_io,
        source_binding=controller.execution_binding, initial_state_sha256=fingerprint(initial),
        initial_counts=dict(R=trainer.r_updates, T=trainer.t_updates, phase=controller.progress['phase']),
        original_scheduler=initial['inner']['scheduler'])
    write_json(directory / 'report.json', report)
    report['training_profile'] = profile_training(controller, data, directory, args.profile_updates, log)
    write_json(directory / 'report.json', report)
    # Throughput of inference is measured at the authenticated input, after
    # discarding every acceptance update. No short-run weight is selected.
    controller.restore(initial)
    exact_equal(initial, controller.checkpoint(), 'discard_training_updates')
    report['decode_profile'] = profile_decode(controller, assets, args.validation_samples, directory, log)
    exact_equal(initial, controller.checkpoint(), 'final_parent_state')
    require(sha256(parent) == BRANCHES[branch]['parent_sha256'], 'Original parent file changed')
    report.update(status='passed', finished_at_exact_parent_state=True,
                  formal_resume_from_preflight_forbidden=True)
    write_json(directory / 'report.json', report)
    log.event('branch_passed', branch=branch)
    del trainer, controller, data, assets, initial
    gc.collect()
    torch.cuda.empty_cache()
    return report


def full_h100_identity():
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose exactly one authorized GPU')
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, 'uuid', ''))
    require('h100' in props.name.lower() and torch.cuda.is_bf16_supported(), 'H100 BF16 required')
    require(props.total_memory >= 75_000_000_000, 'A full H100 with at least 75 GB is required')
    require(uuid and 'mig' not in uuid.lower()
            and 'mig' not in os.environ.get('CUDA_VISIBLE_DEVICES', '').lower(), 'MIG is not permitted')
    # PyTorch may return the UUID without the GPU- prefix; correlate it with
    # the physical GPU and its explicit MIG mode, not the host GPU index.
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,mig.mode.current',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    canonical = lambda value: value.strip().lower().removeprefix('gpu-')
    rows = [row for row in csv.reader(raw.splitlines()) if len(row) == 4 and canonical(row[0]) == canonical(uuid)]
    require(len(rows) == 1 and rows[0][3].strip().lower() == 'disabled',
            'Cannot prove the visible H100 is a full physical GPU with MIG disabled')
    return dict(name=props.name, uuid=uuid, total_memory_bytes=props.total_memory,
                visible_gpu_count=1, mig_mode='disabled', nvidia_smi_row=[v.strip() for v in rows[0]])


def worker(args):
    if os.environ.get('LISTWISE_RECOVERY35_PREFLIGHT_SUPERVISED') != '1':
        raise RuntimeError('Use the bounded preflight supervisor')
    identity = full_h100_identity()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    log = StageLog(args.output, timeout_seconds=240)
    report = dict(format='stage09_recovery35_preflight_v1', status='running', preflight_only=True,
        formal_training_started=False, gpu=identity,
        torch_version=torch.__version__, cuda_version=torch.version.cuda, max_seconds=args.max_seconds,
        profile_updates=args.profile_updates, validation_samples=args.validation_samples,
        sequence=['A', 'B', 'C'], branches={})
    write_json(Path(args.output) / 'report.json', report)
    for branch in 'ABC':
        report['branches'][branch] = branch_worker(args, branch, log)
        write_json(Path(args.output) / 'report.json', report)
    report.update(status='passed', worker_seconds=time.monotonic() - log.started)
    write_json(Path(args.output) / 'report.json', report)


def run_bounded_process(command, *, directory, environment, deadline):
    """Terminate then kill the sole worker before the caller's hard deadline."""
    reason = None
    with (Path(directory) / 'worker.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline - SHUTDOWN_RESERVE:
                    reason = 'overall_preflight_deadline'
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(.1)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    return dict(returncode=process.returncode, reason=reason)


def supervise(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Preflight output must be a new isolated directory')
    protected = [Path(p).resolve().parent for p in (args.parent_a, args.parent_b, args.parent_c, args.ancestor)]
    protected += [ROOT.resolve() / 'scripts', ROOT.resolve() / 'external']
    protected += [Path(p).resolve() for p in (args.dataset, args.assets) if p is not None]
    if any(output.is_relative_to(p) for p in protected):
        raise ValueError('Preflight output cannot be inside source/run/data/asset directories')
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    args.output = str(output)
    write_json(output / 'request.json', vars(args))
    env = os.environ.copy()
    env.update(LISTWISE_RECOVERY35_PREFLIGHT_SUPERVISED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    command = [sys.executable, '-B', '-u', '-m', 'scripts.listwise_diffgrm.recovery35_preflight', '--worker']
    for key, value in vars(args).items():
        if key != 'worker' and value is not None:
            command.extend(['--' + key.replace('_', '-'), str(value)])
    result = run_bounded_process(command, directory=output, environment=env, deadline=started + args.max_seconds)
    elapsed = time.monotonic() - started
    success = False
    if (result['returncode'] == 0 and result['reason'] is None and elapsed <= args.max_seconds
            and (output / 'report.json').is_file()):
        saved = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        success = saved.get('status') == 'passed' and set(saved.get('branches', {})) == set('ABC')
        success = success and all(saved['branches'][b].get('status') == 'passed' for b in 'ABC')
    result.update(status='timeout' if result['reason'] else ('passed' if success else 'failed'),
        preflight_only=True, wall_seconds=elapsed, max_seconds=args.max_seconds,
        recommended_platform_max_time_hours=.45,
        allocated_gpu_seconds_during_script=elapsed,
        accounting_scope='One visible GPU during the supervised script only. Platform initialization/release '
                         'is additional and must fit within the separately approved cumulative 1800 GPU seconds.')
    write_json(output / 'supervisor.json', result)
    print(json.dumps(result), flush=True)
    return 0 if success else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-a', 'parent-b', 'parent-c', 'ancestor', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--assets')
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--profile-updates', type=int, default=3)
    parser.add_argument('--validation-samples', type=int, default=128, choices=(32, 64, 128))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or not 60 <= args.max_seconds <= MAX_SECONDS or not 3 <= args.profile_updates <= 5:
        parser.error('max-seconds must be 60..1620; profile-updates must be 3..5')
    if args.worker:
        try:
            worker(args)
        except BaseException as error:
            write_json(Path(args.output) / 'failure.json', dict(preflight_only=True,
                error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()))
            raise
        return 0
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
