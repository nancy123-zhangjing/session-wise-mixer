"""R2 paired-process resource diagnostic; never a formal train runner.

Run only after CPU acceptance and release of both shared warmup Jobs. Scientific
modules are imported from the frozen Goal code tree. Only this diagnostic's new
directory is writable; no checkpoint is ever saved. No MPS or platform API calls.
"""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')
OUTPUT = ROOT / 'diagnostics/shared_gpu_pair_probe_r2'
REFERENCE = ROOT / 'delivery/SHARED_GPU_PROBE_REPORT_R1.json'
REFERENCE_SHA = '2e0cc1d3cae3572e442f13fec7c98151ddb0ab8438ec1d145b94d8779158161f'
CONFIG_SHA = 'd8e3a40f9c7ac446dbca6f4e8b01811ee4598c0d4d07e18310f524afda9bd2bb'
BINDING_SHA = '871e480d2a77d7e22bfed091d02065f0cdc42eb1db42cf20d43519634959e80a'
ACCEPTANCE_SHA = '2c255470cdd58b2c13dd50fb130aa750f1609306720081ab0e3cd33050cb2831'
ARMS = ('J0', 'J1', 'H0', 'H1')
PAIRS = (('J0', 'H0'), ('J1', 'H1'))
UPDATES, MICRO, EFFECTIVE, EVAL_ROWS, EVAL_BATCH = 20, 256, 1024, 512, 32


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest_file(path, opener=open):
    digest = hashlib.sha256()
    with opener(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def owned(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe owned relative path')
    path = ROOT / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()), 'Owned path escapes root')
    return path


def validate_arguments(args):
    for name in ('warmup_j_sha256', 'warmup_h_sha256'):
        require(bool(re.fullmatch('[0-9a-f]{64}', getattr(args, name))), 'Replace both warmup SHA placeholders')
    require(90 <= args.max_seconds <= 480, 'Total process deadline must be between 90 and 480 seconds')


def precheck(args):
    validate_arguments(args)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Diagnostic output already exists; no overwrite/resume')
    require(not os.environ.get('CUDA_MPS_PIPE_DIRECTORY'), 'This probe must not opt into MPS')
    for arm in ARMS:
        require(not owned('runs/' + arm).exists(), 'A formal branch directory already exists: ' + arm)
    helper = owned('delivery/formal_branch_acceptance.py')
    require(digest_file(helper) == ACCEPTANCE_SHA, 'CPU acceptance helper SHA changed')
    manifest_path = owned('delivery/FORMAL_BRANCH_BINDING.json')
    require(digest_file(manifest_path) == BINDING_SHA, 'Full binding manifest SHA changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('_probe_source_check', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.source_check(manifest)  # stdlib-only path; do not call its CPU torch helpers here.
    require(digest_file(owned('config/efficiency_r2.json')) == CONFIG_SHA, 'Efficiency config SHA changed')
    warmups = {}
    for architecture in ('J', 'H'):
        directory = owned('runs/warmup_' + architecture)
        require(not (directory / '.running').exists(), 'Warmup has an active or unresolved runner lock')
        complete = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
        require((complete.get('arm'), complete.get('r_updates'), complete.get('t_updates'), complete.get('best_epoch'))
                == ('warmup', 3465, 0, 5), 'Warmup is not complete at R5')
        expected = getattr(args, 'warmup_' + architecture.lower() + '_sha256')
        path = directory / 'last.pt'
        require(digest_file(path) == expected, 'Accepted warmup checkpoint SHA changed')
        warmups[architecture] = dict(path=str(path), sha256=expected, complete=complete)
    require(owned('delivery/SHARED_GPU_PROBE_REPORT_R1.json').is_file(), 'R1 report is not a regular owned file')
    require(digest_file(REFERENCE) == REFERENCE_SHA, 'Read-only R1 report SHA differs')
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    require(digest_file(REFERENCE) == REFERENCE_SHA, 'R1 report changed during read')
    validate_reference(reference, warmups)
    return manifest, warmups, reference


def validate_reference(reference, warmups):
    # Accept only the bound historical measurement record, never relabel it PASS.
    require(reference.get('status') == 'failed'
            and reference.get('error') == 'A worker failed to exit cleanly', 'Unexpected R1 failure scope')
    require(reference.get('arms') == list(ARMS) and len(reference.get('blocks', [])) == 15,
            'R1 full measurement inventory differs')
    require((reference.get('microbatch'), reference.get('effective_batch'), reference.get('updates_per_phase'))
            == (MICRO, EFFECTIVE, UPDATES), 'R1 update settings differ')
    evaluation = reference['evaluation']
    require((evaluation['split'], evaluation['prefix_rows'], evaluation['batch'], evaluation['beam'],
             evaluation['actual_inference_precision'], evaluation['test_read'])
            == ('validation', EVAL_ROWS, EVAL_BATCH, 32, 'fp32', False), 'R1 validation settings differ')
    for arm in ARMS:
        require(reference['warmups'][arm[0]] == warmups[arm[0]], 'R1 warmup identity differs')
        require(reference['read_only_warmup_shutdown_checks'][arm]['warmup_sha256'] == warmups[arm[0]]['sha256'],
                'R1 shutdown warmup check is missing')
        require(reference['processes'][arm]['exitcode'] == -15 and not reference['processes'][arm]['alive'],
                'Unexpected R1 worker shutdown outcome')
    for phase in ('R', 'T', 'validation'):
        for arm in ARMS:
            require(reference['comparison'][phase]['arms'][arm]['fingerprints_equal'] is True,
                    'R1 recorded a measurement mismatch')
            baseline = solo_reference(reference, phase, arm)
            for part, field in (('prepared', 'initial_state_sha256'), ('inspected', 'final_state_sha256')):
                require(bool(re.fullmatch('[0-9a-f]{64}', baseline[part][field])), 'Invalid R1 state fingerprint')
            if phase == 'validation':
                require(bool(re.fullmatch('[0-9a-f]{64}', baseline['inspected']['prediction_records_sha256'])),
                        'Invalid R1 prediction fingerprint')


def solo_reference(reference, phase, arm):
    matches = [block for block in reference['blocks']
               if block['mode'] == 'solo' and block['phase'] == phase and list(block['arms']) == [arm]]
    require(len(matches) == 1, 'R1 must have exactly one matching solo record')
    return matches[0]['arms'][arm]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def receive(clients, expected, deadline):
    pending, results = dict(clients), {}
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Diagnostic stage deadline reached; reserve time for bounded shutdown')
        ready = wait(list(pending.values()), timeout=min(remaining, 1.0))
        for connection in ready:
            arm = next(key for key, value in pending.items() if value is connection)
            try:
                message = connection.recv()
            except EOFError as exc:
                raise RuntimeError(arm + ' worker exited without a result') from exc
            require(message.get('event') != 'error', arm + ': ' + message.get('error', 'worker error'))
            require(message.get('event') == expected and message.get('arm') == arm,
                    'Unexpected worker protocol message: ' + repr(message))
            results[arm] = message
            del pending[arm]
    return results


def block(clients, mode, phase, deadline):
    for connection in clients.values():
        connection.send(dict(command='prepare', mode=mode, phase=phase))
    prepared = receive(clients, 'prepared', deadline)
    for connection in clients.values():
        connection.send(dict(command='go'))
    measured = receive(clients, 'measured', deadline)
    # Delay device-to-host state hashing until every process has finished timing.
    for connection in clients.values():
        connection.send(dict(command='inspect'))
    inspected = receive(clients, 'inspected', deadline)
    return dict(mode=mode, phase=phase, arms={arm: dict(prepared=prepared[arm],
                measured=measured[arm], inspected=inspected[arm]) for arm in clients},
                makespan_seconds=max(m['ended'] for m in measured.values())
                    - min(m['started'] for m in measured.values()),
                start_skew_seconds=max(m['started'] for m in measured.values())
                    - min(m['started'] for m in measured.values()))


def state_fingerprint(trainer, fingerprint, rng_state):
    return fingerprint(dict(model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict(),
        r_optimizer=trainer.r_optimizer.state_dict(), t_optimizer=trainer.t_optimizer.state_dict(),
        scheduler=trainer.scheduler.state_dict(), rng=rng_state(),
        r_updates=trainer.r_updates, t_updates=trainer.t_updates))


def worker(connection, arm, warmup, expected_binding):
    log = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log = (OUTPUT / (arm + '.log')).open('x', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = log
        sys.path.insert(0, str(owned('code')))
        import resource
        import torch
        from scripts.listwise_diffgrm.runtime import read_config
        from scripts.listwise_diffgrm.run import build_context, evaluate, microbatches
        from scripts.listwise_diffgrm.training import rng_state
        from scripts.listwise_diffgrm.preflight import fingerprint, fixed_indices, measure
        require(torch.cuda.device_count() == 1, 'Exactly one allocated visible GPU is required')
        torch.cuda.set_device(0)
        require('H100' in torch.cuda.get_device_name(0), 'The diagnostic requires the measured H100 hardware')
        config = read_config(owned('config/efficiency_r2.json'), architecture=arm[0], seed=2024)
        require((config['training']['microbatch_size'], config['training']['effective_batch_size'],
                 config['evaluation']['batch_size']) == (MICRO, EFFECTIVE, EVAL_BATCH), 'Frozen batch sizes changed')
        trainer, assets, data, binding = build_context(config, 'cuda:0', 'bf16', arm.endswith('1'))
        require(binding == expected_binding, 'Runtime source/config/data/assets binding differs')
        checkpoint = Path(warmup['path'])
        require(digest_file(checkpoint) == warmup['sha256'], 'Warmup SHA differs before read')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(digest_file(checkpoint) == warmup['sha256'], 'Warmup changed during read')
        indices = fixed_indices(len(data), UPDATES * EFFECTIVE, 2024 + 1701)
        index_hash = hashlib.sha256(indices.tobytes()).hexdigest()
        connection.send(dict(event='ready', arm=arm, pid=os.getpid(), device='cuda:0',
                             training_index_sha256=index_hash, warmup_sha256=warmup['sha256']))
        mode = phase = last_result = None
        while True:
            message = connection.recv()
            command = message['command']
            if command == 'stop':
                require(digest_file(checkpoint) == warmup['sha256'], 'Warmup changed at shutdown')
                connection.send(dict(event='stopped', arm=arm, warmup_sha256=warmup['sha256']))
                break
            if command == 'prepare':
                mode, phase = message['mode'], message['phase']
                require(mode == 'pair' and phase in ('R', 'T', 'validation'), 'Invalid stage')
                # deepcopy is necessary: optimizer restoration can retain CPU step-tensor references.
                trainer.restore(copy.deepcopy(state), binding, arm=arm, fork=True)
                trainer.r_optimizer.zero_grad(set_to_none=True)
                trainer.t_optimizer.zero_grad(set_to_none=True)
                require(all(p.dtype == torch.float32 for p in trainer.model.parameters()), 'Master parameter dtype changed')
                torch.cuda.synchronize()
                connection.send(dict(event='prepared', arm=arm, mode=mode, phase=phase,
                    initial_state_sha256=state_fingerprint(trainer, fingerprint, rng_state),
                    r_updates=trainer.r_updates, t_updates=trainer.t_updates))
            elif command == 'go':
                require(phase is not None, 'Stage was not prepared')
                cpu0 = resource.getrusage(resource.RUSAGE_SELF)
                started = time.perf_counter()
                if phase in ('R', 'T'):
                    records = []
                    for iteration in range(UPDATES):
                        selected = indices[iteration * EFFECTIVE:(iteration + 1) * EFFECTIVE]
                        def real_update():
                            # Rebuild real CPU batches every step INSIDE the timed region.
                            batches = list(microbatches(data, selected, MICRO))
                            return trainer.update(batches, phase)
                        values, timing = measure(real_update, trainer.device)
                        records.append(dict(iteration=iteration, **timing, values=values))
                    result = dict(records=records, updates=UPDATES, effective_lists=UPDATES * EFFECTIVE,
                        median_update_seconds=statistics.median(row['seconds'] for row in records),
                        warm_median_update_seconds=statistics.median(row['seconds'] for row in records[2:]),
                        timing_scope='real CPU microbatches + H2D + forward/backward + optimizer')
                else:
                    destination = OUTPUT / (mode + '_' + arm + '_validation_beam32')
                    summary, timing = measure(lambda: evaluate(trainer, assets, split='validation',
                        destination=destination, beam=32, limit=EVAL_ROWS), trainer.device)
                    require(summary['examples'] == EVAL_ROWS and summary['diagnostic_prefix_limit'] == EVAL_ROWS
                        and summary['split'] == 'validation' and summary['beam'] == 32,
                        'Evaluation did not use the fixed diagnostic validation prefix')
                    result = dict(timing=timing, examples=EVAL_ROWS, destination=str(destination),
                        timing_scope='entire frozen evaluate: validation load/verification + catalog + real CPU batches'
                            ' + FP32 generation + metrics + gzip + summary write',
                        inference_precision=summary['inference_precision'])
                torch.cuda.synchronize()
                ended = time.perf_counter()
                cpu1 = resource.getrusage(resource.RUSAGE_SELF)
                last_result = dict(event='measured', arm=arm, mode=mode, phase=phase,
                    started=started, ended=ended, wall_seconds=ended - started,
                    cpu_seconds=(cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime),
                    maximum_rss_kib=cpu1.ru_maxrss, result=result)
                connection.send(last_result)
            elif command == 'inspect':
                require(last_result is not None, 'No measured result')
                result = dict(event='inspected', arm=arm, mode=mode, phase=phase,
                    final_state_sha256=state_fingerprint(trainer, fingerprint, rng_state))
                if phase == 'validation':
                    path = Path(last_result['result']['destination']) / 'per_sample.jsonl.gz'
                    # gzip container timestamps differ; compare the complete uncompressed records.
                    result['prediction_records_sha256'] = digest_file(path, gzip.open)
                connection.send(result)
                phase, last_result = None, None
            else:
                raise RuntimeError('Unknown worker command')
    except BaseException:
        error = traceback.format_exc()
        if log:
            print(error, file=log, flush=True)
        try:
            connection.send(dict(event='error', arm=arm, error=error))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if log:
            log.close()


def gpu_inventory():
    query = ['nvidia-smi', '--query-gpu=uuid,name,compute_mode,memory.total', '--format=csv,noheader,nounits']
    rows = subprocess.run(query, check=True, text=True, capture_output=True, timeout=5).stdout.strip().splitlines()
    require(len(rows) == 1, 'nvidia-smi must expose exactly the one allocated GPU')
    uuid, name, mode, memory = [part.strip() for part in rows[0].split(',')]
    require('H100' in name and mode.lower() == 'default', 'H100 default compute mode is required; do not change it')
    return dict(uuid=uuid, name=name, compute_mode=mode, total_memory_mib=int(memory))


def telemetry(stop, status, processes, inventory):
    query = ['nvidia-smi', '-i', inventory['uuid'],
             '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits']
    with (OUTPUT / 'telemetry.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        while not stop.is_set():
            row = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.perf_counter(),
                       stage=status.get('stage', 'initializing'), processes={})
            try:
                values = subprocess.run(query, check=True, text=True, capture_output=True, timeout=3).stdout.strip().split(',')
                row['gpu'] = dict(zip(('utilization_percent', 'used_memory_mib', 'total_memory_mib', 'power_watts'),
                                     [float(value.strip()) for value in values]))
                for arm, process in list(processes.items()):
                    path = Path('/proc') / str(process.pid)
                    try:
                        memory = [line for line in (path / 'status').read_text().splitlines() if line.startswith(('VmRSS:', 'Threads:'))]
                        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                        row['processes'][arm] = dict(pid=process.pid, status=memory,
                            cpu_seconds=(int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'))
                    except (OSError, ValueError):
                        row['processes'][arm] = dict(pid=process.pid, unavailable=True)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row['error'] = str(exc)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stop.wait(1.0)  # host blocking wait, never artificial GPU work


def compare(blocks, reference):
    require(len(blocks) == 6, 'R2 requires exactly six paired measurement blocks')
    result = {}
    for pair in PAIRS:
        key = '+'.join(pair)
        result[key] = {}
        for phase in ('R', 'T', 'validation'):
            matches = [b for b in blocks if b['mode'] == 'pair' and b['phase'] == phase and tuple(b['arms']) == pair]
            require(len(matches) == 1, 'Expected one R2 block per exact pair/phase')
            block_result, rows = matches[0], {}
            for arm in pair:
                solo = solo_reference(reference, phase, arm)
                shared = block_result['arms'][arm]
                require(solo['prepared']['initial_state_sha256'] == shared['prepared']['initial_state_sha256'],
                        arm + '/' + phase + ' initial state differs from R1 solo')
                require(solo['inspected']['final_state_sha256'] == shared['inspected']['final_state_sha256'],
                        arm + '/' + phase + ' final model/optimizer/RNG state differs from R1 solo')
                if phase == 'validation':
                    require(solo['inspected']['prediction_records_sha256'] == shared['inspected']['prediction_records_sha256'],
                            arm + ' validation predictions/metrics differ from R1 solo')
                a, b = solo['measured']['wall_seconds'], shared['measured']['wall_seconds']
                rows[arm] = dict(r1_solo_seconds=a, r2_pair_seconds=b, slowdown=b / a, fingerprints_equal=True)
                if phase in ('R', 'T'):
                    solo_warm = solo['measured']['result']['warm_median_update_seconds']
                    paired_warm = shared['measured']['result']['warm_median_update_seconds']
                    rows[arm]['steady_updates_3_to_20'] = dict(r1_solo_median_seconds=solo_warm,
                        r2_pair_median_seconds=paired_warm, slowdown=paired_warm / solo_warm,
                        scope='Real CPU batches and update; discard first two updates in both recorded modes')
            solo_total = sum(row['r1_solo_seconds'] for row in rows.values())
            result[key][phase] = dict(arms=rows, r1_solo_sequential_seconds=solo_total,
                r2_pair_makespan_seconds=block_result['makespan_seconds'],
                aggregate_speedup_over_r1_solo_sequential=solo_total / block_result['makespan_seconds'],
                estimated_slowdown_vs_two_independent_gpus=block_result['makespan_seconds']
                    / max(row['r1_solo_seconds'] for row in rows.values()),
                limitation='R1 and R2 are separate Jobs with possible host/cache differences. R2 has two active '
                    'workers plus two blocked resident CUDA contexts on one card; not a real two-card formal run.')
    return result


def wait_for_exit(processes, until):
    """One shared deadline, not one short timeout followed by immediate killing."""
    while True:
        alive = [process for process in processes.values() if process.is_alive()]
        if not alive or time.monotonic() >= until:
            return
        for process in alive:
            process.join(timeout=min(0.25, max(0.0, until - time.monotonic())))


def shutdown_children(processes, *, normal, deadline, normal_grace=60.0, term_grace=8.0):
    require(0 <= normal_grace <= 60 and 0 <= term_grace <= 8, 'Invalid shutdown grace')
    started = time.monotonic()
    normal_until = min(started + normal_grace, deadline - 10) if normal else started
    if normal:
        wait_for_exit(processes, normal_until)
    terminated = [arm for arm, process in processes.items() if process.is_alive()]
    for arm in terminated:
        processes[arm].terminate()
    wait_for_exit(processes, min(time.monotonic() + term_grace, deadline - 2))
    killed = [arm for arm, process in processes.items() if process.is_alive()]
    for arm in killed:
        processes[arm].kill()
    wait_for_exit(processes, deadline)
    for process in processes.values():
        process.join(timeout=0)
    return dict(normal_stop_acknowledged=normal, normal_grace_max_seconds=normal_grace if normal else 0,
        actual_normal_grace_budget_seconds=max(0.0, normal_until - started),
        cleanup_seconds=time.monotonic() - started, terminated_arms=terminated, killed_arms=killed,
        all_exited=all(not process.is_alive() for process in processes.values()))


def telemetry_summary(blocks):
    rows = [json.loads(line) for line in (OUTPUT / 'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    result = []
    def window_summary(start, end):
        selected = [row for row in rows if 'gpu' in row and start <= row['monotonic'] <= end]
        require(end > start and len(selected) >= 3, 'Insufficient GPU telemetry in a measured active window')
        utilization = [row['gpu']['utilization_percent'] for row in selected]
        return dict(start_monotonic=start, end_monotonic=end, duration_seconds=end - start,
            samples=len(selected), mean_gpu_utilization_percent=statistics.mean(utilization),
            min_gpu_utilization_percent=min(utilization), max_gpu_utilization_percent=max(utilization),
            peak_used_memory_mib=max(row['gpu']['used_memory_mib'] for row in selected))
    for block_result in blocks:
        starts = [arm['measured']['started'] for arm in block_result['arms'].values()]
        ends = [arm['measured']['ended'] for arm in block_result['arms'].values()]
        result.append(dict(mode=block_result['mode'], phase=block_result['phase'], arms=list(block_result['arms']),
            sampling_scope='periodic nvidia-smi samples, not platform average or proof of three-hour compliance',
            full_span=window_summary(min(starts), max(ends)),
            common_active_intersection=window_summary(max(starts), min(ends)),
            interpretation='Full span includes the faster worker waiting at the phase boundary; intersection '
                'includes only times when both workers execute their measured phase. Formal mixed-phase execution remains unmeasured.'))
    return dict(blocks=result, sampling_errors=sum('error' in row for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup-j-sha256', required=True)
    parser.add_argument('--warmup-h-sha256', required=True)
    parser.add_argument('--max-seconds', type=float, default=480)
    parser.add_argument('--precheck-only', action='store_true', help='stdlib-only CPU/source precheck, writes nothing')
    args = parser.parse_args()
    validate_arguments(args)
    started = time.monotonic()
    deadline = started + args.max_seconds
    work_deadline = deadline - 75  # reserve shared 60s exit wait plus bounded cleanup/report time
    def interrupted(signum, frame):
        raise TimeoutError('Diagnostic deadline or termination signal: ' + str(signum))
    if not args.precheck_only:
        require(sys.platform == 'linux', 'Actual probe execution is Linux-only')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    manifest, warmups, reference = precheck(args)
    if args.precheck_only:
        print(json.dumps(dict(status='cpu_precheck_passed', warmups=warmups, r1_report_sha256=REFERENCE_SHA,
            r1_status_preserved=reference['status'],
            warning='Platform release and full CPU warmup acceptance are separate required gates')), flush=True)
        return
    inventory = gpu_inventory()
    owned('diagnostics').mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    processes, clients, blocks = {}, {}, []
    stop, status = threading.Event(), dict(stage='initializing')
    monitor = threading.Thread(target=telemetry, args=(stop, status, processes, inventory), daemon=True)
    report = dict(status='running', purpose='resource_diagnostic_not_formal_training_or_model_selection',
        created_utc=datetime.now(timezone.utc).isoformat(), allocated_gpus=1, max_process_seconds=args.max_seconds,
        platform_cap_hours=0.15, warmups=warmups, inventory=inventory, arms=list(ARMS), pairs=PAIRS,
        reference=dict(path=str(REFERENCE), sha256=REFERENCE_SHA, status_preserved=reference['status'],
            failure_preserved=reference['error'], purpose='read-only completed R1 solo measurement records'),
        resource_proxy='Four worker contexts resident; only J0+H0 or J1+H1 active, never both pairs together',
        work_deadline_seconds=args.max_seconds - 75, shared_normal_exit_grace_max_seconds=60,
        microbatch=MICRO, effective_batch=EFFECTIVE, updates_per_phase=UPDATES,
        evaluation=dict(split='validation', prefix_rows=EVAL_ROWS, batch=EVAL_BATCH, beam=32,
            actual_inference_precision='fp32', checkpoint_restore_identity='bf16', test_read=False),
        blocks=blocks)
    write_json(OUTPUT / 'report.json', report)
    monitor.start()
    try:
        context = mp.get_context('spawn')
        for arm in ARMS:
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, arm, warmups[arm[0]], manifest['expected_bindings'][arm[0]]))
            process.start()
            child.close()
            processes[arm], clients[arm] = process, parent
            ready = receive({arm: parent}, 'ready', work_deadline)[arm]
            require(ready['training_index_sha256'] == reference['workers'][arm]['training_index_sha256'],
                    'Training sample indices differ from R1 solo')
            report.setdefault('workers', {})[arm] = ready
            write_json(OUTPUT / 'report.json', report)
        for pair in PAIRS:
            for phase in ('R', 'T', 'validation'):
                group = {arm: clients[arm] for arm in pair}
                status['stage'] = 'pair/' + phase + '/' + ','.join(group)
                result = block(group, 'pair', phase, work_deadline)
                blocks.append(result)
                write_json(OUTPUT / 'report.json', report)
                print(json.dumps(dict(event='probe_block_complete', stage=status['stage'],
                                      seconds=result['makespan_seconds'])), flush=True)
        report['comparison'] = compare(blocks, reference)
        status['stage'] = 'shutdown'
        for connection in clients.values():
            connection.send(dict(command='stop'))
        report['read_only_warmup_shutdown_checks'] = receive(clients, 'stopped', work_deadline)
        report['status'] = 'measurements_complete_pending_clean_exit'
        write_json(OUTPUT / 'report.json', report)
    except BaseException:
        report['status'], report['error'] = 'failed', traceback.format_exc()
        raise
    finally:
        # Outer timeout independently kills the whole spawn process group at 480s.
        # Leave time for telemetry/report closure, never restart a grace timer per worker.
        normal = report['status'] == 'measurements_complete_pending_clean_exit'
        report['shutdown'] = shutdown_children(processes, normal=normal, deadline=deadline - 5)
        stop.set()
        monitor.join(timeout=4)
        for connection in clients.values():
            connection.close()
        report['processes'] = {arm: dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for arm, p in processes.items()}
        if normal:
            clean = all(p.exitcode == 0 and not p.is_alive() for p in processes.values())
            clean = clean and not report['shutdown']['terminated_arms'] and not report['shutdown']['killed_arms']
            report['status'] = 'passed' if clean else 'failed'
            if not clean:
                report['error'] = 'A worker failed to exit within the shared bounded normal grace'
        report['elapsed_process_seconds'] = time.monotonic() - started
        report['gpu_accounting_scope'] = 'One allocation only; final allocation-to-release duration must be added by the operator'
        if report['status'] == 'passed':
            try:
                require(not monitor.is_alive(), 'Telemetry monitor did not exit')
                require(digest_file(REFERENCE) == REFERENCE_SHA, 'Read-only R1 reference changed')
                report['telemetry_summary'] = telemetry_summary(blocks)
            except BaseException:
                report['status'], report['error'] = 'failed', traceback.format_exc()
        write_json(OUTPUT / 'report.json', report)
        signal.setitimer(signal.ITIMER_REAL, 0)
    require(report['status'] == 'passed', 'Probe did not finish cleanly')


if __name__ == '__main__':
    main()
