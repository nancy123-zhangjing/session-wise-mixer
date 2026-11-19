"""Launch one separately allocated H100 job after formal-budget authorization.

This stdlib wrapper does not submit jobs, allocate resources, authorize spending,
retry, or resume training. Importing/inspecting it starts no subprocess. The
caller must obtain formal-budget approval before invoking this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import traceback


PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
MANIFEST_SHA256 = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
SOURCE_COMMIT = '8b5acc8853101975a0ba09fe904f990549643dcc'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_path(code, relative):
    relative_path = PurePosixPath(relative)
    if (not relative or '\\' in relative or relative_path.is_absolute()
            or '..' in relative_path.parts):
        raise RuntimeError(f'Invalid source path: {relative!r}')
    path = (code / relative).resolve()
    if not path.is_relative_to(code) or not path.is_file():
        raise RuntimeError(f'Execution source missing or outside code root: {relative}')
    return path


def verify_package(code, parent, manifest_sha):
    if manifest_sha.lower() != MANIFEST_SHA256:
        raise RuntimeError('This launcher is locked to the reviewed recovery manifest')
    manifest_file = code / 'RECOVERY_EXECUTION_MANIFEST.json'
    if sha256(manifest_file) != MANIFEST_SHA256:
        raise RuntimeError('Execution manifest bytes changed')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('format') != 'stage09_h1_recovery_execution_package_v1':
        raise RuntimeError('Unexpected execution manifest format')
    workspace = manifest['workspace']
    if (workspace.get('package_sources_uncommitted') is not False
            or workspace.get('commit') != SOURCE_COMMIT):
        raise RuntimeError('Execution requires the reviewed committed recovery source')
    expected_parent = manifest['parent']
    if (expected_parent.get('sha256') != PARENT_SHA256 or expected_parent.get('arm') != 'H1'
            or expected_parent.get('r_updates') != 13860 or expected_parent.get('t_updates') != 200
            or parent.stat().st_size != expected_parent['bytes'] or sha256(parent) != PARENT_SHA256):
        raise RuntimeError('Parent must be the authenticated H1 R20/T200 checkpoint')
    files = manifest['execution_files']
    if len(files) != manifest['execution_file_count']:
        raise RuntimeError('Execution file count differs from manifest')
    for relative, expected in files.items():
        path = source_path(code, relative)
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise RuntimeError(f'Execution source bytes changed: {relative}')
    metadata = {'RECOVERY_EXECUTION_MANIFEST.json', 'RECOVERY_PARENT_BINDING.json'}
    observed = {path.relative_to(code).as_posix() for path in code.rglob('*') if path.is_file()}
    if observed != set(files) | metadata:
        raise RuntimeError('Code tree contains missing or unbound files')
    binding_file = code / 'RECOVERY_PARENT_BINDING.json'
    if sha256(binding_file) != manifest['parent_binding_json_sha256']:
        raise RuntimeError('Original checkpoint binding metadata changed')
    binding = json.loads(binding_file.read_text(encoding='utf-8'))
    for group, prefix in (('code', ''), ('engine', 'external/DiffGRM/')):
        for relative, expected in binding[group].items():
            if sha256(source_path(code, prefix + relative)) != expected:
                raise RuntimeError(f'Original parent-bound {group} changed: {relative}')
    return manifest, binding


def single_process_environment():
    environment = os.environ.copy()
    # Node topology is checked before removing inherited distributed settings.
    for name in ('NNODES', 'GROUP_WORLD_SIZE', 'SLURM_JOB_NUM_NODES', 'SLURM_NNODES'):
        if name in environment and int(environment[name]) != 1:
            raise RuntimeError(f'Single-node job required; {name}={environment[name]}')
    for name in ('NODE_RANK', 'GROUP_RANK', 'SLURM_NODEID'):
        if name in environment and int(environment[name]) != 0:
            raise RuntimeError(f'Only the single node may run this entry; {name}={environment[name]}')
    names = {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
             'NODE_RANK', 'GROUP_RANK', 'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
             'NNODES', 'NPROC_PER_NODE', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS',
             'SLURM_NTASKS_PER_NODE', 'SLURM_STEP_NUM_TASKS'}
    prefixes = ('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'MV2_', 'ACCELERATE_', 'RDZV_')
    for name in list(environment):
        if name in names or name.startswith(prefixes):
            environment.pop(name)
    environment.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1')
    # Preserve the scheduler's CUDA_VISIBLE_DEVICES: do not conceal an extra
    # allocated GPU by replacing its mask with cuda:0 before inventory.
    return environment


def forward_signal(process, signum):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def wait_child(process, shutdown, *, timeout=None):
    started = time.monotonic()
    deadline_hit = False
    termination_started = None
    while process.poll() is None:
        now = time.monotonic()
        if timeout is not None and now - started >= timeout and not deadline_hit:
            deadline_hit = True
            forward_signal(process, signal.SIGTERM)
            termination_started = now
        if shutdown['signal'] is not None and termination_started is None:
            forward_signal(process, shutdown['signal'])
            termination_started = now
        if termination_started is not None and now - termination_started >= 15:
            forward_signal(process, signal.SIGKILL)
        time.sleep(.2)
    return process.wait(), deadline_hit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--policy', required=True, choices=('decay', 'freeze'))
    parser.add_argument('--subset', required=True)
    args = parser.parse_args(argv)
    if os.name != 'posix':
        raise RuntimeError('This job launcher targets the Linux H100 runtime')
    started = time.monotonic()
    root, parent, subset = Path(args.root).resolve(), Path(args.parent).resolve(), Path(args.subset).resolve()
    code = (root / 'code').resolve()
    executable = Path(args.python).resolve()
    if not root.is_dir() or not code.is_dir() or not executable.is_file():
        raise RuntimeError('Root, immutable code directory and explicit Python executable must exist')
    run_output = root / 'runs' / f'H1-{args.policy}'
    if run_output.exists():
        raise FileExistsError(f'Refusing to restart or overwrite an existing formal run: {run_output}')
    log_directory = root / 'formal_jobs' / f'H1-{args.policy}'
    log_directory.mkdir(parents=True, exist_ok=False)
    active = {'process': None}
    shutdown = {'signal': None}

    def interrupted(signum, _frame):
        if shutdown['signal'] is None:
            shutdown['signal'] = signum
        forward_signal(active['process'], signum)

    old_handlers = {name: signal.signal(name, interrupted) for name in (signal.SIGTERM, signal.SIGINT)}
    result = dict(status='not_started', policy=args.policy, training_started=False,
                  automatic_retry=False, run_output=str(run_output), launcher_sha256=sha256(__file__))
    exit_code = 1
    try:
        manifest, binding = verify_package(code, parent, args.manifest_sha)
        locked_subset = json.loads(subset.read_text(encoding='utf-8'))
        if (locked_subset.get('format') != 'stage09_h1_recovery_subset_v1'
                or locked_subset.get('count') != 8192 or locked_subset.get('population') != 71657
                or locked_subset.get('validation_sha256') != binding['data']['files']['validation.parquet']['sha256']):
            raise RuntimeError('Diagnostic subset does not match the fixed validation release')
        environment = single_process_environment()
        request = dict(root=str(root), python=str(executable), parent=str(parent), parent_sha256=PARENT_SHA256,
            source_commit=manifest['workspace']['commit'], manifest_sha256=MANIFEST_SHA256,
            subset=str(subset), subset_sha256=sha256(subset), policy=args.policy,
            single_gpu_independent_job=True, threads=4,
            authorization_note='Execution must be separately authorized after formal-budget approval.')
        write_json(log_directory / 'request.json', request)
        if shutdown['signal'] is not None:
            raise InterruptedError('Shutdown requested before GPU inventory')
        inventory_code = (
            "import json,torch; print(json.dumps([{'name':torch.cuda.get_device_name(i),"
            "'uuid':str(torch.cuda.get_device_properties(i).uuid),"
            "'total_memory':int(torch.cuda.get_device_properties(i).total_memory)} "
            "for i in range(torch.cuda.device_count())]))"
        )
        with (log_directory / 'inventory.stdout.json').open('wb') as out, \
                (log_directory / 'inventory.stderr.log').open('wb') as err:
            active['process'] = subprocess.Popen([str(executable), '-B', '-c', inventory_code],
                cwd=code, env=environment, stdout=out, stderr=err, start_new_session=True)
            inventory_exit, timed_out = wait_child(active['process'], shutdown, timeout=45)
        active['process'] = None
        if shutdown['signal'] is not None:
            raise InterruptedError('Shutdown requested during GPU inventory')
        if inventory_exit != 0 or timed_out:
            raise RuntimeError(f'GPU inventory failed: exit={inventory_exit}, timeout={timed_out}')
        visible = json.loads((log_directory / 'inventory.stdout.json').read_text(encoding='utf-8'))
        if (len(visible) != 1 or 'h100' not in visible[0]['name'].lower()
                or str(visible[0]['uuid']).upper().startswith('MIG-')):
            raise RuntimeError('This job must expose exactly one full H100 GPU')
        write_json(log_directory / 'inventory.json', visible)
        command = [str(executable), '-u', '-B', '-m', 'scripts.listwise_diffgrm.recovery_run', 'train',
                   '--parent', str(parent), '--policy', args.policy, '--device', 'cuda:0',
                   '--output', str(run_output), '--subset', str(subset)]
        request['command'] = command
        write_json(log_directory / 'request.json', request)
        if shutdown['signal'] is not None:
            raise InterruptedError('Shutdown requested before training')
        with (log_directory / 'training.log').open('wb') as stream:
            active['process'] = subprocess.Popen(command, cwd=code, env=environment, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
            result['training_started'] = True
            child_exit, _ = wait_child(active['process'], shutdown)
        active['process'] = None
        result['child_returncode'] = child_exit
        exit_code = child_exit if child_exit >= 0 else 128 - child_exit
        result['status'] = 'child_exited' if child_exit == 0 else 'child_failed'
        # A zero process exit is recorded, not promoted to scientific acceptance.
        result['completion_requires_independent_audit'] = True
    except BaseException as exc:
        result.update(status='launcher_failed', error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
    finally:
        process = active['process']
        if process is not None and process.poll() is None:
            forward_signal(process, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forward_signal(process, signal.SIGKILL)
                process.wait(timeout=5)
        if shutdown['signal'] is not None:
            exit_code = 128 + shutdown['signal']
            result.update(status='interrupted', signal=shutdown['signal'])
        result.update(exit_code=exit_code, script_seconds=time.monotonic() - started,
                      accounting_scope='Wrapper lifetime only; platform allocation/release billing is external.')
        write_json(log_directory / 'launcher_result.json', result)
        for name, handler in old_handlers.items():
            signal.signal(name, handler)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
