"""Two independent H100 probes; stdlib supervisor, no allocation API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--run-name', default='preflight_r1')
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(args.root).resolve()
    code = root / 'code'
    manifest_file = code / 'RECOVERY_EXECUTION_MANIFEST.json'
    if sha(manifest_file) != args.manifest_sha:
        raise RuntimeError('Execution manifest bytes changed')
    manifest = json.loads(manifest_file.read_text())
    if manifest['workspace']['package_sources_uncommitted']:
        raise RuntimeError('GPU execution requires committed recovery source')
    for relative, record in manifest['execution_files'].items():
        path = code / relative
        if not path.resolve().is_relative_to(code) or sha(path) != record['sha256']:
            raise RuntimeError(f'Execution source changed: {relative}')
    if sha(args.parent) != manifest['parent']['sha256']:
        raise RuntimeError('Parent weights changed')
    inventory_script = "import json,torch; print(json.dumps([{'uuid':str(torch.cuda.get_device_properties(i).uuid),'name':torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())]))"
    visible = json.loads(subprocess.run([args.python, '-c', inventory_script], check=True,
                                       capture_output=True, text=True, timeout=45).stdout)
    if len(visible) != 2 or any('H100' not in row['name'] for row in visible):
        raise RuntimeError('Expected exactly two visible H100 GPUs')
    ids = [row['uuid'] if row['uuid'].startswith('GPU-') else 'GPU-' + row['uuid'] for row in visible]
    if len(set(ids)) != 2:
        raise RuntimeError('GPU identity collision')
    output = root / args.run_name
    output.mkdir(exist_ok=False)
    (output / 'inventory.json').write_text(json.dumps(visible, indent=2))
    processes, streams = {}, []
    reason = None

    def interrupted(*_):
        raise KeyboardInterrupt('Platform or operator requested shutdown')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for index, policy in enumerate(('decay', 'freeze')):
            env = os.environ.copy()
            for name in list(env):
                if name in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or name.startswith(('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'MV2_')):
                    env.pop(name)
            env.update(CUDA_VISIBLE_DEVICES=ids[index], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            stream = (output / f'{policy}_supervisor.log').open('wb')
            streams.append(stream)
            command = [args.python, '-u', '-B', '-m', 'scripts.listwise_diffgrm.recovery_preflight',
                       '--parent', args.parent, '--policy', policy, '--output', str(output / policy),
                       '--max-seconds', '480', '--profile-updates', '3', '--validation-samples', '128']
            processes[policy] = subprocess.Popen(command, cwd=code, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
        while any(process.poll() is None for process in processes.values()):
            if any(process.poll() not in (None, 0) for process in processes.values()):
                reason = 'worker_failed'
                break
            if time.monotonic() - started > 600:
                reason = 'whole_script_deadline'
                break
            time.sleep(.5)
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for stream in streams:
            stream.close()
    passed = reason is None and len(processes) == 2 and all(process.returncode == 0 for process in processes.values())
    reports = {}
    if passed:
        for policy in processes:
            report = json.loads((output / policy / 'report.json').read_text())
            reports[policy] = sha(output / policy / 'report.json')
            passed = passed and report['status'] == 'passed'
        left = json.loads((output / 'decay' / 'report.json').read_text())
        right = json.loads((output / 'freeze' / 'report.json').read_text())
        for key in ('model', 'alignment', 'r_optimizer', 'scheduler'):
            if left['initial']['fork_hashes'][key] != right['initial']['fork_hashes'][key]:
                passed, reason = False, 'initial_pair_mismatch_' + key
        if left['catalog_diagnostic']['effective_codebooks_sha256'] != right['catalog_diagnostic']['effective_codebooks_sha256']:
            passed, reason = False, 'initial_effective_codebook_mismatch'
    result = dict(status='passed' if passed else 'failed', reason=reason,
                  process_exit_codes={k: p.returncode for k, p in processes.items()}, report_sha256=reports,
                  script_seconds=time.monotonic() - started,
                  accounting_note='Does not include platform allocation/startup/release; obtain events separately.')
    (output / 'SUMMARY.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
