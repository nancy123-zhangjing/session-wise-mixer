"""Version-locked CPU acceptance / GPU launch for the approved own directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TRAIN = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training')
PROJECT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
PYTHON = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python')


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def checked_run(command, log, env):
    with log.open('x', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=TRAIN / 'code', env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
        result = process.wait()
    if result:
        raise RuntimeError(f'Command exited {result}; see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'smoke', 'formal'), required=True)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    code, delivery = TRAIN / 'code', args.delivery.resolve()
    if delivery.parent != TRAIN / 'delivery' or not delivery.is_dir() or args.delivery.is_symlink():
        raise ValueError('Use an existing own direct delivery child')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0',
               DIFFGRM_ENGINE_ROOT=str(PROJECT / 'external/DiffGRM'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    head = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], env=env, text=True).strip()
    if head != args.expected_head:
        raise RuntimeError('Execution commit differs from the submitted commit')
    if subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], env=env, text=True).strip():
        raise RuntimeError('Execution checkout must be clean')
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Use the isolated grm-exp Python')
    import torch
    import numpy as np
    if torch.__version__ != '2.6.0+cu118' or np.__version__ != '2.2.6':
        raise RuntimeError('Retain verified Torch/NumPy versions')
    config = code / 'configs/05_crosspse_joint/pilots/diger_mixer_factorial_seed2024.json'
    receipt = {'mode': args.mode, 'commit': head, 'python': str(PYTHON),
               'torch': torch.__version__, 'numpy': np.__version__,
               'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
               'historical_project_readonly_input': str(PROJECT)}
    if args.mode == 'cpu':
        checked_run([str(PYTHON), '-B', str(code / 'tests/test_diger_mixer_factorial.py'), '--require-torch'],
                    delivery / 'cpu_tests.log', env)
        checked_run([str(PYTHON), '-B', str(code / 'scripts/checks/diger_mixer_factorial_protocol.py'),
                     '--project-root', str(PROJECT), '--config', str(config)],
                    delivery / 'input_preflight.log', env)
        write_json(delivery / 'CPU_PASS.json', receipt)
        return
    previous = json.loads((delivery / 'CPU_PASS.json').read_text())
    if previous['commit'] != head or previous['config_sha256'] != receipt['config_sha256']:
        raise RuntimeError('CPU acceptance belongs to a different source version')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Require exactly one visible BF16 GPU')
    receipt['gpu'] = torch.cuda.get_device_name(0)
    if 'H100' not in receipt['gpu']:
        raise RuntimeError('This launch requests the matched H100 resource group')
    # A real allocation verifies this mount/quota from the GPU, not just df.
    fd, probe = tempfile.mkstemp(prefix='owned_gpu_allocation_', dir=delivery)
    try:
        os.posix_fallocate(fd, 0, 2 * 1024**3)
        os.write(fd, b'0' * 1024**2)
        os.fsync(fd)
    finally:
        os.close(fd)
        os.unlink(probe)
    receipt['gpu_allocation_2gib_and_fsync_passed'] = True
    output = TRAIN / 'outputs/pilots/stage05_mixer_factorial' / (delivery.name + '_' + args.mode)
    if args.mode == 'formal':
        smoke_receipt = json.loads((delivery / 'SMOKE_PASS.json').read_text())
        if smoke_receipt['commit'] != head or smoke_receipt['config_sha256'] != receipt['config_sha256']:
            raise RuntimeError('GPU smoke belongs to a different source version')
    write_json(delivery / (args.mode + '_started.json'), dict(receipt, output=str(output), time=time.time()))
    command = [str(PYTHON), '-u', '-B', str(code / 'scripts/train/run_diger_mixer_factorial.py'),
               '--project-root', str(PROJECT), '--config', str(config),
               '--output-base', str(output.parent), '--output', str(output)]
    if args.mode == 'smoke':
        command.append('--smoke')
    checked_run(command, delivery / (args.mode + '.log'), env)
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text())
    complete = json.loads((output / 'COMPLETE.json').read_text())
    if (summary.get('status') != 'complete' or not summary.get('all_protocol_arms_complete')
            or complete.get('summary_sha256') != hashlib.sha256(summary_path.read_bytes()).hexdigest()
            or not summary.get('frozen_negative_control', {}).get('complete_final_tensor_hash_equal')
            or not summary['frozen_negative_control'].get('all_validation_metrics_equal')
            or not summary.get('no_historical_artifact_changes')):
        raise RuntimeError('Four-arm completion / Frozen negative control failed')
    write_json(delivery / (args.mode.upper() + '_PASS.json'),
               dict(receipt, output=str(output), elapsed_seconds=summary['elapsed_seconds'],
                    summary_sha256=complete['summary_sha256']))
    print(f'[{args.mode.upper()} PASS] {output}', flush=True)


if __name__ == '__main__':
    main()
