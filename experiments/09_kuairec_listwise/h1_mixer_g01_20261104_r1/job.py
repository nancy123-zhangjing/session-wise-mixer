"""Run the single H100 G01 train, validation, and locked tests once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    inputs = json.loads((root / 'INPUTS.json').read_text(encoding='utf-8'))
    output = root / 'runs' / 'G01'
    (root / 'jobs').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',
               DIFFGRM_ENGINE_ROOT=str(root / 'code' / 'external' / 'DiffGRM'))
    common = ['--parent', inputs['parent'], '--dataset', inputs['dataset'],
              '--assets', inputs['assets'], '--subset', str(root / 'validation_subset_8192.json'),
              '--r20-reference', inputs['r20_reference'], '--output', str(output), '--device', 'cuda:0']
    stages = []
    started = time.monotonic()
    status = 'incomplete'
    try:
        for stage in ('train', 'test'):
            command = [args.python, '-B', '-u', '-m', 'scripts.listwise_diffgrm.recovery_g01_run', stage] + common
            if stage == 'train' and (output / 'last.pt').is_file():
                command += ['--resume', str(output / 'last.pt')]
            with (root / 'jobs' / (stage + '.log')).open('a', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=root / 'code', env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            stages.append({'stage': stage, 'exit_code': result.returncode})
            if result.returncode:
                raise RuntimeError(f'{stage} exited {result.returncode}')
        status = 'training_validation_test_complete'
    finally:
        receipt = dict(status=status, stages=stages, elapsed_seconds=time.monotonic()-started)
        (root / 'jobs' / 'result.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
