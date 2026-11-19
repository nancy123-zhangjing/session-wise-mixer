"""Launch one SHA-bound smoke or formal feedback study on a single GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
DELIVERY = ROOT / 'outputs/pilots_delivery/feedback_20261012_2150'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
HEADS = {'.': '299e5fa2a25793fcb7be094fbbb602d41f71f546',
         'external/DiffGRM': '39e94e6153c97e59877ba02d4ef1d9a202ed5384',
         'external/genrec': 'b6c8b41d1d20a812590dae11408b69b1ed2a8313'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = ROOT / ('outputs/pilots/stage05_feedback/feedback_' +
                     ('smoke' if args.mode == 'smoke' else 'pilot') + '_20261012_2150')
    if output.exists():
        raise RuntimeError('Refusing to overwrite existing experiment output')
    for relative, expected in HEADS.items():
        directory = ROOT / relative
        head = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain'], text=True).strip()
        if head != expected or dirty:
            raise RuntimeError('Source version or cleanliness mismatch: ' + relative)
    preflight = json.loads((DELIVERY / 'preflight_passed.json').read_text())
    if preflight['status'] != 'passed' or preflight['commit'] != HEADS['.']:
        raise RuntimeError('Missing matching CPU preflight')
    if args.mode == 'formal':
        storage = json.loads((DELIVERY / 'formal_storage_passed.json').read_text())
        if storage['status'] != 'passed' or storage['allocated_probe_bytes'] < 2 * 1024**3:
            raise RuntimeError('Missing post-smoke storage write gate')
        smoke = ROOT / 'outputs/pilots/stage05_feedback/feedback_smoke_20261012_2150'
        gate = json.loads((DELIVERY / 'smoke_gate.json').read_text())
        if gate.get('status') != 'passed' or gate.get('commit') != HEADS['.']:
            raise RuntimeError('Missing matching GPU smoke gate')
        if hashlib.sha256((smoke / 'summary.json').read_bytes()).hexdigest() != gate['summary_sha256']:
            raise RuntimeError('Smoke evidence changed after acceptance')
    os.environ.update({'HF_HOME': str(ROOT / 'data/huggingface'), 'HF_HUB_OFFLINE': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'TORCHDYNAMO_DISABLE': '1',
                       'WANDB_MODE': 'disabled', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'})
    import torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('One BF16-capable GPU required')
    command = [PYTHON, '-u', '-B', str(ROOT / 'scripts/train/run_diger_feedback_pilot.py'),
               '--project-root', str(ROOT), '--arm', 'all', '--output', str(output)]
    if args.mode == 'smoke':
        command.append('--smoke')
    started = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'mode': args.mode, 'heads': HEADS, 'gpu': torch.cuda.get_device_name(0),
               'torch': torch.__version__, 'command': command, 'output': str(output)}
    with (DELIVERY / (args.mode + '_started.json')).open('x', encoding='utf-8') as handle:
        json.dump(started, handle, indent=2)
    print('[FEEDBACK_START] mode=' + args.mode + ' commit=' + HEADS['.'], flush=True)
    print('[GPU] ' + started['gpu'] + ' torch=' + started['torch'], flush=True)
    with (DELIVERY / (args.mode + '.log')).open('xb') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            block = os.read(process.stdout.fileno(), 8192)
            if not block:
                break
            log.write(block)
            log.flush()
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
        code = process.wait()
    complete = code == 0 and (output / 'COMPLETE.json').is_file()
    with (DELIVERY / (args.mode + '_finished.json')).open('x', encoding='utf-8') as handle:
        json.dump({'exit_code': code, 'business_complete': complete}, handle, indent=2)
    if not complete:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
