"""One-GPU forward-only fixed-recommender diagnostic with unique outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
SSD = Path('/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis')
CODE = SSD / 'code'
DELIVERY = SSD / 'delivery/feedback_hard_compat_20261013_1616_r5'
BASE = SSD / 'outputs/analysis/stage05_feedback'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'formal'), required=True)
    args = parser.parse_args()
    os.chdir(CODE)
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    manifest = json.loads((DELIVERY / 'delivery_manifest.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    if head != manifest['code_head'] or dirty:
        raise RuntimeError('Require clean committed diagnostic')
    preflight = json.loads((DELIVERY / 'preflight_passed.json').read_text())
    if preflight['status'] != 'passed' or preflight['executed_ssd_commit'] != head:
        raise RuntimeError('Matching CPU preflight required')
    # User requested a shorter read-only diagnostic workflow after repeated
    # tooling failures. Run one full capture directly; the evaluator retains
    # all numerical, fixed-state, input, identity and completion guards.
    # This does NOT authorize training or describe any failed smoke as passed.
    output = BASE / ('hard_compat_' + args.mode + '_20261013_1616_r5')
    if output.exists():
        raise FileExistsError(output)
    probe = DELIVERY / (args.mode + '_space_probe.tmp')
    with probe.open('xb') as out:
        try:
            os.posix_fallocate(out.fileno(), 0, 128 * 1024**2)
            out.write(b'\0' * 1024**2)
            out.flush()
            os.fsync(out.fileno())
        finally:
            probe.unlink(missing_ok=True)
    os.environ.update({'HF_HOME': str(ROOT / 'data/huggingface'), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'TORCHDYNAMO_DISABLE': '1', 'WANDB_MODE': 'disabled', 'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'})
    import torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('One BF16 GPU required')
    command = [sys.executable, '-u', '-B', str(CODE / 'scripts/analysis/evaluate_feedback_hard_compatibility.py'),
        '--project-root', str(ROOT), '--run-dir', str(ROOT / 'outputs/pilots/stage05_feedback/feedback_pilot_20261012_2150'),
        '--output-base', str(BASE), '--output', str(output)]
    if args.mode == 'smoke':
        command.append('--smoke')
    with (DELIVERY / (args.mode + '_started.json')).open('x') as out:
        json.dump({'utc': datetime.now(timezone.utc).isoformat(), 'code_head': head,
            'command': command, 'gpu': torch.cuda.get_device_name(0),
            'launch_sha256': digest(Path(__file__)),
            'launch_strategy': 'direct full read-only capture; shortened at user request; no passed smoke claimed'}, out, indent=2)
    print('[HARD COMPAT START] ' + args.mode + ' ' + head, flush=True)
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
    with (DELIVERY / (args.mode + '_finished.json')).open('x') as out:
        json.dump({'exit_code': code, 'business_complete': complete}, out, indent=2)
    if not complete:
        raise SystemExit(code or 1)

if __name__ == '__main__':
    main()
