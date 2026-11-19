"""Sync a committed diagnostic to own SSD and verify before any GPU run."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
SSD = Path('/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis')
CODE = SSD / 'code'
DELIVERY = SSD / 'delivery/feedback_hard_compat_20261013_1616_r5'

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-C', str(CODE), *args],
        text=True, stderr=subprocess.STDOUT).strip()

def main():
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    manifest = json.loads((DELIVERY / 'delivery_manifest.json').read_text())
    target = manifest['code_head']
    bundle = DELIVERY / 'code.bundle'
    if digest(bundle) != manifest['bundle_sha256']:
        raise RuntimeError('Bundle SHA256 mismatch')
    head = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or head not in (manifest['base_head'], target):
        raise RuntimeError('Unexpected SSD source or dirty working tree')
    if head != target:
        git('bundle', 'verify', str(bundle))
        git('fetch', str(bundle), 'HEAD:refs/codex/feedback-hard-compat-20261013_1616_r5')
        if git('rev-parse', 'refs/codex/feedback-hard-compat-20261013_1616_r5') != target:
            raise RuntimeError('Unexpected fetched target')
        git('sparse-checkout', 'add', '--no-cone',
            '/scripts/analysis/evaluate_feedback_hard_compatibility.py',
            '/scripts/analysis/feedback_hard_compat_analysis.py',
            '/tests/test_feedback_hard_compat_analysis.py',
            '/tests/test_feedback_hard_compat_evaluator.py',
            '/experiments/05_crosspse_joint/pilots/feedback_update/hard_sid_compatibility/*',
            '/experiments/05_crosspse_joint/pilots/feedback_update/README.md')
        git('merge', '--ff-only', 'refs/codex/feedback-hard-compat-20261013_1616_r5')
    # Configuration tests read this tracked protocol fixture as well as YAML.
    git('sparse-checkout', 'add', '--no-cone',
        '/configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json',
        '/configs/shared/diff_formal_common.yaml',
        '/configs/shared/diff_formal_repo_v6.yaml',
        '/configs/05_crosspse_joint/diff_formal_crosspse_joint_diger_v2.yaml')
    if git('rev-parse', 'HEAD') != target or git('status', '--porcelain'):
        raise RuntimeError('Clean committed target required')
    probe = DELIVERY / 'preflight_space_probe.tmp'
    if probe.exists():
        raise FileExistsError(probe)
    with probe.open('xb') as out:
        try:
            os.posix_fallocate(out.fileno(), 0, 128 * 1024**2)
            out.write(b'\0' * 16 * 1024**2)
            out.flush()
            os.fsync(out.fileno())
        finally:
            probe.unlink(missing_ok=True)
    counts = {}
    for test in ('test_feedback_paired_analysis.py', 'test_feedback_paired_evaluator.py',
                 'test_feedback_hard_compat_analysis.py', 'test_feedback_hard_compat_evaluator.py'):
        result = subprocess.run([sys.executable, '-B', str(CODE / 'tests' / test)], cwd=CODE,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        if result.returncode or 'OK (skipped=' in result.stdout:
            raise RuntimeError('Standard-library test failure or skip: ' + test)
        count = re.search(r'Ran (\d+) tests?', result.stdout)
        if not count:
            raise RuntimeError('Missing unittest count')
        counts[test] = int(count.group(1))
    import torch
    import numpy as np
    torch.set_num_threads(4)
    if torch.__version__ != '2.6.0+cu118' or np.__version__ != '2.2.6':
        raise RuntimeError('Original Torch/NumPy runtime required')
    entry_path = CODE / 'scripts/analysis/evaluate_feedback_hard_compatibility.py'
    spec = importlib.util.spec_from_file_location('_hard_preflight', entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    run_dir = ROOT / 'outputs/pilots/stage05_feedback/feedback_pilot_20261012_2150'
    summary = json.loads((run_dir / 'summary.json').read_text())
    completion = json.loads((run_dir / 'COMPLETE.json').read_text())
    if completion['summary_sha256'] != digest(run_dir / 'summary.json'):
        raise RuntimeError('Training completion SHA mismatch')
    common = None
    models = {}
    for arm in summary['arms']:
        for label in ('initial', 'post_tokenizer'):
            row = next(row for row in arm['snapshots'] if row['label'] == label)
            directory = run_dir / arm['arm'] / 'snapshots' / label
            for name, expected in row['files'].items():
                if digest(directory / name) != expected:
                    raise RuntimeError('Snapshot byte checksum differs')
            state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
            if label == 'initial':
                if common is None:
                    common = state
                elif any(not torch.equal(common[key], state[key]) for key in common):
                    raise RuntimeError('Initial states differ')
            else:
                graft = entry.graft_tokenizer_state(common, state, equal=torch.equal)
                for key in entry.recommender_state(common):
                    if not torch.equal(common[key], graft[key]):
                        raise RuntimeError('Graft changed recommender')
                if arm['arm'] == 'frozen' and any(not torch.equal(common[key], state[key]) for key in common):
                    raise RuntimeError('Frozen complete state changed')
                models[arm['arm']] = {'all_non_tokenizer_tensors_equal': True}
    proof = {'status': 'passed', 'executed_ssd_commit': target, 'bundle_sha256': digest(bundle),
        'standard_library_tests': counts, 'tests_total': sum(counts.values()),
        'actual_snapshot_graft_cpu_tensor_check': models,
        'ssd_allocated_probe_bytes': 128 * 1024**2, 'ssd_written_fsync_bytes': 16 * 1024**2,
        'hdd_inputs_only_read': True, 'no_training_no_optimizer': True,
        'code': str(CODE), 'note': 'Scoped SSD diagnostic source; not complete HDD project sync.'}
    with (DELIVERY / 'preflight_passed.json').open('x') as out:
        json.dump(proof, out, indent=2)
    print(json.dumps(proof), flush=True)

if __name__ == '__main__':
    main()
