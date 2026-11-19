"""Verify the uploaded trace payload and run actual CPU acceptance before GPU use."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'DEPLOYMENT_MANIFEST.json').read_text())
for row in manifest['files']:
    path = root / row['path']
    assert path.resolve().is_relative_to(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', ACCELERATE_USE_CPU='true',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                       '-p', 'test_item_local_beam_trace.py', '-v'], cwd=root, env=env,
                      text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
(root / 'CPU_TESTS.log').write_text(test.stdout, encoding='utf-8')
assert test.returncode == 0 and 'skipped' not in test.stdout, test.stdout
goal = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_item_local_joint_r2'
reference = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/beam_validation_20261021_r1/results'
cmd = [sys.executable, '-B', str(root / 'scripts/analysis/evaluate_item_local_beam_trace.py'),
       '--goal-root', goal, '--output', str(root / 'results'), '--mode', 'preflight', '--reference-root', reference]
with (root / 'CPU_PREFLIGHT.log').open('x') as handle:
    subprocess.run(cmd, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=300)
receipt = {'status': 'CPU_TESTS_AND_PREFLIGHT_PASSED', 'cuda_initialized': False,
           'test_log_sha256': hashlib.sha256((root / 'CPU_TESTS.log').read_bytes()).hexdigest(),
           'preflight_sha256': hashlib.sha256((root / 'results/GRID_PREFLIGHT.json').read_bytes()).hexdigest(),
           'manifest_sha256': hashlib.sha256((root / 'DEPLOYMENT_MANIFEST.json').read_bytes()).hexdigest()}
(root / 'CPU_ACCEPTANCE.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
