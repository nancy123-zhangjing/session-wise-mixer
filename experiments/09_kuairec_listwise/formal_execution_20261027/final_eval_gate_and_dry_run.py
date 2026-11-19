"""Read-only final evaluation gates and native create dry-runs; never submits."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).absolute().parents[3]
FORMAL = ROOT / 'experiments/09_kuairec_listwise/formal_execution_20261027'
PLAN = FORMAL / 'FINAL_EVAL_BOUND_PLAN_20261028.json'
CLI = shutil.which('inspire') or 'inspire'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def call(argv, timeout=150):
    result = subprocess.run([CLI, '--account', 'sii-shared', *argv], capture_output=True,
                            encoding='utf-8', errors='replace', timeout=timeout)
    require(result.returncode == 0, 'CLI gate failed; private output withheld')
    return result.stdout


def save(path, text):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text.rstrip() + '\n')


def main():
    os.environ['NO_PROXY'] = '.sii.edu.cn,<institution-host>'
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    require(plan['status'] == 'BOUND_NOT_SUBMITTED', 'Wrong plan state')
    names = [j['name'] for j in plan['jobs']]
    absent = call(['--json', 'job', 'list', '--workspace', '分布式训练空间',
                   '--keyword', 'nancy-stage09-final-eval', '--all'])
    existing = json.loads(absent)
    require(existing['success'] and not existing['data']['items'], 'Final evaluation name already exists')
    absence_path = FORMAL / ('FINAL_EVAL_GATE_JOB_ABSENCE_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    save(absence_path, absent)
    freeze_path = FORMAL / 'FOUR_BEST_FREEZE_20261028.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    remote_freeze = plan['root'] + '/delivery/' + freeze_path.name
    paths = [j['log'] for j in plan['jobs']]
    for j in plan['jobs']:
        for stage in j['stages']:
            argv = stage['python_argv']
            paths.append(argv[argv.index('--output') + 1])
    require(len(paths) == len(set(paths)) == 16, 'Expected 12 outputs and four logs')
    remote_code = f'''import hashlib,json
from pathlib import Path
root=Path({plan['root']!r})
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
freeze=Path({remote_freeze!r})
assert freeze.is_file() and not freeze.is_symlink() and digest(freeze)=={sha(freeze_path)!r}
f=json.loads(freeze.read_text())
assert f['status']=='passed' and len(f['arms'])==4
for a,row in f['arms'].items():
 p=Path(row['best']['path'])
 assert p.resolve().is_relative_to(root.resolve()) and digest(p)==row['best']['sha256']
 assert not (root/'runs'/a/'.running').exists()
for name in {paths!r}:
 p=Path(name)
 assert p.resolve().is_relative_to(root.resolve()) and not p.exists() and not p.is_symlink(),name
print('STAGE09_FINAL_GATE '+json.dumps(dict(status='PASS',checked_output_absence_count={len(paths)},freeze_sha256={sha(freeze_path)!r},four_best_rehashed=True)))
'''
    raw = call(['notebook', 'ssh', 'nancy-stage09-prep-20261027-r1', '--workspace', 'CPU资源空间',
                '--command-timeout', '90', '--timeout', '60', '--',
                '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python',
                '-B', '-c', remote_code], timeout=180)
    rows = [json.loads(line[len('STAGE09_FINAL_GATE '):]) for line in raw.splitlines()
            if line.startswith('STAGE09_FINAL_GATE ')]
    require(len(rows) == 1 and rows[0]['status'] == 'PASS', 'Remote output/freeze gate missing')
    gate = dict(schema='stage09_final_eval_execution_gate_v1', status='PASS',
                checked_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(PLAN),
                freeze_sha256=sha(freeze_path), remote=rows[0], job_names_absent=names,
                job_absence_source=absence_path.name, job_absence_sha256=sha(absence_path),
                gpu_jobs_submitted=0, dry_runs={})
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    require(bash.is_file(), 'Git Bash syntax checker unavailable')
    for j in plan['jobs']:
        shell = shlex.split(j['command_template'])
        require(len(shell) == 3 and shell[:2] == ['bash', '-c'], 'Unexpected outer shell command')
        checked = subprocess.run([str(bash), '-n'], input=shell[2], text=True, capture_output=True)
        require(checked.returncode == 0, 'Bound shell syntax check failed')
        argv = j['dry_run_argv_template']
        require(argv[-1] == '--dry-run' and argv[:-1] == j['cli_argv_template'], 'Dry-run differs from submission argv')
        dry = call(argv)
        parsed = json.loads(dry)
        require(parsed['success'], 'Native dry-run unsuccessful')
        path = FORMAL / ('FINAL_EVAL_' + j['arm'] + '_DRY_RUN_20261028.json')
        save(path, dry)
        gate['dry_runs'][j['arm']] = dict(file=path.name, sha256=sha(path), bash_syntax_passed=True)
        print(json.dumps(dict(arm=j['arm'], dry_run='PASS', path=str(path))), flush=True)
    save(FORMAL / 'FINAL_EVAL_EXECUTION_GATE_20261028.json', json.dumps(gate, indent=2))
    print(json.dumps(dict(status='GATES_AND_DRY_RUNS_PASS', gpu_jobs_submitted=0)), flush=True)


if __name__ == '__main__':
    main()
