"""Read the existing job and verify its completed test on the shared CPU."""
import base64
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'h1_no_code_tag_20261108_v1'))
import control as c

c.HERE = HERE
intent = json.loads((HERE / 'SUBMISSION_INTENT.json').read_text(encoding='utf-8'))
c.REMOTE = intent['root']
c.JOB = intent['job']


def status():
    job = c.cli(['job', 'status', c.JOB, '--workspace', c.WORKSPACE], 'PLATFORM_LATEST.json')
    log = c.remote(f'cd {c.REMOTE} && if test -f evaluation.log; then tail -n 5 evaluation.log; fi')
    print(json.dumps(job, ensure_ascii=False), flush=True)
    print(log, flush=True)
    return job


def accept():
    if status()['status'] != 'SUCCEEDED':
        raise ValueError('Awaiting successful terminal state')
    worker = c.cli(['job', 'instances', c.JOB, '--workspace', c.WORKSPACE], 'TERMINAL_INSTANCES.json')['items']
    if len(worker) != 1 or worker[0]['status'] != 'instance_success':
        raise ValueError('GPU release not confirmed')
    code = '''
import hashlib, json, sys
from pathlib import Path
root = Path.cwd()
stage = root.parent
sys.path[:0] = [str(stage / 'h_freeze35_20261029_r1/code'), str(stage / 'h1_no_code_tag_20261108_v1')]
from audit_analysis_core import audit_predictions
result = json.loads((root / 'RESULT.json').read_text())
audit = audit_predictions(result['predictions'], dataset_dir=stage / 'goal_20261027_r1/inputs/dataset')
if not audit['passed'] or not audit['source_labels_verified'] or audit['examples'] != 53194:
    raise ValueError('Independent metric audit failed')
for key in ('predictions_sha256', 'summary_sha256'):
    if audit[key] != result[key]:
        raise ValueError('Artifact hash mismatch: ' + key)
if result['comparator_sha256'] != '110182cceccb3a376701c49ca505e03eb620cbca44e80e962509d6065fdd0195':
    raise ValueError('Comparator changed')
result.update(audit_passed=True, max_absolute_error=audit['max_absolute_error'], gpu_released=True)
encoded = json.dumps(result, indent=2) + '\\n'
temp = root / 'accepted_result.tmp'
temp.write_text(encoded, encoding='utf-8')
temp.replace(root / 'ACCEPTED_RESULT.json')
print((root / 'ACCEPTED_RESULT.json').read_text())
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = f'cd {c.REMOTE} && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {c.PYTHON} -B -c '
    command += shlex.quote(f'import base64;exec(base64.b64decode({encoded!r}))')
    result = json.loads(c.remote(command, 'ACCEPTANCE_READBACK.json', timeout=100))
    c.write('ACCEPTED_RESULT.json', result)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['accept']:
        accept()
    else:
        status()
