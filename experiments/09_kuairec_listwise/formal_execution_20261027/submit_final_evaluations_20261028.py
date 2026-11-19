"""Submit exactly the four accepted, bounded final evaluation Jobs once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from inspire.services.utils.json_formatter import format_json

ROOT = Path(__file__).absolute().parents[3]
FORMAL = ROOT / 'experiments/09_kuairec_listwise/formal_execution_20261027'
CLI = shutil.which('inspire') or 'inspire'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def invoke(argv, timeout=120):
    result = subprocess.run([CLI, '--account', 'sii-shared', *argv],
                            capture_output=True, encoding='utf-8', errors='replace', timeout=timeout)
    require(result.returncode == 0, 'CLI returned failure; private connection output withheld')
    payload = json.loads(result.stdout)
    require(payload.get('success') is True, 'Unsuccessful CLI response')
    return payload


def main():
    os.environ['NO_PROXY'] = '.sii.edu.cn,<institution-host>'
    plan_path = FORMAL/'FINAL_EVAL_BOUND_PLAN_20261028.json'
    plan, gate = read(plan_path), read(FORMAL/'FINAL_EVAL_EXECUTION_GATE_20261028.json')
    require(gate['status'] == 'PASS' and gate['plan_sha256'] == sha(plan_path)
            and gate['gpu_jobs_submitted'] == 0, 'Plan/gate identity differs')
    require(plan['budget']['cumulative_with_caps_gpu_seconds'] <= 270000
            and plan['budget']['reserved_evaluation_gpu_seconds'] == 25200, 'Budget gate differs')
    require([j['arm'] for j in plan['jobs']] == ['J0', 'J1', 'H0', 'H1'], 'Unexpected job set/order')
    for job in plan['jobs']:
        arm = job['arm']
        dfile = FORMAL/gate['dry_runs'][arm]['file']
        require(sha(dfile) == gate['dry_runs'][arm]['sha256'], 'Dry-run SHA differs')
        d = read(dfile)['data']
        # CLI public JSON redacts Python path-expression /a as a private path.
        # Compare through the exact installed formatter; submit original argv.
        display = json.loads(format_json({'command': job['command_template']}))['data']['command']
        require(d['command'] == display and d['dry_run'] is True
                and d['name'] == job['name'] and d['max_time_hours'] == job['max_time_hours']
                and d['nodes'] == 1 and d['resource']['gpu'] == 1
                and d['resource']['cpu'] == 20 and d['resource']['memory_gib'] == 200
                and d['shared_memory_gib'] == 32 and not d['auto_fault_tolerance']
                and not d['enable_notification'] and d['workspace'] == '分布式训练空间'
                and d['project'] == '公共科研项目'
                and d['image'] == 'pytorch:25.06-py3:25.06', 'Native dry-run resource/command mismatch')
        require(job['dry_run_argv_template'][:-1] == job['cli_argv_template'], 'Submission argv differs')
        require(not (FORMAL/('FINAL_EVAL_'+arm+'_SUBMIT_INTENT_20261028.json')).exists(),
                'Existing submission intent: inspect native state; never blindly resubmit')
    before = invoke(['--json', 'job', 'list', '--workspace', '分布式训练空间',
                     '--keyword', 'nancy-stage09-final-eval', '--all'])
    require(not before['data']['items'], 'Evaluation Job already exists')
    save(FORMAL/'FINAL_EVAL_PRE_SUBMIT_ABSENCE_20261028.json', before)
    for job in plan['jobs']:
        arm = job['arm']
        intent = dict(name=job['name'], arm=arm, created_at_utc=datetime.now(timezone.utc).isoformat(),
                      status='SUBMISSION_ATTEMPT_RECORDED', plan_sha256=sha(plan_path),
                      command_sha256=hashlib.sha256(job['command_template'].encode()).hexdigest(),
                      gpus=1, max_time_hours=job['max_time_hours'],
                      uncertain_response_rule='Inspect this exact name before any retry; do not duplicate')
        save(FORMAL/('FINAL_EVAL_'+arm+'_SUBMIT_INTENT_20261028.json'), intent)
        try:
            response = invoke(job['cli_argv_template'], timeout=180)
            require(response['data']['name'] == job['name'], 'Created identity differs')
            save(FORMAL/('FINAL_EVAL_'+arm+'_CREATE_20261028.json'), response)
            print(json.dumps(dict(arm=arm, name=job['name'], submitted=True,
                                  max_time_hours=job['max_time_hours'])), flush=True)
        except Exception as error:
            save(FORMAL/('FINAL_EVAL_'+arm+'_SUBMIT_UNCERTAIN_20261028.json'),
                 dict(name=job['name'], error_type=type(error).__name__,
                      status='CHECK_NATIVE_STATE_DO_NOT_RETRY', checked_at_utc=datetime.now(timezone.utc).isoformat()))
            raise RuntimeError('Submission response uncertain; inspect native named Job before any further action') from None


if __name__ == '__main__':
    main()
