"""Run the pinned stdout-only script once through the existing CPU Notebook."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
TASK = OUT.parent
SCRIPT = OUT / 'spotcheck.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


assert {p.name for p in OUT.iterdir()} == {'spotcheck.py', 'run_spotcheck.py'}
source = SCRIPT.read_bytes()
digest = hashlib.sha256(source).hexdigest()
evidence = TASK / 'C_R20_REPLAY_FREEZE_EVIDENCE.json'
assert sha(evidence) == 'e5d9c79eeb955432db1ad09d4cd489f0393d7ec57dbd7eebc36b451350a0058d'
bootstrap = ("import hashlib,sys; source=sys.stdin.buffer.read(); digest=hashlib.sha256(source).hexdigest(); "
             f"assert digest == {digest!r}; "
             "exec(compile(source,'<pinned-c-frozen-spotcheck>','exec'), "
             "{'EXECUTED_SOURCE_SHA256':digest,'__name__':'__main__'})")
python = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
command = ('env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '
           'timeout 100s ' + shlex.quote(python) + ' -B -c ' + shlex.quote(bootstrap) + ' # end of command')
args = [shutil.which('inspire'), '--account', 'sii-shared', 'notebook', 'exec',
        'nancy-stage09-prep-20261027-r1', '--workspace', 'CPU资源空间', '--timeout', '110', '--stdin', command]
assert args[0]
request = dict(format='freeze35_c_frozen_state_spotcheck_request_v1',
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    script_sha256=digest, runner_sha256=sha(Path(__file__)),
    prior_freeze_evidence=dict(path=str(evidence), sha256=sha(evidence)),
    prior_evidence_scope='Accepted replay/freeze gate at R20; no previous independent R21/R22 tensor comparison found.',
    cpu_notebook_precheck=dict(name='nancy-stage09-prep-20261027-r1', account='sii-shared',
        checked_begin_utc='2026-10-29T14:06:08.057362+00:00', checked_end_utc='2026-10-29T14:06:10.162061+00:00',
        status='RUNNING', node='hpc-compute792', node_status='READY', gpu_count=0, cpu_count=4, memory_gib=16),
    command_argv=args, remote_command=command, input_mode='Exact saved script bytes over confirmed CPU SSH stdin; remote SHA checked before exec',
    limits=dict(remote_script_alarm_seconds=90, remote_timeout_seconds=100, cli_timeout_seconds=110,
        local_timeout_seconds=140, max_checkpoint_count=3, max_bytes_per_checkpoint=100*1024*1024,
        cpu_threads=1, remote_writes=False, last_checkpoint_read=False),
    local_runner_command=[sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve())])
write('REQUEST.json', request)
try:
    process = subprocess.run(args, input=source, capture_output=True, timeout=140)
    stdout = process.stdout.decode('utf-8')
    stderr = process.stderr.decode('utf-8', errors='replace')
    assert process.returncode == 0
    marker = 'C_FROZEN_SPOTCHECK_RESULT='
    lines = [line[len(marker):] for line in stdout.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    remote_stdout = lines[0]
    result = json.loads(remote_stdout)
    assert result['status'] == 'passed' and result['executed_source_sha256'] == digest
    assert set(result['checkpoints']) == {'20', '21', '22'}
    assert set(result['comparisons_to_replayed_R20_pre_freeze']) == {'21', '22'}
    assert result['all_input_file_sha256_after_equal'] is True
    assert sha(SCRIPT) == digest and sha(evidence) == request['prior_freeze_evidence']['sha256']
    write('RESULT.json', result)
    write('COMPLETE.json', dict(format='freeze35_c_frozen_state_spotcheck_complete_v1', status='passed',
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        request_sha256=sha(OUT / 'REQUEST.json'), script_sha256=digest,
        runner_sha256=sha(Path(__file__)), result_sha256=sha(OUT / 'RESULT.json'),
        cli_exit_code=process.returncode, unique_remote_success_marker=True,
        stdout_utf8_sha256=hashlib.sha256(remote_stdout.encode('utf-8')).hexdigest(),
        platform_access='Existing CPU Notebook exec only; no Job or Notebook mutation command',
        is_model_restore=False, is_final_delivery_acceptance=False, goal_complete=False))
    print(json.dumps(dict(status='passed', result_sha256=sha(OUT / 'RESULT.json'),
        complete_sha256=sha(OUT / 'COMPLETE.json'), seconds=result['seconds'],
        summaries={k: {field: v[field] for field in ('alignment_tensor_count', 'alignment_all_exact_bytes_equal',
            'changed_catalog_sid_items', 'r_updates_delta', 't_updates_delta', 'changed_model_tensor_count',
            'diger_optimizer_step', 'diger_total_optimizer_steps', 'recommender_sid_embedding_changed')}
            for k, v in result['comparisons_to_replayed_R20_pre_freeze'].items()}), indent=2))
except Exception as error:
    combined = (locals().get('stdout', '') + locals().get('stderr', ''))
    write('FAILED.json', dict(status='failed', error_type=type(error).__name__,
        error_class_matches=[s for s in ('ECONNRESET', 'cached target unavailable', 'Automatic tunnel rebuild',
            'PermissionError', 'KeyError', 'AssertionError', 'No such file', 'Timeout') if s in combined],
        cli_exit_code=locals().get('process').returncode if 'process' in locals() else None,
        partial_outputs_not_accepted=True, script_sha256=digest))
    print(json.dumps(dict(status='failed', error_type=type(error).__name__, failed_sha256=sha(OUT / 'FAILED.json'))))
    raise SystemExit(2)
