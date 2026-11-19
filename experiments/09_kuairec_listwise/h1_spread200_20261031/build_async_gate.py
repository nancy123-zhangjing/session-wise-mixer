"""Bind verified recovery/concurrency evidence and conservative remaining-work budget."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    structure, queue = read(ROOT/'CPU_ASYNC_STRUCTURE.json'), read(ROOT/'CPU_ASYNC_QUEUE.json')
    runtime_sha = sha(ROOT/'async_resume.py')
    assert structure['runtime_sha256'] == queue['runtime_sha256'] == runtime_sha
    assert structure['status'] == 'passed_CPU_async_structure_only'
    assert queue['status'] == 'passed_CPU_real_subprocess_queue' and len(queue['checks']) == 5
    probe_path = ROOT/'concurrency_evidence_local_r1/concurrency_preflight_r1/CONCURRENCY_PREFLIGHT.json'
    probe = read(probe_path)
    assert probe['status'] == 'passed_concurrency_numerical_equivalence' and probe['measured_overlap_adequate']
    assert probe['complete_state_and_loss_sequence_exact'] and probe['FP32_rows_exact'] == dict(eval_a=128,eval_b=128)
    restore = read(ROOT/'cancellation_r1/RESTORE_READY.json')
    assert restore['status'] == 'passed_CPU_restore_and_journal_alignment' and restore['replayed_rows_required'] == 0
    assert all(restore['restored_equal'].values())
    terminal = read(ROOT/'cancellation_r1/PLATFORM_STATUS.json')
    assert terminal['status'] == 'CANCELLED'
    release = read(ROOT/'CONCURRENCY_RELEASE_EVENTS.json')
    assert any(x['reason']=='SuccessfulDeletePod' for x in release['data']['items'])
    ledger = read(ROOT/'GPU_LEDGER.json')
    assert all(j['status'] in ('SUCCEEDED','CANCELLED') for j in ledger['jobs'])
    consumed = round(ledger['completed_created_to_finished_gpu_hours_upper_bound'] * 3600)
    assert consumed == 35813 and ledger['preflight_completed_seconds_upper_bound'] == 663
    logs = [json.loads(s) for s in (ROOT/'cancellation_r1/training.jsonl').read_text().splitlines()]
    t_cost = max(x['seconds'] for x in logs if x['phase'] == 'T')
    r_cost = max(probe['sequential_R_seconds'], probe['concurrent_R_seconds']) / 30
    eval_cost = max(.05, probe['baseline_eval_seconds']/256)
    # This is a conservative serial work sum. Actual two-worker overlap has
    # aggregate throughput faster than this single-stream assumption.
    forecast = (7916*r_cost + 60*t_cost + 450360*eval_cost + 600)*1.25
    assert forecast < 49620 - 120
    assert consumed + 49680 + 180 <= 86400
    assert sha(ROOT/'cancellation_r1/last.pt') == restore['checkpoint_sha256'] == probe['checkpoint_sha256']
    assert sha(ROOT/'cancellation_r1/training.jsonl') == probe['journal_sha256']
    dependencies = {
        'CPU_ASYNC_STRUCTURE.json': ROOT/'CPU_ASYNC_STRUCTURE.json',
        'CPU_ASYNC_QUEUE.json': ROOT/'CPU_ASYNC_QUEUE.json',
        'recovery_r2/RESTORE_READY.json': ROOT/'cancellation_r1/RESTORE_READY.json',
        'recovery_r2/PLATFORM_STATUS.json': ROOT/'cancellation_r1/PLATFORM_STATUS.json',
        'concurrency_preflight_r1/CONCURRENCY_PREFLIGHT.json': probe_path,
        'CONCURRENCY_PLATFORM_STATUS.json': ROOT/'CONCURRENCY_PLATFORM_STATUS.json',
        'CONCURRENCY_RELEASE_EVENTS.json': ROOT/'CONCURRENCY_RELEASE_EVENTS.json',
        'ASYNC_PREVIOUS_PLATFORM_STATUS.json': ROOT/'ASYNC_PREVIOUS_PLATFORM_STATUS.json',
    }
    result = dict(status='accepted_async_resume', runtime_sha256=runtime_sha,
        parent_sha256='991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1',
        scientific_manifest_sha256=structure['scientific_manifest_sha256'],
        checkpoint=REMOTE+'/runs/H1-Spread200-R35/last.pt', checkpoint_sha256=probe['checkpoint_sha256'],
        journal_sha256=probe['journal_sha256'], consumed_gpu_seconds_upper_bound=consumed,
        preflight_seconds_upper_bound=663, platform_max_seconds=49680, inner_max_seconds=49620,
        release_allowance_seconds=180, remaining_R_updates=7916, remaining_T_updates=60,
        remaining_eval_examples_max=450360, forecast_seconds=forecast, future_margin=1.25,
        R_seconds_assumption=r_cost,T_seconds_assumption=t_cost,eval_seconds_per_example_assumption=eval_cost,
        extra_setup_seconds=600,policy_basis='Genuine required-work concurrency; no admin exemption asserted',
        low_utilization_stop_seconds=9000, utilization_sample_seconds=30,
        evidence={name:sha(path) for name,path in dependencies.items()},
        limitation='Short concurrency evidence cannot guarantee future three-hour utilization; stop and save if persistently unsuitable.')
    (ROOT/'ASYNC_RESUME_GATE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
