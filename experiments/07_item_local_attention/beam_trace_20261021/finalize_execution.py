"""Independent local prediction replay check and platform resource reconciliation."""
from datetime import datetime, timezone, timedelta
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
out = root / 'outputs/stage07_item_local_joint/beam_trace_20261021_r1'
new = out / 'verified/results'
old = root / 'outputs/stage07_item_local_joint/beam_validation_20261021_r1/verified/results'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
cells = ('control_beam32', 'control_beam256', 'item_local_beam32', 'item_local_beam256')
checks = {}
for cell in cells:
    old_files = list((old/cell/'attempts').glob('*/per_user.jsonl'))
    new_files = list((new/cell/'attempts').glob('*/per_user.jsonl'))
    assert len(old_files) == len(new_files) == 1
    n = 0
    with old_files[0].open(encoding='utf-8') as a, new_files[0].open(encoding='utf-8') as b:
        for aa, bb in zip_longest(a, b):
            assert aa and bb, 'Truncated data'
            before, after = json.loads(aa), json.loads(bb)
            assert before == {k: after[k] for k in before}, (cell, n)
            n += 1
    assert n == 22363
    checks[cell] = {'count': n, 'all_original_fields_exact': True,
                    'old_per_user_sha256': sha(old_files[0]), 'new_per_user_sha256': sha(new_files[0])}
status = read(here/'JOB_STATUS_FINAL.json')['data']
events = read(here/'JOB_EVENTS_FINAL.json')['data']['items']
assert status['status'] == 'SUCCEEDED' and status['resource']['gpu'] == 4
tz = timezone(timedelta(hours=8))
def event_time(event):
    return datetime.strptime(event['time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz).timestamp()
scheduled = min(event_time(e) for e in events if e['reason'] == 'Scheduled')
released = max(event_time(e) for e in events if e['reason'] == 'SuccessfulDeletePod')
started = min(event_time(e) for e in events if e['reason'] == 'Started' and e['message'] == 'Started container pytorch')
finished = int(status['finished_at']) / 1000
process = read(new/'GPU_PROCESS_RECEIPT.json')
assert process['status'] == 'process_completed_pending_independent_result_review'
assert read(here/'SUMMARY.json')['status'] == 'LOCAL_TRACE_VERIFICATION_PASSED'
complete = read(new/'COMPLETE.json')
assert complete['all_original_predictions_reproduced'] and complete['original_artifacts_unchanged']
assert not complete['training_performed'] and not complete['test_data_evaluated']
receipt = {
    'status': 'VALIDATION_TRACE_FULLY_ACCEPTED', 'gpu_released': True,
    'gpu_count': 4, 'cells': checks, 'total_user_evaluations': 4*22363,
    'allocation_seconds_scheduled_to_deleted_pod': released-scheduled,
    'allocation_gpu_hours': 4*(released-scheduled)/3600,
    'platform_runtime_seconds_started_to_finished': finished-started,
    'platform_runtime_gpu_hours': 4*(finished-started)/3600,
    'process_seconds': process['elapsed_process_seconds'],
    'process_gpu_hours': process['process_allocation_gpu_hours'],
    'billing': 'Resource-time accounting only; no independently attributable settled bill was queried.',
    'training_performed': False, 'test_data_evaluated': False,
    'archive_sha256': sha(out/'validation_evidence.tar.gz'),
    'summary_sha256': sha(here/'SUMMARY.json'),
    'local_archive_verification_sha256': sha(out/'verified_LOCAL_DELIVERY_VERIFICATION.json'),
    'source_and_weight_immutability': 'Remote pre/post checks passed; frozen sources and original evidence retained.',
}
(here/'FINAL_ACCEPTANCE.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
