"""Read-only operational gates for one continuation; no submission or training."""
import hashlib
import json
from pathlib import Path
import time

PARENT_SHA='991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path,expected):
    if sha(path)!=expected:raise ValueError('Operational receipt SHA mismatch: '+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))
def confined(root,path):
    p=Path(path).resolve();p.relative_to(root)
    if p.is_symlink():raise ValueError('Symlink receipt prohibited')
    return p
def validate_resume(root,resume,plan_path,plan_sha,approval_path,approval_sha,max_seconds):
    root=Path(root).resolve()
    if not 120<=max_seconds<=50400:raise ValueError('Continuation cap is 14 hours')
    resume=confined(root,resume)
    plan=read(confined(root,plan_path),plan_sha)
    if plan['status']!='prepared_not_submitted' or plan['parent_sha256']!=PARENT_SHA:
        raise ValueError('Unexpected continuation plan')
    if plan['completed_gpu_seconds_upper_bound']+max_seconds+180>86400:
        raise ValueError('Cumulative 24 GPU-hour budget exceeded')
    if plan['remaining_forecast_seconds']>max_seconds-90:
        raise ValueError('Remaining forecast does not fit allocation')
    if sha(resume)!=plan['checkpoint_sha256']:raise ValueError('Continuation checkpoint changed')
    run=root/'runs/H1-Spread200-R35'
    if sha(run/'training.jsonl')!=plan['journal_sha256']:
        raise ValueError('Original journal differs from verified checkpoint alignment')
    proof=read(confined(root,root/plan['cpu_restore_proof']),plan['cpu_restore_proof_sha256'])
    if (proof['status']!='passed_CPU_restore_and_journal_alignment'
            or proof['checkpoint_sha256']!=plan['checkpoint_sha256']
            or not proof['journal_exactly_matches_checkpoint'] or proof['replayed_rows_required']!=0
            or not all(proof['restored_equal'].values())
            or (proof['r_updates'],proof['t_updates'],proof['progress']['phase'],proof['progress']['update'])!=(16339,140,30,400)):
        raise ValueError('Unverified continuation state')
    terminal=read(confined(root,root/plan['terminal_receipt']),plan['terminal_receipt_sha256'])
    if terminal['name']!='nancy-s09-d-1031-r1' or terminal['status']!='CANCELLED':
        raise ValueError('Original job not terminal')
    approval=read(confined(root,approval_path),approval_sha)
    if (approval.get('status')!='approved' or approval.get('authority')!='workspace_administrator'
            or approval.get('scope')!='stage09_h1_spread200_continuation'
            or approval.get('gpu_count')!=1 or approval.get('gpu_type')!='H100'
            or approval.get('max_seconds',0)<max_seconds
            or not approval.get('approving_person') or not approval.get('approved_at')):
        raise ValueError('Administrator approval of continuation conditions is missing')
    evidence=confined(root,root/approval['source_evidence_file'])
    if sha(evidence)!=approval['source_evidence_sha256'] or evidence.stat().st_size==0:
        raise ValueError('Administrator approval source is not preserved')
    if approval.get('expires_unix') is not None and approval['expires_unix']<time.time()+max_seconds:
        raise ValueError('Approval expires before requested allocation')
    if (run/'.running').exists():raise ValueError('Preserve/archive the confirmed stale lock before continuation')
    return dict(status='continuation_gates_passed',plan_sha256=plan_sha,approval_sha256=approval_sha,
        checkpoint_sha256=plan['checkpoint_sha256'],journal_sha256=plan['journal_sha256'],
        completed_gpu_seconds_upper_bound=plan['completed_gpu_seconds_upper_bound'],
        max_seconds=max_seconds,next_phase=30,next_update=401)
