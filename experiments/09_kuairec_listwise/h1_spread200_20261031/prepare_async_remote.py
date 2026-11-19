"""Verify a separately bound operational package; archive only a proved stale lock."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest-sha',required=True)
    a=p.parse_args();root=Path(a.root).resolve()
    if root!=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1'):raise ValueError('Wrong owned root')
    manifest=root/'ASYNC_DEPLOYMENT.json'
    if sha(manifest)!=a.manifest_sha:raise ValueError('Operational manifest changed')
    package=read(manifest)
    for name,digest in package['files'].items():
        path=root/name;path.resolve().relative_to(root)
        if path.is_symlink() or sha(path)!=digest:raise ValueError('Operational file changed: '+name)
    gate=read(root/'ASYNC_RESUME_GATE.json');run=root/'runs/H1-Spread200-R35'
    for name,digest in gate['evidence'].items():
        if sha(root/name)!=digest:raise ValueError('Recovery evidence changed: '+name)
    if sha(gate['checkpoint'])!=gate['checkpoint_sha256'] or sha(run/'training.jsonl')!=gate['journal_sha256']:
        raise ValueError('Formal checkpoint or journal changed')
    if read(root/'ASYNC_PREVIOUS_PLATFORM_STATUS.json')['status']!='CANCELLED':raise ValueError('Prior job not terminal')
    if (root/'jobs/formal_async_r2').exists():raise ValueError('Attempt already exists; inspect, never overwrite')
    # This invocation follows a live platform status/list check by the caller.
    # Keep the original lock bytes and record why it is safe to move this lock.
    lock=run/'.running';archive=root/'recovery_async_r2/ORIGINAL_RUNNING_LOCK.json'
    receipt=root/'recovery_async_r2/STALE_LOCK_ARCHIVED.json'
    if archive.exists() or receipt.exists():raise ValueError('Lock was already handled; inspect existing evidence')
    old=read(lock)
    if old['pid']!=201 or old['created_at']!=1793382279.9311302:raise ValueError('Unrecognized lock owner')
    digest=sha(lock);lock.rename(archive)
    with receipt.open('x',encoding='utf-8') as f:
        json.dump(dict(status='archived_stale_lock_after_live_terminal_confirmation',unix=time.time(),
            old_job='nancy-s09-d-1031-r1',terminal_receipt_sha256=sha(root/'ASYNC_PREVIOUS_PLATFORM_STATUS.json'),
            archived_lock=str(archive),archived_lock_sha256=digest,checkpoint_sha256=sha(gate['checkpoint']),
            journal_sha256=sha(run/'training.jsonl'),package_sha256=a.manifest_sha),f,indent=2);f.write('\n')
    print(json.dumps(dict(status='prepared_async_continuation',package_sha256=a.manifest_sha,
        gate_sha256=sha(root/'ASYNC_RESUME_GATE.json'),runtime_sha256=sha(root/'async_resume.py'),
        launcher_sha256=sha(root/'job_async_r2.py'),lock_receipt_sha256=sha(receipt))))


if __name__=='__main__':main()
