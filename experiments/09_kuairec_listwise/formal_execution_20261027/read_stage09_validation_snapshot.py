"""Read one locked Stage09 validation epoch through the official SDK.

No remote writes or workload changes. Each successful observation creates a new
local snapshot; the snapshot is evidence, not final checkpoint acceptance.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import shlex

JOB = 'nancy-stage09-four-arm-20261027-r1'
WORKSPACE = '分布式训练空间'
ARMS = ('J0', 'J1', 'H0', 'H1')
EPOCHS = tuple(range(5, 36, 3))
MARKER = 'STAGE09_VALIDATION_SNAPSHOT_JSON='

REMOTE_READ = r'''
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path

G = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')

def tail_training(path):
    if not path.is_file(): return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    rows = []
    for line in reversed(lines):
        try: row = json.loads(line)
        except ValueError: continue
        if isinstance(row, dict): rows.append(row)
        if len(rows) == 3: break
    return rows[::-1]

def main(epoch):
    if epoch not in range(5, 36, 3): raise ValueError('Invalid epoch')
    out = {'remote_observed_at_utc': datetime.now(timezone.utc).isoformat(),
           'epoch': epoch, 'arms': {}}
    for arm in ('J0', 'J1', 'H0', 'H1'):
        run = G / 'runs' / arm
        # R5 is computed once per architecture before the paired branches.
        owner = G / 'runs' / ('warmup_' + arm[0]) if epoch == 5 else run
        path = owner / ('validation_R%02d' % epoch) / 'summary.json'
        row = {'summary_path': str(path),
               'summary_scope': 'shared_warmup' if epoch == 5 else 'arm',
               'summary': None, 'summary_sha256': None, 'summary_bytes': None,
               'training_tail': [], 'run_complete': None}
        out['arms'][arm] = row
        try:
            row['run_complete'] = (run / 'complete.json').is_file()
            row['training_tail'] = tail_training(run / 'training.jsonl')
            if path.is_file():
                raw = path.read_bytes()
                row['summary_sha256'] = hashlib.sha256(raw).hexdigest()
                row['summary_bytes'] = len(raw)
                parsed = json.loads(raw)
                if not isinstance(parsed, dict): raise ValueError('Expected summary object')
                row['summary'] = parsed
        except Exception as error:
            # A newly published summary can still be mid-write. Keep its raw
            # fingerprint but never call a failed/incomplete read complete.
            row['read_exception_type'] = type(error).__name__
    print('STAGE09_VALIDATION_SNAPSHOT_JSON=' + json.dumps(out, separators=(',', ':'), allow_nan=False))
'''


def snapshot(epoch):
    from inspire import InspireClient
    with InspireClient(account='sii-shared', allow_browser=False, timeout=15,
                       operation_timeout=60, catalog_disk_cache=False) as client:
        job = client.jobs.get(JOB, workspace=WORKSPACE)
        if job.status.upper() != 'RUNNING':
            raise RuntimeError('Original Job is not RUNNING')
        command = REMOTE_READ + '\nmain(' + str(epoch) + ')\n'
        result = client.jobs.exec(job.ref, command='python -B -c ' + shlex.quote(command),
                                  timeout=40, max_output_bytes=262144)
    if not result.completed or result.returncode != 0 or result.truncated:
        raise RuntimeError('Read command did not complete cleanly')
    rows = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if len(rows) != 1:
        raise ValueError('Expected one snapshot marker')
    data = json.loads(rows[0])
    if data.get('epoch') != epoch or set(data.get('arms', {})) != set(ARMS):
        raise ValueError('Unexpected snapshot identity')
    data.update(schema='stage09_full_validation_snapshot_v1',
                observed_at_utc=datetime.now(timezone.utc).isoformat(),
                job_name=job.name, job_status=job.status,
                channel='Official SDK jobs.get/exec; remote read-only')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, choices=EPOCHS, required=True,
                        help='Requested R epoch; R5 explicitly reads the shared architecture warmup.')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        # SDK diagnostics can contain access URLs. Never echo their stdout,
        # stderr, exception text, raw job objects, or command results.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data = snapshot(args.epoch)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = Path(__file__).resolve().with_name(f'FORMAL_VALIDATION_R{args.epoch:02d}_{stamp}.json')
        encoded = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        with path.open('xb') as stream:
            stream.write(encoded)
        arms = {}
        for arm, row in data['arms'].items():
            summary = row['summary']
            arms[arm] = dict(summary_present=summary is not None,
                            full_validation=bool(summary and summary.get('full_split') is True
                                                 and summary.get('examples') == 71657),
                            examples=summary.get('examples') if summary else None,
                            run_complete=row['run_complete'])
            if 'read_exception_type' in row:
                arms[arm]['read_exception_type'] = row['read_exception_type']
        print(json.dumps(dict(file=str(path), sha256=hashlib.sha256(encoded).hexdigest(),
                              bytes=len(encoded), epoch=args.epoch, arms=arms), ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'snapshot_failed', 'exception_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
