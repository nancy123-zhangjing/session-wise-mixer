"""Create isolated frozen multiseed checkouts on the existing CPU notebook."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07')
OLD = BASE / 'goal_20261020_item_local_joint_r2'
GOAL = BASE / 'goal_20261021_multiseed_r1'
OLD_CODE = '842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1'
ENGINE = '1c879252f6395d6e64462511823c3929b9c42c00'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    args = parser.parse_args()
    delivery = GOAL / 'delivery'
    bundle = delivery / 'multiseed.bundle'
    need(Path(__file__).resolve() == delivery / 'bootstrap.py', 'Unexpected bootstrap location')
    need(sha(bundle) == args.bundle_sha256, 'Uploaded bundle SHA differs')
    for label, expected in (('code', OLD_CODE), ('engine', ENGINE)):
        need(git(OLD / label, 'rev-parse', 'HEAD') == expected
             and not git(OLD / label, 'status', '--porcelain'), 'Historical source is not frozen: ' + label)
        need(not (GOAL / label).exists() and not (GOAL / label).is_symlink(),
             'Inspect existing deployment instead of overwriting: ' + label)
    receipt_path = delivery / 'BOOTSTRAP.json'
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'code_commit': args.code_commit, 'engine_commit': ENGINE,
              'bundle_sha256': args.bundle_sha256, 'bootstrap_sha256': sha(__file__), 'steps': []}
    with receipt_path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)

    def save():
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(receipt_path)

    def run(label, command):
        log = delivery / ('bootstrap_' + label + '.log')
        with log.open('x', encoding='utf-8') as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=180, cwd=delivery)
        record['steps'].append({'label': label, 'returncode': result.returncode,
                                'log': str(log), 'sha256': sha(log)})
        save()
        need(result.returncode == 0, 'Bootstrap step failed: ' + str(log))

    try:
        for label in ('code', 'engine'):
            run('clone_' + label, ['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout',
                                  str(OLD / label), str(GOAL / label)])
            need(not (GOAL / label / '.git/objects/info/alternates').exists(), 'Unexpected shared Git object store')
        run('verify_bundle', ['git', '-C', str(GOAL / 'code'), 'bundle', 'verify', str(bundle)])
        heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True)
        refs = [line.split(maxsplit=1)[1] for line in heads.splitlines()
                if line.split(maxsplit=1)[0] == args.code_commit]
        need(refs, 'Bundle does not advertise requested commit')
        run('fetch', ['git', '-C', str(GOAL / 'code'), 'fetch', str(bundle), refs[0]])
        need(git(GOAL / 'code', 'rev-parse', 'FETCH_HEAD') == args.code_commit, 'Wrong fetched commit')
        need(git(GOAL / 'code', 'merge-base', OLD_CODE, args.code_commit) == OLD_CODE,
             'New source must descend from the frozen base')
        for label, expected in (('code', args.code_commit), ('engine', ENGINE)):
            run('checkout_' + label, ['git', '-C', str(GOAL / label), 'checkout', '--detach', expected])
            need(git(GOAL / label, 'rev-parse', 'HEAD') == expected
                 and not git(GOAL / label, 'status', '--porcelain'), 'New checkout differs: ' + label)
        link = GOAL / 'code/external/DiffGRM'
        link.parent.mkdir(exist_ok=True)
        need(not link.exists() and not link.is_symlink(), 'Unexpected test engine link')
        link.symlink_to(GOAL / 'engine', target_is_directory=True)
        need(git(GOAL / 'code', 'check-ignore', 'external/DiffGRM') == 'external/DiffGRM',
             'Test engine link must be ignored')
        need(not git(GOAL / 'code', 'status', '--porcelain'), 'Deployment changed source')
        record.update(status='passed', gpu_started=False,
                      cpu_prepare_command=[sys.executable, '-B', str(GOAL / 'code/experiments/07_item_local_attention/multiseed_20261021/prepare_cpu.py'),
                                           '--goal-root', str(GOAL)])
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
