"""Deploy verified joint source bundles into the already-created CPU Goal.

This script never creates/replaces the Goal root, deletes partial checkouts,
installs packages, starts GPU work, or changes either old source checkout.
All four CPU suites (20 tests) and both Torch-blocked prepare subprocesses are
mandatory. Invoke with the full frozen commits and both transferred bundle SHAs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07')
OLD = BASE / 'goal_20261020_ocn_repair_r1'
NEW = BASE / 'goal_20261020_item_local_joint_r1'
PROJECT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
OLD_ROOT = '2fc97b62693628f6507ad2ca60472f6063a84250'
OLD_ENGINE = '8a48a300daa6621d6c705966ff73fa837ef8f12e'
JOINT_ENGINE = '1c879252f6395d6e64462511823c3929b9c42c00'
CAMPAIGN = 'stage07_item_local_alternating_joint_seed2024_20261020_r1'
TESTS = (('test_item_local_alternating_joint.py', 5),
         ('test_item_local_joint_training_state.py', 3),
         ('test_ocn_autocast_gradient.py', 4),
         ('test_item_local_training_state.py', 8))


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def git(path, *arguments):
    return subprocess.check_output(['git', '-C', str(path), *arguments], text=True).strip()


def persist(path, record, *, first=False):
    if first:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    else:
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        temporary.replace(path)


def input_receipt(path):
    value = read_json(path)
    need(value.get('status') == 'JOINT_INPUT_ASSETS_VERIFIED'
         and value.get('new_joint_goal_root') == str(NEW)
         and value.get('project_read_only') == str(PROJECT)
         and value.get('training_code_deployed') is False and value.get('gpu_started') is False,
         'Existing 13-input receipt does not authorize this new CPU Goal')
    records = value['files']
    need(len(records) == 13 and len({row['path'] for row in records}) == 13, 'Need thirteen unique verified inputs')
    for row in records:
        relative = PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts
             and row.get('passed') is True and row.get('exists') is True
             and re.fullmatch('[0-9a-f]{64}', row['sha256'])
             and row.get('observed_sha256') == row['sha256']
             and row['absolute_path'] == str(PROJECT / row['path'])
             and type(row['size_bytes']) is int and row['size_bytes'] > 0,
             'Invalid/mismatched input receipt record: ' + row['path'])
    return value


PREPARE_GUARD = r'''
import builtins, runpy, sys
original_import = builtins.__import__
def cpu_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in {'torch', 'cuda', 'cupy', 'pycuda'}:
        raise RuntimeError('CPU prepare must not import Torch/CUDA: ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = cpu_import
runner = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(runner, run_name='__main__')
if any(name.split('.')[0] in {'torch', 'cuda', 'cupy', 'pycuda'} for name in sys.modules):
    raise RuntimeError('A Torch/CUDA module escaped the CPU preparation import guard')
print('PREPARE_CPU_IMPORT_GUARD_PASS')
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-commit', '--expected-root-commit', dest='root_commit', required=True)
    parser.add_argument('--engine-commit', '--expected-engine-commit', dest='engine_commit', required=True)
    parser.add_argument('--root-bundle-sha', required=True)
    parser.add_argument('--engine-bundle-sha', required=True)
    args = parser.parse_args(argv)
    for value, width in ((args.root_commit, 40), (args.engine_commit, 40),
                         (args.root_bundle_sha, 64), (args.engine_bundle_sha, 64)):
        need(re.fullmatch('[0-9a-f]{' + str(width) + '}', value), 'Commits and bundle hashes must be full lowercase hexadecimal identities')
    need(args.engine_commit == JOINT_ENGINE, 'Expected engine must be the approved 1c87925 joint implementation')
    delivery = NEW / 'delivery'
    need(NEW.is_dir() and not NEW.is_symlink() and delivery.is_dir() and not delivery.is_symlink(),
         'The existing Goal and delivery directory must already exist as real directories')
    receipt_path = delivery / 'JOINT_PREPARATION.json'
    need(not receipt_path.exists(), 'Preparation receipt already exists; inspect it instead of rerunning or overwriting')
    assets_path = delivery / 'INPUT_ASSET_CHECK.json'
    assets = input_receipt(assets_path)
    assets_sha = sha(assets_path)
    for name in ('code', 'engine'):
        target = NEW / name
        need(not target.exists() and not target.is_symlink(),
             'Refuse existing/partial checkout; nothing will be deleted or overwritten: ' + str(target))
    bundles = {'code': (args.root_commit, args.root_bundle_sha), 'engine': (args.engine_commit, args.engine_bundle_sha)}
    for name, (_, expected_hash) in bundles.items():
        bundle = delivery / (name + '_joint.bundle')
        need(bundle.is_file() and not bundle.is_symlink() and sha(bundle) == expected_hash, name + ' bundle transfer SHA mismatch')
    for name, expected in (('code', OLD_ROOT), ('engine', OLD_ENGINE)):
        need(git(OLD / name, 'rev-parse', 'HEAD') == expected and git(OLD / name, 'status', '--porcelain') == '',
             'Old read-only source checkout is not the expected clean baseline: ' + name)

    record = {'schema': 'stage07_item_local_joint_CPU_preparation_v1', 'status': 'CPU_PREPARATION_RUNNING',
              'campaign_id': CAMPAIGN, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'new_root': str(NEW), 'old_root_read_only': str(OLD),
              'root_commit': args.root_commit, 'engine_commit': args.engine_commit,
              'bootstrap_sha256': sha(Path(__file__).resolve()), 'input_receipt_sha256': assets_sha,
              'sources': {}, 'checks': [], 'tests_expected': 20, 'gpu_started': False, 'goal_complete': False}
    persist(receipt_path, record, first=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')

    def command(label, arguments, *, timeout=1800):
        log = delivery / f'joint_bootstrap_{len(record["checks"]):02d}_{label}.log'
        check = {'label': label, 'argv': list(map(str, arguments)), 'log': str(log), 'returncode': None}
        record['checks'].append(check)
        try:
            with log.open('x', encoding='utf-8') as handle:
                result = subprocess.run(check['argv'], env=environment, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
            check['returncode'] = result.returncode
        finally:
            if log.is_file(): check['sha256'] = sha(log)
            persist(receipt_path, record)
        need(check['returncode'] == 0, 'CPU command failed; preserve its log and partial checkout: ' + str(log))
        return log.read_text(encoding='utf-8', errors='replace')

    try:
        for name, (commit, expected_hash) in bundles.items():
            target, bundle = NEW / name, delivery / (name + '_joint.bundle')
            command('clone_' + name, ['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout', str(OLD / name), str(target)])
            need(not (target / '.git/objects/info/alternates').exists(), 'A source clone must not share an alternate object store')
            command('verify_' + name + '_bundle', ['git', '-C', str(target), 'bundle', 'verify', str(bundle)])
            advertised = command('list_' + name + '_bundle_heads', ['git', 'bundle', 'list-heads', str(bundle)])
            refs = [line.split(maxsplit=1)[1] for line in advertised.splitlines()
                    if len(line.split(maxsplit=1)) == 2 and line.split(maxsplit=1)[0] == commit]
            need(refs, 'Bundle does not advertise the exact expected commit: ' + name)
            command('fetch_' + name, ['git', '-C', str(target), 'fetch', str(bundle), sorted(refs)[0]])
            need(git(target, 'rev-parse', 'FETCH_HEAD') == commit, 'Fetched source is not the expected frozen commit')
            command('checkout_' + name, ['git', '-C', str(target), 'checkout', '--detach', commit])
            need(git(target, 'rev-parse', 'HEAD') == commit and git(target, 'status', '--porcelain') == '',
                 'Deployed checkout is not clean and detached at its expected source')
            need(subprocess.run(['git', '-C', str(target), 'symbolic-ref', '-q', 'HEAD'], capture_output=True).returncode == 1,
                 'Deployed source must use detached HEAD')
            record['sources'][name] = {'commit': commit, 'bundle_sha256': expected_hash, 'clone_mode': 'no-hardlinks_no-alternates'}
            persist(receipt_path, record)

        link = NEW / 'code/external/DiffGRM'
        need(not link.exists() and not link.is_symlink(), 'Unexpected existing engine path; refuse replacement')
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(NEW / 'engine', target_is_directory=True)
        need(link.resolve() == (NEW / 'engine').resolve(), 'Engine symlink points outside the new Goal')
        command('verify_ignored_engine_link', ['git', '-C', str(NEW / 'code'), 'check-ignore', 'external/DiffGRM'])
        package = NEW / 'code/experiments/05_crosspse_joint/reproducibility'
        recipe, lock = read_json(package / 'recipes/refined_joint.json'), read_json(package / 'assets.lock.json')['artifacts']
        need({row['path'] for row in assets['files']} == set(recipe['required_artifacts']) and len(recipe['required_artifacts']) == 13,
             'Existing asset receipt does not cover the frozen thirteen-input joint recipe')
        for row in assets['files']:
            need(row['sha256'] == lock[row['path']]['sha256'] and row['size_bytes'] == lock[row['path']]['size_bytes'],
                 'Input receipt differs from frozen asset lock: ' + row['path'])

        total = 0
        for filename, expected_count in TESTS:
            output = command('test_' + filename.removesuffix('.py'), [sys.executable, '-X', 'utf8', '-B', str(NEW / 'code/tests' / filename)])
            counts = re.findall(r'Ran (\d+) tests? in ', output)
            need(counts and int(counts[-1]) == expected_count and 'skipped=' not in output and re.search(r'^OK\s*$', output, re.MULTILINE),
                 'CPU suite did not execute its exact required test count without skips: ' + filename)
            total += expected_count
        need(total == 20, 'All twenty CPU tests are mandatory')
        record['tests_passed'] = total
        for variant in ('control', 'item_local'):
            run_id = f'stage07_item_local_joint_{variant}_seed2024_20261020_r1'
            runner = NEW / 'code/scripts/train/run_item_local_joint.py'
            output = command('prepare_' + variant, [sys.executable, '-B', '-c', PREPARE_GUARD, str(runner),
                '--mode', 'prepare', '--config', str(NEW / 'code/configs/07_item_local_attention/joint_alternating_seed2024.json'),
                '--variant', variant, '--output', str(NEW / 'runs' / run_id), '--engine-root', str(NEW / 'engine'),
                '--expected-engine-commit', args.engine_commit, '--project-root', str(PROJECT)])
            need('PREPARE_CPU_IMPORT_GUARD_PASS' in output, 'Prepare did not prove its Torch/CUDA import guard')
            identity = read_json(NEW / 'runs' / run_id / 'identity.json')
            need(identity['campaign_id'] == CAMPAIGN and identity['variant'] == variant and identity['run_id'] == run_id
                 and identity['engine_commit'] == args.engine_commit, 'Prepared run identity/source differs')
        for name, (commit, expected_hash) in bundles.items():
            need(git(NEW / name, 'rev-parse', 'HEAD') == commit and git(NEW / name, 'status', '--porcelain') == '',
                 'CPU tests/preparation modified frozen source: ' + name)
            need(sha(delivery / (name + '_joint.bundle')) == expected_hash, 'Transferred bundle changed during preparation')
        need(sha(assets_path) == assets_sha, 'The preserved original asset receipt changed during preparation')
        record['status'] = 'CPU_PREPARATION_PASS'
        record['prepared_variants'] = ['control', 'item_local']
        record['prepare_torch_imports_blocked'] = True
    except BaseException as error:
        record['status'] = 'CPU_PREPARATION_FAILED'
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        persist(receipt_path, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
