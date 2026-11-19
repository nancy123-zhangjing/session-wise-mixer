"""One-off, allowlisted retirement of the abandoned stage08 execution copies.

Default is inventory only. --apply requires the exact inventory fingerprint.
This script never imports torch, starts a job, or touches another stage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_session_codeword')
NAMES = ('goal_20261024_session_codeword_v1', 'topk_test_20261025_r1',
         'topk_validation_20261024_r1', 'topk_validation_20261024_r2')
RECEIPT = ROOT.parent / 'stage08_retirement_20261026.json'

def inventory():
    if ROOT.is_symlink() or ROOT.resolve() != ROOT:
        raise RuntimeError('Root identity changed')
    unexpected = {p.name for p in ROOT.iterdir()} - set(NAMES)
    if unexpected:
        raise RuntimeError(f'Unexpected children: {sorted(unexpected)}')
    rows = []
    links = []
    for name in NAMES:
        target = ROOT / name
        if not target.is_dir() or target.resolve().parent != ROOT:
            raise RuntimeError(f'Unexpected target: {target}')
        for directory, dirs, files in os.walk(target, followlinks=False):
            for leaf in dirs + files:
                p = Path(directory) / leaf
                if p.is_symlink():
                    # os.walk(followlinks=False) and shutil.rmtree unlink these
                    # entries without visiting their targets, including shared data.
                    destination = os.readlink(p)
                    links.append({'path': str(p.relative_to(ROOT)), 'target': destination,
                                  'resolved': str(p.resolve())})
                    rows.append([str(p.relative_to(ROOT)), 0, p.lstat().st_mtime_ns,
                                 'symlink', destination])
                elif not p.resolve().is_relative_to(ROOT):
                    raise RuntimeError(f'Escape: {p}')
            for leaf in files:
                p = Path(directory) / leaf
                if p.is_symlink():
                    continue
                stat = p.stat()
                rows.append([str(p.relative_to(ROOT)), stat.st_size, stat.st_mtime_ns])
    rows.sort()
    digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return rows, digest, links

def active_consumers():
    found = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name == str(os.getpid()):
            continue
        try:
            cmd = p.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if str(ROOT) in cmd:
                found.append(int(p.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--expected-sha')
    args = ap.parse_args()
    rows, digest, links = inventory()
    consumers = active_consumers()
    summary = {'root': str(ROOT), 'targets': list(NAMES), 'file_count': len(rows),
               'bytes': sum(r[1] for r in rows), 'inventory_sha256': digest,
               'active_consumers_on_cpu_bridge': consumers, 'symlinks_unlinked_only': links,
               'applied': False}
    if args.apply:
        if digest != args.expected_sha or consumers:
            raise RuntimeError('Changed inventory or active consumer; deletion refused')
        if RECEIPT.exists():
            raise RuntimeError('Existing retirement receipt; refuse duplicate action')
        receipt = dict(summary, files=rows)
        RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        for name in NAMES:
            shutil.rmtree(ROOT / name)
        summary['applied'] = True
        summary['remaining_children'] = [p.name for p in ROOT.iterdir()]
        receipt.update(summary)
        RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
