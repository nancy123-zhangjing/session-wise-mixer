"""Inventory/delete the explicit Stage06/07 remote run copies authorized on 2026-10-27.

Never follows symlinks and never operates on Stage05, shared environments, the
project root, or local backups. The dry-run inventory hash is required to apply.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import stat

BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy')
GROUPS = {
    'experiment1_stage06': ['goal_20261016_precision_v2', 'goal_20261016'],
    'experiment1_stage07': ['beam_trace_20261021_r1', 'goal_20261020_ocn_repair_r1',
        'backup_20261021_multiseed_r1', 'joint_delivery_package_20261020_final_r3',
        'goal_20261020_0045', 'beam_validation_20261021_r1',
        'goal_20261020_item_local_joint_r2', 'goal_20261021_multiseed_r1',
        'goal_20261020_item_local_joint_r1'],
}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def inventory():
    require(BASE.resolve() == BASE, 'Unexpected base resolution')
    targets, records = [], []
    for group, names in sorted(GROUPS.items()):
        allowed = BASE / group
        require(not allowed.is_symlink() and allowed.resolve() == allowed, 'Unexpected group resolution')
        actual = sorted(p.name for p in allowed.iterdir())
        require(actual == sorted(names), f'Unexpected entries under {allowed}: {actual}')
        for name in sorted(names):
            target = allowed / name
            require(target.is_dir() and not target.is_symlink(), f'Unexpected target type: {target}')
            require(target.resolve().parent == allowed, f'Target escaped group: {target}')
            targets.append(str(target))
            pending = [target]
            while pending:
                path = pending.pop()
                info = path.lstat()
                rel = str(path.relative_to(BASE))
                row = {'path': rel, 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                       'mode': info.st_mode, 'inode': info.st_ino, 'device': info.st_dev}
                if stat.S_ISLNK(info.st_mode):
                    row['kind'] = 'symlink'
                    row['link_target'] = os.readlink(path)
                elif stat.S_ISDIR(info.st_mode):
                    row['kind'] = 'directory'
                    pending.extend(sorted(path.iterdir(), reverse=True))
                elif stat.S_ISREG(info.st_mode):
                    row['kind'] = 'file'
                else:
                    raise RuntimeError(f'Refuse special file: {path}')
                records.append(row)
    records.sort(key=lambda row: row['path'])
    stable = {'base': str(BASE), 'targets': targets, 'records': records}
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return stable, digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-inventory-sha256')
    args = parser.parse_args()
    plan, digest = inventory()
    totals = {group: {'files': 0, 'logical_file_bytes': 0, 'symlinks': 0} for group in GROUPS}
    for row in plan['records']:
        group = row['path'].split('/')[0]
        totals[group]['files'] += row['kind'] == 'file'
        totals[group]['symlinks'] += row['kind'] == 'symlink'
        if row['kind'] == 'file':
            totals[group]['logical_file_bytes'] += row['size']
    if args.apply_inventory_sha256:
        require(args.apply_inventory_sha256 == digest, 'Inventory changed; refuse deletion')
        require(getattr(shutil.rmtree, 'avoids_symlink_attacks', False), 'Require Linux fd-safe rmtree')
        for target in plan['targets']:
            path = Path(target)
            require(path.parent.name in GROUPS and path.name in GROUPS[path.parent.name], 'Target not allowlisted')
            require(path.resolve() == path and path.is_relative_to(BASE), 'Target escaped authorized root')
            shutil.rmtree(path)
        leftovers = {group: sorted(p.name for p in (BASE / group).iterdir()) for group in GROUPS}
        require(not any(leftovers.values()), f'Unexpected leftovers: {leftovers}')
        print(json.dumps({'status': 'deleted_and_absence_verified', 'inventory_sha256': digest,
            'deleted_targets': plan['targets'], 'totals': totals, 'remaining': leftovers,
            'finished_at_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False))
    else:
        print(json.dumps({'status': 'inventory_only', 'inventory_sha256': digest,
                          'totals': totals, **plan}, ensure_ascii=False))

if __name__ == '__main__':
    main()
