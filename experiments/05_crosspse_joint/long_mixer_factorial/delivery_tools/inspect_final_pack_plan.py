"""Read-only file collection forecast after final audit and release, no packing.

Calls the frozen collector, requires its full final gates, checks all nine
formal states and three bundles, and rejects accidentally embedded old backups.
Writes only a new receipt. Never changes the frozen packer or training code.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--artifact-audit', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--packer', type=Path, required=True)
    parser.add_argument('--planned-output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    goal, planned = args.goal_root.resolve(), args.planned_output.resolve()
    receipt = args.receipt.resolve()
    need(not planned.exists() and not receipt.exists(), 'Require new package output and inspection receipt')
    need(not receipt.is_relative_to(planned), 'Do not create the inspection receipt inside the future package directory')
    if receipt.is_relative_to(goal):
        need(receipt.relative_to(goal).parts[0] in ('delivery', 'prep'), 'Keep the receipt out of frozen source/input/formal directories')
    if planned.is_relative_to(goal):
        relative = planned.relative_to(goal)
        need(relative.parts[0] == 'delivery' and any(p.startswith('final_package') for p in relative.parts[1:]),
             'Keep in-goal package output below delivery/final_package* so future collections exclude it')
    spec = importlib.util.spec_from_file_location('_inspect_frozen_final_packer', args.packer.resolve())
    packer = importlib.util.module_from_spec(spec); spec.loader.exec_module(packer)
    sources, metadata = packer.collect(goal, args.artifact_audit.resolve(), args.release.resolve())
    need(packer.mandatory_names().issubset(sources), 'Collection omitted a required state/source/input member')
    states = {'formal/common/initial.pt'} | {'formal/' + arm + '/' + name
              for arm in packer.ARMS for name in ('best.pt', 'last.pt')}
    need(states.issubset(sources) and len(states) == 9, 'Common plus four best/last full states are required')
    embedded = [name for name in sources if name.startswith(('delivery/', 'prep/'))
                and Path(name).name in ('mixer_long_backup.tar.gz', 'PACKAGE.json', 'BYTE_VERIFICATION.json')]
    need(not embedded, 'Prior package/extraction is being recursively included: ' + repr(embedded))
    groups = defaultdict(lambda: {'files': 0, 'size_bytes': 0})
    inventory = []
    for name, path in sorted(sources.items()):
        size = path.stat().st_size
        group = name.split('/')[0]
        groups[group]['files'] += 1; groups[group]['size_bytes'] += size
        inventory.append(dict(member=name, size_bytes=size))
    total = sum(row['size_bytes'] for row in inventory)
    result = dict(status='FINAL_PACKAGE_COLLECTION_FORECAST_NOT_ARCHIVE_OR_RESTORE',
        utc=datetime.now(timezone.utc).isoformat(), campaign_id=packer.CAMPAIGN,
        collector_sha256=sha(args.packer), inspector_sha256=sha(Path(__file__)),
        artifact_audit_sha256=sha(args.artifact_audit), release_sha256=sha(args.release),
        planned_package_output=str(planned), files=len(inventory), uncompressed_bytes=total,
        groups=dict(groups), formal_full_state_count=9,
        formal_full_state_bytes=sum(sources[name].stat().st_size for name in states),
        all_14_locked_inputs_included=metadata['all_14_locked_inputs_included'],
        three_git_bundles_present=True, previously_created_packages_collected=False,
        largest_members=sorted(inventory, key=lambda row: row['size_bytes'], reverse=True)[:20],
        source_locks=metadata['source_locks'],
        note='This receipt is created after collection. If stored under delivery it is an additional final-pack member; final MANIFEST is authoritative.',
        local_download_guidance='Download PACKAGE.json plus only the listed parts; remote retains unsplit archive as a redundant copy.',
        final_archive_created=False, full_backup_restore_verified=False, goal_complete=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
