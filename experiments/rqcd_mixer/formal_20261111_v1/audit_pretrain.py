import json
from pathlib import Path
from audit_checkpoints import digest, torch

root = Path(__file__).resolve().parent
records = []
for arm in ('B0', 'B1', 'B2'):
    rq = root / (arm + '_rq')
    complete = json.loads((rq / 'COMPLETE.json').read_text())
    events = [json.loads(x) for x in (rq / 'events.jsonl').read_text().splitlines()]
    checks = [e for e in events if e['kind'] == 'pretrain_validation']
    assert [e['epoch'] for e in checks] == list(range(50, 10001, 50))
    selected_epoch = min(checks, key=lambda e: e['encoding']['collision_redundancy'])['epoch']
    for name in ('last', 'best_collision'):
        s = torch.load(rq / (name + '.pt'), map_location='cpu', weights_only=False)
        assert s['format'] == 'rqcd_mixer_pretrain_v1'
        assert s['epochs'] == (10000 if name == 'last' else selected_epoch)
        assert s['updates'] == s['epochs'] * 11
        assert s['rng']['cuda'] and s['stream'] and s['optimizer']['state']
        assert all(int(v['step']) == s['updates'] for v in s['optimizer']['state'].values())
        assert s['best_collision_epoch'] == selected_epoch
        del s
    assert complete['best_collision_epoch'] == selected_epoch
    records.append({'arm': arm, 'selected_epoch': selected_epoch, 'all_200_diagnostics': True, 'last_and_selected_recovery_state': 'passed'})
content = root.parent / 'goal_20261027_r1/assets/pse_v1/content_features.npy'
binding = json.loads((root / 'B0/binding.json').read_text())
assert digest(content) == binding['data']['content_sha256']
assert digest(root / 'prepared/PREPARED.json') == binding['prepared_sha256']
receipt = {'passed': True, 'pretraining': records,
           'external_content': {'path': str(content), 'bytes': content.stat().st_size, 'sha256': binding['data']['content_sha256']}}
(root / 'pretrain_recovery_audit.json').write_text(json.dumps(receipt, indent=2))
manifest = json.loads((root / 'ASSET_MANIFEST.json').read_text())
manifest['external_data'].append(receipt['external_content'])
for name in ('audit_pretrain.py', 'pretrain_recovery_audit.json'):
    path = root / name
    manifest['files'].append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)})
(root / 'ASSET_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(receipt))
