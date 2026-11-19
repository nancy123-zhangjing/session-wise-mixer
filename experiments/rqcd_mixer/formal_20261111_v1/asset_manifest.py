"""Post-run read-only recovery asset inventory; no model execution."""
import json
from pathlib import Path
from datetime import datetime, timezone
from audit_checkpoints import digest, finite, torch

root = Path(__file__).resolve().parent
audit = json.loads((root / 'checkpoint_audit.json').read_text())
assert audit['passed']
known = {r['path']: r for r in audit['checkpoints']}
rows = []
for path in sorted(root.rglob('*')):
    rel = path.relative_to(root)
    if not path.is_file() or '__pycache__' in rel.parts or '.mplconfig' in rel.parts:
        continue
    if rel.parts[0] == 'formal_20261111_v1' or path.name in ('ASSET_MANIFEST.json', 'asset_manifest.log', 'asset_manifest.exit'):
        continue
    if path.suffix == '.pt' and str(path) not in known:
        state = torch.load(path, map_location='cpu', weights_only=False)
        finite(state)
        if path.parent.name in ('B0', 'B1', 'B2') and path.name == 'last.pt':
            assert state['r_epoch'] == 35 and state['t_updates'] == 300
            assert state['validated_epochs'] == list(range(5, 36))
        del state
    if str(path) in known:
        record = known[str(path)]
        assert path.stat().st_size == record['bytes']
        sha = record['sha256']  # Already hashed/read in completed checkpoint audit; immutable files.
    else:
        sha = digest(path)
    rows.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha})
external = []
for arm in ('B0', 'B1', 'B2'):
    binding = json.loads((root / arm / 'binding.json').read_text())
    paths = json.loads((root / arm / 'artifact_paths.json').read_text())
    if not external:
        dataset = Path(paths['dataset'])
        for name, expected in binding['data']['dataset']['files'].items():
            p = dataset / name
            assert p.stat().st_size == expected['bytes'] and digest(p) == expected['sha256']
            external.append({'path': str(p), **expected})
output = {'created_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
          'checkpoint_audit': 'checkpoint_audit.json', 'files': rows, 'external_data': external,
          'note': 'All large assets remain remote. Checkpoint hashes reused from preceding immutable-file audit. Reports may be supplemented by separate closure receipts.'}
(root / 'ASSET_MANIFEST.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
print(json.dumps({'files': len(rows), 'external_data_files': len(external), 'bytes': sum(r['bytes'] for r in rows)}), flush=True)
