"""Read-only parent/source checks; create only this branch's manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'multiseed_freeze_20261109_v1'
CODE = ROOT.parent / 'h_freeze35_20261029_r1/code'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old = json.loads((PARENT / 'MANIFEST.json').read_text())
for relative, digest in old['source_sha256'].items():
    assert sha(CODE / relative) == digest, relative
parents = {}
for seed in (2025, 2026):
    cfg = ROOT / f'config_seed{seed}.json'
    assert cfg.read_bytes() == (PARENT / cfg.name).read_bytes()
    run = PARENT / f'seed{seed}_H0'
    receipt = json.loads((run / 'checkpoints/R20.json').read_text())
    assert receipt['epoch'] == 20 and receipt['T'] == 200
    checkpoint = Path(receipt['checkpoint'])
    assert checkpoint.is_file() and checkpoint.parent == run / 'checkpoints'
    assert (run / 'GPU_COMPLETE.json').exists()
    scores = [json.loads(p.read_text()) for p in sorted((run / 'checkpoints').glob('R??.json')) if int(p.stem[1:]) <= 20]
    assert max(scores, key=lambda v: v['ndcg5'])['epoch'] == 20
    parents[str(seed)] = receipt
manifest = dict(format='h0_multiseed_r20_freeze_v1', parents=parents,
                runner_sha256={n: sha(ROOT / n) for n in ('experiment.py', 'protocol.py')},
                config_sha256={str(s): sha(ROOT / f'config_seed{s}.json') for s in (2025, 2026)},
                source_sha256=old['source_sha256'])
(ROOT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
