"""Read existing artifacts only; never invoke training or inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for seed in (2025, 2026):
    run = ROOT / f'seed{seed}_H0'
    nodes = json.loads((run / 'PARENT_VALIDATIONS.json').read_text())
    nodes += [json.loads(p.read_text()) for p in sorted((run / 'checkpoints').glob('R??.json'))]
    events = []
    for line in (ROOT / f'{seed}_H0.log').read_text().splitlines():
        try:
            v = json.loads(line)
            if 'event' in v: events.append(v)
        except ValueError:
            pass
    rows.append(dict(seed=seed, validations=[{k: v[k] for k in ('epoch', 'T', 'ndcg5')} for v in nodes],
                     last_event=events[-1] if events else None,
                     tests=json.loads((run / 'TEST_RESULTS.json').read_text()) if (run / 'TEST_RESULTS.json').exists() else [],
                     complete=(run / 'GPU_COMPLETE.json').exists()))
print(json.dumps(rows))
