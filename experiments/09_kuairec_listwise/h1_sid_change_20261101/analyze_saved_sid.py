"""Compare saved original H1 hard SID tensors only; no model or inference."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/stage09_restore/goal_20261027_r1/runs/H1'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

states = {}
sources = {}
for name in ('initial', 'best', 'last'):
    path = SOURCE / (name + '.pt')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    sid = cp['alignment']['hard_codes'].detach().cpu().clone()
    assert tuple(sid.shape) == (10729, 4)
    assert sid[0].eq(-1).all()
    states[name] = sid[1:]
    sources[name] = dict(path=str(path), sha256=sha(path), t_updates=cp['t_updates'],
                         r_updates=cp['r_updates'], real_items=10728, slots=4)
    del cp
assert [sources[n]['t_updates'] for n in ('initial', 'best', 'last')] == [0, 200, 400]
comparisons = []
for a, b in (('initial', 'best'), ('best', 'last'), ('initial', 'last')):
    changed = states[a].ne(states[b])
    count = int(changed.any(dim=1).sum())
    comparisons.append(dict(start=a, end=b, t_start=sources[a]['t_updates'],
        t_end=sources[b]['t_updates'], changed_items=count, total_items=10728,
        changed_item_fraction=count/10728,
        changed_items_per_slot=[int(changed[:, j].sum()) for j in range(4)]))
result = dict(status='passed_saved_tensor_comparison_only', sources=sources,
    definition='Fraction of 10728 real items whose four-code SID differs in at least one slot between saved endpoints; PAD excluded.',
    comparisons=comparisons,
    limitations=['Original H1 refreshes hard SID only after a 40T block.',
        'Original local run preserves initial/best/last checkpoints, not per-20T or every-40T snapshots.',
        'Endpoint difference is not cumulative switching frequency, and cannot be divided by 10 to estimate a 20T rate.'],
    model_constructed=False, model_forward=False, training=False, gpu_used=False,
    source_files_changed=False)
with (HERE/'SAVED_SID_COMPARISON.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))
