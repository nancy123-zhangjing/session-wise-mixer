"""One-time stopped-run migration into an isolated diagnostic-only revision."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.session_wise_ar.runtime import source_identity
from scripts.rqcd_mixer.common import read_state, atomic_save, write_json, sha256


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def main():
    old = Path(sys.argv[1]).resolve()
    assert old != ROOT and not (ROOT/'runs').exists()
    current = source_identity()
    parents = {}
    records = []
    for family in ('tiger', 'diger'):
        src = old/'parents_devicefix'/family/'initial.pt'
        state = read_state(src)
        assert {k for k in current if current[k] != state['sources'][k]} == {'scripts/session_wise_ar/runtime.py'}
        previous = sha256(src)
        state['sources'] = current
        dst = ROOT/'parents_devicefix'/family
        dst.mkdir(parents=True)
        atomic_save(dst/'initial.pt', state)
        equal(state, read_state(dst/'initial.pt'))
        new_sha = sha256(dst/'initial.pt')
        parents[family] = (previous, new_sha)
        write_json(dst/'manifest.json', dict(family_config=state['family_config'], binding=state['binding'],
                   initial_sha256=new_sha, sources=current))
        records.append(dict(source=str(src), source_sha256=previous, destination=str(dst/'initial.pt'), sha256=new_sha))
    for arm in ('AR-T0', 'AR-T1', 'AR-D0', 'AR-D1'):
        src = old/'runs'/arm/'pretrain'
        dst = ROOT/'runs'/arm/'pretrain'
        dst.mkdir(parents=True)
        old_sha, new_sha = parents['tiger' if arm.startswith('AR-T') else 'diger']
        assert not (src/'complete.json').exists(), 'Only unfinished pretraining migration is supported'
        for path in src.iterdir():
            if path.suffix == '.pt':
                state = read_state(path)
                assert state['parent_sha256'] == old_sha
                assert {k for k in current if current[k] != state['sources'][k]} == {'scripts/session_wise_ar/runtime.py'}
                state['sources'], state['parent_sha256'] = current, new_sha
                atomic_save(dst/path.name, state)
                equal(state, read_state(dst/path.name))
                records.append(dict(source=str(path), source_sha256=sha256(path),
                           destination=str(dst/path.name), sha256=sha256(dst/path.name), step=state['progress']['step']))
            elif path.name == 'progress.json':
                shutil.copy2(path, dst/path.name)
        print(arm, 'preserved tensor/optimizer/RNG/data-stream state', flush=True)
    write_json(ROOT/'DIAGNOSTIC_MIGRATION.json', dict(records=records,
               unchanged_training_state=True, changed_metadata_only=['sources', 'parent_sha256']))


if __name__ == '__main__':
    main()
