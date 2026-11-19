"""Read-only, post-training CPU audit. Does not run inference or optimizer steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
torch.set_num_threads(4)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), 'Nonfinite tensor'
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            finite(v)


def audit(root):
    records = []
    shared_config = shared_runtime = None
    shared_binding = None
    noises = {split: digest(root / 'prepared' / f'{split}_noise.npy') for split in ('validation', 'test')}
    assert digest(root / 'prepared' / 'initial.pt') == '359b3815bbaa8e79929ddf5349392096e6826b28263369e6f2b2f48f239cd4dd'
    for arm in ('B0', 'B1', 'B2'):
        run = root / arm
        done = read(run / 'TRAINING_COMPLETE.json')
        best = read(run / 'BEST.json')
        tests = read(run / 'TEST_COMPLETE.json')
        pre = read(root / (arm + '_rq') / 'COMPLETE.json')
        assert pre['epochs'] == 10000 and pre['updates'] == 110000
        assert done['R'] == 35 and done['T'] == 300 and done['R_updates'] == 35 * 693
        selected = torch.load(root / (arm + '_rq') / 'best_collision.pt', map_location='cpu', weights_only=False)
        frozen = {k: v for k, v in selected['tokenizer'].items() if k.startswith(('encoder.', 'decoder.'))}
        assert any(k.startswith('encoder.') for k in frozen)
        assert any(k.startswith('decoder.') for k in frozen)
        checkpoints = sorted((run / 'checkpoints').glob('R*_T*.pt'))
        assert len(checkpoints) == 31, (arm, len(checkpoints))
        scores = []
        for r in range(5, 36):
            t = (r - 5) * 10
            path = run / 'checkpoints' / f'R{r:02d}_T{t:03d}.pt'
            sha = digest(path)
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['format'] == 'rqcd_mixer_joint_v1'
            assert state['r_epoch'] == r and state['t_updates'] == t and state['r_updates'] == r * 693
            assert state['catalog']['version'] == r - 5
            assert state['t_in_block'] == 0
            assert state['rng']['cuda'] and state['generators']
            for k in ('python', 'numpy', 'cpu', 'cuda'):
                assert k in state['rng']
            for k in ('model', 'tokenizer', 'diffusion', 'catalog', 'optimizer_r', 'optimizer_t'):
                finite(state[k])
            assert all(torch.equal(v, state['tokenizer'][k]) for k, v in frozen.items()), (arm, r, 'Frozen RQ changed')
            for p in ('r', 't'):
                stream = state[p + '_stream']
                updates = stream['epoch'] * 693 + math.ceil(stream['cursor'] / 1024)
                assert updates == state[p + '_updates']
                steps = [int(v['step']) for v in state['optimizer_' + p]['state'].values()]
                assert (steps and all(s == updates for s in steps)) if updates else not steps
            if shared_config is None:
                shared_config, shared_runtime = state['config'], state['runtime']
            assert state['config'] == shared_config and state['runtime'] == shared_runtime
            if r == 5:
                binding = state['binding']
                common_binding = {k: binding[k] for k in ('prepared_sha256', 'source', 'data', 'environment')}
                if shared_binding is None:
                    shared_binding = common_binding
                assert common_binding == shared_binding
                for relative, expected_sha in binding['source'].items():
                    assert digest(root / relative) == expected_sha, relative
                assert digest(root / (arm + '_rq') / 'catalog.pt') == binding['catalog_sha256']
            result = read(run / 'validation' / f'R{r:02d}' / 'summary.json')
            assert result['count'] == 71657
            assert result['identity']['R'] == r and result['identity']['T'] == t
            assert result['identity']['checkpoint_sha256'] == sha
            assert result['identity']['binding'] == state['binding']
            assert result['catalog_hash'] == state['catalog']['version_hash']
            assert result['noise_sha256'] == noises['validation']
            assert result['ddim_steps'] == 20 and result['candidate_lists'] == 1 and result['eta'] == 0
            scores.append((r, result['metrics']['ndcg@5'], sha))
            records.append({'arm': arm, 'R': r, 'T': t, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha})
            del state
        winning = max(scores, key=lambda row: row[1])
        assert best['R'] == winning[0] and best['sha256'] == winning[2]
        assert done['final_sha256'] == scores[-1][2]
        for name, expected in (('best', winning), ('final', scores[-1])):
            result = tests[name]
            assert result['count'] == 53194
            assert result['identity']['R'] == expected[0]
            assert result['identity']['checkpoint_sha256'] == expected[2]
            assert result['noise_sha256'] == noises['test']
            assert result['ddim_steps'] == 20 and result['candidate_lists'] == 1 and result['eta'] == 0
        print(json.dumps({'arm': arm, 'checkpoint_audit': 'passed', 'nodes': 31, 'best_R': winning[0]}), flush=True)
    return {'scope': 'CPU state/clock/frozen-RQ/validation/test identity audit; GPU recovery benchmark and resource closure audited separately',
            'passed': True, 'checkpoints': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
