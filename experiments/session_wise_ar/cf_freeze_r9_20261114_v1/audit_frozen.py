"""Read-only comparison of an actual continuation checkpoint to its R9 parent."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def audit(parent, checkpoint):
    p = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)
    s = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    keys = ('config', 'runtime', 'binding', 'tokenizer', 'catalog', 'optimizer_t', 't_stream', 'diffusion')
    checks = {key: equal(p[key], s[key]) for key in keys}
    checks.update({key: equal(p['generators'][key], s['generators'][key])
                   for key in ('T_times', 'T_epsilon', 'T_gumbel')})
    checks['T40'] = s['t_updates'] == p['t_updates'] == 40
    checks['R_advanced'] = s['r_updates'] > p['r_updates']
    checks['model_changed'] = not equal(p['model'], s['model'])
    checks['R_optimizer_clock'] = all(int(v['step']) == s['r_updates'] for v in s['optimizer_r']['state'].values())
    checks['policy'] = s.get('freeze_policy', {}).get('name') == 'cf_freeze_r9_v1'
    return {'passed': all(checks.values()), 'checks': checks, 'R_updates': s['r_updates'],
            'T_updates': s['t_updates'], 'checkpoint': str(checkpoint), 'parent': str(parent),
            'scope': 'CPU state equality on actual GPU-produced checkpoint; not final acceptance'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('parent', type=Path)
    p.add_argument('checkpoint', type=Path)
    a = p.parse_args()
    result = audit(a.parent, a.checkpoint)
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)
