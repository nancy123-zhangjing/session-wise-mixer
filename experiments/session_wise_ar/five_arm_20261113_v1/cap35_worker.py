"""Authorized execution cap, retaining immutable model code and 120-epoch LR schedule."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

OLD = "    total = c['max_steps'] or c['max_epochs']*per_epoch"
NEW = "    total = min(c['max_steps'] or c['max_epochs']*per_epoch, 35*per_epoch)"


def capped_source(source):
    if source.count(OLD) != 1:
        raise ValueError('Unexpected training implementation; do not patch heuristically')
    return source.replace(OLD, NEW, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=['AR-D0', 'AR-D1'], required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from scripts.session_wise_ar import runtime
    from scripts.session_wise_ar.config import load
    from scripts.rqcd_mixer.common import write_json, sha256, read_state
    cfg = load(args.arm)
    assert cfg['training']['max_epochs'] == 120 and cfg['training']['patience'] == 15
    source = inspect.getsource(runtime.train)
    changed = capped_source(source)
    namespace = dict(runtime.__dict__)
    exec(compile(changed, str(Path(__file__).resolve()), 'exec'), namespace)
    train = namespace['train']
    target = root/'runs'/args.arm
    policy = {'max_execution_epochs':35, 'scheduler_epochs':120, 'patience':15,
              'base_runtime_sha256':sha256(runtime.__file__), 'wrapper_sha256':sha256(__file__),
              'original_function_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'capped_function_sha256':hashlib.sha256(changed.encode()).hexdigest(),
              'change':{'old':OLD.strip(),'new':NEW.strip()}}
    policy_path = target/'train/EXECUTION_CAP.json'
    if policy_path.exists():
        assert json.loads(policy_path.read_text()) == policy
    else:
        write_json(policy_path, policy)
    state = read_state(target/'train/last.pt')
    assert state['config'] == cfg and state['progress']['step'] < 35*13857
    write_json(target/'train/CAP_RESUME_START.json', {
        'checkpoint_sha256':sha256(target/'train/last.pt'),
        'step':state['progress']['step'], 'best_step':state['progress']['best_step'],
        'policy':policy, 'scope':'pre-restore snapshot; success established by subsequent progress'})
    del state
    base = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1'
    common = dict(cfg=cfg, prepared=root/'parents_devicefix/diger',
                  pretrain_path=target/'pretrain', dataset=base+'/inputs/dataset',
                  content_path=base+'/assets/pse_v1/content_features.npy', device='cuda')
    if not (target/'train/complete.json').exists():
        train(**common, output=target/'train', resume=True)
    if not (target/'test/complete.json').exists():
        runtime.test(**common, training=target/'train', output=target/'test')


if __name__ == '__main__':
    main()
