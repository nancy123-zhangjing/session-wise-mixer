"""Read-only finished-run checkpoint audit; no training, inference or CUDA resume."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def audit_stop_policy(cfg, progress, policy=None):
    """Check actual stopping and best selection separately from scheduler budget."""
    c = cfg['training']
    per_epoch = math.ceil(cfg['split_counts']['train'] * cfg['list_length'] / c['batch_size'])
    original_limit = c['max_steps'] or c['max_epochs'] * per_epoch
    limit = original_limit
    if policy is not None:
        require(cfg['family'] == 'diger', 'cap only authorized for DIGER')
        require(policy['max_execution_epochs'] == 35, 'execution cap')
        require(policy['scheduler_epochs'] == c['max_epochs'] == 120, 'scheduler horizon changed')
        require(policy['patience'] == c['patience'] == 15, 'patience changed')
        limit = min(limit, 35 * per_epoch)
    records = progress['records']
    best, best_step, bad = None, None, 0
    for i, row in enumerate(records):
        score = row['validation']['metrics']['ndcg@5']
        require(math.isfinite(score), 'nonfinite validation')
        if best is None or score > best:
            best, best_step, bad = score, row['step'], 0
        else:
            bad += 1
        stopped = row['step'] >= limit or (c['patience'] is not None and bad >= c['patience'])
        require(not stopped or i == len(records)-1, 'continued after stop condition')
    require(bool(records) and progress['done'], 'not complete')
    require(progress['step'] == records[-1]['step'] <= limit, 'execution budget exceeded')
    require(progress['best_step'] == best_step and progress['best_score'] == best, 'best selection')
    require(progress['bad_epochs'] == bad, 'patience counter')
    cap = progress['step'] == limit
    early = c['patience'] is not None and bad >= c['patience']
    require(cap or early, 'no valid stop reason')
    return {'per_epoch_updates': per_epoch, 'execution_limit': limit,
            'scheduler_epochs': c['max_epochs'], 'budget_reached': cap,
            'patience_reached': early, 'best_step': best_step}


def audit(root):
    done = json.loads((root/'complete.json').read_text())
    records = done['progress']['records']
    require(done['progress']['done'], 'training not complete')
    rows = []
    for record in records:
        path = root/'checkpoints'/('step_%07d.pt' % record['step'])
        actual = sha(path)
        require(actual == record['checkpoint_sha256'], 'checkpoint SHA')
        v = record['validation']
        require(v['count'] == 71657 and v['split'] == 'validation', 'validation scope')
        require(v['identity']['checkpoint_sha256'] == actual, 'validation identity')
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        cfg = s['config']
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        require(digest == done['config_sha256'] == v['identity']['config_sha256'], 'configuration')
        require(s['sources'] == done['sources'], 'source identity')
        require(s['parent_sha256'] == done['parent_sha256'], 'parent identity')
        require(s['environment'] == done['environment'], 'environment identity')
        require(s['progress']['step'] == record['step'], 'checkpoint clock')
        for key in ('tokenizer', 'recommender', 'optimizers', 'schedulers', 'stream', 'rng'):
            require(bool(s[key]), 'missing '+key)
        require(len(s['optimizers']) == len(s['schedulers']) == (1 if cfg['family']=='tiger' else 2), 'optimizer count')
        for opt in s['optimizers']:
            require(bool(opt['state']) and bool(opt['param_groups']), 'optimizer state missing')
            require(all(int(x['step']) == record['step'] for x in opt['state'].values()), 'optimizer clock')
        stream = s['stream']
        batch = cfg['training']['batch_size']
        require(stream['size'] == cfg['split_counts']['train'] * cfg['list_length'], 'stream size')
        require(stream['epoch'] * math.ceil(stream['size']/batch) + math.ceil(stream['cursor']/batch) == record['step'], 'stream clock')
        require(torch.equal(stream['order'].sort().values, torch.arange(stream['size'])), 'stream permutation')
        torch.Generator().set_state(stream['rng'].cpu())
        require(len(s['rng']['cuda']) == 1, 'CUDA RNG missing')
        torch.Generator().set_state(s['rng']['cpu'].cpu())
        require(tuple(s['sid'].shape) == (cfg['catalog_size']+1, cfg['rq']['levels']+1), 'SID shape')
        require(bool((s['sid'][0] == -1).all()), 'PAD SID')
        require(bool(((s['sid'][1:] >= 0) & (s['sid'][1:] < cfg['rq']['codes'])).all()), 'SID range')
        require(len(torch.unique(s['sid'][1:], dim=0)) == cfg['catalog_size'], 'full SID uniqueness')
        rows.append({'step': record['step'], 'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual})
        del s
    require(rows[-1]['step'] == done['final']['step'] == done['progress']['step'], 'final coverage')
    require(any(r['sha256'] == done['best']['checkpoint_sha256'] for r in rows), 'best coverage')
    policy_path = root/'EXECUTION_CAP.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
    stop = audit_stop_policy(cfg, done['progress'], policy)
    return {'status': 'passed_cpu_structure_and_hash_only', 'arm': done['arm'], 'nodes': rows,
            'stop_policy': stop,
            'boundary': 'No CUDA numerical resume, model forward, or optimizer update performed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
