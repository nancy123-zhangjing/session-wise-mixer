"""Final CPU-only audit; no training or model inference."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
sys.path[:0] = [str(STAGE / 'h_freeze35_20261029_r1/code'), str(STAGE / 'h1_no_code_tag_20261108_v1')]


def main():
    import torch
    from audit_analysis_core import audit_predictions
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.training import sha256
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    rows, checks = [], []
    for seed in (2025, 2026):
        root = ROOT / f'seed{seed}_H0'
        train = json.loads((root / 'TRAINING_COMPLETE.json').read_text())
        complete = json.loads((root / 'GPU_COMPLETE.json').read_text())
        parent = json.loads((root / 'PARENT_RESTORE.json').read_text())
        assert parent['passed'] and all(parent['matching'].values())
        assert (train['R'], train['T']) == (35, 200) and complete['status'] == 'passed'
        assert sha256(parent['parent']['checkpoint']) == manifest['parents'][str(seed)]['sha256']
        last = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
        assert (last['r_updates'], last['t_updates']) == (24255, 200)
        frozen = json.loads((root / 'FREEZE_R20.json').read_text())['hashes']
        assert {k: fingerprint(last[k]) for k in ('alignment', 't_optimizer')} == frozen
        del last
        log = [json.loads(s) for s in (root / 'training.jsonl').read_text().splitlines()]
        assert len(log) == 10395 and all(r['phase'] == 'R' and r['t_updates'] == 200 for r in log)
        assert [r['r_updates'] for r in log] == list(range(13861, 24256))
        scores = {v['epoch']: v['ndcg5'] for v in json.loads((root / 'PARENT_VALIDATIONS.json').read_text())}
        for epoch in (23, 26, 29, 32, 35):
            v = json.loads((root / 'checkpoints' / f'R{epoch:02d}.json').read_text())
            assert v['T'] == 200 and sha256(v['checkpoint']) == v['sha256']
            summary = json.loads((Path(v['validation']) / 'summary.json').read_text())
            assert summary['examples'] == 71657 and summary['beam'] == 32 and summary['full_split']
            scores[epoch] = summary['metrics']['list/ndcg@5']
        assert max(scores, key=scores.get) == train['best_epoch']
        for record in complete['tests']:
            if record.get('reused'):
                src = next(r for r in rows if r['seed'] == seed and r['selection'] == record['reused'])
                rows.append(dict(src, selection=record['label']))
                continue
            audit = audit_predictions(record['predictions'], dataset_dir=STAGE / 'goal_20261027_r1/inputs/dataset')
            assert audit['passed'] and audit['source_labels_verified'] and audit['examples'] == 53194
            assert audit['predictions_sha256'] == record['predictions_sha256']
            assert audit['summary_sha256'] == record['summary_sha256']
            assert sha256(record['checkpoint']) == record['checkpoint_sha256']
            rows.append(dict(seed=seed, selection=record['label'],
                             epoch=train['best_epoch'] if record['label'] == 'best' else 35,
                             metrics={k: record['metrics'][k] for k in ('list/ndcg@5', 'list/hr@5', 'list/recall@5')}))
            checks.append(dict(seed=seed, selection=record['label'], max_absolute_error=audit['max_absolute_error']))
    comparisons = []
    for row in rows:
        seed, selection = row['seed'], row['selection']
        comparators = {}
        for arm in ('H0', 'H1'):
            tests = json.loads((STAGE / 'multiseed_freeze_20261109_v1' / f'seed{seed}_{arm}' / 'TEST_RESULTS.json').read_text())
            target = next(t for t in tests if t['label'] == selection)
            if target.get('reused'): target = next(t for t in tests if t['label'] == target['reused'])
            comparators[arm] = {k: target['metrics'][k] for k in row['metrics']}
        comparisons.append(dict(seed=seed, selection=selection, H0Freeze=row['metrics'],
                                H0Continued=comparators['H0'], H1Freeze=comparators['H1']))
    result = dict(status='passed', rows=rows, audits=checks, comparisons=comparisons)
    (ROOT / 'FINAL_ACCEPTANCE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
