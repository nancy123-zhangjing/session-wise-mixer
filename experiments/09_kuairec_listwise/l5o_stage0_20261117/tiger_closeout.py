"""Read back and verify the finished L5O TIGER arms (T0A, T1A) on Inspire storage.

Runs on a CPU notebook next to the shared disk. Read-only for the run directories:
it hashes checkpoints/predictions, recomputes test metrics from the stored
predictions, checks test-to-validation identities and computes the same paired
user bootstrap used for H and CF (scripts/kuairec_data/l5o_paired.py, 2000 draws,
seed 2024). Writes one JSON document to --output.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.kuairec_data.l5o_metrics import load_predictions, load_split, slate_metrics  # noqa: E402
from scripts.kuairec_data.l5o_paired import paired_delta  # noqa: E402

ARMS = ('T0A', 'T1A')
METRICS = ('imp_hit@5', 'imp_recall@5', 'imp_ndcg@5', 'ev_hit@5', 'ev_recall@5', 'ev_ndcg@5')
TEST_SHA = 'ce63b6b3954cc97cdf592056b795133acd2e5eefd7eb8639cad00c6c5a7b3b29'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    return abs(a - b) <= 1e-9


def arm_readback(run, ids, target, ratios):
    train = json.loads((run/'train/complete.json').read_text())
    test = json.loads((run/'test/complete.json').read_text())
    progress = train['progress']
    records = progress['records']
    curve = [dict(step=r['step'], val_ev_ndcg5=r['validation']['metrics']['ev_ndcg@5'],
                  val_imp_hit5=r['validation']['metrics']['imp_hit@5'],
                  val_ev2_ndcg5=r['validation']['sensitivity_ev2']['ev_ndcg@5'],
                  train_seconds=r.get('train_seconds'), loss=r.get('loss'),
                  checkpoint_sha256=r['checkpoint_sha256'], encoding=r.get('encoding'))
             for r in records]
    steps = [r['step'] for r in records]
    nodes, preds = {}, {}
    for node in ('best', 'final'):
        sel = train[node]
        step = sel['step']
        rec = records[steps.index(step)]
        ckpt = run/f'train/checkpoints/step_{step:07d}.pt'
        summary_path = run/f'test/step_{step:07d}/summary.json'
        pred_path = run/f'test/step_{step:07d}/predictions.jsonl.gz'
        summary = json.loads(summary_path.read_text())
        pred = load_predictions(pred_path, ids)
        recomputed = {}
        for threshold, key in ((1.0, 'metrics'), (2.0, 'sensitivity_ev2')):
            m, _ = slate_metrics(pred, target, ratios, threshold, return_per_sample=True)
            recomputed[key] = all(close(float(m[k]), summary[key][k]) for k in METRICS)
        ckpt_sha = sha(ckpt)
        nodes[node] = dict(
            step=step, checkpoint=str(ckpt.relative_to(run.parent.parent)), checkpoint_bytes=ckpt.stat().st_size,
            checkpoint_sha256=ckpt_sha, validation=sel['validation']['metrics'],
            test=summary['metrics'], test_ev2=summary['sensitivity_ev2'],
            prediction=str(pred_path.relative_to(run.parent.parent)), prediction_bytes=pred_path.stat().st_size,
            prediction_sha256=summary['prediction_sha256'],
            checks=dict(
                checkpoint_file_matches_progress_record=ckpt_sha == rec['checkpoint_sha256'],
                test_identity_matches_validation_identity=summary['identity'] == sel['validation']['identity'],
                test_complete_matches_step_summary=test[node] == summary,
                prediction_file_sha_matches_summary=sha(pred_path) == summary['prediction_sha256'],
                metrics_recomputed_from_predictions=recomputed['metrics'],
                ev2_recomputed_from_predictions=recomputed['sensitivity_ev2'],
                full_test=summary['count'] == len(ids) == 7176,
                sid_legal_rate=summary['sid_legal_rate']))
        preds[node] = pred
    last_sha = sha(run/'train/last.pt')
    selected = run/'pretrain/selected.pt'
    selected_sha = sha(selected)
    budget = dict(done=progress.get('done'), step=progress['step'], evaluated=progress['evaluated'],
                  best_step=progress['best_step'], best_score=progress['best_score'],
                  bad_epochs=progress['bad_epochs'], train_seconds=progress.get('train_seconds'),
                  validation_steps=steps,
                  validation_interval_uniform=len({b - a for a, b in zip(steps, steps[1:-1])}) == 1)
    return dict(
        budget=budget, curve=curve, nodes=nodes,
        last_pt=dict(sha256=last_sha, bytes=(run/'train/last.pt').stat().st_size,
                     equals_final_checkpoint=last_sha == nodes['final']['checkpoint_sha256']),
        parent=dict(recorded_sha256=train['parent_sha256'], pretrain_selected_sha256=selected_sha,
                    matches=selected_sha == train['parent_sha256']),
        config_sha256=train['config_sha256'], prepared_sha256=train['prepared_sha256'],
        environment=train['environment'], parameter_count=train['parameter_count']), preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    if sha(args.dataset/'test.parquet') != TEST_SHA:
        raise ValueError('Test split does not match the L5O binding')
    ids, target, ratios, _ = load_split(args.dataset, 'test')
    arms, preds = {}, {}
    for arm in ARMS:
        arms[arm], preds[arm] = arm_readback(args.wave/'runs'/arm, ids, target, ratios)
    paired = {}
    for node in ('best', 'final'):
        thresholds = {}
        for threshold in (1.0, 2.0):
            _, a = slate_metrics(preds['T0A'][node], target, ratios, threshold, return_per_sample=True)
            _, b = slate_metrics(preds['T1A'][node], target, ratios, threshold, return_per_sample=True)
            thresholds[str(threshold)] = {m: paired_delta(a[m], b[m], np.random.default_rng(2024), args.draws)
                                          for m in METRICS}
        paired[node] = dict(direction='variant-minus-base', base='T0A', variant='T1A', draws=args.draws, seed=2024,
                            interpretation='Paired user sampling for a single training seed',
                            thresholds=thresholds)
    result = dict(wave_root=str(args.wave), test_parquet_sha256=TEST_SHA, arms=arms, paired=paired,
                  scope='Read-only closeout of finished TIGER runs; no training, inference or resume test.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
