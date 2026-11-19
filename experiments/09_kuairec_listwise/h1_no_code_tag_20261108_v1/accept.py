"""Audit the completed run from saved artifacts on the shared CPU."""
from pathlib import Path
import sys

from experiment import (ROOT, SOURCE, OUTPUT, ORIGINAL, BOUNDARIES, PARENT_SHA,
                        FLAG, read, sha256, write_json)

sys.path.insert(0, str(SOURCE))
from audit_analysis_core import audit_predictions


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit_training():
    import json
    r_updates, t_updates = 3465, 0
    count = 0
    baseline_path = ORIGINAL / 'runs/H1/training.jsonl'
    fields = ('phase_index', 'epoch', 'update', 'phase', 'r_updates', 't_updates', 'lists')
    with (OUTPUT / 'training.jsonl').open(encoding='utf-8') as actual, baseline_path.open(encoding='utf-8') as baseline:
        for line in actual:
            row = json.loads(line)
            reference = json.loads(next(baseline))
            check(all(row[k] == reference[k] for k in fields), f'Original update schedule diverged at row {count + 1}')
            r_updates += row['phase'] == 'R'
            t_updates += row['phase'] == 'T'
            check((row['r_updates'], row['t_updates']) == (r_updates, t_updates), 'Update clock is discontinuous')
            count += 1
    check((r_updates, t_updates, count) == (13860, 200, 10595), 'Unexpected final update budget')
    return dict(R_added=10395, T_added=200, R_total=r_updates, T_total=t_updates, schedule_rows=count)


def audit_evaluation(directory, reference, expected, dataset):
    summary = read(directory / 'summary.json')
    baseline = read(reference / 'summary.json')
    for key in ('split', 'examples', 'full_split', 'dataset_binding', 'beam',
                'inference_precision', 'architecture', 'seed', 'metric_protocol'):
        check(summary[key] == baseline[key], f'{directory}: evaluation protocol mismatch in {key}')
    check((summary['r_updates'], summary['t_updates']) == expected, 'Evaluation checkpoint clock mismatch')
    check(summary['full_split'] and summary['beam'] == 32, 'Expected full-split beam32')
    audit = audit_predictions(directory / 'per_sample.jsonl.gz', dataset_dir=dataset)
    check(audit['passed'] and audit['source_labels_verified'], f'{directory}: metric audit failed')
    metrics = ('list/ndcg@5', 'list/hr@5', 'list/recall@5')
    return dict(ours={k: summary['metrics'][k] for k in metrics},
                original={k: baseline['metrics'][k] for k in metrics},
                examples=audit['examples'], max_absolute_error=audit['max_absolute_error'],
                predictions_sha256=audit['predictions_sha256'], summary_sha256=audit['summary_sha256'])


def main():
    gpu = read(ROOT / 'GPU_COMPLETE.json')
    check((gpu['status'], gpu['R'], gpu['T']) == ('passed', 20, 200), 'GPU gate not complete')
    migration = read(ROOT / 'MIGRATION.json')
    check(migration['restore_exact'] and migration['same_effective_codebook_and_sid']
          and migration['original_sha256'] == PARENT_SHA, 'Parent migration gate failed')
    config = migration['new_binding']['config']
    check(config['alignment'][FLAG] is False, 'Wrong ablation flag')
    dataset = config['dataset']
    updates = audit_training()
    selection = read(ROOT / 'SELECTION.json')
    rows = selection['candidates']
    check([(r['epoch'], r['t_updates']) for r in rows] == [(5, 0), *BOUNDARIES.values()], 'Wrong validation grid')
    best = max(rows, key=lambda r: (r['ndcg5'], -r['epoch']))
    check(best == selection['best'], 'Best selection mismatch')
    evaluations = {}
    for row in rows:
        epoch, t = row['epoch'], row['t_updates']
        path = Path(row['validation'])
        check(row['restore_exact'] and sha256(row['checkpoint']) == row['checkpoint_sha256'], 'Checkpoint gate failed')
        check(sha256(path / 'summary.json') == row['summary_sha256'], 'Validation summary changed')
        check(read(path / 'summary.json')['metrics']['list/ndcg@5'] == row['ndcg5'], 'Incorrect selection score')
        if epoch == 5:
            evaluations[epoch] = dict(ours={'list/ndcg@5': row['ndcg5']}, original={'list/ndcg@5': row['ndcg5']},
                                      inherited=True)
            continue
        evaluations[epoch] = audit_evaluation(path, ORIGINAL / 'runs/H1' / f'validation_R{epoch:02d}',
                                              (epoch * 693, t), dataset)
        print(f'R{epoch} full validation audit passed', flush=True)
    test = audit_evaluation(OUTPUT / 'test_best', ORIGINAL / 'evaluations/H1_test_beam32_r2',
                            (best['epoch'] * 693, best['t_updates']), dataset)
    test_receipt = read(ROOT / 'TEST_RECEIPT.json')
    check(test_receipt['checkpoint_sha256'] == best['checkpoint_sha256']
          and test_receipt['selection_sha256'] == sha256(ROOT / 'SELECTION.json')
          and all(test[k] == test_receipt[k] for k in ('summary_sha256', 'predictions_sha256')),
          'Test artifacts do not match the locked selection')
    files = [ROOT / name for name in ('MIGRATION.json', 'SOURCE_DELTA.json', 'SELECTION.json',
                                     'TEST_RECEIPT.json', 'GPU_COMPLETE.json', 'training.log',
                                     'experiment.py', 'replay_support.py', 'parent_R05_T000.pt')]
    files += [path for path in OUTPUT.rglob('*') if path.is_file()]
    manifest = {path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha256(path))
                for path in sorted(files)}
    write_json(ROOT / 'ARTIFACT_MANIFEST.json', manifest)
    write_json(ROOT / 'ACCEPTED_RESULT.json', dict(status='passed', best_epoch=best['epoch'], updates=updates,
        validation=evaluations, best_test=test, artifact_count=len(manifest),
        manifest_sha256=sha256(ROOT / 'ARTIFACT_MANIFEST.json')))
    print('Independent acceptance passed', flush=True)


if __name__ == '__main__':
    main()
