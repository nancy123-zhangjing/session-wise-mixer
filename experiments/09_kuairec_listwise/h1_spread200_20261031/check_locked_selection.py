"""Audit the locked common-grid selection using accepted validation receipts."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
destination = here / 'saved_validation_R35'
selection = read(destination / 'selection.json')
complete = read(destination / 'training_complete.json')
assert sha(destination / 'selection.json') == complete['selection_sha256']
assert complete['status'] == 'training_and_validation_complete'
assert (complete['r_updates'], complete['t_updates']) == (24255, 200)
assert complete['last_sha256'] == read(here / 'R35_training_endpoint_r1/BACKUP_ACCEPTED.json')['checkpoint_sha256']
grid = (20, 23, 26, 29, 32, 35)
records, evidence = {}, {}
for epoch in (8, 11, 14, 17, *grid):
    folder = here / ('first_validation_R08' if epoch == 8 else f'saved_validation_R{epoch:02d}')
    record, summary, accepted = [read(folder / name) for name in ('record.json', 'summary.json', 'ACCEPTED.json')]
    assert record['epoch'] == accepted['epoch'] == epoch
    assert accepted['status'] == 'passed_one_completed_validation_only'
    assert accepted['summary_sha256'] == record['summary_sha256'] == sha(folder / 'summary.json')
    assert record['predictions_sha256'] == accepted['predictions_sha256']
    assert record['ndcg5'] == summary['metrics']['list/ndcg@5']
    audit = read(folder / 'audit.json')
    assert sha(folder / 'audit.json') == accepted['audit_sha256']
    assert audit['passed'] and audit['mismatches'] == 0
    assert audit['examples'] == accepted['examples'] == 71657
    records[epoch] = record
    evidence[str(epoch)] = dict(accepted_sha256=sha(folder / 'ACCEPTED.json'), production_ndcg5=record['ndcg5'])
best = min(grid, key=lambda epoch: (-records[epoch]['ndcg5'], epoch))
assert selection['criterion'] == 'validation/list/ndcg@5; tie earlier'
assert selection['candidates'] == [records[epoch] for epoch in grid]
assert selection['selected'] == records[best]
result = dict(status='passed_locked_validation_selection', selected_epoch=best,
    selected_t_updates=(best - 5) // 3 * 20, grid=list(grid), tie_rule='earlier epoch',
    score_rule='authenticated production FP32 summary; independent FP64 metrics do not reselect',
    selection_sha256=sha(destination / 'selection.json'),
    training_complete_sha256=sha(destination / 'training_complete.json'),
    validation_evidence=evidence, allowed_test_epochs=sorted({35, best}),
    auditor_sha256=sha(Path(__file__)),
    limitation='Validation selection and endpoint receipts only. Tests, full fresh extraction and final acceptance remain separate.')
with (destination / 'SELECTION_ACCEPTED.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result))
