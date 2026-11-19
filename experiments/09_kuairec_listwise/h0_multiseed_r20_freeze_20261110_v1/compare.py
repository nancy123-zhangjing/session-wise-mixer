"""Fetch small result records and compare completed nodes with the existing arms."""
import csv
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
m = runpy.run_path(str(HERE / 'control.py'))
data = json.loads(m['c'].remote(m['c'].PYTHON + ' -B ' + m['REMOTE'] + '/collect_results.py', 'RESULTS_READBACK.json', 45))
(HERE / 'RESULTS_LATEST.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
prior = json.loads(json.loads((HERE.parent / 'multiseed_freeze_20261109_v1/RESULTS_LATEST.json').read_text(encoding='utf-8'))['data']['output'])
rows = []
reference = json.loads((HERE / 'REFERENCE_SEED2024.json').read_text(encoding='utf-8'))['validation']
for epoch_text, score in reference['H0_freeze'].items():
    epoch = int(epoch_text)
    if epoch < 20: continue
    h0, h1 = reference['H0_continued'][epoch_text], reference['H1_freeze'][epoch_text]
    rows.append(dict(seed=2024, epoch=epoch, H0_continued=h0, H0_freeze=score, H1_freeze=h1,
                     H0_freeze_vs_continued_pct=(score / h0 - 1) * 100,
                     H1_freeze_vs_H0_freeze_pct=(h1 / score - 1) * 100))
for result in data:
    seed = result['seed']
    baseline = {r['arm']: {v['epoch']: v['ndcg'] for v in r['validations']} for r in prior if r['seed'] == seed}
    for v in result['validations']:
        if v['epoch'] < 20: continue
        epoch, score = v['epoch'], v['ndcg5']
        rows.append(dict(seed=seed, epoch=epoch, H0_continued=baseline['H0'][epoch], H0_freeze=score,
                         H1_freeze=baseline['H1'][epoch],
                         H0_freeze_vs_continued_pct=(score / baseline['H0'][epoch] - 1) * 100,
                         H1_freeze_vs_H0_freeze_pct=(baseline['H1'][epoch] / score - 1) * 100))
with (HERE / 'VALIDATION_COMPARISON.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(dict(comparison=rows, progress=[dict(seed=r['seed'], event=r['last_event'], complete=r['complete']) for r in data]), ensure_ascii=False))
