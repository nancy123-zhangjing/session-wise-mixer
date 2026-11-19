"""Check audit aggregation and matching evaluation protocols; CPU only."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESTORE = ROOT / 'outputs/stage09_restore/goal_20261027_r1'
result = json.loads((HERE / 'RESULTS.json').read_text(encoding='utf-8-sig'))
assert hashlib.sha256((HERE / 'analyze.py').read_bytes()).hexdigest() == result['script_sha256']

daily = list(csv.DictReader((HERE / 'daily_best.csv').open(encoding='utf-8-sig')))
for split in ('validation', 'test'):
    rows = [x for x in daily if x['split'] == split]
    n = sum(int(x['examples']) for x in rows)
    assert n == result['splits'][split]['examples']
    for arm in ('H0', 'H1'):
        weighted = sum(int(x['examples']) * float(x[arm]) for x in rows) / n
        assert abs(weighted - result['paired'][split][arm]) < 1e-12

temporal = result['temporal_validation']
early_n, late_n = temporal['early_examples'], temporal['late_examples']
curves = list(csv.DictReader((HERE / 'validation_temporal_curves.csv').open(encoding='utf-8-sig')))
for row in curves:
    full = (float(row['early']) * early_n + float(row['late']) * late_n) / (early_n + late_n)
    assert abs(full - float(row['full'])) < 1e-12

paths = [RESTORE / 'runs/H0/validation_R32/summary.json',
         RESTORE / 'runs/H1/validation_R20/summary.json',
         RESTORE / 'sidecars/evaluations/evaluations/H0_test_beam32/summary.json',
         RESTORE / 'sidecars/evaluations/evaluations/H1_test_beam32_r2/summary.json']
summaries = [json.loads(p.read_text(encoding='utf-8-sig')) for p in paths]
keys = ['dataset_binding', 'metric_protocol', 'beam', 'inference_precision', 'architecture', 'seed', 'catalog_size']
for key in keys:
    assert all(x[key] == summaries[0][key] for x in summaries), key
assert all(x['full_split'] for x in summaries)
assert summaries[0]['beam'] == 32 and summaries[0]['inference_precision'] == 'fp32'

def curve(arm, epoch):
    return next(x for x in curves if x['arm'] == arm and int(x['epoch']) == epoch)

h0, h1 = curve('H0', 32), curve('H1', 20)
out = dict(status='PASS', checks=['analysis source SHA matches saved result',
    'daily weighted means match overall NDCG', 'all 20 temporal partition means match full validation',
    'four best evaluations share dataset, metric, beam, precision, architecture and seed'],
    best_pair_early_relative=float(h1['early']) / float(h0['early']) - 1,
    best_pair_late_relative=float(h1['late']) / float(h0['late']) - 1,
    same_R20_late_relative=float(h1['late']) / float(curve('H0', 20)['late']) - 1,
    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
with (HERE / 'VERIFY.json').open('x', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({k: v for k, v in out.items() if k != 'sources'}, ensure_ascii=False))
