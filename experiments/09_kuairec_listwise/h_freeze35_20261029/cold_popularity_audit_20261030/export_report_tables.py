"""Export report tables from the audited initial CPU result; no inference."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'initial_r1'
OUT = HERE / 'tables'
OUT.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

verify = read(HERE / 'VERIFY.json')
assert verify['status'] == 'passed'
assert hashlib.sha256((SOURCE / 'COMPLETE.json').read_bytes()).hexdigest() == verify['analysis_complete_sha256']
groups = {
    'three_classes': ['never_seen', 'history_only', 'target_seen'],
    'five_classes': ['never_seen', 'history_only', 'target_head', 'target_mid', 'target_tail'],
}
composition = read(SOURCE / 'DATA_COMPOSITION.json')
rows = []
for segment, data in composition.items():
    for level, names in groups.items():
        for name in names:
            rows.append(dict(segment=segment, grouping=level, target_class=name,
                             samples=data['samples'], **data['classes'][name]))
write('target_composition.csv', rows)
target_rows, bias_rows, top_rows = [], [], []
for model in read(SOURCE / 'RESULTS.json')['models']:
    for segment, data in read(SOURCE / f'MODEL_{model}.json')['segments'].items():
        for level, names in groups.items():
            for name in names:
                cls = data['classes'][name]
                for k, metrics in cls['metrics'].items():
                    target_rows.append(dict(model=model, segment=segment, grouping=level,
                                            target_class=name, k=int(k),
                                            target_occurrences=cls['target_occurrences'], **metrics))
        for ranking, bias in data['predictions'].items():
            row = dict(model=model, segment=segment, ranking=ranking)
            row.update({k: v for k, v in bias.items() if k not in ['class_shares', 'top10_items']})
            row.update({k + '_share': v for k, v in bias['class_shares'].items()})
            row['mean_real_items_per_sample'] = bias['real_item_slots'] / composition[segment]['samples']
            bias_rows.append(row)
            for rank, item in enumerate(bias['top10_items'], 1):
                top_rows.append(dict(model=model, segment=segment, ranking=ranking, rank=rank, **item))
write('target_metrics.csv', target_rows)
write('prediction_bias.csv', bias_rows)
write('recommended_top10.csv', top_rows)
manifest = {
    'source_complete_sha256': verify['analysis_complete_sha256'],
    'verify_sha256': hashlib.sha256((HERE / 'VERIFY.json').read_bytes()).hexdigest(),
    'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))},
    'note': 'Three-class and five-class rows are alternative partitions; do not add both partitions together.'
}
(OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'passed', 'csv_files': len(manifest['files']), 'target_metric_rows': len(target_rows)}))
