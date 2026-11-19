"""Export report tables only from the independently verified C supplement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ('H1_freeze_R32', 'C_freeze_R32', 'C_best_R23')
PAIRS = tuple('H1_freeze_R32_minus_' + model for model in MODELS[1:])


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csvsave(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=HERE / 'c_supplement_r1')
    parser.add_argument('--verification', type=Path, default=HERE / 'VERIFY_C_SUPPLEMENT.json')
    parser.add_argument('--output', type=Path, default=HERE / 'c_supplement_tables_r1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New tables directory required')
    verification = read(args.verification)
    if verification['status'] != 'passed':
        raise ValueError('Independent C supplement verification must pass first')
    result = args.result_dir.resolve()
    if (Path(verification['result_dir']).resolve() != result
            or verification['analysis_complete_sha256'] != sha(result / 'COMPLETE.json')):
        raise ValueError('Verification is not bound to this exact statistics directory')
    complete = read(result / 'COMPLETE.json')
    if complete['status'] != 'passed':
        raise ValueError('Supplement statistics are incomplete')
    for name, digest in complete['files'].items():
        if sha(result / name) != digest:
            raise ValueError('Verified statistic changed: ' + name)
    request = read(result / 'REQUEST.json')
    if not all(name in request['models'] for name in MODELS):
        raise ValueError('Missing authorized C model')
    metrics, targets, recommendations, tops = [], [], [], []
    for model in MODELS:
        source = read(result / ('MODEL_' + model + '.json'))
        for segment in ('early', 'late', 'full'):
            row = source['segments'][segment]
            metrics.append(dict(model=model, segment=segment, **row['metrics']))
            for group, summary in row['classes'].items():
                for k, values in summary['metrics'].items():
                    targets.append(dict(model=model, segment=segment, group=group,
                                        k=int(k), target_occurrences=summary['target_occurrences'], **values))
            for protocol, values in row['predictions'].items():
                recommendations.append(dict(model=model, segment=segment, protocol=protocol,
                    **{key: value for key, value in values.items() if key not in ('top10_items', 'class_shares')},
                    **{'share_' + key: value for key, value in values['class_shares'].items()}))
                for rank, item in enumerate(values['top10_items'], 1):
                    tops.append(dict(model=model, segment=segment, protocol=protocol, rank=rank, **item))
    bootstrap = []
    for segment, values in read(result / 'PAIRED_BOOTSTRAP.json').items():
        for pair in PAIRS:
            for feature, estimate in values[pair].items():
                bootstrap.append(dict(segment=segment, pair=pair, feature=feature, **estimate))
    decomposition, details = [], []
    source = read(result / 'DECOMPOSITION.json')
    for pair in PAIRS:
        for k, values in source[pair].items():
            decomposition.append(dict(pair=pair, k=int(k), **{key: value for key, value in values.items() if key != 'classes'}))
            for item in values['classes']:
                details.append(dict(pair=pair, k=int(k), **item))
    args.output.mkdir(parents=True, exist_ok=False)
    tables = dict(METRICS=metrics, TARGET_CLASSES=targets, RECOMMENDATIONS=recommendations,
                  TOP10_ITEMS=tops, PAIRED_BOOTSTRAP=bootstrap,
                  DECOMPOSITION=decomposition, DECOMPOSITION_CLASSES=details)
    for name, rows in tables.items():
        csvsave(args.output / (name + '.csv'), rows)
    receipt = dict(status='passed', result_directory=str(result),
                   complete_sha256=sha(result / 'COMPLETE.json'),
                   verification=str(args.verification.resolve()), verification_sha256=sha(args.verification),
                   models=MODELS, pairs=PAIRS, table_rows={key: len(value) for key, value in tables.items()},
                   files={p.name: sha(p) for p in sorted(args.output.iterdir())}, script_sha256=sha(__file__))
    save(args.output / 'EXPORT_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
