"""Offline item-metric replay and the three prespecified H1 recovery comparisons.

This adapter imports only the independent CPU analysis core, never training,
decoding, production metric code, or torch. Inputs are read-only; outputs must
be new. It accepts recovery summaries without groups and the unsalted SHA subset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from audit_analysis_core import (
    _metadata, _rows, _training_seen_items, _truth, paired_comparison, recompute_row,
)
from scripts.listwise_diffgrm.subset import canonical_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA256 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
R20_RELATIVE = 'outputs/stage09_restore/goal_20261027_r1/runs/H1/validation_R20/per_sample.jsonl.gz'
PRIOR_ANALYSIS = ROOT / 'experiments/09_kuairec_listwise/final_analysis_20261027'
R20_HASHES = {
    'predictions_sha256': 'd43da08bbd4288d0fe75209b1b5c59a0369894477de1538e528d9031ea337ea6',
    'summary_sha256': 'd457c373647fa0d80369e986858deb7a62c27bcd0d5b94224874e3929769be50',
    'audit_sha256': '8cbe2e41b55eecb315d158dd0f6888c26a801e9dda64fb3b9c82f018fda10d98',
}
PRIMARY_METRIC = 'list/ndcg@5'
FULL_ROWS = 71657
SUBSET_ROWS = 8192
INTERPRETATION = (
    'Fixed R26 endpoint, validation only. R23 and the 8192-row subset are process diagnostics, '
    'not model selection. User-cluster intervals describe validation-user sampling for these '
    'fixed models; they do not cover training-seed variation, selection of R20, or the '
    'post-hoc proposal of the recovery policy. Failure to detect a decline is not equivalence.'
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_subset(manifest, truth, validation_sha256):
    """Independently reconstruct the complete, unsalted, locked selection."""
    size = manifest.get('count')
    if type(size) is not int or not 0 < size <= len(truth):
        raise ValueError('Invalid diagnostic subset count')
    ranked = sorted((hashlib.sha256(key.encode('utf-8')).hexdigest(), key, value[0])
                    for key, value in truth.items())[:size]
    expected = dict(format='stage09_h1_recovery_subset_v1', count=size, population=len(truth),
                    validation_sha256=validation_sha256, selection='sha256_utf8_sample_id_then_sample_id',
                    rows=[dict(index=index, sample_id=key, selection_sha256=digest)
                          for digest, key, index in ranked])
    if manifest != expected:
        raise ValueError('Diagnostic manifest differs from unsalted source SHA selection')
    return expected


def audit_recovery_predictions(predictions_path, *, dataset_dir, subset_manifest=None,
                               tolerance=2e-6, recomputed_path=None, require_extracted_subset=False):
    """Replay every prediction; source-bind labels, user, index, and cold group.

    Numerical mismatches return passed=False; malformed artifacts raise ValueError.
    Optional per-sample output contains independent metrics, not saved metric values.
    A manifest also permits independent extraction from a full validation artifact.
    """
    path = Path(predictions_path)
    meta, arguments = _metadata(path)
    if meta['split'] != 'validation' or meta.get('diagnostic_prefix_limit') is not None:
        raise ValueError('Recovery analysis accepts validation and no prefix diagnostic')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    truth = _truth(dataset_dir, meta)
    seen_items, seen_sha = _training_seen_items(dataset_dir, meta)
    full = meta.get('full_split')
    if type(full) is not bool or (meta.get('subset') is None) != full:
        raise ValueError('Full-split and subset declarations disagree')
    selected_rows, wanted = None, set()
    if subset_manifest is not None:
        validate_subset(subset_manifest, truth, meta['dataset_binding']['files']['validation.parquet']['sha256'])
        selected_rows = subset_manifest['rows']
        wanted = {row['sample_id'] for row in selected_rows}
    if not full:
        if selected_rows is None or meta['subset'] != {k: v for k, v in subset_manifest.items() if k != 'rows'}:
            raise ValueError('Subset evaluation requires its exact unsalted manifest')
    if require_extracted_subset and (not full or selected_rows is None):
        raise ValueError('Full extraction audit requires the locked subset manifest')

    mismatches, max_error = 0, 0.0

    def compare(saved, actual):
        nonlocal mismatches, max_error
        if not isinstance(saved, dict) or set(saved) != set(actual):
            raise ValueError('Saved metric names differ from independent metric protocol')
        for name, value in actual.items():
            item = saved[name]
            error = abs(value - item) if type(item) in (int, float) and math.isfinite(item) else math.inf
            mismatches += error > tolerance
            max_error = max(max_error, error)

    ids, sums, users, groups, subset_sums = set(), defaultdict(float), {}, {}, defaultdict(float)
    selected_seen, count = set(), 0
    output_context = (gzip.open(recomputed_path, 'xt', encoding='utf-8')
                      if recomputed_path is not None else nullcontext(None))
    with output_context as output:
        for row in _rows(path):
            key, user = row['sample_id'], row['user_id']
            if key in ids:
                raise ValueError('Duplicate prediction sample ID')
            ids.add(key)
            expected = truth.get(key)
            if (type(row.get('dataset_index')) is not int
                    or expected != (row.get('dataset_index'), user, row['targets'])):
                raise ValueError('Saved sample index, user, or targets disagree with source data')
            cold = any(item not in seen_items for item in expected[2])
            if type(row.get('cold_target')) is not bool or row['cold_target'] != cold:
                raise ValueError('Saved cold_target disagrees with source train_seen.json')
            if not full:
                if count >= len(selected_rows) or key != selected_rows[count]['sample_id']:
                    raise ValueError('Subset prediction membership/order differs from manifest')
            values = recompute_row(row, **arguments)
            compare(row.get('metrics'), values)
            user_row = users.setdefault(user, [0, defaultdict(float)])
            group_row = groups.setdefault('cold_target' if cold else 'seen_targets', [0, defaultdict(float)])
            count += 1
            user_row[0] += 1
            group_row[0] += 1
            for name, value in values.items():
                sums[name] += value
                user_row[1][name] += value
                group_row[1][name] += value
                if key in wanted:
                    subset_sums[name] += value
            if key in wanted:
                selected_seen.add(key)
            if output is not None:
                output.write(json.dumps(dict(sample_id=key, dataset_index=row['dataset_index'], user_id=user,
                                             cold_target=cold, metrics=values), allow_nan=False) + '\n')
    if not count or count != meta['examples']:
        raise ValueError('Prediction count differs from evaluation summary')
    if full and ids != truth.keys():
        raise ValueError('Claimed full evaluation omitted or added source samples')
    if not full and ids != wanted:
        raise ValueError('Subset evaluation omitted source samples')
    means = {name: value / count for name, value in sums.items()}
    user_means = {name: sum(values[name] / n for n, values in users.values()) / len(users) for name in sums}
    group_means = {group: dict(examples=n, metrics={name: value / n for name, value in values.items()})
                   for group, (n, values) in groups.items()}
    compare(meta.get('metrics'), means)
    compare(meta.get('user_average'), user_means)
    if 'groups' in meta:
        if set(meta['groups']) != set(group_means):
            raise ValueError('Saved summary group names differ from source groups')
        for group, values in group_means.items():
            saved = meta['groups'][group]
            if saved['examples'] != values['examples']:
                raise ValueError('Saved summary group membership differs')
            compare(saved['metrics'], values['metrics'])
    extracted = None
    extracted_path = path.parent / 'diagnostic_subset.json'
    if selected_rows is not None:
        if selected_seen != wanted:
            raise ValueError('Evaluation does not contain every locked diagnostic sample')
        extracted = dict(examples=len(wanted), metrics={key: value / len(wanted) for key, value in subset_sums.items()})
        if full and (require_extracted_subset or extracted_path.is_file()):
            saved = read_json(extracted_path)
            if saved['examples'] != len(wanted):
                raise ValueError('Extracted diagnostic subset count differs')
            compare(saved['metrics'], extracted['metrics'])
    return dict(format='stage09_h1_recovery_independent_metric_audit_v1', passed=mismatches == 0,
                examples=count, users=len(users), source_labels_verified=True, source_cold_groups_verified=True,
                train_seen_sha256=seen_sha, validation_sha256=meta['dataset_binding']['files']['validation.parquet']['sha256'],
                predictions_sha256=file_sha256(path), summary_sha256=file_sha256(path.parent / 'summary.json'),
                recomputed_rows_sha256=file_sha256(recomputed_path) if recomputed_path is not None else None,
                tolerance=tolerance, mismatches=mismatches,
                max_absolute_error=max_error if math.isfinite(max_error) else None,
                nonfinite_saved_metric=not math.isfinite(max_error),
                metrics=means, user_average=user_means, groups=group_means, diagnostic_subset=extracted,
                source_group_definition='any target internal item ID absent from bound train_seen.json item_ids',
                subset_manifest_sha256=canonical_sha256(subset_manifest) if subset_manifest is not None else None,
                primary_metric=PRIMARY_METRIC, primary_estimand='sample mean', secondary_estimand='mean of per-user sample means',
                implementation='Independent analysis.recompute_row math/set replay; no production metric or decoder import.')


def verify_r20(predictions_path, audit_path, index_path):
    """Bind the already accepted historical R20 by recorded hashes, not a guessed directory."""
    rows = [row for row in read_json(index_path)['rows'] if 'H1:selected_best' in row.get('roles', [])]
    audit = read_json(audit_path)
    observed = dict(predictions_sha256=file_sha256(predictions_path),
                    summary_sha256=file_sha256(Path(predictions_path).parent / 'summary.json'),
                    audit_sha256=file_sha256(audit_path))
    if observed != R20_HASHES or len(rows) != 1 or rows[0]['epoch'] != 20 or rows[0]['audit_sha256'] != observed['audit_sha256']:
        raise ValueError('Historical H1 R20 differs from the accepted source/audit index')
    if (audit.get('passed') is not True or audit.get('source_labels_verified') is not True
            or audit.get('source_cold_groups_verified') is not True or audit.get('examples') != FULL_ROWS
            or any(audit.get(key) != observed[key] for key in ('predictions_sha256', 'summary_sha256'))):
        raise ValueError('Historical R20 audit is incomplete or failed')
    return dict(**observed, index_sha256=file_sha256(index_path), recorded_remote_path=rows[0]['remote_path'])


def expected_evaluations(policy):
    if policy not in ('decay', 'freeze'):
        raise ValueError('Unknown recovery policy')
    values = [('initial_R20', 13860, 200, False)]
    for epoch in range(21, 27):
        if policy == 'decay' and epoch in (21, 24):
            values.append((f'after_T{240 if epoch == 21 else 280}', 693 * (epoch - 1), 240 if epoch == 21 else 280, False))
        values.append((f'R{epoch}', 693 * epoch, (240 if epoch <= 23 else 280) if policy == 'decay' else 200,
                       epoch in (23, 26)))
    return values


def collect_evaluations(run_dir, policy):
    """Resolve only committed evaluation receipts; do not choose the latest attempt."""
    root = Path(run_dir)
    fork, complete = read_json(root / 'fork.json'), read_json(root / 'complete.json')
    protocol = fork['protocol']
    if (fork.get('preflight_only') or protocol.get('preflight_only') or complete.get('preflight_only')
            or protocol.get('format') != 'stage09_h1_recovery_v1' or protocol.get('policy') != policy
            or protocol.get('parent_sha256') != PARENT_SHA256 or protocol.get('start_phase') != 25
            or protocol.get('stop_phase') != 33 or not fork.get('parent_hashes')):
        raise ValueError('Run is not an authenticated formal recovery fork')
    expected_endpoint = dict(policy=policy, r_updates=18018, t_updates=280 if policy == 'decay' else 200,
                             phase=33, endpoint_epoch=26)
    if any(complete.get(key) != value for key, value in expected_endpoint.items()):
        raise ValueError('Run has not completed the fixed R26 endpoint')
    artifacts, shared_source, shared_subset = {}, None, None
    for label, r_updates, t_updates, full in expected_evaluations(policy):
        receipt_path = root / 'evaluations' / label / 'complete.json'
        receipt = read_json(receipt_path)
        attempt = receipt['attempt']
        if not isinstance(attempt, str) or re.fullmatch(r'attempt[1-9][0-9]*', attempt) is None:
            raise ValueError('Unsafe or invalid evaluation attempt name')
        path = receipt_path.parent / attempt / 'per_sample.jsonl.gz'
        if (file_sha256(path) != receipt['predictions_sha256']
                or file_sha256(path.parent / 'summary.json') != receipt['summary_sha256']):
            raise ValueError('Accepted evaluation bytes differ from its completion receipt')
        binding = receipt['binding']
        meta, _ = _metadata(path)
        expected = dict(split='validation', architecture='H', seed=2024, beam=32, inference_precision='fp32',
                        r_updates=r_updates, t_updates=t_updates, full_split=full, examples=FULL_ROWS if full else SUBSET_ROWS)
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Evaluation protocol/count/clock differs: {policy}/{label}')
        if (binding.get('full') is not full or binding.get('r_updates') != r_updates
                or binding.get('t_updates') != t_updates):
            raise ValueError('Receipt training clocks differ from the accepted summary')
        for key in ('source', 'subset', 'state'):
            if re.fullmatch(r'[0-9a-f]{64}', str(binding.get(key))) is None:
                raise ValueError('Evaluation receipt lacks a state/source/subset hash')
        if shared_source is None:
            shared_source, shared_subset = binding['source'], binding['subset']
        if (binding['source'], binding['subset']) != (shared_source, shared_subset):
            raise ValueError('Evaluation source/subset changed within an arm')
        artifacts[label] = dict(path=path, receipt_sha256=file_sha256(receipt_path), binding=binding)
    return fork, artifacts


def compare_endpoints(r20_path, decay_path, freeze_path):
    """The fixed three treatment-minus-baseline comparisons; no best-epoch selection."""
    specifications = (
        ('decay_R26_minus_freeze_R26', freeze_path, decay_path),
        ('decay_R26_minus_H1_R20', r20_path, decay_path),
        ('freeze_R26_minus_H1_R20', r20_path, freeze_path),
    )
    comparisons = {}
    for name, baseline, treatment in specifications:
        value = paired_comparison(baseline, treatment, repeats=2000, seed=2024, confidence=.95)
        value.update(comparison=name, difference='treatment minus baseline', primary_metric=PRIMARY_METRIC,
                     primary_result=value['primary'][PRIMARY_METRIC], interpretation=INTERPRETATION,
                     other_metric_intervals='Descriptive auxiliary intervals; not adjusted for multiplicity.')
        comparisons[name] = value
    return comparisons


def analyze_recovery(*, decay_run, freeze_run, dataset_dir, subset_path, output_root,
                     r20_predictions=ROOT / R20_RELATIVE,
                     r20_audit=PRIOR_ANALYSIS / 'fullval_H1_R20_audit.json',
                     r20_index=PRIOR_ANALYSIS / 'full_validation_selected_audits.json'):
    """Create new audit/recomputed-row artifacts and all three 2000-draw comparisons."""
    r20_receipt = verify_r20(r20_predictions, r20_audit, r20_index)
    manifest = read_json(subset_path)
    if (manifest.get('count'), manifest.get('population')) != (SUBSET_ROWS, FULL_ROWS):
        raise ValueError('Formal diagnostic/full validation size changed')
    decay_fork, decay = collect_evaluations(decay_run, 'decay')
    freeze_fork, freeze = collect_evaluations(freeze_run, 'freeze')
    for key in ('parent_hashes', 'execution_binding'):
        if decay_fork[key] != freeze_fork[key]:
            raise ValueError(f'Arms differ in common starting/source identity: {key}')
    if ({key: value for key, value in decay_fork['protocol'].items() if key != 'policy'}
            != {key: value for key, value in freeze_fork['protocol'].items() if key != 'policy'}):
        raise ValueError('Arm protocols differ beyond their assigned policy')
    if decay['initial_R20']['binding'] != freeze['initial_R20']['binding']:
        raise ValueError('Initial state/source/subset differ across arms')
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        audits = {}
        collections = [('H1_R20', {'full_validation': dict(path=Path(r20_predictions))}), ('decay', decay), ('freeze', freeze)]
        for arm, artifacts in collections:
            audits[arm] = {}
            for label, artifact in artifacts.items():
                destination = output / arm / label
                destination.mkdir(parents=True)
                audit = audit_recovery_predictions(
                    artifact['path'], dataset_dir=dataset_dir, subset_manifest=manifest,
                    recomputed_path=destination / 'recomputed_per_sample.jsonl.gz',
                    require_extracted_subset=arm != 'H1_R20' and label in ('R23', 'R26'))
                audit.update(input=str(artifact['path'].resolve()), receipt_sha256=artifact.get('receipt_sha256'))
                write_json(destination / 'audit.json', audit)
                if not audit['passed']:
                    raise ValueError(f'Independent metric audit failed: {arm}/{label}')
                audits[arm][label] = dict(audit=str((destination / 'audit.json').relative_to(output)),
                    audit_sha256=file_sha256(destination / 'audit.json'), examples=audit['examples'],
                    metrics=audit['metrics'], user_average=audit['user_average'], diagnostic_subset=audit['diagnostic_subset'])
        comparisons = compare_endpoints(r20_predictions, decay['R26']['path'], freeze['R26']['path'])
        (output / 'comparisons').mkdir()
        for name, value in comparisons.items():
            write_json(output / 'comparisons' / (name + '.json'), value)
        result = dict(format='stage09_h1_recovery_analysis_v1', status='passed', parent_sha256=PARENT_SHA256,
            historical_R20=r20_receipt, subset_file_sha256=file_sha256(subset_path),
            implementation_sha256=dict(adapter=file_sha256(__file__),
                independent_core=file_sha256(ROOT / 'scripts/listwise_diffgrm/analysis.py')),
            primary_metric=PRIMARY_METRIC, primary_estimand='sample mean',
            secondary_estimand='mean of per-user sample means', endpoint_epoch=26,
            bootstrap=dict(repeats=2000, seed=2024, confidence=.95, unit='user cluster'),
            comparisons={name: value['primary_result'] for name, value in comparisons.items()},
            evaluations=audits, interpretation=INTERPRETATION)
        write_json(output / 'analysis_report.json', result)
        return result
    except Exception as error:
        write_json(output / 'FAILED.json', dict(status='failed', exception=type(error).__name__, message=str(error),
                                               partial_artifacts_are_not_accepted=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decay-run', required=True)
    parser.add_argument('--freeze-run', required=True)
    parser.add_argument('--dataset', default=str(ROOT / 'outputs/kuairec_listwise_v1/run1'))
    parser.add_argument('--subset', required=True, help='Locked unsalted recovery subset JSON, including rows')
    parser.add_argument('--output-root', required=True, help='New analysis directory; existing directories are refused')
    parser.add_argument('--r20-predictions', default=str(ROOT / R20_RELATIVE))
    parser.add_argument('--r20-audit', default=str(PRIOR_ANALYSIS / 'fullval_H1_R20_audit.json'))
    parser.add_argument('--r20-index', default=str(PRIOR_ANALYSIS / 'full_validation_selected_audits.json'))
    args = parser.parse_args(argv)
    result = analyze_recovery(decay_run=args.decay_run, freeze_run=args.freeze_run, dataset_dir=args.dataset,
        subset_path=args.subset, output_root=args.output_root, r20_predictions=args.r20_predictions,
        r20_audit=args.r20_audit, r20_index=args.r20_index)
    print(json.dumps(dict(status=result['status'], comparisons=result['comparisons']), allow_nan=False))


if __name__ == '__main__':
    main()
