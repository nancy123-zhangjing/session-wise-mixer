"""Read-only replay of the fixed H1 R26 test comparison; never trains or decodes.

Use --decay/--freeze for the completed launcher's evaluation/ directories.
The historical R20 test artifact is locked by path and SHA, not regenerated.
All output goes to a new directory; existing reports and inputs are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from scripts.listwise_diffgrm.analysis import audit_predictions, paired_comparison
from scripts.listwise_diffgrm.subset import file_sha256


R20_DIR = ROOT / 'outputs/stage09_restore/goal_20261027_r1/sidecars/evaluations/evaluations/H1_test_beam32_r2'
R20_AUDIT = ROOT / 'experiments/09_kuairec_listwise/final_analysis_20261027/H1_test_beam32_audit.json'
R20_HASHES = {
    'per_sample.jsonl.gz': '10a2edb7baeb337eba781e26cd9a4d396ff8e6de7cd97bebcc3efd48d3c6b1a2',
    'summary.json': '2e588690946d14d6c406acc0f00612c9c631c65aaa94adb201eec351afeae1e0',
    'audit.json': '8ce33eec2edfdb94e99764915a6850445868882ace1825676097f41240c1c339',
}
CHECKPOINT_HASHES = {
    'R20': 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1',
    'decay': '2ccf71e14ade6dbc58848a249f9396cd36d061f64abb65f56152e8afb84585c6',
    'freeze': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
}
TEST_SHA = 'f1946e99dfcb1bd1603164a6481fdc3ce7ceacb37bedcd755a025bbf14aa9cee'
TRAIN_SEEN_SHA = 'f8a0205149fc22d418bf4c28f43037c44c8d7539d1aa9b837f636f6df636efdb'
EXECUTION_SHA = '5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f'
EXAMPLES, USERS = 53194, 4153
PRIMARY = 'list/ndcg@5'
INTERPRETATION = (
    'Fixed R26 endpoints, complete held-out test split, one training seed. '
    'The policies were proposed after inspecting validation results; this is not '
    'a preregistered discovery experiment. R20 is the historically validation-selected '
    'parent. No test-based epoch, beam, policy or metric selection is performed here. '
    'User-cluster intervals describe sampling uncertainty for these fixed models, '
    'not training-seed variation or proof of equivalence. Aggregate @10/@20/@50 '
    'measure the fixed aggregation of multiple candidate five-item lists, not a '
    'single five-item list. Auxiliary intervals are not multiplicity-adjusted.'
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_metadata(meta, policy, historical):
    """Reject a different split, endpoint, precision or candidate protocol."""
    expected = dict(split='test', examples=EXAMPLES, full_split=True, subset=None,
                    diagnostic_prefix_limit=None, catalog_size=10728, beam=32,
                    inference_precision='fp32', architecture='H', seed=2024,
                    r_updates=13860 if policy == 'R20' else 18018,
                    t_updates=280 if policy == 'decay' else 200)
    if policy not in ('R20', 'decay', 'freeze'):
        raise ValueError('Unsupported fixed model')
    for name, value in expected.items():
        if name not in meta or type(meta[name]) is not type(value) or meta[name] != value:
            raise ValueError(f'{policy}: evaluation protocol differs at {name}')
    protocol = dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32,
                    pad_id=0, target_relevance='unordered_set_of_five_future_distinct_items',
                    aggregation='list_score_order_then_item_position_first_occurrence')
    if meta.get('metric_protocol') != protocol:
        raise ValueError(f'{policy}: unsupported ranking/aggregation protocol')
    if meta.get('dataset_binding') != historical['dataset_binding']:
        raise ValueError(f'{policy}: dataset binding differs from accepted R20 test')
    if (meta['dataset_binding']['files']['test.parquet']['sha256'] != TEST_SHA
            or meta['dataset_binding']['files']['train_seen.json']['sha256'] != TRAIN_SEEN_SHA):
        raise ValueError(f'{policy}: source test/train-seen binding differs')
    if not isinstance(meta.get('groups'), dict):
        raise ValueError(f'{policy}: summary must contain the source cold/seen group metrics')


def verify_r20():
    observed = {name: file_sha256(R20_AUDIT if name == 'audit.json' else R20_DIR / name)
                for name in R20_HASHES}
    if observed != R20_HASHES:
        raise ValueError('Historical R20 test bytes differ from the accepted artifact')
    audit = read_json(R20_AUDIT)
    if (audit.get('passed') is not True or audit.get('source_labels_verified') is not True
            or audit.get('source_cold_groups_verified') is not True
            or audit.get('examples') != EXAMPLES or audit.get('users') != USERS
            or audit.get('predictions_sha256') != observed['per_sample.jsonl.gz']
            or audit.get('summary_sha256') != observed['summary.json']):
        raise ValueError('Historical R20 test audit is incomplete or mismatched')
    return dict(directory=str(R20_DIR), checkpoint_sha256=CHECKPOINT_HASHES['R20'],
                artifact_sha256=observed, reused_without_gpu_evaluation=True)


def verify_completion(directory, policy):
    """Check the launcher's completion evidence before trusting its predictions."""
    receipt_path = directory.parent / 'complete.json'
    request_path = directory.parent / 'request.json'
    receipt = read_json(receipt_path)
    request = read_json(request_path)
    expected = dict(policy=policy, checkpoint_sha256=CHECKPOINT_HASHES[policy],
                    parent_sha256=CHECKPOINT_HASHES['R20'], execution_manifest_sha256=EXECUTION_SHA,
                    launcher_sha256=file_sha256(Path(__file__).with_name('evaluate_test.py')),
                    fixed_endpoint='R26', training_performed=False)
    if any(type(source.get(k)) is not type(v) or source.get(k) != v
           for source in (receipt, request) for k, v in expected.items()):
        raise ValueError(f'{policy}: completion/request source, policy or endpoint differs')
    if (receipt.get('status') != 'complete' or receipt.get('examples') != EXAMPLES
            or receipt.get('predictions_sha256') != file_sha256(directory / 'per_sample.jsonl.gz')
            or receipt.get('summary_sha256') != file_sha256(directory / 'summary.json')
            or receipt.get('request_sha256') != file_sha256(request_path)):
        raise ValueError(f'{policy}: completed evaluation SHA binding differs')
    before = receipt.get('before_state_sha256')
    if (not isinstance(before, str) or re.fullmatch('[0-9a-f]{64}', before) is None
            or before != receipt.get('after_state_sha256') or before != request.get('before_state_sha256')):
        raise ValueError(f'{policy}: evaluation changed model/training state or lacks its proof')
    if (request.get('split') != 'test' or request.get('inference_precision') != 'fp32'
            or request.get('dataset_test_sha256') != TEST_SHA or request.get('r_updates') != 18018
            or request.get('t_updates') != (280 if policy == 'decay' else 200)):
        raise ValueError(f'{policy}: request does not describe the fixed full-test evaluation')
    config, evaluation = request['config'], request['config']['evaluation']
    if (any(config.get(k) != v for k, v in dict(architecture='H', list_length=5, history_max=50, seed=2024).items())
            or any(evaluation.get(k) != v for k, v in dict(test_beam=32, batch_size=32, candidate_lists=32,
                aggregate_ks=[10, 20, 50], filter_history=False,
                aggregation='list_score_order_then_item_position_first_occurrence',
                collision_rule='reconstruction_error_then_original_id').items())):
        raise ValueError(f'{policy}: request configuration differs from the locked protocol')
    return dict(path=str(receipt_path), sha256=file_sha256(receipt_path),
                request_path=str(request_path), request_sha256=file_sha256(request_path),
                checkpoint_sha256=CHECKPOINT_HASHES[policy], state_unchanged=True,
                state_sha256=before, training_performed=False)


def analyze(*, decay, freeze, dataset, output):
    directories = {'R20': R20_DIR.resolve(), 'decay': Path(decay).resolve(), 'freeze': Path(freeze).resolve()}
    dataset, output = Path(dataset).resolve(), Path(output).absolute()
    if len(set(directories.values())) != 3:
        raise ValueError('The three prediction directories must be distinct')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Analysis output must be a new directory')
    if any(output.resolve().is_relative_to(p) for p in [*directories.values(), dataset, R20_AUDIT.parent]):
        raise ValueError('Output must not modify a source prediction/data/prior-report directory')
    historical = verify_r20()
    metas = {name: read_json(path / 'summary.json') for name, path in directories.items()}
    for name, meta in metas.items():
        validate_metadata(meta, name, metas['R20'])
    receipts = {name: verify_completion(directories[name], name) for name in ('decay', 'freeze')}
    if (file_sha256(dataset / 'test.parquet') != TEST_SHA
            or file_sha256(dataset / 'train_seen.json') != TRAIN_SEEN_SHA):
        raise ValueError('Actual local test/seen data differs from the locked source')
    inputs = [R20_AUDIT, dataset / 'test.parquet', dataset / 'train_seen.json']
    inputs.extend(path / name for path in directories.values() for name in ('summary.json', 'per_sample.jsonl.gz'))
    inputs.extend(Path(row['path']) for row in receipts.values())
    inputs.extend(Path(row['request_path']) for row in receipts.values())
    implementation = [Path(__file__).resolve(), ROOT / 'scripts/listwise_diffgrm/analysis.py',
                      ROOT / 'scripts/listwise_diffgrm/subset.py', Path(__file__).with_name('evaluate_test.py')]
    original = {str(path): file_sha256(path) for path in [*inputs, *implementation]}
    output.mkdir(parents=True, exist_ok=False)
    try:
        audits = {}
        for name, directory in directories.items():
            audit = audit_predictions(directory / 'per_sample.jsonl.gz', dataset_dir=dataset)
            if (audit.get('passed') is not True or audit['examples'] != EXAMPLES or audit['users'] != USERS
                    or audit.get('source_labels_verified') is not True
                    or audit.get('source_cold_groups_verified') is not True):
                raise ValueError(f'{name}: independent full-test replay failed')
            audit.update(split='test', test_parquet_sha256=TEST_SHA,
                         primary_estimand='sample mean', input=str(directory),
                         completion_receipt=receipts.get(name))
            write_json(output / (name + '_audit.json'), audit)
            audits[name] = audit
            print(json.dumps(dict(event='audit_passed', model=name, examples=audit['examples'],
                                  users=audit['users'], ndcg5=audit['metrics'][PRIMARY])), flush=True)
        comparisons = {}
        for name, baseline, treatment in (
            ('decay_R26_minus_freeze_R26', 'freeze', 'decay'),
            ('decay_R26_minus_H1_R20', 'R20', 'decay'),
            ('freeze_R26_minus_H1_R20', 'R20', 'freeze'),
        ):
            value = paired_comparison(directories[baseline] / 'per_sample.jsonl.gz',
                                      directories[treatment] / 'per_sample.jsonl.gz',
                                      repeats=2000, seed=2024, confidence=.95)
            if value['examples'] != EXAMPLES or value['users'] != USERS:
                raise ValueError('Paired test population differs')
            value.update(comparison=name, split='test', difference='treatment minus baseline',
                         primary_metric=PRIMARY, primary_result=value['primary'][PRIMARY],
                         interpretation=INTERPRETATION)
            write_json(output / (name + '.json'), value)
            comparisons[name] = value['primary_result']
            print(json.dumps(dict(event='comparison_complete', name=name,
                                  primary=value['primary_result'])), flush=True)
        if any(file_sha256(path) != expected for path, expected in original.items()):
            raise ValueError('An input or analysis implementation changed during replay')
        if 'torch' in sys.modules:
            raise RuntimeError('The independent local analysis unexpectedly imported torch')
        report = dict(format='stage09_h1_recovery_fixed_R26_test_analysis_v1', status='passed',
            created_utc=datetime.now(timezone.utc).isoformat(), split='test', examples=EXAMPLES, users=USERS,
            primary_metric=PRIMARY, primary_estimand='sample mean',
            secondary_estimand='mean of per-user sample means', fixed_recovery_endpoint_epoch=26,
            bootstrap=dict(repeats=2000, seed=2024, confidence=.95, unit='user cluster'),
            historical_R20=historical, completion_receipts=receipts, inputs_unchanged=True,
            input_and_implementation_sha256=original,
            evaluations={name: dict(audit=name + '_audit.json', audit_sha256=file_sha256(output / (name + '_audit.json')),
                                   metrics=row['metrics'], user_average=row['user_average'], groups=row['groups'])
                         for name, row in audits.items()},
            comparisons=comparisons, interpretation=INTERPRETATION,
            gpu_evaluation_performed_by_this_script=False, torch_imported=False)
        write_json(output / 'analysis_report.json', report)
        return report
    except Exception as error:
        write_json(output / 'FAILED.json', dict(status='failed', exception=type(error).__name__,
            message=str(error), partial_artifacts_are_not_accepted=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decay', required=True, help='Decay launcher evaluation/ directory')
    parser.add_argument('--freeze', required=True, help='Freeze launcher evaluation/ directory')
    parser.add_argument('--dataset', default=str(ROOT / 'outputs/kuairec_listwise_v1/run1'))
    parser.add_argument('--output', required=True, help='New analysis output directory')
    args = parser.parse_args(argv)
    report = analyze(decay=args.decay, freeze=args.freeze, dataset=args.dataset, output=args.output)
    print(json.dumps(dict(status=report['status'], examples=report['examples'], users=report['users'],
                          comparisons=report['comparisons']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
