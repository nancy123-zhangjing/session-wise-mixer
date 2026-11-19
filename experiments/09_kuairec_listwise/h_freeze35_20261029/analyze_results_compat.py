"""Versioned, local CPU adapter for two already-audited historical validations.

Uses the unchanged analyze_results.py CLI for --branch and --all-ready. Only
the exact A/R23 and A/R26 validation references may lack summary.groups. Their
accepted audits are reused with explicit provenance, never called new replay.
All other metadata checks, sample audits, selection and bootstrap stay in v1.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VERSION = 'freeze35_historical_validation_compat_v1'
HELPER_SHA = '47522df77e2eab06e598454ed331e6a939609bf0f63ddb9087d78f7f19aa9677'
REFS_SHA = '4c0a0e988f9073a89ab6fac37d6d05b394ff4271ccbb553ace9c75ffc5bf7734'
API_PATH = ROOT / 'scripts/analysis/analyze_h1_recovery.py'
API_SHA = 'addeb170c093e9fa9c312bc282242fe52794053a5166490f7b0378b52efe10ce'
PRIOR = ROOT / 'experiments/09_kuairec_listwise/h1_recovery_20261028/final_analysis'
REPORT_SHA = '61cae3b9b4fa000f0a0ce7903fcbdbb584ff7ac1aac9d565e026b10295469d60'
HISTORICAL_PARENT_SHA = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
LEGACY = {
    23: dict(
        summary_sha256='9910012060faae7c4e22c604b326fc3832bc4fbdd6564b95f13e166daef90f91',
        predictions_sha256='92ed6add54a876ade99baa25d885cdc63238050558b601f410e84aacc4439948',
        checkpoint_sha256='1dd0ceea3d6dd93492db5a5ab5d29762abe1fa73bcff18651efce7eac5f37300',
        audit_sha256='ead723207791a7a4a6e4f7c99db904d5076d7f0b166f7db895b6580a07ee0d90'),
    26: dict(
        summary_sha256='b375e20ca859f2982019a8d37b882bd9035355bbe88250964fd9706f0036f784',
        predictions_sha256='1236e808836bc29b2fc493cc2c4da210d3ab24f49dd2adf769f002224ea63393',
        checkpoint_sha256='1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
        audit_sha256='b5b855a2983aa45dea421ac5c44bc816446dfe30e02a6d0bba7156f75e224b9c'),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


HELPER_PATH = HERE / 'analyze_results.py'
need(digest(HELPER_PATH) == HELPER_SHA, 'Accepted source analyzer changed')
ADAPTER_SHA = digest(__file__)
_spec = importlib.util.spec_from_file_location('_freeze35_compat_source_v1', HELPER_PATH)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
_original_context, _original_audit, _original_finish = helper.Context, helper.audit, helper.finish


class Context(_original_context):
    def __init__(self, runs_root, dataset, references):
        super().__init__(runs_root, dataset, references)
        self.bind(__file__, ADAPTER_SHA)
        self.bind(HELPER_PATH, HELPER_SHA)
        self.legacy_audit_reuse = {}

    def point(self, branch, epoch, split, record, *, reference=False, model=None, coverage=None):
        if reference and branch == 'A' and split == 'validation' and epoch in LEGACY:
            need(model is None and coverage is None, 'Historical compatibility cannot validate a new model')
            return self._legacy_point(branch, epoch, split, record)
        return super().point(branch, epoch, split, record, reference=reference, model=model, coverage=coverage)

    def _legacy_point(self, branch, epoch, split, record):
        expected = dict(self.refs['validation']['A'][str(epoch)])
        for key in ('summary', 'predictions', 'checkpoint'):
            if key in expected.get('local', {}):
                expected[key] = expected['local'][key]
        locked = LEGACY[epoch]
        need(self.refs_sha == REFS_SHA and record == expected,
             'Legacy compatibility requires the exact locked historical reference')
        need(all(record[key] == locked[key] for key in
                 ('summary_sha256', 'predictions_sha256', 'checkpoint_sha256')),
             'Unrecognized missing-groups validation bytes')
        paths = {key: Path(record[key]).resolve() for key in ('summary', 'predictions', 'checkpoint')}
        for key, path in paths.items():
            self.bind(path, locked[key + '_sha256'])
        need(paths['summary'].parent == paths['predictions'].parent,
             'Historical prediction and summary directories differ')
        meta = helper.read(paths['summary'])
        need('groups' not in meta, 'Expected the exact historical missing-groups schema')
        # Disposable schema check only. No fabricated groups enter original
        # metadata, source bytes, audit values, selection, or report metrics.
        helper.validate_metadata(dict(meta, groups={}), branch, epoch, split)
        value, provenance = self._accepted_audit(epoch, paths, locked)
        self.legacy_audit_reuse[epoch] = (value, provenance)
        return dict(branch=branch, epoch=epoch, split=split, reference=True,
                    **{key: str(path) for key, path in paths.items()},
                    **{key + '_sha256': locked[key + '_sha256'] for key in paths},
                    metadata=meta, model_fingerprint=None, execution_fingerprint=None,
                    historical_audit_reuse=provenance)

    def _accepted_audit(self, epoch, paths, locked):
        self.bind(API_PATH, API_SHA)
        report_path = PRIOR / 'analysis_report.json'
        report = self.json(report_path, REPORT_SHA)
        need(report.get('status') == 'passed'
             and report.get('format') == 'stage09_h1_recovery_analysis_v1'
             and report.get('parent_sha256') == HISTORICAL_PARENT_SHA
             and report.get('primary_estimand') == 'sample mean'
             and report.get('implementation_sha256') == dict(adapter=API_SHA, independent_core=helper.ANALYSIS_SHA),
             'Historical report provenance or estimand differs')
        index = report['evaluations']['freeze'][f'R{epoch}']
        audit_path = PRIOR / 'freeze' / f'R{epoch}' / 'audit.json'
        need((PRIOR / index['audit']).resolve() == audit_path.resolve()
             and index['audit_sha256'] == locked['audit_sha256'], 'Historical audit index differs')
        value = self.json(audit_path, locked['audit_sha256'])
        need(value.get('format') == 'stage09_h1_recovery_independent_metric_audit_v1'
             and value.get('passed') is True and value.get('mismatches') == 0
             and value.get('nonfinite_saved_metric') is False and value.get('tolerance') == 2e-6
             and value.get('source_labels_verified') is True and value.get('source_cold_groups_verified') is True
             and (value.get('examples'), value.get('users')) == helper.POPULATION['validation']
             and value.get('primary_estimand') == 'sample mean', 'Historical full-source audit did not pass')
        need(all(value.get(key) == locked[key] for key in ('summary_sha256', 'predictions_sha256'))
             and value.get('validation_sha256') == helper.DATA_SHA['validation.parquet']
             and value.get('train_seen_sha256') == helper.DATA_SHA['train_seen.json']
             and Path(value['input']).resolve() == paths['predictions'], 'Historical audit input binding differs')
        need(index['examples'] == value['examples']
             and all(index[key] == value[key] for key in ('metrics', 'user_average', 'diagnostic_subset')),
             'Historical report metrics differ from its accepted audit')
        need({key: group['examples'] for key, group in value['groups'].items()}
             == dict(cold_target=8269, seen_targets=63388), 'Historical source-group coverage differs')
        receipt_path = paths['predictions'].parent.parent / 'complete.json'
        receipt = self.json(receipt_path, value['receipt_sha256'])
        need(all(receipt.get(key) == locked[key] for key in ('summary_sha256', 'predictions_sha256'))
             and receipt.get('attempt') == paths['predictions'].parent.name
             and receipt['binding'].get('full') is True
             and receipt['binding'].get('r_updates') == epoch * 693
             and receipt['binding'].get('t_updates') == 200, 'Historical completion receipt differs')
        rows_path = audit_path.parent / 'recomputed_per_sample.jsonl.gz'
        self.bind(rows_path, value['recomputed_rows_sha256'])
        provenance = dict(adapter_version=VERSION, operation='reuse_accepted_historical_audit',
                          recomputed_in_this_run=False, saved_summary_groups_present=False,
                          saved_summary_groups_verified=False,
                          groups_origin='Prior independent replay verified against source train_seen.json',
                          original_audit=str(audit_path), original_audit_sha256=locked['audit_sha256'],
                          original_report=str(report_path), original_report_sha256=REPORT_SHA,
                          original_api=str(API_PATH), original_api_sha256=API_SHA,
                          original_complete=str(receipt_path), original_complete_sha256=value['receipt_sha256'],
                          original_recomputed_rows=str(rows_path),
                          original_recomputed_rows_sha256=value['recomputed_rows_sha256'])
        return value, provenance


def audit(ctx, label, point, output):
    if 'historical_audit_reuse' not in point:
        return _original_audit(ctx, label, point, output)
    need((point['branch'], point['split'], point['reference']) == ('A', 'validation', True)
         and point['epoch'] in ctx.legacy_audit_reuse, 'Unexpected historical audit reuse')
    accepted, provenance = ctx.legacy_audit_reuse[point['epoch']]
    need(point['historical_audit_reuse'] == provenance
         and all(point[key] == LEGACY[point['epoch']][key] for key in
                 ('summary_sha256', 'predictions_sha256', 'checkpoint_sha256')),
         'Historical point changed after authentication')
    value = deepcopy(accepted)
    value.update(model=label, branch=point['branch'], epoch=point['epoch'], split=point['split'],
                 checkpoint_sha256=point['checkpoint_sha256'], primary_estimand='sample mean',
                 historical_audit_reuse=provenance, recomputed_in_this_run=False)
    path = Path(output) / 'audits' / (label + '.json')
    helper.write(path, value)
    return dict(point, audit_path=str(path), audit_sha256=helper.sha(path), metrics=value['metrics'],
                user_average=value['user_average'], groups=value['groups'])


def finish(ctx, output, report, text):
    reused = {f'A_R{epoch}_validation': provenance
              for epoch, (_, provenance) in sorted(ctx.legacy_audit_reuse.items())}
    report['historical_schema_compatibility'] = dict(
        version=VERSION, adapter_sha256=ADAPTER_SHA, source_analyzer_sha256=HELPER_SHA,
        normal_metadata_audit_and_bootstrap_unchanged=True, accepted_audits_reused=reused)
    if reused:
        text += ('\n历史格式兼容：A 的 R23/R26 验证 summary 原本没有 groups；本报告复用已验收、'
                 '重新核验输入及来源 SHA 的逐样本审计，不声称本次重新计算了这两份预测。'
                 '冷/热组来自原审计对真实标签与 train_seen.json 的独立核验，'
                 '没有补写或更改原 summary。其他审计与配对比较沿用原分析器。\n')
    return _original_finish(ctx, output, report, text)


# Adapt only this private imported module instance; source files and the shared
# independent core are untouched. The original B report remains hash-valid.
helper.Context, helper.audit, helper.finish = Context, audit, finish


def main(argv=None):
    return helper.main(argv)


if __name__ == '__main__':
    main()
