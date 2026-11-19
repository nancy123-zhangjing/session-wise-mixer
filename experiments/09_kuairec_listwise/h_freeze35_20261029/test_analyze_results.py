"""Tiny protocol/state-binding checks; no real prediction replay or bootstrap."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).with_name('analyze_results.py')
spec = importlib.util.spec_from_file_location('freeze35_results_under_test', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def metadata(branch='B', epoch=35, split='test'):
    return dict(split=split, examples=m.POPULATION[split][0], full_split=True, subset=None,
        diagnostic_prefix_limit=None, catalog_size=10728, beam=32, inference_precision='fp32',
        architecture='H', seed=2024, r_updates=epoch * 693, t_updates=m.expected_t(branch, epoch),
        metric_protocol=copy.deepcopy(m.PROTOCOL), groups={}, metrics={m.PRIMARY: .01},
        dataset_binding=dict(history_max=50, list_length=5, pad_id=0,
                             files={name: {'sha256': value} for name, value in m.DATA_SHA.items()}))


class ResultsProtocolTests(unittest.TestCase):
    def setUp(self):
        base = m.ROOT / 'tmp/freeze35_results'; base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        # Deliberately tiny file containers, never accepted by audit_predictions.
        ctx = object.__new__(m.Context)
        ctx.runs_root = self.root / 'runs'; ctx.dataset = self.root / 'data'; ctx.inputs = {}
        ctx.references_path = self.root / 'references.json'; dump(ctx.references_path, {})
        ctx.refs_sha = m.sha(ctx.references_path); ctx.refs = {}
        d = ctx.runs_root / m.NAMES['B']; d.mkdir(parents=True)
        weight = d / 'phase44_R_R35_T360.pt'; weight.write_bytes(b'tiny synthetic, not a model')
        records = {}
        for split in ('validation', 'test'):
            e = d / 'evaluations' / f'{split}_R35' / 'attempt1'
            dump(e / 'summary.json', metadata(split=split))
            (e / 'per_sample.jsonl.gz').write_bytes(b'tiny synthetic, not predictions')
            record = dict(kind='new', epoch=35, summary=str(e / 'summary.json'),
                predictions=str(e / 'per_sample.jsonl.gz'), summary_sha256=m.sha(e / 'summary.json'),
                predictions_sha256=m.sha(e / 'per_sample.jsonl.gz'), checkpoint=str(weight),
                checkpoint_sha256=m.sha(weight), ndcg5=.01)
            dump(e.parent / 'complete.json', dict(attempt='attempt1',
                summary_sha256=record['summary_sha256'], predictions_sha256=record['predictions_sha256'],
                binding=dict(branch='B', split=split, epoch=35, r_updates=24255, t_updates=360,
                             model='c' * 64, execution='d' * 64)))
            records[split] = record
        dump(d / 'validation_records/R35.json', records['validation'])
        dump(d / 'selection.json', dict(branch='B', candidates={}, selected=records['validation']))
        coverage = dict(status='passed', expected=5, present=5,
                        records={weight.name: {'sha256': m.sha(weight)}, **{f'other{i}.pt': {} for i in range(4)}})
        dump(d / 'CHECKPOINT_COVERAGE.json', coverage)
        complete = dict(status='training_and_validation_complete', branch='B', phase=45, r_updates=24255, t_updates=360,
                        selection_sha256=m.sha(d / 'selection.json'), checkpoint_coverage_sha256=m.sha(d / 'CHECKPOINT_COVERAGE.json'))
        dump(d / 'complete.json', complete)
        dump(d / 'request.json', dict(branch='B', parent_sha256=m.PARENTS['B'], reference_manifest_sha256=ctx.refs_sha,
             protocol=dict(format='stage09_recovery35_v1', name=m.NAMES['B'], branch='B', parent_sha256=m.PARENTS['B'])))
        dump(d / 'test_predictions_complete.json', dict(branch='B', results=[records['test']],
             selection_sha256=complete['selection_sha256'], independently_audited=False))
        return ctx, d

    def test_metadata_rejects_wrong_split_beam_precision_clock_and_aggregation(self):
        m.validate_metadata(metadata(), 'B', 35, 'test')
        for key, value in [('split', 'validation'), ('beam', 256), ('inference_precision', 'bf16'),
                           ('r_updates', 22176), ('t_updates', 200), ('examples', 32), ('seed', 2025), ('full_split', False)]:
            bad = metadata(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_metadata(bad, 'B', 35, 'test')
        bad = metadata(); bad['metric_protocol']['candidate_lists'] = 256
        with self.assertRaises(ValueError): m.validate_metadata(bad, 'B', 35, 'test')

    def test_c_r5_test_is_rejected_but_r5_validation_is_allowed(self):
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            m.validate_metadata(metadata('C', 5), 'C', 5, 'test')
        m.validate_metadata(metadata('C', 5, 'validation'), 'C', 5, 'validation')

    def test_common_best_is_selected_independently_with_earlier_exact_tie(self):
        values = {e: {'metadata': {'metrics': {m.PRIMARY: .1}}} for e in m.GRID}
        self.assertEqual(m.shared_best(values), 20)
        values[26]['metadata']['metrics'][m.PRIMARY] = .2
        self.assertEqual(m.shared_best(values), 26)
        values[26]['metadata']['metrics'][m.PRIMARY] = float('nan')
        with self.assertRaises(ValueError): m.shared_best(values)
        values.pop(23)
        with self.assertRaises(ValueError): m.shared_best(values)

    def test_only_exact_branch_suffix_can_relocate_downloaded_artifacts(self):
        d = self.root / m.NAMES['A']; (d / 'evaluations/test_R35').mkdir(parents=True)
        file = d / 'evaluations/test_R35/summary.json'; file.write_text('{}')
        remote = '/owned/newgoal/runs/' + d.name + '/evaluations/test_R35/summary.json'
        self.assertEqual(m.safe_local(d, remote), file.resolve())
        for name in ('../elsewhere.json', '/owned/other/summary.json'):
            with self.assertRaises((ValueError, FileNotFoundError)): m.safe_local(d, name)

    def test_actual_completion_receipts_and_selected_model_are_required(self):
        ctx, directory = self.fixture()
        validation, tests, best = m.verify_runtime(ctx, 'B')
        self.assertEqual(best, 35); self.assertEqual(set(tests), {35})
        self.assertEqual(validation[35]['checkpoint_sha256'], tests[35]['checkpoint_sha256'])
        request = m.read(directory / 'request.json'); request['parent_sha256'] = m.PARENTS['A']
        dump(directory / 'request.json', request)
        # A fresh context avoids accidentally reporting a changed-input guard as the only check.
        ctx.inputs = {}
        with self.assertRaisesRegex(ValueError, 'Source/branch'):
            m.verify_runtime(ctx, 'B')

    def test_missing_completion_or_extra_test_never_becomes_an_accepted_report(self):
        ctx, directory = self.fixture()
        done = m.read(directory / 'test_predictions_complete.json')
        done['results'].append(dict(done['results'][0], epoch=32))
        dump(directory / 'test_predictions_complete.json', done)
        with self.assertRaisesRegex(ValueError, 'unauthorized'):
            m.verify_runtime(ctx, 'B')
        (directory / 'test_predictions_complete.json').unlink()
        with self.assertRaises(FileNotFoundError): m.verify_runtime(ctx, 'B')
        self.assertFalse((self.root / 'analysis_report.json').exists())

    def test_changed_prediction_bytes_or_wrong_evaluation_state_are_rejected(self):
        ctx, directory = self.fixture()
        prediction = directory / 'evaluations/test_R35/attempt1/per_sample.jsonl.gz'
        prediction.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Changed source bytes'):
            m.verify_runtime(ctx, 'B')

    def test_test_model_fingerprint_must_match_validation_locked_state(self):
        ctx, directory = self.fixture()
        receipt_path = directory / 'evaluations/test_R35/complete.json'
        receipt = m.read(receipt_path); receipt['binding']['model'] = 'a' * 64
        dump(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'validation-locked model/execution'):
            m.verify_runtime(ctx, 'B')

    def test_paired_bootstrap_locks_repeats_seed_population_and_estimand(self):
        base = dict(split='test', predictions='baseline')
        treatment = dict(split='test', predictions='treatment')
        primary = dict(baseline=.01, treatment=.02, delta=.01, relative_delta=1., confidence_interval=[.005, .015])
        fn = Mock(return_value=dict(examples=53194, users=4153, primary={m.PRIMARY: primary}))
        ctx = SimpleNamespace(module=SimpleNamespace(paired_comparison=fn))
        result = m.pair(ctx, 'tiny_pair', base, treatment, self.root)
        self.assertEqual(fn.call_args.kwargs, dict(repeats=2000, seed=2024, confidence=.95))
        self.assertEqual(result['delta'], .01)
        with self.assertRaisesRegex(ValueError, 'validation to test'):
            m.pair(ctx, 'wrong', dict(base, split='validation'), treatment, self.root)

    def test_analysis_version_mismatch_is_refused_before_import(self):
        with patch.object(m, 'ANALYSIS_SHA', '0' * 64), patch.object(m.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'accepted SHA'): m.implementation()
            imported.assert_not_called()

    def test_locked_validation_bootstrap_plan_is_complete_without_c_origin_test(self):
        own_a = {name for name, _, _ in m.branch_comparison_specs('A')}
        own_b = {name for name, _, _ in m.branch_comparison_specs('B')}
        own_c = m.branch_comparison_specs('C')
        self.assertIn('A_R35_minus_A_R26_validation', own_a)
        self.assertIn('B_R35_minus_B_R32_validation', own_b)
        self.assertEqual({name for name, _, _ in own_c}, {'C_R35_minus_R5_validation', 'C_R35_minus_R20_validation'})
        self.assertTrue(all('_test' not in point for row in own_c for point in row))
        fixed = {name for name, _, _ in m.final_comparison_specs({'A': 35, 'B': 35, 'C': 35})}
        self.assertEqual(fixed, {f'A_minus_{b}_R35_{split}' for b in ('C', 'B') for split in ('test', 'validation')})
        selected = {name for name, _, _ in m.final_comparison_specs({'A': 26, 'B': 35, 'C': 20})}
        self.assertTrue({'A_minus_C_validation_best_test', 'A_minus_C_validation_best_validation'} <= selected)

    def test_cached_early_report_cannot_be_mixed_with_a_different_final_runtime(self):
        point = dict(predictions_sha256='a' * 64, summary_sha256='b' * 64, checkpoint_sha256='c' * 64)
        report = dict(selected_epoch=35, fixed_epoch=35,
                      models={'B_R35_validation': dict(point), 'B_R35_test': dict(point)})
        m.verify_cached_points(report, 'B', {35: point}, {35: point}, 35)
        with self.assertRaisesRegex(ValueError, 'downloaded runtime'):
            m.verify_cached_points(report, 'B', {35: point}, {35: dict(point, checkpoint_sha256='d' * 64)}, 35)

    def test_import_does_not_add_torch_and_all_ready_requires_prior_branch_reports(self):
        before = 'torch' in sys.modules
        with patch.object(m, 'Context', return_value=SimpleNamespace(json=lambda path: m.read(path))):
            with self.assertRaises(FileNotFoundError):
                m.analyze_all(runs_root=self.root, dataset=self.root, references=self.root,
                              branch_reports=self.root / 'not_ready', output=self.root / 'final')
        self.assertEqual('torch' in sys.modules, before)
        self.assertFalse((self.root / 'final').exists())


if __name__ == '__main__': unittest.main()
