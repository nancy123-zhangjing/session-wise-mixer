"""Tiny CPU fixtures for recovery replay/provenance and paired cluster analysis."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analysis import analyze_h1_recovery as analysis
from scripts.listwise_diffgrm.analysis import recompute_row
from scripts.listwise_diffgrm.subset import file_sha256


def save(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def mean(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


class RecoveryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / 'tmp'
        self.temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='recovery_analysis_test_', dir=self.temp_root)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.source = [dict(sample_id=key, user_id=user, target_item_ids=targets)
                       for key, user, targets in [('a', 1, [1, 2, 3, 4, 5]), ('b', 1, [1, 2, 3, 4, 5]),
                                                  ('c', 2, [1, 2, 3, 4, 11])]]
        pq.write_table(pa.Table.from_pylist(self.source), self.dataset / 'validation.parquet')
        save(self.dataset / 'train_seen.json', dict(item_ids=list(range(1, 11))))
        self.binding = dict(files={name: dict(sha256=file_sha256(self.dataset / name))
                                  for name in ('validation.parquet', 'train_seen.json')})
        ranked = sorted((hashlib.sha256(row['sample_id'].encode()).hexdigest(), row['sample_id'], i)
                        for i, row in enumerate(self.source))[:2]
        self.manifest = dict(format='stage09_h1_recovery_subset_v1', count=2, population=3,
            validation_sha256=self.binding['files']['validation.parquet']['sha256'],
            selection='sha256_utf8_sample_id_then_sample_id',
            rows=[dict(index=i, sample_id=key, selection_sha256=digest) for digest, key, i in ranked])
        self.manifest_path = self.root / 'subset.json'
        save(self.manifest_path, self.manifest)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.temp_root.resolve()))
        self.temp.cleanup()

    def evaluation(self, directory, *, hits=(), subset=False, groups=False, r_updates=13860, t_updates=200):
        root = self.root / directory
        root.mkdir(parents=True)
        indices = [row['index'] for row in self.manifest['rows']] if subset else list(range(3))
        rows = []
        for index in indices:
            truth = self.source[index]
            targets = truth['target_item_ids']
            row = dict(sample_id=truth['sample_id'], dataset_index=index, user_id=truth['user_id'],
                       targets=targets, cold_target=11 in targets,
                       predictions=[targets if truth['sample_id'] in hits else [6, 7, 8, 9, 10]],
                       scores=[-1.0], beam_mask=[True], complete_mask=[True])
            row['metrics'] = recompute_row(row, catalog_size=20)
            rows.append(row)
        users = {}
        for row in rows:
            users.setdefault(row['user_id'], []).append(row['metrics'])
        meta = dict(split='validation', examples=len(rows), full_split=not subset,
            subset={k: v for k, v in self.manifest.items() if k != 'rows'} if subset else None,
            dataset_binding=self.binding, catalog_size=20, architecture='H', seed=2024, beam=32,
            inference_precision='fp32', r_updates=r_updates, t_updates=t_updates,
            metrics=mean([row['metrics'] for row in rows]), user_average=mean([mean(value) for value in users.values()]),
            metric_protocol=dict(list_length=5, aggregate_ks=[10, 20, 50], candidate_lists=32, pad_id=0,
                target_relevance='unordered_set_of_five_future_distinct_items',
                aggregation='list_score_order_then_item_position_first_occurrence'))
        if groups:
            grouped = {}
            for row in rows:
                grouped.setdefault('cold_target' if row['cold_target'] else 'seen_targets', []).append(row['metrics'])
            meta['groups'] = {key: dict(examples=len(value), metrics=mean(value)) for key, value in grouped.items()}
        save(root / 'summary.json', meta)
        path = root / 'per_sample.jsonl.gz'
        self.write_rows(path, rows)
        if not subset:
            wanted = {row['sample_id'] for row in self.manifest['rows']}
            values = [row['metrics'] for row in rows if row['sample_id'] in wanted]
            save(root / 'diagnostic_subset.json', dict(examples=len(values), metrics=mean(values)))
        return path

    @staticmethod
    def write_rows(path, rows):
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    @staticmethod
    def read_rows(path):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def test_full_without_groups_and_legacy_groups_recompute_sample_mean(self):
        for groups in (False, True):
            path = self.evaluation('full_' + str(groups), hits=('a', 'b'), groups=groups)
            output = self.root / ('recomputed_' + str(groups) + '.gz')
            report = analysis.audit_recovery_predictions(path, dataset_dir=self.dataset,
                subset_manifest=self.manifest, recomputed_path=output, require_extracted_subset=True)
            self.assertTrue(report['passed'])
            self.assertAlmostEqual(report['metrics']['list/ndcg@5'], 2 / 3)
            self.assertAlmostEqual(report['user_average']['list/ndcg@5'], .5)
            self.assertEqual(report['groups']['cold_target']['examples'], 1)
            self.assertEqual(len(self.read_rows(output)), 3)
            self.assertEqual(report['recomputed_rows_sha256'], file_sha256(output))

    def test_unsalted_subset_requires_full_manifest_and_exact_row_order(self):
        path = self.evaluation('subset', hits=('a',), subset=True)
        report = analysis.audit_recovery_predictions(path, dataset_dir=self.dataset, subset_manifest=self.manifest)
        self.assertTrue(report['passed'])
        with self.assertRaisesRegex(ValueError, 'exact unsalted manifest'):
            analysis.audit_recovery_predictions(path, dataset_dir=self.dataset)
        bad = copy.deepcopy(self.manifest)
        bad['salt'] = 'legacy_salt'
        with self.assertRaisesRegex(ValueError, 'unsalted source SHA'):
            analysis.audit_recovery_predictions(path, dataset_dir=self.dataset, subset_manifest=bad)
        self.write_rows(path, list(reversed(self.read_rows(path))))
        with self.assertRaisesRegex(ValueError, 'membership/order'):
            analysis.audit_recovery_predictions(path, dataset_dir=self.dataset, subset_manifest=self.manifest)

    def test_source_labels_user_cold_and_full_coverage_are_verified(self):
        for field, value in [('targets', [1, 2, 3, 4, 12]), ('user_id', 42), ('cold_target', True)]:
            with self.subTest(field=field):
                path = self.evaluation(field)
                rows = self.read_rows(path)
                rows[0][field] = value
                self.write_rows(path, rows)
                with self.assertRaisesRegex(ValueError, 'source'):
                    analysis.audit_recovery_predictions(path, dataset_dir=self.dataset)
        path = self.evaluation('omitted')
        self.write_rows(path, self.read_rows(path)[:2])
        meta = analysis.read_json(path.parent / 'summary.json')
        meta['examples'] = 2
        save(path.parent / 'summary.json', meta)
        with self.assertRaisesRegex(ValueError, 'omitted'):
            analysis.audit_recovery_predictions(path, dataset_dir=self.dataset)

    def test_numeric_tampering_and_full_extraction_fail_audit(self):
        for location in ('row', 'summary', 'extraction'):
            with self.subTest(location=location):
                path = self.evaluation(location)
                if location == 'row':
                    rows = self.read_rows(path)
                    rows[0]['metrics']['list/ndcg@5'] = .25
                    self.write_rows(path, rows)
                else:
                    target = path.parent / ('summary.json' if location == 'summary' else 'diagnostic_subset.json')
                    value = analysis.read_json(target)
                    value['metrics']['list/ndcg@5'] = .25
                    save(target, value)
                report = analysis.audit_recovery_predictions(path, dataset_dir=self.dataset,
                    subset_manifest=self.manifest, require_extracted_subset=True)
                self.assertFalse(report['passed'])
                self.assertGreater(report['mismatches'], 0)

    def test_three_paired_comparisons_keep_sample_mean_primary_and_fixed_bootstrap(self):
        r20 = self.evaluation('r20')
        decay = self.evaluation('decay', hits=('a', 'b'))
        freeze = self.evaluation('freeze', hits=('c',))
        result = analysis.compare_endpoints(r20, decay, freeze)
        first = result['decay_R26_minus_freeze_R26']
        self.assertEqual(first['bootstrap']['repeats'], 2000)
        self.assertEqual(first['bootstrap']['seed'], 2024)
        self.assertAlmostEqual(first['primary_result']['delta'], 1 / 3)
        self.assertAlmostEqual(first['user_average']['list/ndcg@5']['delta'], 0)
        self.assertEqual(first['primary_result']['confidence_interval'], [-1., 1.])
        self.assertAlmostEqual(result['decay_R26_minus_H1_R20']['primary_result']['delta'], 2 / 3)
        self.assertAlmostEqual(result['freeze_R26_minus_H1_R20']['primary_result']['delta'], 1 / 3)

    def run_artifacts(self, policy):
        root = self.root / ('run_' + policy)
        root.mkdir()
        save(root / 'fork.json', dict(protocol=dict(format='stage09_h1_recovery_v1', policy=policy,
            parent_sha256=analysis.PARENT_SHA256, start_phase=25, stop_phase=33),
            parent_hashes=dict(model='f' * 64), execution_binding=dict(code='same immutable source')))
        save(root / 'complete.json', dict(policy=policy, r_updates=18018,
            t_updates=280 if policy == 'decay' else 200, phase=33, endpoint_epoch=26))
        for label, r_updates, t_updates, full in analysis.expected_evaluations(policy):
            path = self.evaluation(f'run_{policy}/evaluations/{label}/attempt1',
                hits=('a', 'b') if policy == 'decay' else ('c',), subset=not full,
                r_updates=r_updates, t_updates=t_updates)
            save(path.parent.parent / 'complete.json', dict(attempt='attempt1',
                predictions_sha256=file_sha256(path), summary_sha256=file_sha256(path.parent / 'summary.json'),
                binding=dict(source='a' * 64, subset='b' * 64, state='c' * 64,
                             full=full, r_updates=r_updates, t_updates=t_updates)))
        return root

    def prior_artifacts(self):
        path = self.evaluation('historical_r20', groups=True)
        audit_path = self.root / 'prior_audit.json'
        audit = analysis.audit_recovery_predictions(path, dataset_dir=self.dataset)
        save(audit_path, audit)
        index_path = self.root / 'prior_index.json'
        hashes = dict(predictions_sha256=file_sha256(path), summary_sha256=file_sha256(path.parent / 'summary.json'),
                      audit_sha256=file_sha256(audit_path))
        save(index_path, dict(rows=[dict(epoch=20, roles=['H1:selected_best'], remote_path='/archived/H1/validation_R20',
                                       audit_sha256=hashes['audit_sha256'])]))
        return path, audit_path, index_path, hashes

    def test_complete_local_adapter_writes_17_audits_and_never_changes_inputs(self):
        decay, freeze = self.run_artifacts('decay'), self.run_artifacts('freeze')
        r20, audit, index, hashes = self.prior_artifacts()
        # An interrupted, later attempt must never replace the committed receipt.
        (decay / 'evaluations/R26/attempt2').mkdir()
        (decay / 'evaluations/R26/attempt2/per_sample.jsonl.gz').write_bytes(b'not accepted')
        before = {path: file_sha256(path) for path in self.root.rglob('*') if path.is_file()}
        output = self.root / 'analysis'
        with patch.multiple(analysis, FULL_ROWS=3, SUBSET_ROWS=2, R20_HASHES=hashes):
            result = analysis.analyze_recovery(decay_run=decay, freeze_run=freeze, dataset_dir=self.dataset,
                subset_path=self.manifest_path, output_root=output, r20_predictions=r20, r20_audit=audit, r20_index=index)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(list(output.rglob('audit.json'))), 17)
        self.assertEqual(len(list(output.rglob('recomputed_per_sample.jsonl.gz'))), 17)
        self.assertEqual(len(list((output / 'comparisons').glob('*.json'))), 3)
        self.assertEqual(before, {path: file_sha256(path) for path in before})

    def test_receipts_reject_tampering_preflight_and_path_traversal(self):
        for case in ('bytes', 'preflight', 'attempt'):
            with self.subTest(case=case):
                root = self.run_artifacts('freeze')
                if case == 'bytes':
                    (root / 'evaluations/R26/attempt1/per_sample.jsonl.gz').write_bytes(b'changed')
                elif case == 'preflight':
                    path = root / 'complete.json'
                    value = analysis.read_json(path)
                    value['preflight_only'] = True
                    save(path, value)
                else:
                    path = root / 'evaluations/initial_R20/complete.json'
                    value = analysis.read_json(path)
                    value['attempt'] = '../other'
                    save(path, value)
                with patch.multiple(analysis, FULL_ROWS=3, SUBSET_ROWS=2):
                    with self.assertRaises(ValueError):
                        analysis.collect_evaluations(root, 'freeze')
                # Keep test artifacts for inspection until workspace-scoped teardown.
                root.rename(self.root / ('rejected_' + case))

    def test_historical_r20_is_bound_to_accepted_index_and_exact_bytes(self):
        r20, audit, index, hashes = self.prior_artifacts()
        with patch.multiple(analysis, FULL_ROWS=3, R20_HASHES=hashes):
            self.assertEqual(analysis.verify_r20(r20, audit, index)['predictions_sha256'], hashes['predictions_sha256'])
            value = analysis.read_json(index)
            value['rows'][0]['epoch'] = 26
            save(index, value)
            with self.assertRaisesRegex(ValueError, 'accepted source/audit index'):
                analysis.verify_r20(r20, audit, index)


if __name__ == '__main__':
    unittest.main()
