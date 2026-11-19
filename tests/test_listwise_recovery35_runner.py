"""Bounded runner/artifact tests: tiny CPU state, mocked evaluation, no GPU/data."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm import recovery35_artifacts as art
from scripts.listwise_diffgrm import recovery35_run as run
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.training import cpu_copy, restore_rng, rng_state, sha256


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class TinyAlignment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('hard_codes', torch.tensor([[-1], [0]]))
        self.cache = {'marker': torch.tensor(1.)}


class TinyTrainer:
    def __init__(self, root):
        self.model = torch.nn.Linear(1, 1)
        self.alignment = TinyAlignment()
        self.device = torch.device('cpu')
        self.r_updates, self.t_updates = 24254, 200
        self.config = dict(seed=2024, dataset=str(root), training=dict(
            effective_batch_size=4, microbatch_size=2, checkpoint_every_updates=100))
        self.plan = [dict(kind='R', epoch=i, updates=1) for i in range(45)]
        self.plan[44] = dict(kind='R', epoch=35, updates=1)


class TinyController:
    """A single final phase; controller numerical semantics have separate tests."""
    def __init__(self, root, branch='A'):
        self.branch, self.trainer = branch, TinyTrainer(root)
        self.start_phase, self.stop_phase = 44, 45
        self.progress = dict(phase=44, update=0, boundary_prepared=False)
        self.protocol = {'synthetic_runner_fixture': True, 'branch': branch}
        self.protocol_sha256 = 'a' * 64
        self.execution_binding = {'data': {'files': {'validation.parquet': {'sha256': sha256(root / 'validation.parquet')}}}}
        self.steps = self.accepted = 0
        self.restores = []

    @property
    def frozen(self): return True

    @property
    def needs_replay_gate(self): return False

    @property
    def done(self): return self.progress['phase'] == self.stop_phase

    @property
    def row(self): return self.trainer.plan[self.progress['phase']]

    def counts_at(self, progress):
        return {'R': 24255 if progress['phase'] >= 45 else 24254 + progress['update'], 'T': 200}

    def validate_clocks(self, inner):
        assert {'R': inner['r_updates'], 'T': inner['t_updates']} == self.counts_at(inner['progress'])

    def checkpoint(self):
        return cpu_copy(dict(protocol=self.protocol, inner=dict(
            model=self.trainer.model.state_dict(), alignment=self.trainer.alignment.state_dict(),
            r_optimizer={'clock': self.trainer.r_updates}, t_optimizer={'clock': self.trainer.t_updates},
            scheduler={'last_epoch': self.trainer.r_updates}, rng=rng_state(), binding=self.execution_binding,
            progress=self.progress, r_updates=self.trainer.r_updates, t_updates=self.trainer.t_updates)))

    def restore(self, saved):
        self.restores.append(copy.deepcopy(saved))
        inner = saved['inner']; self.progress = copy.deepcopy(inner['progress'])
        self.trainer.model.load_state_dict(inner['model'])
        self.trainer.alignment.load_state_dict(inner['alignment'])
        self.trainer.r_updates, self.trainer.t_updates = inner['r_updates'], inner['t_updates']
        restore_rng(inner['rng'])

    def step(self, batches, kind):
        assert kind == 'R' and not self.progress['boundary_prepared']
        self.steps += 1; self.progress['update'] += 1; self.trainer.r_updates += 1
        with torch.no_grad(): self.trainer.model.weight.add_(.1)
        return dict(loss=1., r_updates=self.trainer.r_updates, t_updates=self.trainer.t_updates)

    def prepare_boundary(self):
        assert self.progress['update'] == 1
        self.progress['boundary_prepared'] = True

    def complete_phase(self):
        assert self.progress['boundary_prepared']
        self.progress = dict(phase=self.progress['phase'] + 1, update=0, boundary_prepared=False)


class TinyReplayController(TinyController):
    """Only exercise runner transitions R20 -> gate -> skipped T -> final R."""
    def __init__(self, root):
        super().__init__(root, 'C')
        self.start_phase, self.stop_phase = 24, 27
        self.progress = dict(phase=24, update=0, boundary_prepared=False)
        self.trainer.r_updates = 13859
        self.trainer.plan[24] = dict(kind='R', epoch=20, updates=1)
        self.trainer.plan[25] = dict(kind='T', epoch=20, updates=40)
        self.trainer.plan[26] = dict(kind='R', epoch=35, updates=1)

    @property
    def frozen(self): return self.progress['phase'] >= 25

    @property
    def needs_replay_gate(self): return self.progress['phase'] == 25 and self.accepted == 0

    def counts_at(self, progress):
        phase = progress['phase']
        r = (13859 + progress['update'] if phase == 24 else
             24255 if phase == 27 or (phase == 26 and progress['update']) else 13860)
        return {'R': r, 'T': 200}

    def step(self, batches, kind):
        assert not self.needs_replay_gate and kind == 'R'
        self.steps += 1; self.progress['update'] += 1
        self.trainer.r_updates = self.row['epoch'] * 693
        return dict(loss=1., r_updates=self.trainer.r_updates, t_updates=200)

    def prepare_boundary(self):
        assert not self.needs_replay_gate
        assert self.progress['update'] == (0 if self.row['kind'] == 'T' else 1)
        self.progress['boundary_prepared'] = True

    def accept_replay_gate(self, receipt):
        assert self.needs_replay_gate and self.progress['update'] == 0
        assert receipt['checkpoint_state_sha256'] == fingerprint(self.checkpoint()['inner'])
        assert receipt['protocol_sha256'] == self.protocol_sha256
        self.accepted += 1


def fake_evaluate(trainer, assets, *, split, destination, beam):
    destination = Path(destination); destination.mkdir(parents=True)
    summary = dict(split=split, examples=71657 if split == 'validation' else 53194,
                   full_split=True, beam=beam, inference_precision='fp32', architecture='H', seed=2024,
                   r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                   metrics={'list/ndcg@5': .02})
    dump(destination / 'summary.json', summary)
    with gzip.open(destination / 'per_sample.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.write('{}\n')
    return summary


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / 'tmp/recovery35_runner'
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temporary_root); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'validation.parquet').write_bytes(b'synthetic, not a parquet dataset')
        self.parent = self.root / 'parent.pt'; self.parent.write_bytes(b'synthetic source')
        self.refs = self.root / 'references.json'; dump(self.refs, {})
        self.output = self.root / 'arm'

    def args(self, **changes):
        result = dict(branch='A', parent=str(self.parent), ancestor=None, device='cuda:0',
                      dataset=None, assets=None, output=str(self.output), references=str(self.refs), resume=None)
        return argparse.Namespace(**dict(result, **changes))

    def lock_request(self, branch='A'):
        dump(self.output / 'request.json', dict(branch=branch, parent=str(self.parent),
             parent_sha256=sha256(self.parent), ancestor=None,
             reference_manifest_sha256=sha256(self.refs),
             protocol={'synthetic_runner_fixture': True, 'branch': branch},
             replay_rule=art.REPLAY_RULE, trajectory_rule=run.TRAJECTORY_RULE,
             validation_epochs=list(art.VALIDATION_EPOCHS[branch])))

    def patches(self, controller):
        def reference(manifest, branch, epoch, **kwargs):
            return ({'epoch': epoch, 'checkpoint_sha256': 'b' * 64},
                    {'metrics': {'list/ndcg@5': .001}, 'r_updates': epoch * 693})
        return patch.multiple(run, build_recovery35=lambda *a, **k: (controller, None, list(range(4))),
            authenticate_references=lambda *a: None, phase_indices=lambda *a: [0, 1, 2, 3],
            microbatches=lambda *a: [None], evaluate=fake_evaluate, reference_record=reference,
            record_phase_order=lambda *a: {'status': 'passed', 'synthetic_runner_fixture': True},
            audit_phase=lambda *a: {'status': 'passed', 'synthetic_runner_fixture': True})

    def test_common_grid_selection_tie_earlier_even_when_records_include_epoch(self):
        records = {str(e): dict(epoch=e, ndcg5=.01, kind='new') for e in art.BEST_EPOCHS}
        self.assertEqual(art.select_best(records)['epoch'], 20)
        records['26']['ndcg5'] = .02
        self.assertEqual(art.select_best(records)['epoch'], 26)
        del records['23']
        with self.assertRaises(ValueError): art.select_best(records)

    def test_boundary_is_immutable_and_coverage_detects_missing_stage(self):
        c = TinyController(self.root); self.output.mkdir()
        initial = art.save_boundary(c, self.output, 'initial.pt')
        self.assertEqual(art.save_boundary(c, self.output, 'initial.pt'), initial)
        c.step([], 'R'); c.prepare_boundary(); c.complete_phase()
        with self.assertRaises(FileExistsError): art.save_boundary(c, self.output, 'initial.pt')
        with self.assertRaises(FileNotFoundError): art.audit_checkpoints(c, self.output)
        art.save_boundary(c, self.output, 'phase44_R_R35_T200.pt')
        coverage = art.audit_checkpoints(c, self.output)
        self.assertEqual((coverage['expected'], coverage['present']), (2, 2))
        with self.assertRaises(ValueError): art.save_boundary(c, self.output, '../escape.pt')

    def test_evaluation_preserves_rng_modes_buffers_cache_and_reuses_without_forward(self):
        c = TinyController(self.root); c.step([], 'R'); c.prepare_boundary()
        before = fingerprint(c.checkpoint()); old_cache = c.trainer.alignment.cache['marker'].clone()
        def evaluate(*args, **kwargs):
            c.trainer.alignment.hard_codes.fill_(8)
            c.trainer.alignment.cache['marker'].add_(9)
            torch.rand(11)
            return fake_evaluate(*args, **kwargs)
        with patch.object(run, 'evaluate', side_effect=evaluate) as mocked:
            result = run.evaluation(c, None, self.output, 'validation', 35)
            reused = run.evaluation(c, None, self.output, 'validation', 35)
            self.assertEqual(mocked.call_count, 1)
        self.assertEqual(result, reused)
        self.assertEqual(fingerprint(c.checkpoint()), before)
        self.assertTrue(torch.equal(c.trainer.alignment.cache['marker'], old_cache))

    def test_partial_evaluation_exception_is_retried_in_fresh_attempt(self):
        c = TinyController(self.root); c.step([], 'R'); c.prepare_boundary()
        before = fingerprint(c.checkpoint())
        def partial(trainer, assets, *, destination, **kwargs):
            Path(destination).mkdir(parents=True); (Path(destination) / 'partial').write_text('partial')
            trainer.alignment.hard_codes.fill_(8); torch.rand(3)
            raise OSError('simulated validation interruption')
        with patch.object(run, 'evaluate', side_effect=partial):
            with self.assertRaises(OSError): run.evaluation(c, None, self.output, 'validation', 35)
        self.assertEqual(fingerprint(c.checkpoint()), before)
        folder = self.output / 'evaluations/validation_R35'
        self.assertFalse((folder / 'complete.json').exists())
        with patch.object(run, 'evaluate', side_effect=fake_evaluate):
            record = run.evaluation(c, None, self.output, 'validation', 35)
        self.assertIn('attempt2', record['summary'])
        self.assertTrue((folder / 'attempt1/partial').is_file())

    def test_invalid_evaluation_protocol_never_publishes_complete(self):
        c = TinyController(self.root); c.step([], 'R'); c.prepare_boundary()
        def wrong(*args, **kwargs):
            result = fake_evaluate(*args, **kwargs); result['inference_precision'] = 'bf16'
            dump(Path(kwargs['destination']) / 'summary.json', result)
        with patch.object(run, 'evaluate', side_effect=wrong):
            with self.assertRaises(ValueError): run.evaluation(c, None, self.output, 'validation', 35)
        self.assertFalse((self.output / 'evaluations/validation_R35/complete.json').exists())

    def test_prepared_final_boundary_resume_does_not_repeat_training_or_early_milestone(self):
        c = TinyController(self.root)
        with self.patches(c), patch.object(run, 'evaluate', side_effect=OSError('validation interrupted')), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError): run.train(self.args())
        self.assertEqual(c.steps, 1)
        early = self.output / 'milestones/training_finished.json'
        early_sha = sha256(early)
        self.assertTrue(json.loads(early.read_text())['validation_pending'])
        saved = torch.load(self.output / 'last.pt', map_location='cpu', weights_only=False)
        self.assertTrue(saved['inner']['progress']['boundary_prepared'])
        resumed = TinyController(self.root)
        with self.patches(resumed), redirect_stdout(io.StringIO()):
            run.train(self.args(resume=str(self.output / 'last.pt')))
        self.assertEqual(resumed.steps, 0)
        self.assertEqual(sha256(early), early_sha)
        self.assertTrue((self.output / 'milestones/validation_finished.json').is_file())
        self.assertTrue((self.output / 'phase44_R_R35_T200.pt').is_file())
        self.assertEqual(json.loads((self.output / 'CHECKPOINT_COVERAGE.json').read_text())['status'], 'passed')

    def test_resume_rejects_changed_reference_manifest_before_advancing(self):
        c = TinyController(self.root)
        with self.patches(c), patch.object(run, 'evaluate', side_effect=OSError('stop')), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError): run.train(self.args())
        dump(self.refs, {'changed_after_start': True})
        resumed = TinyController(self.root)
        with self.patches(resumed), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, '[Rr]eference|[Rr]equest'):
                run.train(self.args(resume=str(self.output / 'last.pt')))
        self.assertEqual(resumed.steps, 0)

    def test_best_test_existing_reference_reused_and_fixed_r35_runs_once(self):
        self.output.mkdir()
        checkpoint = self.output / 'R35.pt'; torch.save({'synthetic': True}, checkpoint)
        fixed = dict(kind='new', epoch=35, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        selected = dict(kind='reference', epoch=20, checkpoint_sha256='a' * 64)
        dump(self.refs, {'test': {'A': {'20': {'synthetic_cached_reference': True}}}})
        dump(self.output / 'validation_records/R35.json', fixed)
        dump(self.output / 'selection.json', dict(branch='A', selected=selected))
        dump(self.output / 'complete.json', dict(status='training_and_validation_complete', branch='A',
             selection_sha256=sha256(self.output / 'selection.json')))
        self.lock_request()
        c = TinyController(self.root); c.trainer.r_updates = 35 * 693
        c.restore = lambda state: None
        cached = dict(checkpoint_sha256='a' * 64, summary='existing', predictions='existing')
        with patch.object(run, 'build_recovery35', return_value=(c, None, None)), \
             patch.object(run, 'evaluation', return_value={'epoch': 35}) as evaluate, \
             patch.object(run, 'reference_record', return_value=(cached, {})) as reference, redirect_stdout(io.StringIO()):
            run.test(self.args())
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(reference.call_args.kwargs['split'], 'test')
        records = json.loads((self.output / 'test_predictions_complete.json').read_text())
        self.assertEqual([x['epoch'] for x in records['results']], [35, 20])
        self.assertTrue(records['results'][1]['reused'])
        self.assertFalse(records['independently_audited'])
        event = json.loads((self.output / 'milestones/test_predictions_finished.json').read_text())
        self.assertTrue(event['independent_audit_pending'])

    def test_best_equal_r35_deduplicates_test_and_b_never_uses_best_grid(self):
        for branch, selected_epoch in [('A', 35), ('B', 32)]:
            with self.subTest(branch=branch):
                self.output = self.root / f'arm_{branch}'
                self.output.mkdir(); checkpoint = self.output / 'R35.pt'; torch.save({}, checkpoint)
                fixed = dict(kind='new', epoch=35, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
                dump(self.output / 'validation_records/R35.json', fixed)
                dump(self.output / 'selection.json', dict(branch=branch, selected=dict(epoch=selected_epoch)))
                dump(self.output / 'complete.json', dict(status='training_and_validation_complete', branch=branch,
                     selection_sha256=sha256(self.output / 'selection.json')))
                self.lock_request(branch)
                c = TinyController(self.root, branch); c.trainer.r_updates = 35 * 693; c.restore = lambda state: None
                with patch.object(run, 'build_recovery35', return_value=(c, None, None)), \
                     patch.object(run, 'evaluation', return_value={'epoch': 35}) as evaluate, \
                     patch.object(run, 'reference_record', side_effect=AssertionError('Unexpected extra test')), redirect_stdout(io.StringIO()):
                    run.test(self.args(branch=branch))
                self.assertEqual(evaluate.call_count, 1)

    def test_uncached_historical_a_best_is_evaluated_via_authenticated_legacy_loader(self):
        self.output.mkdir(); checkpoint = self.output / 'R35.pt'; torch.save({}, checkpoint)
        fixed = dict(kind='new', epoch=35, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        historical = self.root / 'old_R23.pt'; historical.write_bytes(b'synthetic legacy outer')
        selected = dict(kind='reference', epoch=23, checkpoint=str(historical), checkpoint_sha256=sha256(historical))
        dump(self.output / 'validation_records/R35.json', fixed)
        dump(self.output / 'selection.json', dict(branch='A', selected=selected))
        dump(self.output / 'complete.json', dict(status='training_and_validation_complete', branch='A',
             selection_sha256=sha256(self.output / 'selection.json')))
        self.lock_request()
        c = TinyController(self.root); c.trainer.r_updates = 35 * 693; c.restore = lambda state: None
        historical_controller = TinyController(self.root); historical_controller.trainer.r_updates = 23 * 693
        with patch.object(run, 'build_recovery35', return_value=(c, None, None)), \
             patch.object(run, 'historical_h1_controller', return_value=(historical_controller, None)) as loader, \
             patch.object(run, 'evaluation', side_effect=lambda controller, assets, output, split, epoch, **k: {'epoch': epoch}) as evaluate, \
             patch.object(run, 'reference_record', side_effect=AssertionError('Missing cache must be evaluated')), redirect_stdout(io.StringIO()):
            run.test(self.args())
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[1], selected)
        self.assertEqual([c.args[4] for c in evaluate.call_args_list], [35, 23])

    def test_reference_record_rejects_wrong_precision(self):
        summary = self.root / 'summary.json'; predictions = self.root / 'predictions.gz'
        predictions.write_bytes(b'synthetic')
        dump(summary, dict(split='validation', full_split=True, beam=32, examples=71657,
                           inference_precision='bf16', architecture='H', seed=2024))
        record = dict(summary=str(summary), predictions=str(predictions),
                      summary_sha256=sha256(summary), predictions_sha256=sha256(predictions))
        with self.assertRaises(ValueError):
            art.reference_record({'validation': {'A': {'20': record}}}, 'A', 20)

    def test_c_replay_gate_is_accepted_before_skipping_t_and_never_reseeds_skipped_t(self):
        actual_lock = run.run_lock
        @contextmanager
        def locked(*args, **kwargs):
            with actual_lock(*args, **kwargs):
                for epoch in (8, 11, 14, 17):
                    dump(self.output / 'replay' / f'R{epoch:02d}.json',
                         dict(status='passed', epoch=epoch, rule=art.REPLAY_RULE))
                yield
        def replay(controller, output, epoch, candidate, refs):
            result = dict(status='passed', branch='C', epoch=epoch, rule=art.REPLAY_RULE,
                          validation_sha256='b' * 64, reference_prediction_sha256='c' * 64,
                          candidate_prediction_sha256=candidate['predictions_sha256'])
            art.write_once(Path(output) / 'replay' / f'R{epoch:02d}.json', result)
            return result
        for already_prepared in (False, True):
            with self.subTest(already_prepared=already_prepared):
                self.output = self.root / f'replay_{already_prepared}'
                c = TinyReplayController(self.root)
                if already_prepared:
                    c.progress.update(update=1, boundary_prepared=True)
                    c.trainer.r_updates = 13860
                audited = []
                def trajectory(controller, output, phase):
                    self.assertTrue(controller.progress['boundary_prepared'])
                    pending = torch.load(Path(output) / 'last.pt', map_location='cpu', weights_only=False)
                    self.assertEqual(pending['inner']['progress'], controller.progress)
                    audited.append(phase)
                    return {'status': 'passed', 'synthetic_runner_fixture': True}
                with self.patches(c), patch.object(run, 'run_lock', side_effect=locked), \
                     patch.object(run, 'replay_check', side_effect=replay), \
                     patch.object(run, 'audit_phase', side_effect=trajectory), \
                     patch.object(run, 'record_phase_order', return_value={'status': 'passed'}) as observed, \
                     patch.object(run, 'phase_indices', return_value=[0, 1, 2, 3]) as order, redirect_stdout(io.StringIO()):
                    run.train(self.args(branch='C'))
                self.assertEqual(c.accepted, 1)
                self.assertEqual(c.steps, 1 if already_prepared else 2)
                self.assertEqual(audited, [24])  # Prepared resume MUST NOT skip the trajectory gate.
                self.assertEqual([call.args[2] for call in order.call_args_list], [26] if already_prepared else [24, 26])
                self.assertEqual(observed.call_count, 0 if already_prepared else 1)
                self.assertTrue((self.output / 'R20_pre_freeze.pt').is_file())
                self.assertTrue((self.output / 'phase25_T_R20_T200.pt').is_file())
                self.assertTrue((self.output / 'milestones/R20_freeze_accepted.json').is_file())


if __name__ == '__main__':
    unittest.main()
