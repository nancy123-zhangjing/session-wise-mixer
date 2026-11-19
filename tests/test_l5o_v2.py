"""Protocol v2: pass-based budgets, in-run freezes, smoothed selection, TIGER fast paths."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))

from scripts.kuairec_data import l5o_v2
from scripts.kuairec_data.l5o_v2_job import JOBS, RERUNS
from scripts.kuairec_data.selection import raw_choice, smoothed_choice


def same_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def fake_receipt(root):
    receipt = {'passed': True, 'items_sha256': '0'*64,
               'test': {'1.0': {'samples': 7176, 'imp_hit@5': .41, 'ev_ndcg@5': .109}},
               'selection': {'filter_history': True,
                             'binding': {'source': {'x': '1'}, 'calibration': '2'*64}}}
    path = Path(root)/'COMPLETE.json'
    path.write_text(json.dumps(receipt))
    return path


class SelectionTest(unittest.TestCase):
    def test_smoothed_moving_average_edges_and_ties(self):
        points = [('a', 0.1), ('b', 0.5), ('c', 0.2), ('d', 0.4), ('e', 0.4)]
        node, smoothed, raw, values = smoothed_choice(points)
        self.assertEqual([round(v, 6) for v in values], [0.3, round(0.8/3, 6), round(1.1/3, 6), round(1.0/3, 6), 0.4])
        self.assertEqual((node, raw), ('e', 0.4))
        self.assertEqual(smoothed_choice([('x', 1.0), ('y', 1.0)])[0], 'x')      # tie -> earlier
        self.assertEqual(raw_choice(points), ('b', 0.5))
        with self.assertRaises(ValueError):
            smoothed_choice([('x', float('nan'))])


class BudgetTest(unittest.TestCase):
    def test_v2_budgets_follow_old_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = fake_receipt(tmp)
            h = l5o_v2.build('h', 'H0', 2024, receipt=receipt, dataset='D', assets='A')
            self.assertNotIn('r_updates_per_unit', h['training'])
            self.assertEqual(l5o_v2.budgets('h', h), dict(updates_per_pass=2235, passes=35, r_updates=78225,
                                                          t_updates=200, freeze_after_pass=20, validations=35))
            self.assertTrue(h['evaluation']['filter_history'])
            t = l5o_v2.build('tiger', 'T1A', 2026, receipt=receipt)
            b = l5o_v2.budgets('tiger', t)
            self.assertEqual((b['batch'], b['warmup']), (256, 10000))
            self.assertAlmostEqual(b['lr'], 0.01)
            self.assertAlmostEqual(b['passes'], l5o_v2.OLD_TIGER_PASSES, places=4)
            self.assertEqual(b['validation_every'], math.ceil(math.ceil(2288120*5/256)/2))
            self.assertEqual((t['arm'], t['seed'], t['execution']['precision'], t['model']['attention']),
                             ('AR-T1A', 2026, 'tf32', 'eager'))
            c = l5o_v2.build('cf', 'CF-M1', 2026, receipt=receipt)
            self.assertNotIn('r_updates_per_unit', c['training'])
            self.assertEqual(l5o_v2.budgets('cf', c)['t_updates'], 40)
            s = l5o_v2.build('sasrec', 'SASREC', 2025)
            self.assertEqual(s['evaluation'], {'filter_history': True, 'selection_rule': 'smoothed3'})
            with self.assertRaises(ValueError):
                l5o_v2.build('sasrec', 'SASREC', 2027)

    def test_jobs_fit_quota_rows_and_pair_arms(self):
        for name, slots in JOBS.items():
            self.assertIn(len(slots), (1, 2, 4, 8))
            if name in RERUNS:
                self.assertTrue(all(slot in [s for n, ss in JOBS.items() if n not in RERUNS for s in ss] for slot in slots))
                continue
            if not name.startswith('sasrec'):
                seeds = {}
                for family, arm, seed in slots:
                    seeds.setdefault(seed, set()).add(arm)
                self.assertTrue(all(len(arms) == 2 for arms in seeds.values()))
        covered = sorted((f, a, s) for name, slots in JOBS.items() if name not in RERUNS for f, a, s in slots)
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(len(covered), 3*2*3 + 3)


def _lock(args):
    from scripts.kuairec_data.l5o_v2_worker import locked_config
    root, barrier_path, cfg = args
    while not Path(barrier_path).exists():
        pass
    return str(locked_config(Path(root), 'h_2024', cfg))


class OldSplitControlTest(unittest.TestCase):
    """2026-11-19 step 1: current code on the old split; only the v2 execution template and the
    in-run R20 freeze differ from the accepted old freeze configs."""

    def test_reference_is_the_accepted_old_config_and_only_execution_and_freeze_change(self):
        exp = ROOT/'experiments/09_kuairec_listwise'
        for seed in (2025, 2026):
            accepted = json.loads((exp/f'multiseed_freeze_20261109_v1/config_seed{seed}.json').read_text())
            self.assertEqual(l5o_v2.h_old_reference(seed), accepted)
        cfg = l5o_v2.build('h', 'H1', 2024, h_split='old')
        self.assertEqual(cfg, l5o_v2.build('h', 'H0', 2024, h_split='old'))
        ref = l5o_v2.h_old_reference(2024)
        self.assertEqual({k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k)},
                         {'execution', 'training', 'evaluation', 'h_split_control'})
        self.assertEqual({k for k in cfg['training'] if cfg['training'][k] != ref['training'].get(k)},
                         {'microbatch_size', 'freeze_after_epoch', 'checkpoint_every_updates'})
        self.assertEqual({k for k in cfg['evaluation'] if cfg['evaluation'][k] != ref['evaluation'][k]}, {'batch_size'})
        self.assertFalse(cfg['evaluation']['filter_history'])
        self.assertEqual((cfg['evaluation']['selection'], cfg['history_max']), ('ndcg@5', 50))
        self.assertNotIn('r_updates_per_unit', cfg['training'])
        self.assertNotIn('validate_every_r_unit', cfg['training'])
        with self.assertRaises(ValueError):
            l5o_v2.build('h', 'H0', 2024, h_split='old', h_rhythm='old693')
        with self.assertRaises(ValueError):
            l5o_v2.build('tiger', 'T0A', 2024, h_split='old')


class LockedConfigTest(unittest.TestCase):
    def test_concurrent_arms_lock_the_same_config(self):
        import multiprocessing as mp
        cfg = {'seed': 2024, 'arms': ['H0', 'H1'], 'x': list(range(2000))}
        with tempfile.TemporaryDirectory() as tmp:
            go = Path(tmp)/'go'
            with mp.get_context('spawn').Pool(8) as pool:
                pending = pool.map_async(_lock, [(tmp, go, cfg)]*8)
                go.write_text('1')
                paths = pending.get(60)
            self.assertEqual(len(set(paths)), 1)
            self.assertEqual(json.loads(Path(paths[0]).read_text()), cfg)
            self.assertEqual(sorted(p.name for p in (Path(tmp)/'configs').iterdir()), ['h_2024.json'])
            from scripts.kuairec_data.l5o_v2_worker import locked_config
            with self.assertRaises(ValueError):
                locked_config(Path(tmp), 'h_2024', dict(cfg, seed=2025))


class HFreezeTest(unittest.TestCase):
    def test_plan_skips_tokenizer_blocks_after_pass_20(self):
        from scripts.listwise_diffgrm.training import phase_plan, nominal_t_updates, expected_update_counts
        cfg = json.loads((ROOT/'configs/09_kuairec_listwise/l5o.json').read_text())
        cfg['training'].pop('r_updates_per_unit')
        cfg['training']['freeze_after_epoch'] = 20
        plan = phase_plan(cfg, 2288120)
        self.assertEqual({r['updates'] for r in plan if r['kind'] == 'R'}, {2235})
        t_rows = [(r['epoch'], r['updates'], r.get('frozen', False)) for r in plan if r['kind'] == 'T']
        self.assertEqual(t_rows, [(e, 40, False) for e in (5, 8, 11, 14, 17)] +
                                 [(e, 0, True) for e in (20, 23, 26, 29, 32)])
        self.assertEqual(nominal_t_updates(cfg), 400)
        self.assertEqual(expected_update_counts(plan, dict(phase=len(plan), update=0)), {'R': 78225, 'T': 200})
        legacy = json.loads((ROOT/'configs/09_kuairec_listwise/l5o.json').read_text())
        self.assertTrue(all(r['updates'] == 40 for r in phase_plan(legacy, 2288120) if r['kind'] == 'T'))
        bad = copy.deepcopy(cfg); bad['training']['freeze_after_epoch'] = 99
        with self.assertRaises(ValueError):
            phase_plan(bad, 2288120)

    def test_in_run_freeze_end_to_end_on_cpu(self):
        from test_listwise_l5o import L5OIntegration
        from scripts.listwise_diffgrm import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = L5OIntegration.setup_fixture(None, root)
            cfg['training'].update(cycles=3, freeze_after_epoch=2)
            cfg['execution']['keep_validated_checkpoints'] = True
            path = root/'cfg.json'; path.write_text(json.dumps(cfg))
            run.main(['--config', str(path), 'warmup', '--output', str(root/'w')])
            run.main(['--config', str(path), 'train', '--arm', 'H1', '--output', str(root/'h1'),
                      '--shared-warmup', str(root/'w'/'last.pt')])
            last = torch.load(root/'h1'/'last.pt', map_location='cpu', weights_only=False)
            frozen = torch.load(root/'h1'/'checkpoints'/'R02.pt', map_location='cpu', weights_only=False)
            self.assertEqual((last['t_updates'], last['r_updates']), (1, 4))
            self.assertEqual(int(last['alignment']['diger_total_optimizer_steps']), 3)
            for key, value in frozen['alignment'].items():
                self.assertTrue(torch.equal(value, last['alignment'][key]), key)
            self.assertTrue(same_tree(frozen['t_optimizer'], last['t_optimizer']))
            # The final state restores (clock checks) and evaluates.
            run.main(['--config', str(path), 'evaluate', '--checkpoint', str(root/'h1'/'last.pt'),
                      '--split', 'test', '--output', str(root/'test')])
            self.assertTrue((root/'test'/'summary.json').is_file())


class CFFreezeTest(unittest.TestCase):
    def test_in_run_freeze_stops_tokenizer_blocks(self):
        from test_rqcd_mixer import SyntheticData, fixture, tiny_config
        counts = {}
        for freeze in (None, 2):
            cfg = tiny_config()
            cfg['training']['epochs'] = 4
            if freeze is not None:
                cfg['training']['freeze_after_epoch'] = freeze
            cfg, _, trainer = fixture(cfg=cfg)
            data, seen, snapshot = SyntheticData(cfg), [], None
            while trainer.phase != 'done':
                event = trainer.advance(data)
                if event['kind'] == 'R_epoch':
                    seen.append((event['R'], event['T']))
                    if event['R'] == 2:
                        snapshot = copy.deepcopy(trainer.tokenizer.state_dict())
            counts[freeze] = seen
            if freeze is not None:
                self.assertEqual(trainer.catalog.version, 1)
                for key, value in snapshot.items():
                    self.assertTrue(torch.equal(value, trainer.tokenizer.state_dict()[key]), key)
                _, _, other = fixture(cfg=copy.deepcopy(cfg))
                other.restore(trainer.state())
        self.assertEqual(counts[None], [(1, 0), (2, 2), (3, 4), (4, 6)])
        self.assertEqual(counts[2], [(1, 0), (2, 2), (3, 2), (4, 2)])


class TigerFastPathTest(unittest.TestCase):
    def build(self, attention, dropout=0.0):
        from test_session_ar_core import tiny
        from scripts.session_wise_ar.model import ARRecommender
        cfg = tiny('AR-T0')
        cfg['model'].update(attention=attention, dropout=dropout)
        torch.manual_seed(3)
        return cfg, ARRecommender(cfg)

    def batch(self):
        context = torch.tensor([[0, 3, 4, 5], [0, 0, 7, 2], [1, 2, 3, 4]])
        return {'context': context, 'target': torch.tensor([6, 9, 11]), 'users': torch.tensor([1, 5, 7])}

    def test_sdpa_training_matches_eager_loss_and_gradients(self):
        from scripts.session_wise_ar.catalog import add_suffix
        sid = add_suffix(torch.randint(0, 8, (12, 3), generator=torch.Generator().manual_seed(1)), 8)
        _, eager = self.build('eager')
        _, sdpa = self.build('sdpa')
        sdpa.load_state_dict(eager.state_dict())
        eager.train(); sdpa.train()
        a, b = eager.hard_loss(self.batch(), sid), sdpa.hard_loss(self.batch(), sid)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        a.backward(); b.backward()
        for (name, p), q in zip(eager.named_parameters(), sdpa.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)
        # Evaluation keeps the eager kernels bit for bit.
        eager.eval(); sdpa.eval()
        hidden, mask = eager.encode(self.batch()['context'], sid, self.batch()['users'])
        prefix = torch.zeros((3, 1), dtype=torch.long)
        self.assertTrue(torch.equal(eager.next_logprobs(hidden, mask, prefix), sdpa.next_logprobs(hidden, mask, prefix)))
        # Fused CUDA kernels reject a key-axis stride != 1 (T5's bias is a permute).
        seen = []
        original = torch.nn.functional.scaled_dot_product_attention
        def spy(*args, **kwargs):
            seen.append(kwargs['attn_mask'].stride(-1))
            return original(*args, **kwargs)
        from unittest.mock import patch
        sdpa.train()
        with patch('scripts.session_wise_ar.model.F.scaled_dot_product_attention', side_effect=spy):
            sdpa.hard_loss(self.batch(), sid)
        self.assertTrue(seen and all(s == 1 for s in seen))
        _, dropped = self.build('sdpa', dropout=0.1)
        dropped.train()
        self.assertTrue(torch.isfinite(dropped.hard_loss(self.batch(), sid)))

    def test_tf32_scope_restores_strict_fp32_flags(self):
        from test_session_ar_core import tiny
        from scripts.session_wise_ar import runtime
        cfg = tiny('AR-T0')
        cfg['execution']['precision'] = 'tf32'
        self.assertEqual(runtime.training_precision(cfg), 'tf32')
        before = torch.backends.cuda.matmul.allow_tf32
        with runtime.train_kernels(cfg, 'cpu'):
            pass
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, before)
        cfg['model']['sdpa_backend'] = 'flash'
        with self.assertRaises(ValueError):
            with runtime.train_kernels(cfg, 'cpu'):
                pass


if __name__ == '__main__':
    unittest.main()
