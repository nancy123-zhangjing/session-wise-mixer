"""CPU toy tests against the actual frozen H1 engine; no formal training."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
ORIGINAL = WORKSPACE / 'outputs/stage09_restore/goal_20261027_r1'
CODE = ORIGINAL / 'code'
os.environ['DIFFGRM_ENGINE_ROOT'] = str(CODE / 'external/DiffGRM')
sys.path.insert(0, str(CODE))

import numpy as np
import torch
from scripts.listwise_diffgrm.assets import write_assets, load_assets
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.runtime import build_model
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all, phase_plan, rng_state, restore_rng
from scripts.listwise_diffgrm.decoding import SIDCatalog, generate_lists
from grid import HistoryGridAttention
from protocol import GridController, fingerprint, skipped, counts_at, verify_binding

torch.set_num_threads(2)
TEMP_ROOT = WORKSPACE / 'tmp/grid_attention_tests'
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEMP_ROOT)


def fixture(directory):
    cfg = json.loads((ROOT / 'config_B.json').read_text())['base_config']
    cfg.update(history_max=3)
    cfg['model'].update(codebook_size=8, n_embd=16, n_inner=32, dropout=0.0,
                        attn_pdrop=0.0, resid_pdrop=0.0)
    # Preserve all 45 phases / 35 R epochs, with tiny samples and one T step/cycle.
    cfg['training'].update(effective_batch_size=2, microbatch_size=2,
                           t_updates_per_cycle=1, decoder_warmup_updates=1)
    cfg['alignment']['joint_diger_assignment_temperature'] = 0.5
    path = Path(directory) / 'assets'
    if not path.exists():
        rng = np.random.default_rng(2024)
        features = rng.normal(0, .2, (9, 16)).astype('float32')
        write_assets(path, binding={}, original_item_ids=np.arange(9), content_features=features,
                     residual_features=features, centroids=rng.normal(0, .2, (4, 8, 4)).astype('float32'),
                     coarse_centroid=np.zeros(16, dtype='float32'), encoder_metadata={}, toy=True)
    assets = load_assets(path)
    seed_all(2024)
    model = build_model(cfg, assets.make_tokenizer())
    alignment = build_alignment(assets, True, config_overrides=cfg['alignment'])
    trainer = AlternatingTrainer(model, alignment, cfg, 2, precision='bf16')
    batch = dict(history_item_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                 history_mask=torch.tensor([[False, True, True], [True, True, True]]),
                 target_item_ids=torch.tensor([[3, 4, 5, 6, 7], [1, 2, 6, 8, 9]]))
    binding = dict(config=cfg, code={'fixture': 'frozen_H1'}, engine={}, data={}, asset_manifest='toy')
    for _ in range(5): trainer.update([batch], 'R')
    parent = trainer.checkpoint(binding, dict(phase=5, update=0, best_score=.001,
                                              best_epoch=5, validations=[]), 'H1')
    return trainer, batch, parent, binding


def spec(arm):
    return {k: v for k, v in json.loads((ROOT / f'config_{arm}.json').read_text()).items() if k != 'base_config'}


def logits(trainer, batch):
    trainer.model.eval()
    hs = trainer.alignment.hard_codes[batch['history_item_ids']]
    with torch.no_grad():
        encoded = trainer.model._encode(hs, batch['history_mask'])
        return trainer.model._decode(encoded, torch.full((2, 4), -1), item_index=0)


class GridTests(unittest.TestCase):
    def test_identity_and_zero_output_gradients(self):
        x = torch.randn(2, 3, 4, 16, requires_grad=True)
        valid = torch.tensor([[[False] * 4, [True] * 4, [True] * 4], [[False] * 4] * 3])
        for mode in ('row', 'row_column'):
            for train in (True, False):
                grid = HistoryGridAttention(16, mode=mode).train(train)
                y = grid(x, valid)
                self.assertTrue(torch.equal(x, y))
                y.square().sum().backward()
                for layer in (grid.row, grid.column):
                    if layer is not None:
                        self.assertGreater(float(layer.attention.out_proj.weight.grad.abs().sum()), 0)
                        self.assertEqual(float(layer.attention.in_proj_weight.grad.abs().sum()), 0)

    def test_axes_shared_parameters_padding_and_all_padding_after_learning(self):
        grid = HistoryGridAttention(16, mode='row_column').eval()
        for layer in (grid.row, grid.column):
            torch.nn.init.normal_(layer.attention.out_proj.weight, std=.1)
            torch.nn.init.normal_(layer.attention.out_proj.bias, std=.1)
        x = torch.randn(2, 3, 4, 16)
        valid = torch.ones(2, 3, 4, dtype=torch.bool)
        valid[0, 0] = False
        valid[1] = False
        actual = grid(x, valid)
        rows = torch.stack([torch.stack([grid.row(x[b, i][None], valid[b, i][None])[0]
                                        for i in range(3)]) for b in range(2)])
        expected = torch.stack([torch.stack([grid.column(rows[b, :, m][None], valid[b, :, m][None])[0]
                                            for m in range(4)], dim=1) for b in range(2)])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        noisy = x.clone(); noisy[~valid] = torch.randn_like(noisy[~valid]) * 1000
        torch.testing.assert_close(grid(noisy, valid)[valid], actual[valid], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[~valid], x[~valid]))
        self.assertTrue(torch.isfinite(actual).all())

    def test_config_and_exact_formal_schedule(self):
        old = json.loads((ORIGINAL / 'runs/H1/resolved_config.json').read_text())
        for arm in ('B', 'C'):
            cfg = json.loads((ROOT / f'config_{arm}.json').read_text())
            self.assertEqual(cfg['base_config'], old)
            self.assertEqual((cfg['start_epoch'], cfg['freeze_epoch'], cfg['end_epoch']), (5, 20, 35))
        plan = phase_plan(old, 709443)
        self.assertEqual(counts_at(plan, dict(phase=25, update=0)), dict(R=13860, T=200))
        self.assertEqual(counts_at(plan, dict(phase=45, update=0)), dict(R=24255, T=200))
        self.assertEqual(sum(r['updates'] for r in plan if r['kind'] == 'T'), 400)
        self.assertEqual([r['epoch'] for r in plan if skipped(r)], [20, 23, 26, 29, 32])
        with self.assertRaises(ValueError): counts_at(plan, dict(phase=25, update=1))

    def test_parent_logits_adam_scheduler_rng_and_bc_row_initialization(self):
        rows = []
        for arm in ('B', 'C'):
            with tempfile.TemporaryDirectory() as tmp:
                trainer, batch, parent, binding = fixture(tmp)
                before = logits(trainer, batch)
                control = GridController(trainer, parent, binding, spec(arm))
                self.assertTrue(torch.equal(parent['rng']['torch'], torch.get_rng_state()))
                self.assertTrue(torch.equal(before, logits(trainer, batch)))
                rows.append(fingerprint(control.grid.row.state_dict()))
                self.assertEqual(fingerprint(parent['r_optimizer']['state']), fingerprint(trainer.r_optimizer.state_dict()['state']))
                self.assertEqual(fingerprint(parent['scheduler']), fingerprint(trainer.scheduler.state_dict()))
                self.assertEqual(parent['r_optimizer']['param_groups'][0]['lr'], trainer.r_optimizer.param_groups[0]['lr'])
                self.assertEqual(fingerprint(parent['alignment']), fingerprint(trainer.alignment.state_dict()))
                before_grid = fingerprint(control.grid.state_dict())
                control.update([batch], 'T'); trainer.refresh_catalog(); control.finish_phase()
                self.assertEqual(before_grid, fingerprint(control.grid.state_dict()))
                control.update([batch], 'R'); control.finish_phase()
                self.assertNotEqual(before_grid, fingerprint(control.grid.state_dict()))
                # Output projection's first step opens Q/K/V and LayerNorm paths.
                trainer.r_optimizer.zero_grad(set_to_none=True)
                trainer.phase_mode('R')
                loss, _ = trainer.objective(batch, 'R'); loss.backward()
                for layer in (control.grid.row, control.grid.column):
                    if layer is not None:
                        self.assertGreater(float(layer.attention.in_proj_weight.grad.abs().sum()), 0)
        self.assertEqual(rows[0], rows[1])

    def test_exact_next_t_and_r_resume_and_reject_corrupt_optimizer(self):
        for arm in ('B', 'C'):
            with tempfile.TemporaryDirectory() as tmp:
                trainer, batch, parent, binding = fixture(tmp)
                control = GridController(trainer, parent, binding, spec(arm))
                for kind in ('T', 'R'):
                    saved = control.checkpoint()
                    expected_metrics = control.update([batch], kind)
                    expected = control.checkpoint()
                    # A new trainer/controller reproduces cross-process construction.
                    restored, _, _, _ = fixture(tmp)
                    other = GridController(restored, parent, binding, spec(arm))
                    other.restore(saved)
                    self.assertEqual(other.update([batch], kind), expected_metrics)
                    actual = other.checkpoint()
                    for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler'):
                        self.assertEqual(fingerprint(expected[key]), fingerprint(actual[key]), key)
                    if kind == 'R':
                        bad = copy.deepcopy(actual)
                        grid_index = next(i for i,n in enumerate(other.optimizer_names) if n.startswith('backbone.item_local_attention.'))
                        bad['r_optimizer']['state'][grid_index]['step'] += 1
                        with self.assertRaisesRegex(ValueError, 'Adam clock'):
                            other.restore(bad)
                    if kind == 'T': trainer.refresh_catalog()
                    control.finish_phase()

    def test_t_gradient_through_trained_grid_and_prefix_only_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, batch, parent, binding = fixture(tmp)
            control = GridController(trainer, parent, binding, spec('C'))
            for layer in (control.grid.row, control.grid.column):
                torch.nn.init.normal_(layer.attention.out_proj.weight, std=.01)
            trainer.phase_mode('T')
            trainer.r_optimizer.zero_grad(set_to_none=True)
            trainer.t_optimizer.zero_grad(set_to_none=True)
            loss, _ = trainer.objective(batch, 'T', include_aux=False)
            loss.backward()
            self.assertGreater(float(trainer.alignment.mixer.base_codebooks.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in control.grid.parameters()))
            seen = []
            hook = control.grid.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
            trainer.objective(batch, 'R')
            self.assertEqual(seen, [3, 4, 5, 6, 7])
            seen.clear()
            catalog = SIDCatalog(item_ids=torch.arange(1,10), sids=trainer.alignment.hard_codes[1:].cpu(),
                                 reconstruction_errors=torch.zeros(9), original_item_ids=torch.arange(9), codebook_size=8)
            result = generate_lists(trainer.model, trainer.alignment.hard_codes[batch['history_item_ids']],
                                    batch['history_mask'], catalog, route='H', list_length=5, beam_size=2)
            self.assertEqual(result.item_ids.shape[-1], 5)
            self.assertEqual(set(seen), {3, 4, 5, 6, 7})
            hook.remove()

    def test_full_toy_schedule_freezes_tokenizer_but_trains_grid_to_r35(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, batch, parent, binding = fixture(tmp)
            control = GridController(trainer, parent, binding, spec('C'))
            at_freeze = None
            while control.progress['phase'] < len(trainer.plan):
                row = trainer.plan[control.progress['phase']]
                if not skipped(row):
                    control.update([batch], row['kind'])
                    if row['kind'] == 'T': trainer.refresh_catalog()
                else:
                    with self.assertRaises(ValueError): control.update([batch], 'T')
                control.finish_phase()
                if row['kind'] == 'R' and row['epoch'] == 20:
                    at_freeze = (control.freeze_hash, fingerprint(control.grid.state_dict()))
                    frozen = control.checkpoint()
                    control.restore(frozen)
            self.assertEqual((trainer.r_updates, trainer.t_updates), (35, 5))
            self.assertEqual(at_freeze[0], control.frozen_state())
            self.assertNotEqual(at_freeze[1], fingerprint(control.grid.state_dict()))
            self.assertEqual(trainer.scheduler.last_epoch, 35)
            self.assertEqual(int(trainer.alignment.diger_total_optimizer_steps), 10)
            final = control.checkpoint()
            control.restore(final)
            bad = copy.deepcopy(final); bad['freeze_hash'] = None
            with self.assertRaisesRegex(ValueError, 'Freeze anchor'): control.restore(bad)


if __name__ == '__main__':
    unittest.main()
