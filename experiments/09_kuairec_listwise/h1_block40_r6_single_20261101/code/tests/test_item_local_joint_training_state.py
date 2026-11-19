"""CPU recovery of the real alternating joint DiffGRM Trainer and model.

python -B tests/test_item_local_joint_training_state.py

Only orchestration dependencies and validation metric measurement use the
existing CPU fixture. Real DIGER loss, alternating optimizer groups, SID refresh,
state transactions and shuffled data/RNG run unchanged. Temporary artifacts are
small, generated offline inputs; no historical run or platform is touched.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_joint_state_cpu_fixture", ROOT / "tests/test_item_local_training_state.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
fixture, state, runtime = base.fixture, base.state, base.trainer_runtime
Mixer = fixture.runtime.JointCrossPSEAlignment.__init__.__globals__["CrossPSECodebookMixer"]


class JointTokenizer(fixture.TinyTokenizer):
    def __init__(self):
        self.id2item = ["<pad>", *[f"item-{i}" for i in range(1, 8)]]
        self.item2id = {item: i for i, item in enumerate(self.id2item)}
        self._cb2items = None

    def _create_reverse_mapping(self):
        return {tokens: item for item, tokens in self.item2tokens.items()}

    def _compute_collision_stats(self, mapping):
        return {"items": len(mapping), "unique": len(set(mapping.values()))}

    def _finalize_position_stats(self, values):
        return values


def write_artifacts(root):
    root.mkdir()
    rng = np.random.default_rng(1234)
    books = rng.normal(size=(4, 8, 8)).astype(np.float32)
    slots = rng.normal(size=(7, 4, 8)).astype(np.float32)
    codes = ((slots[:, :, None] - books[None]) ** 2).sum(-1).argmin(-1).astype(np.int64)
    np.save(root / "base_pse_codebooks.npy", books)
    np.save(root / "opq_residual_embeddings.npy", slots.reshape(7, 32))
    np.save(root / "semantic_ids.npy", codes)
    (root / "items.json").write_text(json.dumps(JointTokenizer().id2item[1:]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        mixer = Mixer(torch.from_numpy(books), n_heads=2, train_base_codebooks=True)
    torch.save({"model": mixer.state_dict()}, root / "best.pt")


class JointDataset(Dataset):
    def __init__(self):
        self.accesses = []

    def __len__(self):
        return 4

    def __getitem__(self, index):
        draw = (random.randrange(7) + int(np.random.randint(0, 7)) + int(torch.randint(0, 7, ()).item())) % 7
        self.accesses.append((index, draw))
        return {"history_item_ids": torch.tensor([0, 1 + index % 7, 1 + (index + draw) % 7]),
                "history_mask": torch.tensor([False, True, True]), "target_item_id": torch.tensor(1 + (index + 1) % 7)}


def make_trainer(root, artifacts, *, resume=False, seed=2024):
    base.seed_all(seed)
    config = base.trainer_config(root, resume=resume)
    config.update({"epochs": 10, "joint_crosspse_enabled": True,
                   "joint_crosspse_artifact_dir": str(artifacts), "joint_crosspse_items_path": str(artifacts / "items.json"),
                   "crosspse_heads": 2, "joint_gradient_mode": "diger_st",
                   "joint_recommender_warmup_epochs": 5, "joint_cycle_epochs": 4,
                   "joint_tokenizer_epochs_per_cycle": 1, "joint_max_tokenizer_phases": 2,
                   "joint_tokenizer_batches_per_phase": 1, "joint_sid_refresh_batch_size": 3,
                   "joint_evaluate_initial_state": True, "joint_recommender_sia_weight": 0.0,
                   "joint_tokenizer_sia_weight": 0.0, "joint_codebook_lr": .03, "joint_mixer_lr": .02,
                   "joint_weight_decay": .01, "joint_diger_gumbel_noise_start": .8,
                   "joint_diger_gumbel_noise_end": .2, "patience": 0,
                   "dropout": .1, "attn_pdrop": .1, "resid_pdrop": .1})
    tokenizer = JointTokenizer()
    model = fixture.DIFF_GRM(config, object(), tokenizer)
    trainer = runtime.DIFF_GRMTrainer(config, model, tokenizer)
    trainer.validation_epochs = []

    def measure(self, loader, split="val", epoch=None):
        self.validation_epochs.append(epoch)
        # Check actual mapping coherence at every validation, including restore.
        for item, tokens in self.tokenizer.item2tokens.items():
            index = self.tokenizer.item2id[item]
            expected = tuple(int(code) + self.tokenizer.sid_offset + d * 8
                             for d, code in enumerate(self.model.joint_alignment.hard_codes[index]))
            assert tokens == expected
        self.config["current_split"] = split
        # Keep epoch0 the best: it must not be repeated or overwritten on fit end.
        result = {"weighted_score": 1.0 if epoch == 0 else .1}
        self._record_evaluation_metrics(epoch, split, result)
        return result

    trainer.evaluate = types.MethodType(measure, trainer)
    loader = DataLoader(JointDataset(), batch_size=2, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(6789))
    return trainer, loader


class InjectedExit(RuntimeError):
    pass


class JointBoundaryTests(base.NestedAssertMixin, unittest.TestCase):
    def compare_terminal(self, expected, actual):
        for key in ("model", "optimizer", "scheduler", "rng", "joint_runtime"):
            self.assert_nested_equal(expected[key], actual[key], key)
        for key in ("next_epoch", "best_epoch", "best_val_score", "eval_count", "no_improve_count",
                    "cumulative_optimizer_updates", "decoder_epochs_completed", "joint_progress"):
            self.assert_nested_equal(expected["progress"][key], actual["progress"][key], key)

    def test_real_5_plus_1_3_both_transitions_resume_without_repeating_commits(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            artifacts = root / "offline_joint"; write_artifacts(artifacts)
            direct, loader = make_trainer(root / "direct", artifacts)
            boundaries = {}
            original = runtime.save_boundary

            def record(path, **kwargs):
                payload = original(path, **kwargs)
                boundaries[kwargs["progress"]["next_epoch"]] = copy.deepcopy(payload)
                return payload

            with mock.patch.object(runtime, "save_boundary", side_effect=record):
                self.assertEqual(direct.fit(loader, []), (0, 1.0))
            expected = state.load_boundary(direct.last_training_state)
            self.assertEqual(expected["format"], "diffgrm_joint_phase_boundary_v1")
            self.assertEqual(expected["progress"]["decoder_epochs_completed"], 8)
            self.assertEqual(expected["progress"]["cumulative_optimizer_updates"], 16)
            joint = expected["progress"]["joint_progress"]
            self.assertEqual((joint["tokenizer_epochs_completed"], joint["tokenizer_optimizer_updates"], joint["sid_refresh_count"]), (2, 2, 2))
            self.assertEqual(expected["scheduler"]["last_epoch"], 16)
            self.assertEqual(int(expected["model"]["joint_alignment.diger_optimizer_step"]), 2)
            self.assertEqual(boundaries[5]["progress"]["joint_progress"]["next_phase"], "tokenizer")
            self.assertEqual(boundaries[6]["progress"]["joint_progress"]["next_phase"], "recommender")
            # Frozen side stays bitwise constant, including nonzero AdamW decay.
            for epoch in range(1, 11):
                before, after = boundaries[epoch - 1]["model"], boundaries[epoch]["model"]
                phase = boundaries[epoch]["progress"]["joint_progress"]["last_committed_phase"]
                frozen = [name for name in after if name.startswith("joint_alignment.") == (phase == "recommender")]
                active = [name for name in after if name not in frozen]
                self.assertTrue(all(torch.equal(before[name], after[name]) for name in frozen), (epoch, phase))
                self.assertTrue(any(not torch.equal(before[name], after[name]) for name in active), (epoch, phase))
            epoch0_best = boundaries[0]["best_checkpoint_bytes"]
            self.assertEqual(Path(direct.saved_model_ckpt).read_bytes(), epoch0_best)
            for cut in (5, 6):
                with self.subTest(committed_boundary=cut):
                    interrupted, first_loader = make_trainer(root / f"cut{cut}", artifacts)

                    def save_then_exit(path, **kwargs):
                        value = original(path, **kwargs)
                        if kwargs["progress"]["next_epoch"] == cut:
                            raise InjectedExit("after a complete durable phase")
                        return value

                    with mock.patch.object(runtime, "save_boundary", side_effect=save_then_exit):
                        with self.assertRaises(InjectedExit): interrupted.fit(first_loader, [])
                    committed = state.load_boundary(interrupted.last_training_state)
                    self.assertEqual(committed["progress"]["next_epoch"], cut)
                    accesses = len(first_loader.dataset.accesses)
                    train_prefix = Path(interrupted.training_metrics_csv).read_bytes()
                    with Path(interrupted.training_metrics_csv).open("ab") as handle: handle.write(b"uncommitted,tail\n")
                    Path(interrupted.saved_model_ckpt).write_bytes(b"uncommitted best checkpoint")
                    resumed, next_loader = make_trainer(root / f"cut{cut}", artifacts, resume=True, seed=99)
                    self.assertEqual(Path(resumed.training_metrics_csv).read_bytes(), train_prefix)
                    # Restore must recover these nonpersistent/derived states, too.
                    resumed.model.joint_alignment.mixer.initial_codebooks.add_(42)
                    resumed.tokenizer.item2tokens = {"stale": (99,)}
                    resumed.tokenizer._cb2items = {"stale": [99]}
                    self.assertEqual(resumed.fit(next_loader, []), (0, 1.0))
                    actual = state.load_boundary(resumed.last_training_state)
                    self.compare_terminal(expected, actual)
                    self.assertNotIn(0, resumed.validation_epochs)
                    self.assertEqual(loader.dataset.accesses[accesses:], next_loader.dataset.accesses)
                    self.assertEqual(Path(direct.training_metrics_csv).read_bytes(), Path(resumed.training_metrics_csv).read_bytes())
                    self.assertEqual(Path(direct.joint_metrics_csv).read_bytes(), Path(resumed.joint_metrics_csv).read_bytes())
                    self.assertEqual(Path(resumed.saved_model_ckpt).read_bytes(), epoch0_best)

    def test_partial_tokenizer_phase_replays_only_uncommitted_phase(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); artifacts = root / "offline_joint"; write_artifacts(artifacts)
            direct, loader = make_trainer(root / "direct", artifacts); direct.fit(loader, [])
            expected = state.load_boundary(direct.last_training_state)
            interrupted, first_loader = make_trainer(root / "partial", artifacts)
            with mock.patch.object(interrupted.model, "refresh_joint_codes", side_effect=InjectedExit("after tokenizer updates, before SID commit")):
                with self.assertRaises(InjectedExit): interrupted.fit(first_loader, [])
            committed = state.load_boundary(interrupted.last_training_state)
            self.assertEqual(committed["progress"]["next_epoch"], 5)
            self.assertEqual(int(committed["model"]["joint_alignment.diger_optimizer_step"]), 0)
            resumed, next_loader = make_trainer(root / "partial", artifacts, resume=True, seed=99)
            resumed.fit(next_loader, [])
            self.compare_terminal(expected, state.load_boundary(resumed.last_training_state))
            self.assertEqual(Path(direct.training_metrics_csv).read_bytes(), Path(resumed.training_metrics_csv).read_bytes())
            self.assertTrue(list(Path(interrupted.training_metrics_csv).parent.glob("*.interrupted_tail_*.bak")))

    def test_joint_boundary_rejects_nonjoint_format_or_wrong_optimizer_groups(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); artifacts = root / "offline_joint"; write_artifacts(artifacts)
            trainer, loader = make_trainer(root / "run", artifacts); trainer.fit(loader, [])
            payload = state.load_boundary(trainer.last_training_state)
            original = copy.deepcopy(payload)
            payload["format"] = "diffgrm_epoch_boundary_v1"
            state.atomic_torch_save(payload, trainer.last_training_state)
            with self.assertRaisesRegex(ValueError, "cannot be interchanged"):
                make_trainer(root / "run", artifacts, resume=True)
            original["joint_runtime"]["optimizer_groups"][0]["parameters"].reverse()
            state.atomic_torch_save(original, trainer.last_training_state)
            resumed, next_loader = make_trainer(root / "run", artifacts, resume=True)
            with self.assertRaisesRegex(ValueError, "optimizer groups changed"):
                resumed.fit(next_loader, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
