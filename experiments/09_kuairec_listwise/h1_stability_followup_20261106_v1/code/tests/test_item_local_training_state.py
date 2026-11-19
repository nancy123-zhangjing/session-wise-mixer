"""Independent CPU recovery tests: real DiffGRM, optimizer, scheduler and shuffle.

No package installs, GPU calls or skipped tests. The optional trainer integration
uses a single-process CPU Accelerator substitute and a torch cosine scheduler;
the actual trainer loop, state helpers, model, OCN loss and CSV writers run intact.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import random
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
fixture_spec = importlib.util.spec_from_file_location(
    "_item_attention_fixture", TESTS / "test_item_local_attention.py"
)
assert fixture_spec is not None and fixture_spec.loader is not None
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)
state_spec = importlib.util.spec_from_file_location(
    f"{fixture.PACKAGE}.training_state", fixture.SOURCE / "training_state.py"
)
assert state_spec is not None and state_spec.loader is not None
state = importlib.util.module_from_spec(state_spec)
sys.modules[state_spec.name] = state
state_spec.loader.exec_module(state)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CPUAccelerator:
    is_main_process = True
    num_processes = 1
    device = torch.device("cpu")
    scaler = None
    sync_gradients = True

    def __init__(self):
        self.step = 0

    def unwrap_model(self, model):
        return model

    def prepare(self, *objects):
        return objects

    @contextmanager
    def accumulate(self, model):
        self.step += 1
        yield

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, maximum):
        return torch.nn.utils.clip_grad_norm_(parameters, maximum)

    def reduce(self, value, reduction="sum"):
        return value

    def init_trackers(self, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


class RandomHistoryDataset(Dataset):
    """Consume Python, NumPy and torch RNGs as well as sampler generator state."""

    def __init__(self):
        self.accesses = []

    def __len__(self):
        return 8

    def __getitem__(self, index):
        jitter = (int(random.random() * 8) + int(np.random.randint(0, 8))
                  + int(torch.randint(0, 8, ()).item())) % 8
        self.accesses.append((int(index), jitter))
        codes = torch.tensor([(index + digit) % 8 for digit in range(4)])
        history = torch.tensor([
            [-1, -1, -1, -1],
            [(index + digit + jitter) % 8 for digit in range(4)],
            [(index + digit + 2) % 8 for digit in range(4)],
        ])
        return {
            "history_sid": history,
            "history_mask": torch.tensor([False, True, True]),
            "decoder_input_ids": codes,
            "decoder_labels": codes.clone(),
        }


def loader_with_random_generator():
    return DataLoader(
        RandomHistoryDataset(), batch_size=2, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(314159),
    )


def make_components(seed=2024):
    seed_all(seed)
    config = fixture.tiny_config(True)
    config.update(dropout=0.1, attn_pdrop=0.1, resid_pdrop=0.1)
    with redirect_stdout(io.StringIO()):
        model = fixture.DIFF_GRM(config, object(), fixture.TinyTokenizer())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    return model, optimizer, scheduler, loader_with_random_generator(), CPUAccelerator()


def train_epoch(model, optimizer, scheduler, loader):
    losses = []
    model.train()
    with redirect_stdout(io.StringIO()):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).loss
            if not bool(torch.isfinite(loss)):
                raise AssertionError("Non-finite recovery fixture loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
    return losses


def snapshot(model, optimizer, scheduler, loader):
    return copy.deepcopy({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": state.capture_rng(loader),
    })


class NestedAssertMixin:
    def assert_nested_equal(self, left, right, path="root"):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor, path)
            self.assertEqual(left.dtype, right.dtype, path)
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assert_nested_equal(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_nested_equal(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)


class RecoveryHelperTests(NestedAssertMixin, unittest.TestCase):
    def test_two_epochs_equal_epoch_boundary_recovery_including_all_rngs(self):
        direct = make_components()
        first_losses = train_epoch(*direct[:4])
        second_losses = train_epoch(*direct[:4])
        expected = snapshot(*direct[:4])
        expected_accesses = list(direct[3].dataset.accesses)

        interrupted = make_components()
        self.assertEqual(first_losses, train_epoch(*interrupted[:4]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "train.csv"
            csv.write_bytes(b"epoch,loss\n1,0.123\n")
            committed_csv = csv.read_bytes()
            best = root / "best_model.bin"
            state.atomic_torch_save(interrupted[0].state_dict(), best)
            committed_best = best.read_bytes()
            fingerprint = state.configuration_fingerprint(fixture.tiny_config(True))
            boundary = root / "last.pt"
            state.save_boundary(
                boundary, model=interrupted[0], optimizer=interrupted[1],
                scheduler=interrupted[2], accelerator=interrupted[4], loader=interrupted[3],
                progress={"next_epoch": 1, "best_epoch": 1, "best_val_score": 0.2,
                          "no_improve_count": 2, "cumulative_optimizer_updates": 4},
                config_fingerprint=fingerprint, diagnostic_paths=[csv], best_checkpoint=best,
            )
            # Model, scheduler, RNG and best checkpoint advance in an unfinished epoch.
            train_epoch(*interrupted[:4])
            csv.write_bytes(committed_csv + b"2,uncommitted\n")
            best.write_bytes(b"uncommitted candidate checkpoint")
            random.random(); np.random.random(7); torch.rand(11)

            restored = make_components(seed=99)
            restored[4].step = 999
            payload = state.load_boundary(boundary)
            self.assertEqual(payload["best_checkpoint_bytes"], committed_best)
            state.restore_diagnostics(payload)
            progress = state.restore_boundary(
                payload, model=restored[0], optimizer=restored[1], scheduler=restored[2],
                accelerator=restored[4], loader=restored[3], config_fingerprint=fingerprint,
            )
            self.assertEqual(progress["next_epoch"], 1)
            self.assertEqual(progress["no_improve_count"], 2)
            self.assertEqual(restored[4].step, 0)
            self.assertEqual(best.read_bytes(), committed_best)
            self.assertEqual(csv.read_bytes(), committed_csv)
            self.assertEqual(second_losses, train_epoch(*restored[:4]))
            self.assertEqual(expected_accesses[8:], restored[3].dataset.accesses)
            self.assert_nested_equal(expected, snapshot(*restored[:4]))

    def test_diagnostic_tail_is_backed_up_and_rollback_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "curve.csv"
            committed = b"epoch,loss\n1,0.1\n"
            tail = b"2,0.9\n3,partial"
            csv.write_bytes(committed + tail)
            payload = {"diagnostic_offsets": {str(csv): len(committed)}}
            state.restore_diagnostics(payload)
            self.assertEqual(csv.read_bytes(), committed)
            backups = list(root.glob("*.interrupted_tail_*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), tail)
            state.restore_diagnostics(payload)
            self.assertEqual(list(root.glob("*.interrupted_tail_*.bak")), backups)
            csv.write_bytes(b"too short")
            with self.assertRaisesRegex(RuntimeError, "missing or shortened"):
                state.restore_diagnostics(payload)

    def test_atomic_checkpoint_failure_preserves_old_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            original = {"value": torch.tensor([1.0, 2.0])}
            state.atomic_torch_save(original, path)
            original_bytes = path.read_bytes()
            with mock.patch.object(state.os, "replace", side_effect=OSError("injected replace failure")):
                with self.assertRaisesRegex(OSError, "injected"):
                    state.atomic_torch_save({"value": torch.tensor([9.0])}, path)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assert_nested_equal(torch.load(path, weights_only=True), original)

    def test_config_change_rejected_before_touching_model_or_best_checkpoint(self):
        components = make_components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.bin"
            path.write_bytes(b"original candidate")
            payload = {"configuration_fingerprint": "locked"}
            before = copy.deepcopy(components[0].state_dict())
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                state.restore_boundary(
                    payload, model=components[0], optimizer=components[1], scheduler=components[2],
                    accelerator=components[4], loader=components[3], config_fingerprint="different",
                )
            self.assert_nested_equal(before, components[0].state_dict())
            self.assertEqual(path.read_bytes(), b"original candidate")
        config = fixture.tiny_config(True)
        changed = copy.deepcopy(config)
        changed["item_local_attention"]["num_heads"] = 2
        self.assertNotEqual(state.configuration_fingerprint(config), state.configuration_fingerprint(changed))


def load_actual_trainer():
    """Substitute optional orchestration deps, never model/state/trainer logic."""
    transformers = types.ModuleType("transformers")

    def cosine_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
        if name != "cosine" or num_warmup_steps != 0:
            raise AssertionError("CPU integration fixture expects cosine without warmup")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    transformers.get_scheduler = cosine_scheduler
    accelerate = types.ModuleType("accelerate")
    accelerate.Accelerator = CPUAccelerator
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **kwargs: iterable
    utils = types.ModuleType("genrec.utils")
    utils.get_total_steps = lambda config, loader: config["epochs"] * len(loader)
    utils.get_file_name = lambda config, suffix="": "recovery_fixture" + suffix
    utils.config_for_log = lambda config: {}
    path = fixture.SOURCE / "trainer.py"
    spec = importlib.util.spec_from_file_location(f"{fixture.PACKAGE}.trainer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "transformers": transformers, "accelerate": accelerate, "tqdm": tqdm,
        "genrec.utils": utils, "genrec.model": fixture.abstract_module,
        "genrec.tokenizer": fixture.tokenizer_module,
    }):
        spec.loader.exec_module(module)
    return module


trainer_runtime = load_actual_trainer()


def trainer_config(root, *, resume=False):
    config = fixture.tiny_config(True)
    config.update({
        "dataset": "synthetic", "model": "DIFF_GRM", "run_id": "recovery_fixture",
        "training_state_dir": str(root / "state"),
        "diffusion_diagnostics_log_dir": str(root / "diagnostics"),
        "run_metadata_dir": str(root / "metadata"),
        "accelerator": CPUAccelerator(), "use_ddp": False,
        "epochs": 2, "lr": 0.01, "weight_decay": 0.03,
        "warmup_steps": 0, "warmup_ratio": 0.0,
        "train_batch_size": 2, "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0, "eval_start_epoch": 1, "eval_interval": 1,
        "patience": 5, "val_metric": "weighted_score", "numerical_guard": True,
        "dropout": 0.1, "attn_pdrop": 0.1, "resid_pdrop": 0.1,
        # Real evaluate changes this from its pre-evaluation runtime value.
        "current_split": "train",
    })
    if resume:
        config["resume_training_state"] = True
    return config


def make_actual_trainer(root, *, seed=2024, resume=False, overrides=None):
    seed_all(seed)
    config = trainer_config(root, resume=resume)
    config.update(overrides or {})
    model = fixture.DIFF_GRM(config, object(), fixture.TinyTokenizer())
    trainer = trainer_runtime.DIFF_GRMTrainer(config, model, fixture.TinyTokenizer())

    def synthetic_validation(self, loader, split="val", epoch=None):
        # Only metric measurement is substituted; keep genuine selection/save/early stop.
        results = {"weighted_score": float(epoch) / 10.0}
        self.config["current_split"] = split
        self._record_evaluation_metrics(epoch, split, results)
        return results

    trainer.evaluate = types.MethodType(synthetic_validation, trainer)
    return trainer, loader_with_random_generator()


class ActualTrainerRecoveryTests(NestedAssertMixin, unittest.TestCase):
    def test_actual_fit_continuous_equals_committed_boundary_restart(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            direct, direct_loader = make_actual_trainer(root / "direct")
            expected_selection = direct.fit(direct_loader, [])
            expected = state.load_boundary(direct.last_training_state)

            interrupted, interrupted_loader = make_actual_trainer(root / "resumed")
            original_save = trainer_runtime.save_boundary

            class InjectedProcessExit(Exception):
                pass

            def commit_then_interrupt(path, **kwargs):
                result = original_save(path, **kwargs)
                if kwargs["progress"]["next_epoch"] == 1:
                    raise InjectedProcessExit("simulated process exit after durable epoch one")
                return result

            with mock.patch.object(trainer_runtime, "save_boundary", side_effect=commit_then_interrupt):
                with self.assertRaises(InjectedProcessExit):
                    interrupted.fit(interrupted_loader, [])
            committed = state.load_boundary(interrupted.last_training_state)
            self.assertEqual(committed["progress"]["next_epoch"], 1)
            self.assertFalse(committed["progress"]["training_complete"])
            committed_csv = Path(interrupted.training_metrics_csv).read_bytes()
            with open(interrupted.training_metrics_csv, "ab") as handle:
                handle.write(b"2,uncommitted,garbage\n")
            Path(interrupted.saved_model_ckpt).write_bytes(b"uncommitted best candidate")

            resumed, resumed_loader = make_actual_trainer(root / "resumed", seed=99, resume=True)
            self.assertEqual(Path(resumed.training_metrics_csv).read_bytes(), committed_csv)
            actual_selection = resumed.fit(resumed_loader, [])
            actual = state.load_boundary(resumed.last_training_state)
            self.assertEqual(actual_selection, expected_selection)
            self.assertEqual(actual_selection, (2, 0.2))
            self.assertEqual(resumed.completed_epochs, 2)
            self.assertTrue(actual["progress"]["training_complete"])
            for key in ("model", "optimizer", "scheduler", "rng"):
                self.assert_nested_equal(expected[key], actual[key], key)
            for key in ("next_epoch", "best_epoch", "best_val_score", "eval_count",
                        "no_improve_count", "cumulative_optimizer_updates", "decoder_epochs_completed"):
                self.assertEqual(expected["progress"][key], actual["progress"][key], key)
            self.assertEqual(direct_loader.dataset.accesses[8:], resumed_loader.dataset.accesses)
            self.assertEqual(Path(direct.training_metrics_csv).read_bytes(),
                             Path(resumed.training_metrics_csv).read_bytes())
            # fit restores its chosen best, while last.pt retains actual end-of-run optimizer state.
            self.assert_nested_equal(direct.model.state_dict(), resumed.model.state_dict())

    def test_actual_trainer_rejects_resolved_training_config_change(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer, loader = make_actual_trainer(root)
            trainer.fit(loader, [])
            config = trainer_config(root, resume=True)
            config["lr"] *= 0.5
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                trainer_runtime.DIFF_GRMTrainer(config, trainer.model, fixture.TinyTokenizer())

    def test_mid_epoch_budget_exit_keeps_incomplete_recoverable_boundary(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer, loader = make_actual_trainer(root)
            deadline_check = trainer._check_training_deadline
            checks = 0

            def interrupt_second_batch():
                nonlocal checks
                checks += 1
                if checks == 3:  # epoch entry, first batch, second batch
                    trainer.config["training_deadline_unix_seconds"] = 0
                    trainer.config["training_deadline_reserve_seconds"] = 0
                deadline_check()

            trainer._check_training_deadline = interrupt_second_batch
            validation = mock.Mock(wraps=trainer.evaluate)
            trainer.evaluate = validation
            with self.assertRaises(state.TrainingBudgetReached):
                trainer.fit(loader, [])
            self.assertTrue(trainer.training_incomplete)
            self.assertEqual(trainer.training_stop_reason, "wall_deadline")
            validation.assert_not_called()
            boundary = state.load_boundary(trainer.last_training_state)
            self.assertEqual(boundary["progress"]["next_epoch"], 0)
            self.assertEqual(boundary["progress"]["cumulative_optimizer_updates"], 0)
            self.assertFalse(boundary["progress"]["training_complete"])
            self.assertFalse(Path(trainer.saved_model_ckpt).exists())
            self.assertTrue(any(
                not torch.equal(value, trainer.model.state_dict()[name])
                for name, value in boundary["model"].items()
            ), "fixture must have executed a genuine partial epoch")
            resumed, resumed_loader = make_actual_trainer(root, seed=99, resume=True)
            self.assertEqual(resumed.fit(resumed_loader, []), (2, 0.2))
            completed = state.load_boundary(resumed.last_training_state)
            self.assertTrue(completed["progress"]["training_complete"])
            self.assertEqual(completed["progress"]["cumulative_optimizer_updates"], 8)

    def test_early_stop_commits_completion_and_reload_selects_best_not_last(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            overrides = {"epochs": 4, "patience": 1}
            trainer, loader = make_actual_trainer(root, overrides=overrides)

            def falling_validation(self, loader, split="val", epoch=None):
                results = {"weighted_score": 0.2 if epoch == 1 else 0.1}
                self.config["current_split"] = split
                self._record_evaluation_metrics(epoch, split, results)
                return results

            trainer.evaluate = types.MethodType(falling_validation, trainer)
            self.assertEqual(trainer.fit(loader, []), (1, 0.2))
            boundary = state.load_boundary(trainer.last_training_state)
            self.assertTrue(boundary["progress"]["training_complete"])
            self.assertEqual(boundary["progress"]["stop_reason"], "early_stopping")
            self.assertEqual(boundary["progress"]["next_epoch"], 2)
            self.assertEqual(boundary["progress"]["no_improve_count"], 1)
            best = torch.load(trainer.saved_model_ckpt, weights_only=True)
            self.assert_nested_equal(trainer.model.state_dict(), best)
            self.assertTrue(any(
                not torch.equal(value, best[name])
                for name, value in boundary["model"].items()
            ), "fixture must distinguish the last epoch from the chosen best")
            resumed, resumed_loader = make_actual_trainer(root, seed=99, resume=True, overrides=overrides)
            validation = mock.Mock(wraps=resumed.evaluate)
            resumed.evaluate = validation
            self.assertEqual(resumed.fit(resumed_loader, []), (1, 0.2))
            self.assertEqual(resumed.completed_epochs, 2)
            self.assertEqual(resumed_loader.dataset.accesses, [])
            validation.assert_not_called()
            self.assert_nested_equal(resumed.model.state_dict(), best)


if __name__ == "__main__":
    unittest.main(verbosity=2)
