"""CPU-only safety and source-equivalence checks for the seed adapter."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("joint_multiseed_under_test", ROOT / "scripts/train/run_item_local_joint_multiseed.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
BASE = (ROOT / "scripts/train/run_item_local_joint.py").read_text(encoding="utf-8")
CONFIG = json.loads((ROOT / "configs/07_item_local_attention/multiseed_20261021.json").read_text(encoding="utf-8"))


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


class MultiSeedAdapterTests(unittest.TestCase):
    def test_four_unique_run_ids_and_isolated_seed_modules(self):
        identities = {adapter.run_id(seed, variant) for seed in adapter.SEEDS for variant in adapter.VARIANTS}
        self.assertEqual(len(identities), 4)
        runners = [adapter.load_runner(seed) for seed in adapter.SEEDS]
        self.assertNotEqual(runners[0].CAMPAIGN, runners[1].CAMPAIGN)
        for seed, runner in zip(adapter.SEEDS, runners):
            self.assertEqual(runner.run_id("control"), adapter.run_id(seed, "control"))
            self.assertEqual(runner.validate_study(CONFIG)["seed"], seed)
        with self.assertRaises(ValueError):
            adapter.load_runner(2024)

    def test_source_pin_and_prepare_changes_are_only_seed_and_binding(self):
        self.assertEqual(adapter.source_sha256(BASE), adapter.PINNED_BASE_SOURCE_SHA256)
        self.assertEqual(adapter.source_sha256(BASE.replace("\n", "\r\n")), adapter.PINNED_BASE_SOURCE_SHA256)
        original = function(BASE, "prepare")
        changed = adapter.adapted_prepare_ast(BASE)
        compile(changed, "<prepare-test>", "exec")

        class Reverse(ast.NodeTransformer):
            removed = 0
            seeds = 0

            def visit_Name(self, node):
                if node.id == "_MULTISEED_SEED":
                    self.seeds += 1
                    return ast.Constant(value=2024)
                return node

            def visit_Assign(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "_MULTISEED_ADAPTER_IDENTITY":
                    self.removed += 1
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                call = node.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    receiver = ast.unparse(call.func.value)
                    if call.func.attr == "update" and receiver in ("identity", "config['training_state_identity']"):
                        self.removed += 1
                        return None
                return self.generic_visit(node)

        reverse = Reverse()
        recovered = reverse.visit(changed).body[0]
        self.assertEqual((reverse.removed, reverse.seeds), (3, 2))
        self.assertEqual(ast.dump(original), ast.dump(recovered))

    def test_all_original_training_code_is_preserved(self):
        original = function(BASE, "run")
        changed = adapter.adapted_run_ast(BASE)
        compile(changed, "<run-test>", "exec")
        original_tail = next(node.orelse for node in ast.walk(original)
                             if isinstance(node, ast.If) and node.orelse
                             and isinstance(node.orelse[0], ast.Assign)
                             and isinstance(node.orelse[0].value, ast.Constant)
                             and node.orelse[0].value.value == "results.json")

        class Reverse(ast.NodeTransformer):
            budgets = 0
            tails = 0

            def visit_BinOp(self, node):
                if ast.unparse(node) == "prepared['study']['max_gpu_hours'] * 3600":
                    self.budgets += 1
                    return ast.parse("28 * 3600", mode="eval").body
                return self.generic_visit(node)

            def visit_If(self, node):
                node = self.generic_visit(node)
                if (node.orelse and isinstance(node.orelse[0], ast.Assign)
                        and isinstance(node.orelse[0].value, ast.Constant)
                        and node.orelse[0].value.value == "results.json"):
                    self.tails += 1
                    node.orelse = copy.deepcopy(original_tail)
                return node

        reverse = Reverse()
        recovered = reverse.visit(changed).body[0]
        self.assertEqual((reverse.budgets, reverse.tails), (1, 1))
        self.assertEqual(ast.dump(original), ast.dump(recovered))

    def test_config_rejects_drift_and_test_evaluation(self):
        for key, bad in (("seeds", [2024, 2025]), ("final_evaluation", "test"),
                         ("max_gpu_hours", 28), ("training_budget_policy", "original_early_stopping"),
                         ("global_codebook_mixer_enabled", True), ("gpu_count_per_run", 4)):
            with self.subTest(key=key):
                config = copy.deepcopy(CONFIG)
                config[key] = bad
                with self.assertRaises(ValueError):
                    adapter.validate_config(config, 2025)
        config = copy.deepcopy(CONFIG)
        config["variants"]["control"]["item_local_attention"]["enabled"] = True
        with self.assertRaises(ValueError):
            adapter.validate_config(config, 2025)

    def test_fixed_budget_changes_only_patience_and_preserves_selection(self):
        before = {"patience": 5, "epochs": 45, "eval_start_epoch": 5, "eval_interval": 5,
                  "val_metric": "weighted_score", "crosspse_seed": 2024,
                  "vectorized_beam_search": {"val": {"beam_act": 32, "beam_max": 32}},
                  "lr": 0.001, "joint_max_tokenizer_phases": 10, "mixed_precision": "bf16"}
        after, audit = adapter.apply_stopping_policy(before, {"status": "passed"}, "fixed_45")
        self.assertEqual(before["patience"], 5)
        self.assertEqual(after["patience"], 0)
        restored = copy.deepcopy(after)
        restored["patience"] = 5
        self.assertEqual(restored, before)
        self.assertEqual(audit["adapter_overrides"], {"patience": {"before": 5, "after": 0}})
        self.assertTrue(audit["checkpoint_selection_unchanged"])
        before["vectorized_beam_search"]["val"]["beam_act"] = 256
        with self.assertRaises(ValueError):
            adapter.apply_stopping_policy(before, {}, "fixed_45")

    def test_prepare_persists_new_seed_adapter_and_immutable_input_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = adapter.load_runner(2026)
            runner.ROOT = root
            package = root / "experiments/05_crosspse_joint"
            package.mkdir(parents=True)
            project, engine = root / "assets", root / "engine"
            names = ["locked/decoder.bin", "locked/tokenizer/best.pt", "locked/tokenizer/semantic_ids.json"]
            names += [f"locked/input_{i}.bin" for i in range(10)]
            locks = {}
            for i, name in enumerate(names):
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = f"locked fixture {i}".encode()
                path.write_bytes(data)
                locks[name] = {"sha256": hashlib.sha256(data).hexdigest()}
            for name in runner.ENGINE_FILES:
                path = engine / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("engine fixture", encoding="utf-8")
            (package / "reproducibility").mkdir()
            (package / "reproducibility/assets.lock.json").write_text(json.dumps({"artifacts": locks}), encoding="utf-8")
            recipe = {"required_artifacts": names, "missing_from_archive": [],
                      "initial_decoder": names[0], "initial_tokenizer_dir": "locked/tokenizer"}
            config_path = root / "campaign.json"
            config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
            output = root / "goal/seeds/seed2026/runs" / adapter.run_id(2026, "control")
            args = argparse.Namespace(config=str(config_path), output=str(output), variant="control", mode="prepare",
                                      resume=False, engine_root=str(engine), project_root=str(project),
                                      expected_engine_commit=adapter.EXPECTED_ENGINE_COMMIT,
                                      deadline_unix=None, training_deadline_reserve_seconds=1200.0)
            with mock.patch.object(runner, "git_identity", return_value={"head": adapter.EXPECTED_ENGINE_COMMIT, "tracked_dirty": False}), \
                    mock.patch.object(runner.stage05, "load_recipe", return_value=recipe), \
                    mock.patch.object(runner, "merge_recipe", return_value=({"patience": 0}, {"status": "passed"}, [])), \
                    mock.patch.object(runner, "dataset_audit", return_value={"fixture": True}):
                prepared = runner.prepare(args)
                identity = runner.read_json(output / "identity.json")
                config = runner.read_json(output / "resolved_config.json")
                self.assertEqual(identity["seed"], 2026)
                self.assertEqual(config["rand_seed"], 2026)
                self.assertEqual(config["training_state_identity"]["seed"], 2026)
                self.assertEqual(identity["multiseed_adapter"], config["training_state_identity"]["multiseed_adapter"])
                self.assertEqual(len(prepared["records"]), 13)
                self.assertTrue(all(runner.sha(prepared["workspace"] / name) == locks[name]["sha256"] for name in names))
                self.assertFalse((output / "TEST_STARTED.json").exists())
                runner._MULTISEED_ADAPTER_IDENTITY = {**runner._MULTISEED_ADAPTER_IDENTITY, "sha256": "0" * 64}
                with self.assertRaisesRegex(ValueError, "identity/source/config differs"):
                    runner.prepare(args)

    def test_training_only_tail_requires_full_budget_and_does_not_read_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            progress = {"tokenizer_epochs_completed": 10, "tokenizer_optimizer_updates": 60, "sid_refresh_count": 10}
            fit = {"completed_epochs": 45, "training_stop_reason": "max_epochs", "cumulative_recommender_optimizer_updates": 3640}
            namespace = {"work": work, "need": adapter.need, "timings": [{"split": "val"}],
                         "prepared": {"study": {"training_budget_policy": "fixed_45"}}, "fit": fit,
                         "terminal": {"progress": {"decoder_epochs_completed": 35}}, "joint_progress": progress,
                         "audit": SimpleNamespace(records=[{"phase": "recommender"}] * 35 + [{"phase": "tokenizer"}] * 10),
                         "common": {"fit": fit}, "_MULTISEED_SEED": 2025,
                         "_MULTISEED_PARENT_CAMPAIGN": adapter.CAMPAIGN,
                         "initial": {"initial_SID_sha256": "saved-sid"}, "sha": lambda path: "fixture-sha",
                         "time": SimpleNamespace(perf_counter=lambda: 100.0), "started": 0.0}
            exec(compile(adapter.TRAINING_ONLY_TAIL, "<tail>", "exec"), namespace)
            result = namespace["result"]
            self.assertFalse(result["test_data_evaluated"])
            self.assertEqual(result["final_validation_pending"], [32, 256])
            self.assertEqual(result["seed"], 2025)
            fit["completed_epochs"] = 40
            with self.assertRaisesRegex(ValueError, "budget was not completed"):
                exec(compile(adapter.TRAINING_ONLY_TAIL, "<tail>", "exec"), namespace)
            fit["completed_epochs"] = 45
            (work / "TEST_STARTED.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not consume test"):
                exec(compile(adapter.TRAINING_ONLY_TAIL, "<tail>", "exec"), namespace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
