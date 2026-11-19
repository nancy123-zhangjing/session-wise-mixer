"""Controls and optional real-Torch CPU tests for the matched Mixer factorial.

Synthetic CPU tests are not a substitute for the four-arm GPU smoke. The
Windows host may skip unavailable Torch/NumPy; --require-torch forbids that.
No historical evidence, configuration, or model artifact is modified.
Sparse code deployments must set DIFFGRM_ENGINE_ROOT to the readonly engine
checkout. Missing engine source is an error, never an implicit test skip.
"""
from __future__ import annotations

import ast
import copy
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/train/run_diger_mixer_factorial.py"
HELPER_PATH = ROOT / "scripts/checks/diger_mixer_factorial_protocol.py"
CONFIG_PATH = ROOT / "configs/05_crosspse_joint/pilots/diger_mixer_factorial_seed2024.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load factorial control module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_engine_root(project_root, override):
    if override is None:
        candidate = Path(project_root) / "external/DiffGRM"
    else:
        if not str(override).strip():
            raise ValueError("DIFFGRM_ENGINE_ROOT cannot be empty")
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            raise ValueError("DIFFGRM_ENGINE_ROOT must be an explicit absolute engine directory")
    engine = candidate.resolve()
    required = ("genrec/models/DIFF_GRM/model.py", "genrec/models/DIFF_GRM/cross_pse.py")
    missing = [relative for relative in required if not (engine / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"DiffGRM engine source missing under {engine}: {missing}. "
                                "For a sparse deployment, set DIFFGRM_ENGINE_ROOT to the readonly engine checkout; "
                                "these checks cannot be skipped.")
    return engine


ENGINE_ROOT = resolve_engine_root(ROOT, os.environ.get("DIFFGRM_ENGINE_ROOT"))
runner = load_module("_mixer_factorial_runner_tests", RUNNER_PATH)
helper = load_module("_mixer_factorial_protocol_tests", HELPER_PATH)
try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


class VariantAndScopeTests(unittest.TestCase):
    def fake_model(self):
        return SimpleNamespace(config={}, joint_alignment=SimpleNamespace(mixer=SimpleNamespace()))

    def test_mixer_variant_labels_and_config_switch_match(self):
        for enabled, variant in ((False, "refined_pse"), (True, "cross_pse")):
            model = self.fake_model()
            runner.configure_variant(model, dict(mixer_enabled=enabled, tokenizer_variant=variant))
            self.assertIs(model.joint_alignment.mixer.mixer_enabled, enabled)
            self.assertEqual(model.joint_alignment.tokenizer_variant, variant)
            self.assertIs(model.config["crosspse_mixer_enabled"], enabled)

    def test_non_boolean_mixer_flag_is_rejected(self):
        for enabled in (0, 1, "false", None):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                runner.configure_variant(self.fake_model(), dict(mixer_enabled=enabled, tokenizer_variant="refined_pse"))

    def test_mismatched_variant_label_is_rejected(self):
        for enabled, variant in ((False, "cross_pse"), (True, "refined_pse"), (True, "unknown")):
            with self.subTest(enabled=enabled, variant=variant), self.assertRaises(ValueError):
                runner.configure_variant(self.fake_model(), dict(mixer_enabled=enabled, tokenizer_variant=variant))

    def test_new_output_only_in_own_factorial_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "outputs/pilots/stage05_mixer_factorial/new"
            self.assertEqual(runner.validate_output(root, target), target)
            self.assertFalse(target.exists())
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                runner.validate_output(root, target)

    def test_historical_nested_and_arbitrary_ssd_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for value in ("experiments/05_crosspse_joint/evidence/new", "outputs/pilots/stage05_feedback/new",
                          "outputs/pilots/stage05_mixer_factorial", "outputs/pilots/stage05_mixer_factorial/new/nested"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runner.validate_output(root, Path(value))
            with self.assertRaises(ValueError):
                runner.validate_output(root, root / "other_ssd/new", root / "other_ssd")

    def test_output_symlink_parent_is_rejected_when_host_supports_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            try:
                (root / "outputs").symlink_to(elsewhere, target_is_directory=True)
            except OSError:
                self.skipTest("Windows host does not permit creating symbolic links")
            with self.assertRaises(ValueError):
                runner.validate_output(root, Path("outputs/pilots/stage05_mixer_factorial/new"))

    def test_old_feedback_arm_cannot_be_selected_as_factorial_arm(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--arm", "cross_rec", "--output", "unused"])

    def test_cpu_allowance_requires_explicit_smoke_before_asset_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = runner.parse_args(["--project-root", str(root), "--allow-cpu",
                                      "--output", "outputs/pilots/stage05_mixer_factorial/new"])
            with self.assertRaises(ValueError):
                runner.run(args)
            self.assertFalse((root / "outputs").exists())

    def test_resolved_historical_controls_must_match_including_smoothing(self):
        expected = dict(label_smoothing=0.2, train_batch_size=64, eval_batch_size=8,
                        gradient_accumulation_steps=16)
        runner.verify_resolved_controls(expected)
        for key, wrong in (("label_smoothing", 0.1), ("train_batch_size", 128),
                           ("eval_batch_size", 16), ("gradient_accumulation_steps", 8)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.verify_resolved_controls(dict(expected, **{key: wrong}))


class EngineRootSelectionTests(unittest.TestCase):
    def create_engine(self, root):
        directory = root / "genrec/models/DIFF_GRM"
        directory.mkdir(parents=True)
        for name in ("model.py", "cross_pse.py"):
            with (directory / name).open("x", encoding="utf-8") as handle:
                handle.write("# Synthetic filesystem-resolution fixture; never imported.\n")

    def test_default_engine_remains_inside_local_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            engine = root / "external/DiffGRM"
            self.create_engine(engine)
            self.assertEqual(resolve_engine_root(root, None), engine)

    def test_explicit_readonly_engine_can_live_outside_sparse_code_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sparse = root / "sparse_code"
            engine = root / "readonly_engine"
            self.create_engine(engine)
            before = {path: path.read_bytes() for path in engine.rglob("*") if path.is_file()}
            self.assertEqual(resolve_engine_root(sparse, str(engine)), engine)
            self.assertFalse((sparse / "external").exists())
            self.assertEqual(before, {path: path.read_bytes() for path in engine.rglob("*") if path.is_file()})

    def test_missing_or_partial_engine_is_a_failure_not_a_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(FileNotFoundError):
                resolve_engine_root(root, None)
            engine = root / "incomplete"
            directory = engine / "genrec/models/DIFF_GRM"
            directory.mkdir(parents=True)
            with (directory / "cross_pse.py").open("x", encoding="utf-8") as handle:
                handle.write("# Only one source file.\n")
            with self.assertRaises(FileNotFoundError) as error:
                resolve_engine_root(root, str(engine))
            self.assertIn("model.py", str(error.exception))

    def test_empty_or_relative_override_is_rejected(self):
        for override in ("", " ", "external/DiffGRM"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                resolve_engine_root(ROOT, override)


class RunnerStaticControlTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        self.run = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")

    def test_real_execution_explicitly_disables_noise_and_sia_for_both_sides(self):
        expected = {"joint_recommender_sia_weight": 0.0, "joint_tokenizer_sia_weight": 0.0,
                    "joint_diger_gumbel_noise_start": 0.0, "joint_diger_gumbel_noise_end": 0.0,
                    "joint_diger_recommendation_weight": 1.0}
        observed = {}
        for node in ast.walk(self.run):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in expected and isinstance(value, ast.Constant):
                        observed[key.value] = value.value
        self.assertEqual(observed, expected)

    def test_bootstrap_rng_uses_cpu_generator_not_cuda_global_reseed(self):
        rebase = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "rebase_common_mixer")
        calls = [node for node in ast.walk(rebase) if isinstance(node, ast.Call)]
        seeds = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed"]
        self.assertEqual(len(seeds), 1)
        self.assertIsInstance(seeds[0].func.value, ast.Attribute)
        self.assertEqual(seeds[0].func.value.attr, "default_generator")
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "fork_rng" for node in calls))

    def test_common_initial_clone_and_frozen_negative_control_are_enforced(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("model.load_state_dict(common_state, strict=True)", source)
        self.assertIn("Frozen negative-control final complete tensors differ", source)
        self.assertIn("Frozen negative-control validation metrics differ", source)
        for receipt in ("initial_effective", "anchor"):
            self.assertIn(receipt, source) if receipt == "anchor" else self.assertIn("effective_codebooks_equal", source)
        for key in ("phase_seed", "data_seed", "data_order_sha256", "optimizer_updates"):
            self.assertIn(key, source)

    def test_pilot_evaluates_validation_only_not_test_or_best_epoch_fit(self):
        calls = [node for node in ast.walk(self.run) if isinstance(node, ast.Call)]
        attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertNotIn("fit", attributes)
        self.assertNotIn("run", attributes)
        self.assertNotIn("evaluate", attributes)
        self.assertIn("validate_snapshot", attributes)
        self.assertIn("train_phase", attributes)


class FactorialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = helper.load_protocol(CONFIG_PATH)

    def test_four_arms_are_exact_two_by_two_factorial(self):
        expected = ("no_mixer_frozen", "no_mixer_joint", "mixer_frozen", "mixer_joint")
        self.assertEqual(helper.arm_names(self.protocol), expected)
        self.assertEqual([(helper.arm_settings(self.protocol, arm)["mixer_enabled"],
                           helper.arm_settings(self.protocol, arm)["tokenizer_batches"] > 0)
                          for arm in expected], [(False, False), (False, True), (True, False), (True, True)])

    def test_matched_formal_budgets_and_feedback(self):
        for arm in helper.ARMS:
            settings = helper.arm_settings(self.protocol, arm)
            self.assertEqual(settings["tokenizer_batches"], 384 if arm.endswith("_joint") else 0)
            self.assertEqual(settings["gradient_accumulation_steps"], 16)
            self.assertEqual(settings["tokenizer_batches"] // settings["gradient_accumulation_steps"],
                             24 if arm.endswith("_joint") else 0)
            self.assertEqual(settings["decoder_epochs"], 3)
            self.assertEqual(settings["decoder_batch_limit"], 0)
            self.assertEqual(settings["eval_batch_limit"], 0)
            self.assertEqual(settings["recommendation_weight"], 1.0)
            self.assertEqual(settings["noise_start"], 0.0)
            self.assertEqual(settings["noise_end"], 0.0)

    def test_smoke_budget_is_explicit_not_formal_and_probes_do_not_select_epochs(self):
        for arm in helper.ARMS:
            settings = helper.arm_settings(self.protocol, arm, smoke=True)
            joint = arm.endswith("_joint")
            self.assertEqual(settings["tokenizer_batches"], 16 if joint else 0)
            self.assertEqual(settings["decoder_batch_limit"], 16)
            self.assertEqual(settings["decoder_epochs"], 1)
            self.assertEqual(settings["eval_batch_limit"], 2)
            self.assertEqual(settings["tokenizer_probe_updates"], [1] if joint else [])
        self.assertEqual(self.protocol["evaluation"]["checkpoint_selection"], "none_fixed_budget")
        self.assertEqual(self.protocol["evaluation"]["primary_decoder_epoch"], 3)
        self.assertTrue(self.protocol["evaluation"]["no_test"])

    def test_common_initialization_controls_have_effective_anchor_and_live_gate(self):
        controls = self.protocol["mixer_initialization"]
        self.assertEqual(controls["base"], "bound_common_initial_effective_codebooks")
        self.assertEqual(controls["n_heads"], 4)
        self.assertEqual(controls["dropout"], 0)
        self.assertEqual(controls["residual_scale_init"], 1)
        self.assertFalse(controls["learn_residual_scale"])
        self.assertTrue(controls["zero_init_attention_output"])
        self.assertTrue(controls["train_base_codebooks"])
        self.assertEqual(self.protocol["mixer_init_seed"], 2024)

    def test_arm_flags_do_not_change_logical_epoch_rng_seed(self):
        for epoch in (1, 2, 3):
            observations = []
            for arm in helper.ARMS:
                helper.arm_settings(self.protocol, arm)
                observations.append((helper.phase_seed(self.protocol["decoder_seed_base"], "decoder", epoch),
                                     helper.phase_seed(self.protocol["data_seed_base"], "data", epoch)))
            self.assertEqual(len(set(observations)), 1)

    def test_resolved_label_smoothing_is_locked_to_repo_v6_point_two(self):
        values = []
        for relative in self.protocol["config_layers"]:
            matches = re.findall(r"(?m)^label_smoothing:[ \t]*([0-9]+(?:\.[0-9]+)?)[ \t]*(?:#.*)?$",
                                 (ROOT / relative).read_text(encoding="utf-8"))
            values.extend(float(value) for value in matches)
        self.assertEqual(values[-1], 0.2)
        self.assertEqual(self.protocol["expected_label_smoothing"], 0.2)

    def test_changed_protocol_controls_are_rejected(self):
        changes = (("tokenizer_batches", 96), ("decoder_epochs", 4), ("noise_start", 1.0),
                   ("noise_end", 0.2), ("expected_label_smoothing", 0.1), ("mixer_init_seed", 2025),
                   ("gradient_accumulation_steps", 8), ("softmax_temperature", 0.5))
        for key, value in changes:
            protocol = copy.deepcopy(self.protocol)
            protocol[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.validate_protocol(protocol)

    def test_old_base_zero_gate_or_dropout_bootstraps_are_rejected(self):
        for key, value in (("base", "old_base_pse"), ("residual_scale_init", 0),
                           ("dropout", 0.1), ("zero_init_attention_output", False),
                           ("train_base_codebooks", False)):
            protocol = copy.deepcopy(self.protocol)
            protocol["mixer_initialization"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.validate_protocol(protocol)

    def test_missing_duplicate_or_reordered_arms_rejected(self):
        for arms in (list(helper.ARMS[:-1]), [*helper.ARMS[:-1], helper.ARMS[0]], list(reversed(helper.ARMS))):
            protocol = copy.deepcopy(self.protocol)
            protocol["arms"] = arms
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                helper.validate_protocol(protocol)

    def test_joint_feedback_cannot_be_disabled_in_only_one_arm(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["recommendation_weights"]["no_mixer_joint"] = 0
        with self.assertRaises(ValueError):
            helper.validate_protocol(protocol)


class BoundCommonSnapshotTests(unittest.TestCase):
    def bundle(self, root, *, mode="pilot", tokenizer_updates=0):
        protocol = helper.load_protocol(CONFIG_PATH)
        run = root / protocol["common_snapshot_run"]
        snapshot = run / "frozen/snapshots/initial"
        snapshot.mkdir(parents=True)
        files = {}
        for name in ("model.pt", "training_state.pt", "semantic_ids.npy", "crossed_codebooks.npy"):
            with (snapshot / name).open("xb") as handle:
                handle.write(("synthetic " + name).encode("ascii"))
            files[name] = helper.file_sha256(snapshot / name)
        row = dict(label="initial", tokenizer_updates=tokenizer_updates, recommender_updates=0,
                   recommender_epochs=0, files=files, model_tensor_state_sha256="c" * 64,
                   semantic_ids_array_sha256="d" * 64)
        summary = dict(mode=mode, status="complete", arms=[dict(arm="frozen", snapshots=[row])])
        with (run / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle)
        protocol["common_snapshot_summary_sha256"] = helper.file_sha256(run / "summary.json")
        with (run / "COMPLETE.json").open("x", encoding="utf-8") as handle:
            json.dump(dict(summary_sha256=protocol["common_snapshot_summary_sha256"],
                           all_protocol_arms_complete=True), handle)
        return protocol, snapshot, run

    def test_bound_pre_update_initial_snapshot_is_accepted_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protocol, snapshot, run = self.bundle(root)
            before = {path: helper.file_sha256(path) for path in run.rglob("*") if path.is_file()}
            receipt = helper.verify_common_snapshot(root, protocol)
            self.assertEqual(Path(receipt["snapshot_dir"]), snapshot)
            self.assertEqual(receipt["summary_sha256"], protocol["common_snapshot_summary_sha256"])
            self.assertEqual(before, {path: helper.file_sha256(path) for path in run.rglob("*") if path.is_file()})

    def test_changed_common_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protocol, snapshot, _ = self.bundle(root)
            with (snapshot / "model.pt").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(RuntimeError):
                helper.verify_common_snapshot(root, protocol)

    def test_source_smoke_or_already_updated_initial_cannot_supply_common_start(self):
        for mode, updates in (("smoke", 0), ("pilot", 1)):
            with self.subTest(mode=mode, updates=updates), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                protocol, _, _ = self.bundle(root, mode=mode, tokenizer_updates=updates)
                with self.assertRaises(RuntimeError):
                    helper.verify_common_snapshot(root, protocol)

    def test_common_snapshot_missing_completion_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protocol, _, run = self.bundle(root)
            with (run / "COMPLETE.json").open("w", encoding="utf-8") as handle:
                json.dump(dict(summary_sha256=protocol["common_snapshot_summary_sha256"],
                               all_protocol_arms_complete=False), handle)
            with self.assertRaises(RuntimeError):
                helper.verify_common_snapshot(root, protocol)


def isolated_model_methods():
    """Use the actual phase/sync implementation without package-level imports."""
    path = ENGINE_ROOT / "genrec/models/DIFF_GRM/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"joint_tokenizer_parameter_groups", "set_joint_phase", "sync_joint_tokenizer_mapping"}
    methods = [node for klass in tree.body if isinstance(klass, ast.ClassDef)
               for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise RuntimeError("Could not isolate actual model phase/sync methods")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(methods), type_ignores=[])), str(path), "exec"), namespace)
    return namespace


@unittest.skipIf(torch is None or np is None, "Torch/NumPy unavailable; real CPU tensor checks must run with --require-torch on server")
class TensorBootstrapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(31)
        module = load_module("_factorial_cross_module_tests",
                             ENGINE_ROOT / "genrec/models/DIFF_GRM/cross_pse.py")
        self.mixer_class = module.CrossPSECodebookMixer
        methods = isolated_model_methods()

        class Tokenizer:
            id2item = ["<pad>", "a", "b", "c", "d", "e", "f"]
            item2id = {item: index for index, item in enumerate(id2item)}
            sid_offset = 7
            _cb2items = {"stale": True}

            def _create_reverse_mapping(self):
                return {tokens: self.item2id[item] for item, tokens in self.item2tokens.items()}

            def _compute_collision_stats(self, mapping):
                return {"unique": len(set(mapping.values())), "items": len(mapping)}

            def _finalize_position_stats(self, values):
                return values

        class Alignment(torch.nn.Module):
            def __init__(self, mixer, slots):
                super().__init__()
                self.mixer = mixer
                self.register_buffer("item_slots", slots)
                self.register_buffer("hard_codes", torch.full((len(slots), 4), -1, dtype=torch.long))
                self.tokenizer_variant = "cross_pse"

            def set_recommender_cache(self, enabled):
                self.cache_enabled = enabled

        class ToyModel(torch.nn.Module):
            joint_tokenizer_parameter_groups = methods["joint_tokenizer_parameter_groups"]
            set_joint_phase = methods["set_joint_phase"]
            sync_joint_tokenizer_mapping = methods["sync_joint_tokenizer_mapping"]

            def __init__(self, mixer, slots):
                super().__init__()
                self.encoder = torch.nn.Linear(64, 64)
                self.decoder = torch.nn.Linear(64, 64)
                self.joint_alignment = Alignment(mixer, slots)
                self.joint_crosspse_enabled = True
                self.config = {}
                self.tokenizer = Tokenizer()
                self.codebook_size = 256
                self.n_digit = 4

        old_base = torch.randn(4, 256, 64)
        old_mixer = self.mixer_class(old_base, n_heads=4, dropout=0,
                                     residual_scale_init=1, learn_residual_scale=False,
                                     zero_init_attention_output=False, train_base_codebooks=True)
        with torch.no_grad():
            old_mixer.attention.out_proj.bias.fill_(0.25)
            self.effective = old_mixer.crossed_codebooks(False)[0].detach().clone()
        self.old_base = old_base.clone()
        self.model = ToyModel(old_mixer, torch.randn(7, 4, 64))
        with torch.no_grad():
            codes = old_mixer.squared_distances(self.model.joint_alignment.item_slots[1:], self.effective).argmin(-1)
            self.model.joint_alignment.hard_codes[1:].copy_(codes)
        self.initial_codes = self.model.joint_alignment.hard_codes.detach().clone()

    def rebase(self):
        return runner.rebase_common_mixer(self.model, self.effective, init_seed=2024,
                                          mixer_class=self.mixer_class)

    def test_rebase_uses_effective_books_not_old_base_and_keeps_nearest_codes(self):
        self.assertFalse(torch.equal(self.effective, self.old_base))
        mixer = self.rebase()
        self.assertTrue(torch.equal(mixer.base_codebooks, self.effective))
        self.assertTrue(torch.equal(mixer.crossed_codebooks(False)[0], self.effective))
        fresh = mixer.squared_distances(self.model.joint_alignment.item_slots[1:],
                                        mixer.crossed_codebooks(False)[0]).argmin(-1)
        self.assertTrue(torch.equal(fresh, self.initial_codes[1:]))
        self.assertTrue(torch.equal(self.model.joint_alignment.hard_codes, self.initial_codes))

    def test_nonpersistent_anchor_is_common_effective_book_and_not_checkpoint_field(self):
        mixer = self.rebase()
        self.assertTrue(torch.equal(mixer.initial_codebooks, self.effective))
        self.assertFalse(torch.equal(mixer.initial_codebooks, self.old_base))
        self.assertNotIn("initial_codebooks", mixer.state_dict())
        self.assertIsNot(mixer.base_codebooks, mixer.initial_codebooks)
        with torch.no_grad():
            mixer.base_codebooks.add_(0.1)
        self.assertTrue(torch.equal(mixer.initial_codebooks, self.effective))

    def test_zero_output_projection_fixed_unit_scale_has_nonzero_first_gradient(self):
        mixer = self.rebase()
        self.assertTrue(torch.equal(mixer.attention.out_proj.weight, torch.zeros_like(mixer.attention.out_proj.weight)))
        self.assertTrue(torch.equal(mixer.attention.out_proj.bias, torch.zeros_like(mixer.attention.out_proj.bias)))
        self.assertEqual(float(mixer.residual_scale), 1.0)
        self.assertNotIsInstance(mixer.residual_scale, torch.nn.Parameter)
        self.model.set_joint_phase("tokenizer")
        books = mixer.crossed_codebooks(False)[0]
        loss = (books - (self.effective + 0.1)).square().sum()
        loss.backward()
        norm = sum(float(parameter.grad.abs().sum()) for parameter in mixer.attention.out_proj.parameters()
                   if parameter.grad is not None)
        self.assertGreater(norm, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_disabled_mixer_never_revives_interaction_parameters_during_phase_switches(self):
        mixer = self.rebase()
        runner.configure_variant(self.model, dict(mixer_enabled=False, tokenizer_variant="refined_pse"))
        for phase in ("tokenizer", "recommender", "tokenizer", "recommender"):
            self.model.set_joint_phase(phase)
            interactions = [value for name, value in mixer.named_parameters() if name != "base_codebooks"]
            self.assertTrue(all(not value.requires_grad for value in interactions))
            self.assertEqual(mixer.base_codebooks.requires_grad, phase == "tokenizer")
            groups = dict(self.model.joint_tokenizer_parameter_groups())
            self.assertEqual(groups["joint_mixer"], [])
            self.assertTrue(torch.equal(mixer.crossed_codebooks(False)[0], self.effective))

    def test_bootstrap_and_switch_do_not_consume_cpu_rng(self):
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.rebase()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if cuda_before is not None:
            self.assertTrue(all(torch.equal(left, right) for left, right in
                                zip(cuda_before, torch.cuda.get_rng_state_all())))
        for enabled in (False, True):
            runner.configure_variant(self.model, dict(mixer_enabled=enabled,
                                     tokenizer_variant="cross_pse" if enabled else "refined_pse"))
            self.model.set_joint_phase("tokenizer")
            self.model.joint_alignment.mixer.crossed_codebooks(False)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_four_arm_clones_share_initial_weights_effective_books_and_official_mapping(self):
        self.rebase()
        common = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        for enabled, updating in ((False, False), (True, False), (False, True), (True, True)):
            model = copy.deepcopy(self.model)
            model.load_state_dict(common, strict=True)
            runner.configure_variant(model, dict(mixer_enabled=enabled,
                                     tokenizer_variant="cross_pse" if enabled else "refined_pse"))
            model.set_joint_phase("tokenizer" if updating else "recommender")
            self.assertTrue(torch.equal(model.joint_alignment.mixer.crossed_codebooks(False)[0], self.effective))
            self.assertTrue(torch.equal(model.joint_alignment.mixer.initial_codebooks, self.effective))
            self.assertTrue(torch.equal(model.joint_alignment.hard_codes, self.initial_codes))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, common[name]), name)
            model.sync_joint_tokenizer_mapping()
            self.assertIsNone(model.tokenizer._cb2items)
            for item_id, item in enumerate(model.tokenizer.id2item[1:], 1):
                expected = tuple(int(code) + 7 + position * 256
                                 for position, code in enumerate(self.initial_codes[item_id]))
                self.assertEqual(model.tokenizer.item2tokens[item], expected)
                self.assertIn(expected, model.tokenizer.tokens2item)

    def test_rebase_refuses_non_tensor_effective_codebooks(self):
        with self.assertRaises((TypeError, ValueError)):
            runner.rebase_common_mixer(self.model, [[[0.0]]], init_seed=2024,
                                       mixer_class=self.mixer_class)

    def test_three_real_cpu_updates_respect_variant_groups_and_frozen_recommender(self):
        self.rebase()
        for enabled in (False, True):
            model = copy.deepcopy(self.model)
            runner.configure_variant(model, dict(mixer_enabled=enabled,
                                     tokenizer_variant="cross_pse" if enabled else "refined_pse"))
            model.set_joint_phase("tokenizer")
            mixer = model.joint_alignment.mixer
            groups = dict(model.joint_tokenizer_parameter_groups())
            self.assertEqual(bool(groups["joint_mixer"]), enabled)
            non_tokenizer_before = {name: value.detach().clone() for name, value in model.state_dict().items()
                                    if not name.startswith("joint_alignment.")}
            interactions_before = {name: value.detach().clone() for name, value in mixer.named_parameters()
                                   if name != "base_codebooks"}
            optimizer = torch.optim.SGD([value for values in groups.values() for value in values], lr=1e-4)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                model.set_joint_phase("tokenizer")
                loss = mixer.objective(model.joint_alignment.item_slots[1:], balance_weight=0.01,
                                       anchor_weight=0.01, balance_temperature=0.1)["loss"]
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                optimizer.step()
            for name, value in model.state_dict().items():
                if name in non_tokenizer_before:
                    self.assertTrue(torch.equal(value, non_tokenizer_before[name]), name)
            changes = [not torch.equal(value, interactions_before[name]) for name, value in mixer.named_parameters()
                       if name != "base_codebooks"]
            self.assertEqual(any(changes), enabled)


if __name__ == "__main__":
    required = "--require-torch" in sys.argv
    if required:
        sys.argv.remove("--require-torch")
        if torch is None or np is None:
            raise SystemExit("Torch/NumPy missing: required real CPU tensor checks cannot be skipped")
    unittest.main(verbosity=2)
