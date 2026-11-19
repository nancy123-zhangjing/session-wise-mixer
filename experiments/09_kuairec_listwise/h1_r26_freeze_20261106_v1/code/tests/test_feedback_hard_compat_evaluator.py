"""Standard-library control tests for frozen-Decoder hard-SID diagnostics.

Synthetic state dictionaries and static checks do not substitute for a real
Torch/GPU smoke or prove that the archived model is compatible.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/analysis/evaluate_feedback_hard_compatibility.py"
spec = importlib.util.spec_from_file_location("_feedback_hard_compat_evaluator_tests", PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load hard compatibility evaluator: {PATH}")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class StateIsolationTests(unittest.TestCase):
    def setUp(self):
        self.initial = {
            "encoder.weight": [1.0, 2.0],
            "decoder.weight": [3.0, 4.0],
            "joint_alignment.mixer.codebooks": [[0.0, 1.0]],
            "joint_alignment.hard_codes": [[0, 1]],
        }
        self.current = copy.deepcopy(self.initial)
        self.current["joint_alignment.mixer.codebooks"] = [[0.25, 1.25]]
        self.current["joint_alignment.hard_codes"] = [[1, 0]]

    def test_recommender_projection_excludes_only_tokenizer_prefix(self):
        state = dict(self.initial, **{"joint_alignment_fake.weight": [5],
                                     "encoder.joint_alignment.weight": [6]})
        self.assertEqual(set(entry.recommender_state(state)),
                         {"encoder.weight", "decoder.weight", "joint_alignment_fake.weight",
                          "encoder.joint_alignment.weight"})

    def test_graft_replaces_tokenizer_only_from_updated_state(self):
        grafted = entry.graft_tokenizer_state(self.initial, self.current)
        self.assertEqual(set(grafted), set(self.initial))
        self.assertIs(grafted["encoder.weight"], self.initial["encoder.weight"])
        self.assertIs(grafted["decoder.weight"], self.initial["decoder.weight"])
        self.assertIs(grafted["joint_alignment.mixer.codebooks"],
                      self.current["joint_alignment.mixer.codebooks"])
        self.assertEqual(grafted["joint_alignment.hard_codes"], [[1, 0]])

    def test_graft_does_not_mutate_either_input(self):
        before = copy.deepcopy((self.initial, self.current))
        entry.graft_tokenizer_state(self.initial, self.current)
        self.assertEqual((self.initial, self.current), before)

    def test_frozen_state_can_be_grafted_without_changes(self):
        self.assertEqual(entry.graft_tokenizer_state(self.initial, copy.deepcopy(self.initial)), self.initial)

    def test_decoder_recovery_checkpoint_cannot_be_used_as_tokenizer_only_state(self):
        self.current["decoder.weight"] = [3.1, 4.0]
        with self.assertRaises((ValueError, RuntimeError)):
            entry.graft_tokenizer_state(self.initial, self.current)

    def test_encoder_difference_also_rejected(self):
        self.current["encoder.weight"] = [1.0, 2.1]
        with self.assertRaises((ValueError, RuntimeError)):
            entry.graft_tokenizer_state(self.initial, self.current)

    def test_missing_or_extra_state_keys_rejected(self):
        for operation in ("missing_decoder", "missing_tokenizer", "extra"):
            with self.subTest(operation=operation):
                state = copy.deepcopy(self.current)
                if operation == "missing_decoder":
                    state.pop("decoder.weight")
                elif operation == "missing_tokenizer":
                    state.pop("joint_alignment.hard_codes")
                else:
                    state["unexpected.weight"] = [1]
                with self.assertRaises((ValueError, RuntimeError)):
                    entry.graft_tokenizer_state(self.initial, state)

    def test_non_tokenizer_prefix_lookalike_is_not_silently_grafted(self):
        initial = dict(self.initial, **{"joint_alignment_fake.weight": [1]})
        current = dict(self.current, **{"joint_alignment_fake.weight": [2]})
        with self.assertRaises((ValueError, RuntimeError)):
            entry.graft_tokenizer_state(initial, current)

    def test_empty_tokenizer_projection_is_rejected(self):
        initial = {"encoder.weight": [1], "decoder.weight": [2]}
        with self.assertRaises((ValueError, RuntimeError)):
            entry.graft_tokenizer_state(initial, copy.deepcopy(initial))

    def test_custom_tensor_equality_is_used_for_all_non_tokenizer_values(self):
        seen = []

        def equal(a, b):
            seen.append((a, b))
            return a == b

        result = entry.graft_tokenizer_state(self.initial, self.current, equal=equal)
        self.assertEqual(result["decoder.weight"], self.initial["decoder.weight"])
        self.assertEqual(len(seen), 2)
        self.assertIn((self.initial["encoder.weight"], self.current["encoder.weight"]), seen)
        self.assertIn((self.initial["decoder.weight"], self.current["decoder.weight"]), seen)


class ResolvedConfigurationTests(unittest.TestCase):
    def test_original_resolved_repo_v6_smoothing_is_accepted(self):
        self.assertEqual(entry.verify_diagnostic_config(dict(label_smoothing=0.2,
                         train_batch_size=64, eval_batch_size=8)), 0.2)

    def test_common_default_smoothing_is_not_the_resolved_experiment(self):
        with self.assertRaises(ValueError) as error:
            entry.verify_diagnostic_config(dict(label_smoothing=0.1, train_batch_size=64, eval_batch_size=8))
        self.assertIn("actual=", str(error.exception))
        self.assertIn("0.1", str(error.exception))

    def test_different_batch_sizes_are_rejected(self):
        for changes in (dict(train_batch_size=128), dict(eval_batch_size=16)):
            config = dict(label_smoothing=0.2, train_batch_size=64, eval_batch_size=8)
            config.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                entry.verify_diagnostic_config(config)

    def test_missing_resolved_configuration_field_is_rejected(self):
        for missing in ("label_smoothing", "train_batch_size", "eval_batch_size"):
            config = dict(label_smoothing=0.2, train_batch_size=64, eval_batch_size=8)
            config.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                entry.verify_diagnostic_config(config)

    def test_bound_top_level_repo_v6_layer_overrides_common_smoothing(self):
        protocol = json.loads((ROOT / "configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json")
                              .read_text(encoding="utf-8"))
        common, repo = "configs/shared/diff_formal_common.yaml", "configs/shared/diff_formal_repo_v6.yaml"
        self.assertLess(protocol["config_layers"].index(common), protocol["config_layers"].index(repo))
        top_level = re.compile(r"(?m)^label_smoothing:[ \t]*([0-9]+(?:\.[0-9]+)?)[ \t]*(?:#.*)?$")
        observed = []
        values_by_layer = {}
        for relative in protocol["config_layers"]:
            values = top_level.findall((ROOT / relative).read_text(encoding="utf-8"))
            self.assertLessEqual(len(values), 1, f"Ambiguous top-level label_smoothing: {relative}")
            values_by_layer[relative] = values
            observed.extend(float(value) for value in values)
        self.assertEqual(values_by_layer[common], ["0.1"])
        self.assertEqual(values_by_layer[repo], ["0.2"])
        self.assertEqual(observed[-1], 0.2)


class DiagnosticReceiptTests(unittest.TestCase):
    def setUp(self):
        self.actual = {"position_nll_sums": [1, 2, 3, 4], "position_loss_sums": [2, 3, 4, 5],
                       "position_correct_counts": [1, 2, 3, 4]}
        self.original = copy.deepcopy(self.actual)
        self.original["position_nll_sums"][-1] = 4.25
        self.original["position_loss_sums"][-1] = 4.5
        self.per_digit = copy.deepcopy(self.actual)
        self.per_digit["position_nll_sums"][-1] = 4.1
        self.per_digit["position_loss_sums"][-1] = 4.75
        self.native_per_digit = copy.deepcopy(self.original)
        self.dtypes = {"logits": "torch.bfloat16", "flattened_nll": "torch.float32",
                       "per_digit_nll": ["torch.float32"] * 4,
                       "original_sums": {"position_nll_sums": "torch.float32"}}

    def receipt(self):
        return entry.diagnostic_receipt(self.actual, self.original, self.per_digit,
                                        self.native_per_digit, self.dtypes)

    def test_raw_aggregates_dtypes_and_original_tolerances_are_retained(self):
        receipt = self.receipt()
        self.assertEqual(receipt["actual"], self.actual)
        self.assertEqual(receipt["reference"], self.original)
        self.assertEqual(receipt["fp32_per_digit"], self.per_digit)
        self.assertEqual(receipt["original_dtype_per_digit"], self.native_per_digit)
        self.assertEqual(receipt["dtypes"], self.dtypes)
        self.assertEqual(receipt["guard_rtol"], 1e-5)
        self.assertEqual(receipt["guard_atol"], 1e-5)
        self.assertEqual(json.loads(json.dumps(receipt)), receipt)

    def test_same_and_cross_precision_discrepancies_are_distinguished(self):
        differences = self.receipt()["max_absolute_differences"]
        self.assertEqual(differences["flat_fp32_vs_original"]["position_nll_sums"], 0.25)
        self.assertAlmostEqual(differences["flat_fp32_vs_per_digit_fp32"]["position_nll_sums"], 0.1)
        self.assertEqual(differences["native_per_digit_vs_original"]["position_nll_sums"], 0.0)
        self.assertEqual(differences["explicit_fp32_vs_original_dtype"]["position_nll_sums"], 0.25)
        self.assertEqual(differences["flat_fp32_vs_original"]["position_loss_sums"], 0.5)
        self.assertEqual(differences["flat_fp32_vs_per_digit_fp32"]["position_loss_sums"], 0.25)
        self.assertEqual(differences["native_per_digit_vs_original"]["position_loss_sums"], 0.0)
        self.assertEqual(differences["explicit_fp32_vs_original_dtype"]["position_loss_sums"], 0.5)
        for source in differences.values():
            self.assertEqual(source["position_correct_counts"], 0.0)

    def test_mismatched_keys_and_position_shapes_are_rejected(self):
        for operation in ("missing_original", "missing_per_digit", "short_actual", "short_original", "short_per_digit"):
            values = copy.deepcopy([self.actual, self.original, self.per_digit])
            if operation.startswith("missing"):
                values[1 if operation == "missing_original" else 2].pop("position_nll_sums")
            else:
                which = {"short_actual": 0, "short_original": 1, "short_per_digit": 2}[operation]
                values[which]["position_nll_sums"].pop()
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                entry.diagnostic_receipt(*values, self.native_per_digit, self.dtypes)

    def test_native_dtype_vector_shape_is_checked_independently(self):
        self.native_per_digit["position_nll_sums"].pop()
        with self.assertRaises(ValueError):
            self.receipt()

    def test_receipt_does_not_mutate_the_supplied_raw_values(self):
        before = copy.deepcopy((self.actual, self.original, self.per_digit, self.native_per_digit, self.dtypes))
        self.receipt()
        self.assertEqual((self.actual, self.original, self.per_digit, self.native_per_digit, self.dtypes), before)

    def test_nonfinite_receipt_cannot_be_silently_serialized_as_json(self):
        self.actual["position_nll_sums"][0] = float("inf")
        with self.assertRaises(ValueError):
            json.dumps(self.receipt(), allow_nan=False)


class PrecisionComparisonPairTests(unittest.TestCase):
    def setUp(self):
        self.actual = {"position_nll_sums": [10.0] * 4}
        self.fp32_per_digit = copy.deepcopy(self.actual)
        self.original = {"position_nll_sums": [10.2] * 4}
        self.native_per_digit = copy.deepcopy(self.original)

    def pairs(self):
        return entry.comparison_pairs(self.actual, self.original, self.fp32_per_digit, self.native_per_digit)

    def required_comparisons_close(self):
        # Standard-library emulation checks which operand pairs are selected;
        # actual CUDA torch.allclose execution still requires GPU smoke.
        return {name: all(abs(a - b) <= 1e-5 + 1e-5 * abs(b)
                         for key in left for a, b in zip(left[key], right[key]))
                for name, (left, right) in self.pairs().items()}

    def test_both_same_precision_pairs_are_present_with_exact_operands(self):
        pairs = self.pairs()
        self.assertEqual(set(pairs), {"explicit_fp32", "original_dtype"})
        self.assertIs(pairs["explicit_fp32"][0], self.actual)
        self.assertIs(pairs["explicit_fp32"][1], self.fp32_per_digit)
        self.assertIs(pairs["original_dtype"][0], self.native_per_digit)
        self.assertIs(pairs["original_dtype"][1], self.original)

    def test_large_cross_precision_difference_does_not_replace_same_precision_checks(self):
        self.assertGreater(abs(self.actual["position_nll_sums"][0]
                               - self.original["position_nll_sums"][0]), 0.1)
        self.assertEqual(self.required_comparisons_close(), {"explicit_fp32": True, "original_dtype": True})

    def test_fp32_path_discrepancy_cannot_be_hidden_by_native_path_agreement(self):
        self.fp32_per_digit["position_nll_sums"][0] += 0.01
        self.assertEqual(self.required_comparisons_close(), {"explicit_fp32": False, "original_dtype": True})

    def test_native_path_discrepancy_cannot_be_hidden_by_fp32_path_agreement(self):
        self.native_per_digit["position_nll_sums"][0] += 0.01
        self.assertEqual(self.required_comparisons_close(), {"explicit_fp32": True, "original_dtype": False})


class OriginalDiagnosticProjectionTests(unittest.TestCase):
    def projection(self):
        tree = ast.parse(PATH.read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        matches = [node for node in ast.walk(run) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "reference" for target in node.targets)
                   and isinstance(node.value, ast.DictComp)
                   and any(isinstance(generator.iter, ast.Name) and generator.iter.id == "comparisons"
                           for generator in node.value.generators)]
        self.assertEqual(len(matches), 1, "Project original extra diagnostics onto scored fields before checking keys")
        return matches[0], run

    def project(self, original, comparisons):
        assignment, _ = self.projection()
        module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(assignment)], type_ignores=[]))
        namespace = dict(reference=original, comparisons=comparisons)
        exec(compile(module, str(PATH), "exec"), namespace)
        return namespace["reference"]

    def test_extra_original_diagnostic_fields_do_not_reach_three_scoring_guards(self):
        scored = {"position_nll_sums": [1.0] * 4, "position_loss_sums": [2.0] * 4,
                  "position_correct_counts": [3] * 4}
        original = dict(scored, position_token_counts=[64] * 4,
                        position_view_counts=[64] * 4, extra_diagnostic_counts=[0] * 4)
        before = copy.deepcopy(original)
        projected = self.project(original, scored)
        self.assertEqual(len(original), 6)
        self.assertEqual(set(projected), set(scored))
        self.assertEqual(len(projected), 3)
        self.assertEqual(original, before)
        pairs = entry.comparison_pairs(scored, projected, scored, scored)
        self.assertEqual(set(pairs), {"explicit_fp32", "original_dtype"})
        self.assertEqual(pairs["original_dtype"][1], scored)

    def test_missing_scored_original_field_is_not_silently_ignored(self):
        scored = {"position_nll_sums": [1.0] * 4, "position_loss_sums": [2.0] * 4,
                  "position_correct_counts": [3] * 4}
        missing = {key: value for key, value in scored.items() if key != "position_nll_sums"}
        with self.assertRaises(KeyError):
            self.project(missing, scored)

    def test_projection_precedes_actual_precision_guard_calls(self):
        projection, run = self.projection()
        pair_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "comparison_pairs"]
        self.assertTrue(pair_calls)
        self.assertLess(projection.lineno, min(call.lineno for call in pair_calls))


class StaticInferenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(PATH.read_text(encoding="utf-8"))
        self.calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]

    def test_no_training_generation_sid_refresh_or_random_mask_call(self):
        attributes = {node.func.attr for node in self.calls if isinstance(node.func, ast.Attribute)}
        names = {node.func.id for node in self.calls if isinstance(node.func, ast.Name)}
        forbidden = {"fit", "backward", "step", "train_phase", "refresh_joint_codes",
                     "refresh_hard_codes", "generate", "full_mask_diagnostics", "random_mask",
                     "sample", "multinomial", "rand", "randn", "gumbel_softmax"}
        # Original full-mask diagnostic aggregation may be invoked for a
        # consistency check; it is deterministic and not a sampling path.
        forbidden.remove("full_mask_diagnostics")
        self.assertFalse((attributes | names) & forbidden)

    def test_no_optimizer_or_fit_entrypoint_is_constructed(self):
        constructors = {node.func.attr for node in self.calls if isinstance(node.func, ast.Attribute)}
        self.assertFalse(constructors & {"Adam", "AdamW", "SGD", "Optimizer"})

    def test_inference_mode_and_eval_are_explicit(self):
        attributes = {node.func.attr for node in self.calls if isinstance(node.func, ast.Attribute)}
        self.assertIn("inference_mode", attributes)
        self.assertIn("eval", attributes)

    def test_all_model_parameters_are_explicitly_frozen(self):
        calls = [node for node in self.calls
                 if isinstance(node.func, ast.Attribute) and node.func.attr == "requires_grad_"]
        self.assertTrue(any(node.args and isinstance(node.args[0], ast.Constant)
                            and node.args[0].value is False for node in calls))

    def test_prepared_batch_uses_hard_official_labels_not_differentiable_assignments(self):
        calls = [node for node in self.calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == "prepare_joint_batch"]
        self.assertTrue(calls)
        for call in calls:
            keywords = {key.arg: key.value for key in call.keywords}
            for key in ("training", "differentiable"):
                self.assertIsInstance(keywords[key], ast.Constant)
                self.assertIs(keywords[key].value, False)

    def test_only_initial_and_post_tokenizer_snapshots_are_selected(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn("('initial', 'post_tokenizer')", source)
        self.assertNotIn("decoder_3", source)
        self.assertEqual(entry.SPLITS, ("train", "val"))

    def test_decoder_forward_has_no_cache_and_explicit_mask_tensors(self):
        calls = [node for node in self.calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == "forward_decoder_only"]
        self.assertTrue(calls)
        for call in calls:
            keywords = {key.arg: key.value for key in call.keywords}
            self.assertIs(keywords["use_cache"].value, False)
        source = PATH.read_text(encoding="utf-8")
        self.assertIn("'decoder_input_ids': torch.zeros_like(labels)", source)
        self.assertIn("'mask_positions': torch.ones_like(labels", source)

    def test_pipeline_default_optimizer_constructor_is_bypassed(self):
        calls = [node for node in self.calls if isinstance(node.func, ast.Name) and node.func.id == "Pipeline"]
        self.assertEqual(len(calls), 1)
        keywords = {key.arg: key.value for key in calls[0].keywords}
        self.assertIn("trainer", keywords)
        self.assertIsInstance(keywords["trainer"], ast.Call)
        self.assertEqual(keywords["trainer"].func.id, "object")

    def test_full_mask_receipt_is_written_before_unchanged_tolerance_guard(self):
        writes = [node for node in self.calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "write_json" and node.args
                  and isinstance(node.args[0], ast.Name) and node.args[0].id == "comparison_path"]
        guards = [node for node in self.calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "allclose"]
        self.assertEqual(len(writes), 1)
        self.assertTrue(guards)
        self.assertLess(writes[0].lineno, min(guard.lineno for guard in guards))
        for guard in guards:
            keywords = {key.arg: key.value for key in guard.keywords}
            self.assertEqual(keywords["rtol"].value, 1e-5)
            self.assertEqual(keywords["atol"].value, 1e-5)
        source = PATH.read_text(encoding="utf-8")
        self.assertIn("Same-precision full-mask diagnostic mismatch", source)
        self.assertIn("original_dtype_per_digit", source)

    def test_guard_consumes_both_pairs_not_a_cross_precision_comparison(self):
        run = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        pair_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == "comparison_pairs"]
        self.assertEqual(len(pair_calls), 1)
        self.assertEqual(len(pair_calls[0].args), 4)
        guards = [node for node in self.calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "allclose"]
        enclosing = [node for node in ast.walk(self.tree) if isinstance(node, ast.For)
                     and pair_calls[0] in list(ast.walk(node))
                     and any(guard in list(ast.walk(node)) for guard in guards)]
        self.assertTrue(enclosing, "Both selected precision pairs must be iterated by the actual guard")

    def test_main_per_position_statistics_keep_explicit_fp32_logits(self):
        flat_assignments = [node for node in ast.walk(self.tree) if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "flat" for target in node.targets)]
        self.assertEqual(len(flat_assignments), 1)
        attributes = {node.func.attr for node in ast.walk(flat_assignments[0].value)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn("float", attributes)
        self.assertIn("reshape", attributes)
        self.assertIn("('position_nll', own_nll)", PATH.read_text(encoding="utf-8"))


class EvidenceScopeTests(unittest.TestCase):
    def test_input_outside_feedback_pilot_scope_is_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for candidate in ("outputs/pilots/stage05_noise/old", "experiments/05_crosspse_joint/evidence/old",
                              "outputs/pilots/stage05_feedback/old/nested"):
                with self.subTest(candidate=candidate):
                    args = SimpleNamespace(project_root=root, run_dir=Path(candidate),
                                           output=Path("outputs/analysis/stage05_feedback/new"), output_base=None)
                    with self.assertRaises(ValueError):
                        entry.run(args)
                    self.assertFalse((root / "outputs/analysis").exists())

    def test_smoke_training_receipt_is_not_accepted_as_formal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir = root / "outputs/pilots/stage05_feedback/smoke"
            run_dir.mkdir(parents=True)
            summary = dict(mode="smoke", status="complete", all_protocol_arms_complete=True,
                           no_test_evaluation=True)
            with (run_dir / "summary.json").open("x", encoding="utf-8") as handle:
                json.dump(summary, handle)
            shared = entry.load_module("_hard_compat_scope_shared", PATH.with_name("evaluate_feedback_paired.py"))
            with (run_dir / "COMPLETE.json").open("x", encoding="utf-8") as handle:
                json.dump(dict(summary_sha256=shared.sha256(run_dir / "summary.json"), arms=list(entry.ARMS)), handle)
            args = SimpleNamespace(project_root=root, run_dir=run_dir,
                                   output=Path("outputs/analysis/stage05_feedback/new"), output_base=None)
            with self.assertRaises(ValueError):
                entry.run(args)
            self.assertFalse((root / "outputs/analysis").exists())

    def test_historical_or_arbitrary_ssd_output_scope_is_rejected(self):
        shared = entry.load_module("_hard_compat_output_shared", PATH.with_name("evaluate_feedback_paired.py"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for candidate in ("outputs/pilots/stage05_feedback/old", "experiments/05_crosspse_joint/evidence/new"):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    shared.validate_output(root, Path(candidate), None)
            with self.assertRaises(ValueError):
                shared.validate_output(root, root / "arbitrary_ssd/new", root / "arbitrary_ssd")

    def test_existing_analysis_directory_cannot_be_overwritten(self):
        shared = entry.load_module("_hard_compat_existing_shared", PATH.with_name("evaluate_feedback_paired.py"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "outputs/analysis/stage05_feedback/preserved"
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                shared.validate_output(root, target, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
