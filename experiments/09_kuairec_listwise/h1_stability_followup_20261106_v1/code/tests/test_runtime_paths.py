"""CPU-only relocation checks, separate from the sealed historical inventory."""
from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime_paths as runtime


def source_functions(relative, names, namespace):
    """Exercise actual path readers without importing torch/numpy/model code."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError("Requested reader changed or disappeared")
    exec(compile(ast.Module(body=functions, type_ignores=[]), relative, "exec"), namespace)
    return namespace


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime-path-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "new-project"
        self.root.mkdir()
        self.engine = self.root / "external/DiffGRM"
        self.engine.mkdir(parents=True)

    def resolve(self, value, **kwargs):
        return runtime.resolve_project_path(value, project_root=self.root, **kwargs)

    def namespace(self):
        return {"Path": Path, "PROJECT_ROOT": self.root, "DIFF_ROOT": self.engine,
                "resolve_project_path": runtime.resolve_project_path, "json": json, "csv": csv}

    def test_all_known_roots_map_to_new_root(self):
        for prefix in runtime.LEGACY_PROJECT_ROOTS:
            with self.subTest(prefix=prefix):
                self.assertEqual(self.resolve(prefix + "/data/input.bin"), self.root / "data/input.bin")
                self.assertEqual(self.resolve(prefix), self.root)

    def test_windows_backslashes_and_case_are_understood_on_any_host(self):
        self.assertEqual(self.resolve(r"c:\USERS\Nancy\Documents\EXPERIMENT1\data\input.bin"),
                         self.root / "data/input.bin")

    def test_prefix_boundary_does_not_relocate_other_project(self):
        for prefix in runtime.LEGACY_PROJECT_ROOTS:
            value = prefix + "_other/data/input.bin"
            self.assertEqual(self.resolve(value), Path(value))

    def test_unknown_absolute_path_remains_explicit(self):
        for value in ("/unrelated/project/input.bin", "D:/other/project/input.bin"):
            self.assertEqual(self.resolve(value), Path(value))

    def test_new_copy_wins_even_when_old_file_exists(self):
        old = self.base / "old-project"
        old.mkdir()
        (old / "input.bin").write_bytes(b"old bytes")
        (self.root / "input.bin").write_bytes(b"new bytes")
        with mock.patch.object(runtime, "LEGACY_PROJECT_ROOTS", (old.as_posix(),)):
            selected = self.resolve(old / "input.bin")
        self.assertEqual(selected.read_bytes(), b"new bytes")
        self.assertEqual((old / "input.bin").read_bytes(), b"old bytes")

    def test_missing_new_copy_never_falls_back_to_existing_old_file(self):
        old = self.base / "old-project"
        old.mkdir()
        (old / "input.bin").write_bytes(b"old only")
        with mock.patch.object(runtime, "LEGACY_PROJECT_ROOTS", (old.as_posix(),)):
            selected = self.resolve(old / "input.bin")
        self.assertEqual(selected, self.root / "input.bin")
        self.assertFalse(selected.exists())

    def test_relocated_parent_traversal_is_rejected(self):
        for prefix in runtime.LEGACY_PROJECT_ROOTS:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                self.resolve(prefix + "/data/../../outside")

    def test_relocated_embedded_drive_is_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve(runtime.LEGACY_PROJECT_ROOTS[0] + "/D:/outside")

    def test_relative_engine_paths_preserve_existing_semantics(self):
        self.assertEqual(self.resolve("saved/run/model.bin", relative_to=self.engine),
                         self.engine / "saved/run/model.bin")
        self.assertEqual(self.resolve("../../data/input.bin", relative_to=self.engine),
                         self.root / "data/input.bin")

    def test_relative_project_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve("../../../outside", relative_to=self.engine)

    def test_current_root_absolute_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve(self.root / ".." / "outside")

    def test_relocated_symlink_escape_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.root / "link").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest("Symlink privilege unavailable: " + str(exc))
        with self.assertRaises(ValueError):
            self.resolve(runtime.LEGACY_PROJECT_ROOTS[0] + "/link/input.bin")
        with self.assertRaises(ValueError):
            self.resolve(self.root / "link/input.bin")

    def test_relative_base_symlink_escape_is_rejected(self):
        outside = self.base / "outside-engine"
        outside.mkdir()
        linked_base = self.root / "linked-engine"
        try:
            linked_base.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest("Symlink privilege unavailable: " + str(exc))
        with self.assertRaises(ValueError):
            self.resolve("saved/model.bin", relative_to=linked_base)

    def test_ar_csv_metadata_and_checkpoint_reader_relocates(self):
        namespace = self.namespace()
        namespace["VARIANTS"] = {"variant": {"report_column": "variant", "report_value": "variant",
            "report": self.root / "per_seed.csv", "run_id": "recorded-run", "sid_quantizer": "rqvae"}}
        source_functions("scripts/analysis/analyze_ar_self_attention.py",
                         ("project_path", "metadata_from_report", "load_run"), namespace)
        metadata = self.root / "outputs/run_metadata/run.json"
        metadata.parent.mkdir(parents=True)
        checkpoint = self.engine / "saved/run/model.bin"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        prefix = runtime.LEGACY_PROJECT_ROOTS[0]
        metadata.write_text(json.dumps({"run_id": "recorded-run", "seed": 2024, "sid_quantizer": "rqvae",
            "checkpoint": prefix + "/external/DiffGRM/saved/run/model.bin"}), encoding="utf-8")
        with (self.root / "per_seed.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(("variant", "seed", "metadata"))
            writer.writerow(("variant", 2024, prefix + "/outputs/run_metadata/run.json"))
        args = type("Args", (), {"variant": "variant", "seed": 2024, "per_seed_report": None,
                                "metadata": None, "checkpoint": None})()
        _, selected_metadata, selected_checkpoint = namespace["load_run"](args)
        self.assertEqual(selected_metadata, metadata)
        self.assertEqual(selected_checkpoint, checkpoint)

    def test_diff_attention_and_summary_resolvers_relocate(self):
        for script, function in (("analyze_diff_cpd_attention.py", "resolve_diff_path"),
                                 ("summarize_diff_formal.py", "resolve_run_path")):
            namespace = source_functions("scripts/analysis/" + script, (function,), self.namespace())
            selected = namespace[function](runtime.LEGACY_PROJECT_ROOTS[1] + "/outputs/metrics.csv")
            self.assertEqual(selected, self.root / "outputs/metrics.csv")

    def test_four_smoke_check_readers_relocate_without_model_imports(self):
        for script in ("check_diff_crosspse_diger_smoke.py", "check_diff_crosspse_frozen_smoke.py",
                       "check_diff_refinedpse_joint_smoke.py", "check_diff_refinedpse_frozen_smoke.py"):
            namespace = self.namespace()
            namespace["ROOT"] = self.root
            source_functions("scripts/checks/" + script, ("resolve_runtime_path",), namespace)
            with self.subTest(script=script):
                self.assertEqual(namespace["resolve_runtime_path"](
                    runtime.LEGACY_PROJECT_ROOTS[0] + "/outputs/joint_metrics.csv"),
                    self.root / "outputs/joint_metrics.csv")
                self.assertEqual(namespace["resolve_runtime_path"]("../../outputs/joint_metrics.csv"),
                                 self.root / "outputs/joint_metrics.csv")

    def test_ar_summary_reader_prefers_new_path_and_preserves_basename_fallback(self):
        namespace = source_functions("scripts/analysis/summarize_ar_internal_study.py",
                                     ("resolve_position_csv",), self.namespace())
        target = self.root / "outputs/position_loss_logs/nested/metrics.csv"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"csv")
        self.assertEqual(namespace["resolve_position_csv"](
            runtime.LEGACY_PROJECT_ROOTS[0] + "/outputs/position_loss_logs/nested/metrics.csv", self.root), target)
        self.assertEqual(namespace["resolve_position_csv"](
            "../../outputs/position_loss_logs/nested/metrics.csv", self.root), target)
        self.assertEqual(namespace["resolve_position_csv"]("missing/metrics.csv", self.root), target)

    def test_ar_summary_basename_fallback_ignores_external_file_symlink(self):
        namespace = source_functions("scripts/analysis/summarize_ar_internal_study.py",
                                     ("resolve_position_csv",), self.namespace())
        outside = self.base / "outside.csv"
        outside.write_bytes(b"must not read")
        target = self.root / "outputs/position_loss_logs/metrics.csv"
        target.parent.mkdir(parents=True)
        try:
            target.symlink_to(outside)
        except OSError as exc:
            self.skipTest("Symlink privilege unavailable: " + str(exc))
        self.assertIsNone(namespace["resolve_position_csv"]("missing/metrics.csv", self.root))

    def test_plot_position_csv_expression_uses_relocation(self):
        tree = ast.parse((ROOT / "scripts/analysis/plot_ar_internal_study.py").read_text(encoding="utf-8"))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "path" for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        namespace = self.namespace()
        namespace["row"] = {"position_metrics_csv": runtime.LEGACY_PROJECT_ROOTS[0] + "/outputs/metrics.csv"}
        selected = eval(compile(ast.Expression(assignments[0].value), "plot_path", "eval"), namespace)
        self.assertEqual(selected, self.root / "outputs/metrics.csv")

    def test_checkpoint_finder_cli_reads_only_new_metadata_and_weight(self):
        spec = importlib.util.spec_from_file_location("tested_checkpoint_finder", ROOT / "scripts/checks/find_diff_checkpoint.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = self.engine / "saved/run/model.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"checkpoint")
        metadata_dir = self.root / "outputs/run_metadata"
        metadata_dir.mkdir(parents=True)
        metadata = metadata_dir / "run_run_metadata.json"
        metadata.write_text(json.dumps({"run_id": "run", "seed": 2024,
            "checkpoint": runtime.LEGACY_PROJECT_ROOTS[2] + "/external/DiffGRM/saved/run/model.bin"}), encoding="utf-8")
        original = metadata.read_bytes()
        output = io.StringIO()
        with mock.patch.multiple(module, ROOT=self.root, DIFF_ROOT=self.engine), \
                mock.patch.object(sys, "argv", ["find_diff_checkpoint.py", "--run-id", "run"]), \
                contextlib.redirect_stdout(output):
            module.main()
        self.assertEqual(output.getvalue().strip(), str(target))
        self.assertEqual(metadata.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
