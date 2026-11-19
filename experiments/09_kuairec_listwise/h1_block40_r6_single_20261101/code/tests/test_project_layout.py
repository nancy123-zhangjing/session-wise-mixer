"""CPU-only regression tests for the Batch A package relocation."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repro"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

import audit_layout_migration as audit
import project_repro as project
import stage05_repro as stage05


def script_paths(name, names):
    """Read path constants without importing PDF/image dependencies."""
    values = {"ROOT": ROOT}

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return evaluate(node.left) / evaluate(node.right)
        raise ValueError("Unexpected expression in a report path constant")

    tree = ast.parse((ROOT / "scripts" / "reports" / name).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = evaluate(node.value)
    return {name: values[name] for name in names}


class ProjectLayoutTests(unittest.TestCase):
    def test_only_the_singular_output_prefix_is_relocated(self):
        self.assertEqual(audit.relocated("output/01/README.md"), "experiments/01/README.md")
        for path in ("outputs/run.json", "data/output/x", "archive/output/x"):
            self.assertEqual(audit.relocated(path), path)

    def test_package_resolvers_and_all_stage_indexes_exist(self):
        packages = [project.package(s) for s in project.STAGES] + [stage05.PACKAGE]
        self.assertEqual(len(packages), 5)
        for package in packages:
            self.assertEqual(package.parent, ROOT / "experiments")
            self.assertTrue((package / "README.md").is_file())
            self.assertTrue((package / "reproducibility/package_integrity.json").is_file())
        self.assertFalse((ROOT / "output").exists())

    def test_registry_identity_and_all_bound_paths(self):
        before = audit.read(audit.AUDIT / "before.json")["registry"]
        after = audit.read(ROOT / "experiments/experiment_registry.json")
        audit.check_registry(before, after)
        self.assertEqual(len(after["runs"]), 23)
        for run in after["runs"]:
            package = ROOT / run["package"]
            for field in ("metadata", "recipe"):
                self.assertTrue((package / run[field]).is_file(), (run["id"], field))

    def test_registry_guard_rejects_scientific_field_change(self):
        before = {"runs": [{"package": "output/01", "seed": 2024}]}
        after = {"runs": [{"package": "experiments/01", "seed": 2025}]}
        with self.assertRaises(ValueError):
            audit.check_registry(before, after)

    def test_seals_allow_only_reviewed_helper_hash_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "scripts/project_repro.py").write_bytes(b"new helper")
            before = {"files": [{"path": "recipe.json", "sha256": "protected"}],
                      "shared": [{"path": "scripts/project_repro.py", "sha256": "old"}]}
            after = copy.deepcopy(before)
            after["shared"][0]["sha256"] = hashlib.sha256(b"new helper").hexdigest()
            audit.check_seal(before, after, root)
            after["files"][0]["sha256"] = "changed"
            with self.assertRaises(ValueError):
                audit.check_seal(before, after, root)

    def test_seals_do_not_allow_new_or_removed_bindings(self):
        before = {"files": [{"path": "recipe.json", "sha256": "protected"}], "helpers": []}
        with self.assertRaises(ValueError):
            audit.check_seal(before, {"files": [], "helpers": []})

    def test_legacy_report_paths_use_the_existing_stage_package(self):
        pdf = ROOT / "experiments/01_ar_baseline_3x256/pdf/AR_GRM_3x256_stage1_report.pdf"
        source = ROOT / "archive/evidence_bundles/repro_sources_01_04/01_ar_baseline_full_evidence/formal_3x256_evidence_v2"
        build = script_paths("build_stage1_pdf_report.py", {"EVIDENCE", "OUTPUT"})
        assets = script_paths("build_stage1_readme_assets.py", {"EVIDENCE", "PDF", "OUT"})
        render = script_paths("render_stage1_pdf.py", {"PDF", "OUT"})
        self.assertEqual(build, {"EVIDENCE": source, "OUTPUT": pdf})
        self.assertEqual(assets["EVIDENCE"], source)
        self.assertEqual(assets["PDF"], pdf)
        self.assertEqual(assets["OUT"], ROOT / "experiments/01_ar_baseline_3x256/report_assets")
        self.assertEqual(render["PDF"], pdf)
        self.assertTrue(pdf.is_file())
        self.assertTrue(render["OUT"].is_relative_to(ROOT / "tmp"))


if __name__ == "__main__":
    unittest.main()
