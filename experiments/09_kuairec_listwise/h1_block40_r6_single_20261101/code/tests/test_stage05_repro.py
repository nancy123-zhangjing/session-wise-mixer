"""CPU-only tests for the Stage 05 packaging/replay safeguards."""
import argparse
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repro"))

import stage05_repro as stage


class ReproTests(unittest.TestCase):
    def test_reject_unsafe_paths(self):
        for value in ("../outside", "/etc/passwd", "C:/other", "a/../../b", "a\\..\\b", ""):
            with self.subTest(path=value), self.assertRaises(ValueError):
                stage.safe_relative(value)

    def test_historical_path_relocation(self):
        self.assertEqual(stage.project_path("/home/jingzhang/experiment1/outputs/test.pt"), "outputs/test.pt")
        self.assertEqual(stage.project_path("../../outputs/test.pt"), "outputs/test.pt")
        self.assertEqual(stage.project_path("saved/x/model.bin"), "external/DiffGRM/saved/x/model.bin")

    def make_archive(self, folder, link=False):
        archive = folder / "source.tar.gz"
        data = b"original evidence\n"
        name = "experiment1/data/item.bin"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(name)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        records = {name: {"path": "data/item.bin", "sha256": hashlib.sha256(data).hexdigest()}}
        return archive, records, data

    def test_extract_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive, records, data = self.make_archive(folder)
            stage.extract_exact(archive, records, folder / "replay")
            stage.extract_exact(archive, records, folder / "replay")
            self.assertEqual((folder / "replay/data/item.bin").read_bytes(), data)

    def test_extract_preserves_different_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive, records, _ = self.make_archive(folder)
            target = folder / "replay/data/item.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user's newer file")
            with self.assertRaises(ValueError):
                stage.extract_exact(archive, records, folder / "replay")
            self.assertEqual(target.read_bytes(), b"user's newer file")

    def test_extract_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive, records, _ = self.make_archive(folder, link=True)
            with self.assertRaises(ValueError):
                stage.extract_exact(archive, records, folder / "replay")

    def test_missing_member_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive, records, _ = self.make_archive(folder)
            records["experiment1/missing"] = {"path": "missing", "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                stage.extract_exact(archive, records, folder / "replay")

    def test_changed_input_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "data").mkdir()
            (folder / "data/item").write_bytes(b"changed")
            stage.write_json(folder / "reproducibility/assets.lock.json",
                             {"artifacts": {"data/item": {"sha256": "0" * 64}}})
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                stage.check_inputs(folder, {"id": "test", "required_artifacts": ["data/item"]}, folder)

    def test_plan_never_starts_training(self):
        recipe = {"id": "cross_frozen", "initial_decoder": "fixed.bin", "config_layers": []}
        with patch.object(stage, "load_recipe", return_value=recipe), patch.object(stage, "verify_package"), \
                patch.object(stage, "check_inputs"), patch.object(stage.subprocess, "Popen") as launch:
            stage.run_replay(argparse.Namespace(experiment="cross_frozen", mode="smoke", project_root=".", execute=False))
            launch.assert_not_called()

    def test_historical_results_and_phase_counts(self):
        for key, *_ in stage.SPECS:
            stage.analyze_run(stage.PACKAGE, key)
        cross = stage.analyze_run(stage.PACKAGE, "cross_joint_v2")[3]
        refined = stage.analyze_run(stage.PACKAGE, "refined_joint")[3]
        self.assertEqual((cross["best_wall_epoch"], cross["tokenizer_phases_at_best"]), (5, 0))
        self.assertEqual((refined["decoder_epochs_at_best"], refined["tokenizer_phases_at_best"]), (15, 4))

    def test_pinned_source_can_be_unpacked(self):
        lock = stage.read_json(stage.PACKAGE / "reproducibility/code_versions.json")
        source = next(r for r in lock["repositories"] if r["name"] == "DiffGRM")
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine"
            stage.unpack_code(stage.PACKAGE / source["snapshot"], engine)
            self.assertTrue((engine / "main.py").is_file())
            self.assertTrue((engine / "genrec/models/DIFF_GRM/joint_cross_pse.py").is_file())
            with self.assertRaises(ValueError):
                stage.unpack_code(stage.PACKAGE / source["snapshot"], engine)

    def test_recipe_initializations_are_paired(self):
        for first, second in (("cross_frozen", "cross_joint_v2"), ("refined_frozen", "refined_joint")):
            a, b = (stage.load_recipe(stage.PACKAGE, k) for k in (first, second))
            self.assertEqual(a["initial_decoder"], b["initial_decoder"])
            self.assertEqual(a["initial_tokenizer_dir"], b["initial_tokenizer_dir"])
            self.assertIn(a["initial_decoder"], a["required_artifacts"])

    def test_gradient_isolation_guard(self):
        decoder = {"phase": "recommender", "total_loss": "2", "recommender_gradient_norm": "1", "tokenizer_gradient_norm": "0"}
        tokenizer = {"phase": "tokenizer", "total_loss": "1", "recommender_gradient_norm": "0", "tokenizer_gradient_norm": "1", "diger_direct_gradient_norm": ".5"}
        stage.validate_smoke_phases([decoder], frozen=True, direct=False)
        stage.validate_smoke_phases([decoder, tokenizer], frozen=False, direct=True)
        with self.assertRaises(ValueError):
            stage.validate_smoke_phases([decoder, tokenizer], frozen=True, direct=True)
        with self.assertRaises(ValueError):
            stage.validate_smoke_phases([{**decoder, "tokenizer_gradient_norm": "1"}], frozen=True, direct=False)
        with self.assertRaises(ValueError):
            stage.validate_smoke_phases([{**decoder, "total_loss": "nan"}], frozen=True, direct=False)

    def test_formal_config_cannot_silently_change_lr(self):
        meta = {"lr": .001, "train_batch_size": 64}
        self.assertEqual(stage.audit_formal_config(meta.copy(), meta)["status"], "passed")
        with self.assertRaises(ValueError):
            stage.audit_formal_config({**meta, "lr": .01}, meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
