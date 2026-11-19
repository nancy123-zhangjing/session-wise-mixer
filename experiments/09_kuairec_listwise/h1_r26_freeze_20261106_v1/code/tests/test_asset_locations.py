"""CPU-only tests: locations may move; locked asset identities may not."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repro"))

import asset_locations as locations
import project_repro as project
import stage05_repro as stage05


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class AssetLocationTests(unittest.TestCase):
    def test_explicit_path_does_not_read_local_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "asset.tar.gz"
            archive.write_bytes(b"asset")
            with patch.object(Path, "read_text", side_effect=AssertionError("No mapping read allowed")):
                self.assertEqual(locations.resolve_archive(archive, root), archive.resolve())

    def test_relative_map_is_relative_to_project_not_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.tar.gz").write_bytes(b"asset")
            write_json(root / locations.LOCAL_CONFIG, {"schema_version": 1, "migration_archive": "input.tar.gz"})
            self.assertEqual(locations.resolve_archive(root=root), (root / "input.tar.gz").resolve())

    def test_missing_map_does_not_guess_a_download_or_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "no default archive is guessed"):
                locations.resolve_archive(root=temporary)

    def test_explicit_relative_path_keeps_callers_cwd_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            caller = Path(temporary) / "caller"
            root.mkdir()
            caller.mkdir()
            (root / "input.tar.gz").write_bytes(b"mapped asset")
            (caller / "input.tar.gz").write_bytes(b"explicit asset")
            write_json(root / locations.LOCAL_CONFIG, {"schema_version": 1, "migration_archive": "input.tar.gz"})
            with patch("os.getcwd", return_value=str(caller)):
                self.assertEqual(locations.resolve_archive("input.tar.gz", root), caller / "input.tar.gz")
                self.assertEqual(locations.resolve_archive(root=root), root / "input.tar.gz")

    def test_map_cannot_override_identity_and_bad_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [[], {"schema_version": 2}, {"schema_version": 1, "migration_archive": None},
                     {"schema_version": 1, "migration_archive": ""},
                     {"schema_version": 1, "migration_archive": "asset", "sha256": "override"}]
            for value in cases:
                with self.subTest(value=value):
                    write_json(root / locations.LOCAL_CONFIG, value)
                    with self.assertRaises(ValueError):
                        locations.resolve_archive(root=root)

    def test_missing_archive_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                locations.resolve_archive("missing.tar.gz", temporary)

    def test_hash_and_size_both_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset"
            path.write_bytes(b"abc")
            expected = {"size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            self.assertEqual(locations.verify_archive(path, expected), expected["sha256"])
            with self.assertRaisesRegex(ValueError, "size"):
                locations.verify_archive(path, {**expected, "size_bytes": 4})
            with self.assertRaisesRegex(ValueError, "SHA256"):
                locations.verify_archive(path, {**expected, "sha256": "0" * 64})

    def exercise_restore(self, kind, tampered=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            content = b"known input"
            rel = "data/item.bin"
            member = "experiment1/" + rel
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo(member)
                info.size = len(content)
                stream.addfile(info, io.BytesIO(content))
            record = {"path": rel, "archive_member": member, "size_bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()}
            lock = {"sha256": stage05.sha(archive), "archive_sha256": stage05.sha(archive),
                    "artifacts": {rel: record}}
            pack = root / "test-package"
            write_json(pack / "reproducibility/assets.lock.json", lock)
            write_json(root / locations.LOCAL_CONFIG, {"schema_version": 1, "migration_archive": str(archive)})
            recipe = {"id": "fixture", "required_artifacts": [rel], "evaluation_artifacts": [], "tokenizer_artifacts": []}
            destination = root / "restore"
            if tampered:
                archive.write_bytes(b"wrong archive")
            if kind == "01":
                argv = ["project_repro.py", "restore-inputs", "--stage", "01", "--experiment", "fixture", "--project-root", str(destination)]
                with patch.object(project, "ROOT", root), patch.object(project, "package", return_value=pack), \
                     patch.object(project, "verify"), patch.object(project, "load_recipe", return_value=recipe), \
                     patch("sys.argv", argv):
                    if tampered:
                        with self.assertRaisesRegex(ValueError, "hash mismatch"):
                            project.main()
                    else:
                        project.main()
            else:
                args = argparse.Namespace(archive=None, experiment="fixture", include_final=False, project_root=destination)
                with patch.object(stage05, "ROOT", root), patch.object(stage05, "PACKAGE", pack), \
                     patch.object(stage05, "verify_package"), patch.object(stage05, "load_recipe", return_value=recipe):
                    if tampered:
                        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                            stage05.restore_inputs(args)
                    else:
                        stage05.restore_inputs(args)
            if tampered:
                self.assertFalse(destination.exists())
            else:
                self.assertEqual((destination / rel).read_bytes(), content)

    def test_restore_01_04_from_local_map(self):
        self.exercise_restore("01")

    def test_restore_05_from_local_map(self):
        self.exercise_restore("05")

    def test_both_restore_paths_reject_wrong_archive_before_writing(self):
        for kind in ("01", "05"):
            with self.subTest(stage=kind):
                self.exercise_restore(kind, tampered=True)


if __name__ == "__main__":
    unittest.main()
