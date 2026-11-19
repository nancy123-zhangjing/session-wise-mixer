"""Portable CPU tests for byte preservation, refusal paths and actual Git recovery."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import multiseed_backup as backup

byte = backup.byte


def mark_complete(package, receipt):
    byte.write(package / "PACK_COMPLETE.json", {"status": "FULL_MULTISEED_PACKAGE_COMPLETED",
               "campaign_id": backup.CAMPAIGN, "package_sha256": byte.sha(package / "PACKAGE.json"),
               "archive_sha256": receipt["archive_sha256"], "manifest_sha256": receipt["manifest_sha256"],
               "full_source_byte_recheck": True, "final_completed_artifact_binding_recheck": True,
               "source_file_inventory_unchanged": True})


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="multiseed_backup_test_")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def package(self, sources=None, metadata=None):
        if sources is None:
            source = self.root / "weights.bin"
            source.write_bytes(bytes(range(256)) * 200)
            sources = {"seeds/seed2025/runs/example/training_state/weights.bin": source}
        package = self.root / "package"
        package.mkdir()
        receipt = byte.archive_files(sources, package, metadata or {}, 256)
        mark_complete(package, receipt)
        return package, receipt

    def test_split_roundtrip_and_external_manifest_binding(self):
        package, receipt = self.package()
        self.assertGreater(len(receipt["parts"]), 1)
        output = self.root / "extracted"
        byte.verify(package, output)
        manifest, entries = backup.verified_manifest(output, package)
        self.assertEqual(manifest["campaign_id"], backup.CAMPAIGN)
        self.assertEqual(len(entries), 1)
        member = next(iter(entries))
        self.assertEqual((output / member).read_bytes(), (self.root / "weights.bin").read_bytes())

    def test_tampered_volume_rejected_before_extraction(self):
        package, receipt = self.package()
        part = package / receipt["parts"][0]["member"]
        part.write_bytes(part.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "Volume bytes/SHA"):
            byte.verify(package, self.root / "extracted")
        self.assertFalse((self.root / "extracted").exists())

    def test_changed_manifest_and_package_receipts_rejected(self):
        package, _ = self.package()
        output = self.root / "extracted"
        byte.verify(package, output)
        manifest = output / "MANIFEST.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "external package receipt"):
            backup.verified_manifest(output, package)
        receipt = package / "PACKAGE.json"
        receipt.write_bytes(receipt.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "completion does not bind"):
            backup.package_identity(package)

    def test_traversal_and_links_refused_before_extraction(self):
        for case, name, kind in (("traversal", "../escape.bin", tarfile.REGTYPE),
                                 ("symlink", "link", tarfile.SYMTYPE),
                                 ("hardlink", "link", tarfile.LNKTYPE)):
            with self.subTest(case=case):
                package = self.root / case
                package.mkdir()
                archive = package / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as handle:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "../../outside"
                    member.size = 1 if kind == tarfile.REGTYPE else 0
                    handle.addfile(member, io.BytesIO(b"x") if member.size else None)
                row = byte.file_record(archive.name, archive)
                byte.write(package / "PACKAGE.json", {"format": byte.FORMAT, "campaign_id": backup.CAMPAIGN,
                           "parts": [row], "archive_sha256": row["sha256"], "archive_size_bytes": row["size_bytes"],
                           "manifest_sha256": "0" * 64, "files": 1})
                output = self.root / (case + "_extract")
                with self.assertRaises((ValueError, RuntimeError)):
                    byte.verify(package, output)
                self.assertFalse(output.exists())
        self.assertFalse((self.root / "escape.bin").exists())

    def test_inventory_preserves_all_seed_files_and_rejects_symlinks(self):
        for prefix in ("seeds/seed2025", "seeds/seed2026", "delivery"):
            path = self.root / prefix / "unknown_new_state.xyz"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"retain every suffix")
        inventory = backup.source_inventory(self.root)
        self.assertEqual(len(inventory), 3)

        class FakeEntry:
            path = str(self.root / "badlink")
            def stat(self, *, follow_symlinks):
                self_follow = follow_symlinks
                if self_follow:
                    raise AssertionError("Must never follow a link")
                return type("Stat", (), {"st_mode": stat.S_IFLNK})()

        class Entries:
            def __enter__(self):
                return iter([FakeEntry()])
            def __exit__(self, *args):
                return False

        with patch.object(backup.os, "scandir", return_value=Entries()):
            with self.assertRaisesRegex(ValueError, "Symlink found"):
                backup.regular_tree(self.root)

    def test_source_change_late_in_archive_is_rejected(self):
        first, second = self.root / "first", self.root / "second"
        first.write_bytes(b"initial")
        second.write_bytes(b"trigger")
        output = self.root / "late_change"
        output.mkdir()
        original = tarfile.TarFile.add

        def changing(tar, name, *args, **kwargs):
            value = original(tar, name, *args, **kwargs)
            if Path(name) == second:
                first.write_bytes(b"changed after first was packed")
            return value

        with patch.object(tarfile.TarFile, "add", changing):
            with self.assertRaisesRegex(ValueError, "Source changed after"):
                byte.archive_files({"first": first, "second": second}, output, {}, 4096)
        self.assertFalse((output / "PACKAGE.json").exists())

    def test_independent_git_bundle_restoration_and_dirty_refusal(self):
        goal, scratch = self.root / "goal", self.root / "scratch"
        goal.mkdir()
        scratch.mkdir()
        sources, identities = {}, {}
        for name in ("code", "engine"):
            repository = goal / name
            subprocess.run(["git", "init", "--quiet", str(repository)], check=True, env=backup.git_env())
            (repository / "model.py").write_text("VALUE = " + repr(name) + "\n", encoding="utf-8")
            backup.git(repository, "add", "model.py")
            backup.git(repository, "-c", "user.name=Backup CPU test", "-c", "user.email=test@example.invalid",
                       "commit", "--quiet", "-m", "test snapshot")
            commit = backup.git(repository, "rev-parse", "HEAD")
            identities[name], additions = backup.repository_snapshot(goal, name, commit, scratch)
            sources.update(additions)
        package, _ = self.package(sources, {"source_identity": identities})
        extracted = self.root / "extract_git"
        byte.verify(package, extracted)
        restored = self.root / "restored"
        result = backup.restore_git(extracted, package, restored)
        self.assertEqual(result["status"], "INDEPENDENT_GIT_RESTORE_PASSED")
        for name in ("code", "engine"):
            self.assertEqual((restored / name / "model.py").read_bytes(), (goal / name / "model.py").read_bytes())
        (goal / "code/model.py").write_text("dirty", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact clean deployment"):
            backup.repository_snapshot(goal, "code", identities["code"]["commit"], scratch)


if __name__ == "__main__":
    unittest.main()
