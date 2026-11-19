"""Small real-Git regressions for captured shallow history and wrong boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import restore_git_with_shallow as recovery
from test_multiseed_backup import mark_complete

byte = recovery.original.byte


class ShallowRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stage07_shallow_regression_")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, repository, *arguments):
        return recovery.git(repository, *arguments)["stdout"]

    def fixture(self):
        code, engine = self.root / "code_source", self.root / "engine_source"
        self.git(self.root, "init", "--quiet", str(code))
        for value in range(3):
            (code / "model.py").write_text(f"VERSION = {value}\n", encoding="utf-8")
            self.git(code, "add", "model.py")
            self.git(code, "-c", "user.name=CPU regression", "-c", "user.email=fixture@example.invalid",
                     "commit", "--quiet", "-m", f"version {value}")
        self.git(self.root, "clone", "--quiet", "--depth", "2", code.as_uri(), str(engine))
        (engine / "model.py").write_text("VERSION = 3\n", encoding="utf-8")
        self.git(engine, "add", "model.py")
        self.git(engine, "-c", "user.name=CPU regression", "-c", "user.email=fixture@example.invalid",
                 "commit", "--quiet", "-m", "local engine patch")
        sources, identities = {}, {}
        for name, repository in (("code", code), ("engine", engine)):
            bundle = self.root / f"{name}.bundle"
            self.git(repository, "bundle", "create", str(bundle), "HEAD")
            identity = {"commit": self.git(repository, "rev-parse", "HEAD"), "clean": True,
                        "bundle_sha256": byte.sha(bundle), "bundle_has_no_external_prerequisites": True}
            audit = self.root / f"{name}_git_audit.json"
            byte.write(audit, identity)
            sources[f"source/{name}.bundle"] = bundle
            sources[f"source/{name}_git_audit.json"] = audit
            identities[name] = identity
        package = self.root / "package"
        package.mkdir()
        receipt = byte.archive_files(sources, package, {"source_identity": identities, "original_goal_root": "/fixture/goal"}, 1024 * 1024)
        mark_complete(package, receipt)
        verified = self.root / "verified"
        byte.verify(package, verified)
        supplement = self.root / "supplement"
        supplement.mkdir()
        shallow = supplement / "engine.shallow"
        shallow.write_bytes((engine / ".git/shallow").read_bytes())
        capture = {"schema": recovery.SCHEMA, "campaign_id": recovery.original.CAMPAIGN,
                   "source_goal_root": "/fixture/goal", "repository": "engine", "source_repository": "/fixture/goal/engine",
                   "source_commit": identities["engine"]["commit"], "source_is_shallow": True,
                   "source_status_porcelain": "", "captured_utc": "2026-10-22T00:00:00Z", "capture_mode": "read_only",
                   "shallow_file": byte.file_record("engine.shallow", shallow),
                   "shallow_commits": shallow.read_text(encoding="ascii").splitlines(),
                   "original_package_sha256": byte.sha(package / "PACKAGE.json"),
                   "original_manifest_sha256": byte.sha(verified / "MANIFEST.json"),
                   "original_engine_bundle_sha256": byte.sha(verified / "source/engine.bundle"),
                   "original_engine_audit_sha256": byte.sha(verified / "source/engine_git_audit.json")}
        byte.write(supplement / "SOURCE_SHALLOW.json", capture)
        return verified, package, supplement, identities

    def test_actual_shallow_metadata_recovers_exact_clean_source(self):
        verified, package, supplement, identities = self.fixture()
        result = recovery.restore(verified, package, supplement, self.root / "restore")
        self.assertEqual(result["status"], "INDEPENDENT_SOURCE_RESTORE_WITH_CAPTURED_SHALLOW_PASSED")
        self.assertFalse(result["source_audit_correction"]["complete_upstream_engine_history_included"])
        for name in ("code", "engine"):
            row = result["repositories"][name]
            self.assertEqual(row["commit"], identities[name]["commit"])
            self.assertEqual(row["is_shallow_repository"], name == "engine")
            self.assertTrue(row["clean"])
            self.assertEqual(row["repository_local_core_longpaths"], "true")
            self.assertEqual(row["repository_local_core_autocrlf"], "false")

    def test_tampered_boundary_bytes_rejected_before_output(self):
        verified, package, supplement, _ = self.fixture()
        (supplement / "engine.shallow").write_bytes(b"f" * 40 + b"\n")
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "Captured shallow bytes/SHA differ"):
            recovery.restore(verified, package, supplement, output)
        self.assertFalse(output.exists())

    def test_inventing_head_boundary_does_not_hide_missing_history(self):
        verified, package, supplement, identities = self.fixture()
        shallow = supplement / "engine.shallow"
        shallow.write_text(identities["engine"]["commit"] + "\n", encoding="ascii")
        path = supplement / "SOURCE_SHALLOW.json"
        capture = byte.read(path)
        capture["shallow_file"] = byte.file_record("engine.shallow", shallow)
        capture["shallow_commits"] = shallow.read_text(encoding="ascii").splitlines()
        path.write_text(json.dumps(capture), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, 'uncaptured shallow boundary'):
            recovery.restore(verified, package, supplement, self.root / "wrong_boundary")
        self.assertFalse((self.root / "wrong_boundary/GIT_RESTORE_WITH_SHALLOW.json").exists())


if __name__ == "__main__":
    unittest.main()
