"""CPU checks for pinned public-model manifests and quantizer diagnostics."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_listwise_assets import make_assets, make_dataset
from scripts.listwise_diffgrm.asset_diagnostics import audit_assets, collision_statistics
from scripts.listwise_diffgrm.model_manifest import DENSE_FILES, select_dense_files, verify_snapshot
from scripts.listwise_diffgrm.encode_text import encode_catalog


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "outputs" / "listwise_cpu_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_dense_selection_uses_pinned_revision_and_excludes_other_heads(self):
        info = {"sha": "a" * 40, "siblings": [
            {"rfilename": name, "size": 1, "blobId": "b" * 40}
            for name in (*DENSE_FILES, "onnx/model.onnx_data", "sparse_linear.pt", "colbert_linear.pt")]}
        selected = select_dense_files(info, "a" * 40)
        self.assertEqual({row["path"] for row in selected}, set(DENSE_FILES))
        with self.assertRaises(ValueError):
            select_dense_files(info, "c" * 40)

    def test_snapshot_verifies_lfs_sha256_and_nonlfs_git_blobs(self):
        records = []
        for name in DENSE_FILES:
            data = name.encode("ascii")
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append({"path": name, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest() if name.endswith(".bin") else None,
                            "git_blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
        checked = verify_snapshot(self.root, records)
        self.assertEqual(len(checked), 11)
        (self.root / "config.json").write_bytes(b"x" * len(b"config.json"))
        with self.assertRaisesRegex(ValueError, "blob hash"):
            verify_snapshot(self.root, records)

    def test_asset_audit_recomputes_errors_collision_and_behavior_groups(self):
        data = make_dataset(self.root / "dataset")
        assets = make_assets(data, self.root / "assets")
        report = audit_assets(assets.root, data)
        self.assertTrue(report["toy"])
        self.assertTrue(report["nearest_sid_exact_match"])
        self.assertTrue(report["error_recompute_matches"])
        self.assertEqual(report["collision"]["items"], 9)
        self.assertEqual(report["behavior_groups"]["behavior_train_seen"]["items"], 5)
        self.assertEqual(len(report["collision"]["per_digit"]), 4)
        self.assertGreater(report["residual_squared_error_sum"]["mean"], 0)

    def test_local_model_checks_all_files_and_identity_survives_directory_move(self):
        snapshot = self.root / "model"
        snapshot.mkdir()
        records = []
        for name in DENSE_FILES:
            data = name.encode("ascii")
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append({"path": name, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest() if name.endswith(".bin") else None,
                            "git_blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
        manifest = self.root / "model_manifest.json"
        manifest.write_text(json.dumps({"model_id": "BAAI/bge-m3", "revision": "a" * 40, "files": records}))
        data = make_dataset(self.root / "dataset")
        class ToyEncoder:
            @staticmethod
            def tokenizer(texts, **kwargs):
                return {"input_ids": [[0, 1] for _ in texts]}
            @staticmethod
            def encode(texts, **kwargs):
                result = np.zeros((len(texts), 1024), dtype=np.float32)
                result[:, 0] = 1
                return result
        with patch("scripts.listwise_diffgrm.encode_text._make_encoder", return_value=ToyEncoder()) as factory:
            result = encode_catalog(data, self.root / "encoded", revision="a" * 40,
                                    model_path=snapshot, model_manifest=manifest)
            self.assertEqual(factory.call_args.args, ("a" * 40, "cpu", False, snapshot))
            self.assertEqual(result["loading_provenance"]["source"], "verified_local_snapshot")
            relocated = self.root / "relocated_model"
            snapshot.rename(relocated)
            before = factory.call_count
            again = encode_catalog(data, self.root / "encoded", revision="a" * 40, resume=True,
                                   model_path=relocated, model_manifest=manifest)
            self.assertEqual(result["fingerprint"], again["fingerprint"])
            self.assertEqual(factory.call_count, before)
            with (relocated / "pytorch_model.bin").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                encode_catalog(data, self.root / "encoded", revision="a" * 40, resume=True,
                               model_path=relocated, model_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
