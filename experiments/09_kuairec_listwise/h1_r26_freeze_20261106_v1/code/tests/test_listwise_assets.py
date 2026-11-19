"""CPU-only asset/identity tests; no encoder download or real catalog fit."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.listwise_diffgrm.assets import (
    BOUND_FILES, TEXT_TEMPLATE_ID, catalog_original_ids, dataset_binding,
    encode_residuals, load_assets, normalized_pca, sha256,
    validate_embedding_metadata, write_assets,
)
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.encode_text import encode_catalog, untruncated_token_lengths


def make_dataset(root: Path, n_items: int = 9):
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"history_max": 7, "list_length": 5}))
    (root / "item_map.json").write_text(json.dumps({"pad_id": 0, "entries": [
        {"id": i + 1, "original_id": i * 2} for i in range(n_items)]}))
    (root / "train_seen.json").write_text(json.dumps({"item_ids": [1, 2, 3, 4, 5], "user_ids": [1]}))
    with (root / "item_text.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(n_items):
            stream.write(json.dumps({"video_id": i * 2, "text": f"标题：商品{i}" if i else ""}, ensure_ascii=False) + "\n")
    pq.write_table(pa.table({"item_id": np.arange(1, n_items + 1)}), root / "item_catalog.parquet")
    rows = []
    for i, history in enumerate(([1, 2], [1, 2, 3], [1, 1, 2, 3, 4, 5, 6])):
        rows.append({"sample_id": f"sample-{i}", "user_id": i + 1,
                     "history_item_ids": history,
                     "history_timestamp_us": list(range(1, len(history) + 1)),
                     "target_item_ids": [1, 6, 7, 8, 9], "target_mask": [True] * 5,
                     "target_timestamp_us": list(range(100, 105))})
    for split in ("train", "validation", "test"):
        pq.write_table(pa.Table.from_pylist(rows), root / f"{split}.parquet", row_group_size=1)
    rebind_dataset(root)
    return root


def rebind_dataset(root):
    manifest = {"schema_version": 1, "artifacts": [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
        for name in BOUND_FILES]}
    (root / "manifest.json").write_text(json.dumps(manifest))


def make_assets(dataset, output):
    rng = np.random.default_rng(12)
    n = len(catalog_original_ids(dataset))
    # Unequal codewords and nonzero center expose accidental non-residual encoding.
    books = rng.normal(size=(4, 8, 8)).astype(np.float32)
    codes = np.arange(n * 4).reshape(n, 4) % 8
    residual = books[np.arange(4)[None], codes].reshape(n, 32).copy()
    residual += rng.normal(scale=0.02, size=residual.shape).astype(np.float32)
    center = np.full(32, 2.0, dtype=np.float32)
    return write_assets(output, binding=dataset_binding(dataset),
                        original_item_ids=catalog_original_ids(dataset),
                        content_features=residual + center, residual_features=residual,
                        centroids=books, coarse_centroid=center,
                        encoder_metadata={"test_fixture": True}, toy=True)


class AssetTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs" / "listwise_cpu_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temp.name)
        self.dataset = make_dataset(self.root / "dataset")

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_pad_original_ids_and_residual_reconstruction(self):
        assets = make_assets(self.dataset, self.root / "assets")
        loaded = load_assets(assets.root, self.dataset)
        self.assertEqual(loaded.catalog_sids.shape, (10, 4))
        self.assertTrue(loaded.catalog_sids[0].eq(-1).all())
        self.assertEqual(loaded.original_item_ids.tolist(), [-1] + list(range(0, 18, 2)))
        self.assertTrue(torch.isinf(loaded.reconstruction_errors[0]))
        self.assertTrue(torch.equal(loaded.residual_features[0], torch.zeros(4, 8)))
        recovered, errors = encode_residuals(loaded.residual_features[1:].numpy(), loaded.centroids.numpy())
        np.testing.assert_array_equal(recovered, loaded.catalog_sids[1:])
        np.testing.assert_allclose(errors, loaded.reconstruction_errors[1:])
        wrong, _ = encode_residuals(loaded.content_features[1:].numpy(), loaded.centroids.numpy())
        self.assertFalse(np.array_equal(wrong, recovered))
        tokenizer = loaded.make_tokenizer()
        self.assertEqual(tokenizer.vocab_size, 3 + 4 * 8)
        self.assertEqual(tokenizer.item2id["0"], 1)

    def test_both_alignment_arms_start_at_same_E0_and_anchor(self):
        assets = make_assets(self.dataset, self.root / "assets")
        off = build_alignment(assets, False)
        on = build_alignment(assets, True, config_overrides={"crosspse_heads": 4})
        self.assertTrue(torch.equal(off.mixer.crossed_codebooks()[0], on.mixer.crossed_codebooks()[0]))
        for alignment in (off, on):
            self.assertTrue(torch.equal(alignment.mixer.initial_codebooks, assets.centroids))
            self.assertTrue(torch.equal(alignment.hard_codes, assets.catalog_sids))
            self.assertEqual(alignment.diger_total_optimizer_steps.item(), 400)
        self.assertFalse(off.mixer.attention.out_proj.weight.requires_grad)
        self.assertTrue(on.mixer.attention.out_proj.weight.requires_grad)
        loss = on.tokenizer_objective(None, torch.tensor([[1, 2, 3, 4, 5]]), .01, .01, .1, 0.)["loss"]
        loss.backward()
        self.assertGreater(on.mixer.base_codebooks.grad.abs().sum().item(), 0.)
        self.assertGreater(on.mixer.attention.out_proj.weight.grad.abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            build_alignment(assets, True, config_overrides={"crosspse_residual_scale_init": 0.})

    def test_binding_rejects_changed_mapping_even_after_dataset_remanifest(self):
        assets = make_assets(self.dataset, self.root / "assets")
        mapping = json.loads((self.dataset / "item_map.json").read_text())
        mapping["entries"][0]["original_id"] = 99
        (self.dataset / "item_map.json").write_text(json.dumps(mapping))
        rebind_dataset(self.dataset)
        with self.assertRaisesRegex(ValueError, "different dataset"):
            load_assets(assets.root, self.dataset)

    def test_asset_corruption_and_replacement_are_rejected(self):
        assets = make_assets(self.dataset, self.root / "assets")
        with (assets.root / "semantic_ids.npy").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaises(ValueError):
            load_assets(assets.root)
        with self.assertRaises(FileExistsError):
            make_assets(self.dataset, assets.root)

    def test_split_bytes_verified_not_only_recorded(self):
        with (self.dataset / "test.parquet").open("ab") as stream:
            stream.write(b"x")
        with self.assertRaises(ValueError):
            dataset_binding(self.dataset)

    def test_pca_is_whitened_normalized_and_repeatable(self):
        data = np.random.default_rng(1).normal(size=(32, 16)).astype(np.float32)
        a, transform = normalized_pca(data, 8)
        b, _ = normalized_pca(data, 8)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, atol=1e-6)
        self.assertEqual(transform["components"].shape, (8, 16))
        with self.assertRaises(ValueError):
            normalized_pca(np.zeros_like(data), 8)

    def test_bge_metadata_requires_fixed_revision_and_bound_text(self):
        binding = dataset_binding(self.dataset)
        metadata = {"model_id": "BAAI/bge-m3", "revision": "a" * 40,
                    "pooling": "cls", "normalize_embeddings": True, "max_length": 512,
                    "text_template_id": TEXT_TEMPLATE_ID, "dimension": 1024,
                    "item_order": "internal_item_id_ascending",
                    "item_map_sha256": binding["files"]["item_map.json"]["sha256"],
                    "item_text_sha256": binding["files"]["item_text.jsonl"]["sha256"]}
        validate_embedding_metadata(metadata, binding, 1024)
        with self.assertRaises(ValueError):
            validate_embedding_metadata({**metadata, "revision": "main"}, binding, 1024)
        with self.assertRaises(ValueError):
            validate_embedding_metadata({**metadata, "item_text_sha256": "old"}, binding, 1024)

    def test_encoder_resume_checks_chunks_and_never_downloads_in_test(self):
        class ToyEncoder:
            calls = 0
            fail = True
            @staticmethod
            def tokenizer(texts, **kwargs):
                assert kwargs["truncation"] is False
                # Deliberately make one fixture item exceed the locked cap.
                return {"input_ids": [list(range(520 if "商品1" in text else len(text) + 2)) for text in texts]}
            def encode(self, texts, **kwargs):
                self.calls += 1
                if self.calls == 2 and self.fail:
                    raise RuntimeError("interruption")
                result = np.zeros((len(texts), 1024), dtype=np.float32)
                result[:, 0] = 1
                return result
        encoder = ToyEncoder()
        output = self.root / "encoded"
        with patch("scripts.listwise_diffgrm.encode_text._make_encoder", return_value=encoder) as factory:
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                encode_catalog(self.dataset, output, revision="a" * 40, chunk_size=4)
            first = sha256(output / "chunk_00000000.npy")
            encoder.fail = False
            result = encode_catalog(self.dataset, output, revision="a" * 40, chunk_size=4, resume=True)
            self.assertEqual(first, sha256(output / "chunk_00000000.npy"))
            self.assertEqual(result["empty_text_count"], 1)
            self.assertEqual(result["truncation_statistics"]["truncated_items"], 1)
            self.assertEqual(result["truncation_statistics"]["discarded_token_count"], 8)
            self.assertEqual(np.load(output / "untruncated_token_lengths.npy")[1], 520)
            self.assertEqual(np.load(output / "dense_embeddings.npy").shape, (9, 1024))
            self.assertEqual(factory.call_args.args, ("a" * 40, "cpu", False))
            before = factory.call_count
            encode_catalog(self.dataset, output, revision="a" * 40, chunk_size=4, resume=True)
            self.assertEqual(factory.call_count, before)
            with self.assertRaises(ValueError):
                encode_catalog(self.dataset, output, revision="b" * 40, chunk_size=4, resume=True)
            with (output / "dense_embeddings.npy").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaises(ValueError):
                encode_catalog(self.dataset, output, revision="a" * 40, chunk_size=4, resume=True)


if __name__ == "__main__":
    unittest.main()
