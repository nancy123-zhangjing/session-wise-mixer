"""True item IDs, split binding, chronology and live SID lookup checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_listwise_assets import make_assets, make_dataset
from scripts.listwise_diffgrm.data import KuaiRecListDataset, ListwiseCollator, iter_batches, resolve_sids


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs" / "listwise_cpu_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temp.name)
        self.dataset = make_dataset(self.root / "dataset")
        self.assets = make_assets(self.dataset, self.root / "assets")

    def tearDown(self):
        self.temp.cleanup()

    def test_collator_preserves_items_targets_and_left_pads_history(self):
        data = KuaiRecListDataset(self.dataset, "validation", self.assets)
        batch = data.collator()([data[0], data[2]])
        self.assertEqual(batch["history_item_ids"][0].tolist(), [0, 0, 0, 0, 0, 1, 2])
        self.assertEqual(batch["history_item_ids"][1].tolist(), [1, 1, 2, 3, 4, 5, 6])
        self.assertTrue(torch.equal(batch["history_item_ids"].ne(0), batch["history_mask"]))
        # Rewatching history items is permitted; targets are not filtered.
        self.assertEqual(batch["target_item_ids"][0].tolist(), [1, 6, 7, 8, 9])
        self.assertEqual(batch["target_cold_start"][0].tolist(), [False, True, True, True, True])
        self.assertEqual(batch["user_cold_start"].tolist(), [False, True])
        self.assertNotIn("history_sids", batch)

    def test_lookup_uses_current_catalog_and_never_overwrites_original_ids(self):
        data = KuaiRecListDataset(self.dataset, "train", self.assets)
        batch = data.collator()([data[0]])
        first = resolve_sids(batch, self.assets.catalog_sids)
        changed = self.assets.catalog_sids.clone()
        changed[1] = (changed[1] + 1) % self.assets.codebook_size
        second = resolve_sids(batch, changed)
        self.assertFalse(torch.equal(first["target_sids"][0, 0], second["target_sids"][0, 0]))
        self.assertTrue(torch.equal(second["target_item_ids"], batch["target_item_ids"]))
        self.assertTrue(second["history_sids"][0, :5].eq(-1).all())

    def test_batches_keep_tail_and_shuffle_repeats_for_same_seed(self):
        batches = list(iter_batches(self.dataset, "train", 2, assets=self.assets))
        self.assertEqual([len(batch["sample_id"]) for batch in batches], [2, 1])
        order = lambda: [item for b in iter_batches(self.dataset, "train", 2, shuffle=True, seed=17)
                         for item in b["sample_id"]]
        self.assertEqual(order(), order())
        self.assertEqual(set(order()), {"sample-0", "sample-1", "sample-2"})

    def test_rejects_future_leakage_invalid_ids_duplicate_targets_and_truncation(self):
        data = KuaiRecListDataset(self.dataset, "train", self.assets)
        collate = data.collator()
        row = data[0]
        for invalid in (
            {**row, "history_timestamp_us": [1, 101]},
            {**row, "target_item_ids": [1, 1, 2, 3, 4]},
            {**row, "target_item_ids": [0, 1, 2, 3, 4]},
            {**row, "target_item_ids": [1, 2, 3, 4, 10]},
            {**row, "history_item_ids": [1] * 8, "history_timestamp_us": list(range(8))},
        ):
            with self.assertRaises(ValueError):
                collate([invalid])
        with self.assertRaises(ValueError):
            KuaiRecListDataset(self.dataset, "val", self.assets)
        with self.assertRaises(ValueError):
            ListwiseCollator()( [])


if __name__ == "__main__":
    unittest.main()
