"""KuaiRec list rows stay as item IDs; SID lookup happens at the training step.

Do not keep an alignment object in a DataLoader worker. Its cached SID snapshot
would remain stale after a Tokenizer phase. Use ``resolve_sids`` in the parent
process with the current ``alignment.hard_codes`` on every batch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .assets import PSEAssets, catalog_original_ids, dataset_binding


SPLITS = {"train", "validation", "test"}
ROW_COLUMNS = ["sample_id", "user_id", "history_item_ids", "history_timestamp_us",
               "target_item_ids", "target_mask", "target_timestamp_us"]


class KuaiRecListDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, assets: PSEAssets | None = None,
                 *, verify: bool = True):
        if split not in SPLITS:
            raise ValueError(f"Unsupported split {split!r}")
        self.root, self.split = Path(dataset_dir), split
        self.binding = dataset_binding(self.root, verify=verify)
        if assets is not None:
            # Already hashed above; compare exact split/catalog identities here.
            if self.binding != assets.manifest["dataset_binding"]:
                raise ValueError("Dataset does not match bound PSE assets")
            if not np.array_equal(catalog_original_ids(self.root), assets.original_item_ids[1:].numpy()):
                raise ValueError("Dataset item mapping differs from asset row order")
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.n_items = len(catalog_original_ids(self.root))
        seen = json.loads((self.root / "train_seen.json").read_text(encoding="utf-8"))
        self.item_seen, self.user_seen = set(seen["item_ids"]), set(seen["user_ids"])
        # Arrow retains compact columns instead of a Python dictionary per row.
        self.table = pq.read_table(self.root / f"{split}.parquet", columns=ROW_COLUMNS,
                                   memory_map=True)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = {name: self.table[name][index].as_py() for name in ROW_COLUMNS}
        row["user_cold_start"] = row["user_id"] not in self.user_seen
        row["history_cold_start"] = [item not in self.item_seen for item in row["history_item_ids"]]
        row["target_cold_start"] = [item not in self.item_seen for item in row["target_item_ids"]]
        return row

    def collator(self) -> "ListwiseCollator":
        return ListwiseCollator(history_max=int(self.config["history_max"]),
                               list_length=int(self.config["list_length"]), n_items=self.n_items)


class ListwiseCollator:
    def __init__(self, history_max: int = 50, list_length: int = 5,
                 n_items: int | None = None):
        if history_max < 1 or list_length < 1:
            raise ValueError("History/list lengths must be positive")
        self.history_max, self.list_length, self.n_items = history_max, list_length, n_items

    def __call__(self, rows: Iterable[dict]) -> dict:
        rows = list(rows)
        if not rows:
            raise ValueError("Cannot collate an empty batch")
        count = len(rows)
        result = {
            "sample_id": [row["sample_id"] for row in rows],
            "user_id": torch.tensor([row["user_id"] for row in rows], dtype=torch.long),
            "history_item_ids": torch.zeros(count, self.history_max, dtype=torch.long),
            "history_mask": torch.zeros(count, self.history_max, dtype=torch.bool),
            "history_timestamp_us": torch.zeros(count, self.history_max, dtype=torch.long),
            "history_cold_start": torch.zeros(count, self.history_max, dtype=torch.bool),
            "target_item_ids": torch.empty(count, self.list_length, dtype=torch.long),
            "target_mask": torch.ones(count, self.list_length, dtype=torch.bool),
            "target_timestamp_us": torch.empty(count, self.list_length, dtype=torch.long),
            "target_cold_start": torch.zeros(count, self.list_length, dtype=torch.bool),
            "user_cold_start": torch.tensor([row.get("user_cold_start", False) for row in rows]),
        }
        for i, row in enumerate(rows):
            history, target = row["history_item_ids"], row["target_item_ids"]
            history_times, target_times = row["history_timestamp_us"], row["target_timestamp_us"]
            if not 1 <= len(history) <= self.history_max:
                raise ValueError("History exceeds its locked length or is empty; never silently truncate")
            if len(target) != self.list_length or len(set(target)) != self.list_length:
                raise ValueError("Targets must be the locked number of distinct items")
            if len(history_times) != len(history) or len(target_times) != len(target):
                raise ValueError("Item/timestamp length mismatch")
            if (any(a >= b for a, b in zip(history_times, history_times[1:]))
                    or any(a >= b for a, b in zip(target_times, target_times[1:]))
                    or history_times[-1] >= target_times[0]):
                raise ValueError("History must be strictly earlier than the ordered target list")
            target_mask = row.get("target_mask", [True] * self.list_length)
            if len(target_mask) != self.list_length or not all(target_mask):
                raise ValueError("Prepared lists must contain every target; PAD targets are unsupported")
            all_ids = history + target
            if min(all_ids) <= 0 or (self.n_items is not None and max(all_ids) > self.n_items):
                raise ValueError("Real item IDs must be in 1..N; PAD is not an item")
            n = len(history)
            result["history_item_ids"][i, -n:] = torch.tensor(history)
            result["history_mask"][i, -n:] = True
            result["history_timestamp_us"][i, -n:] = torch.tensor(history_times)
            result["history_cold_start"][i, -n:] = torch.tensor(row.get("history_cold_start", [False] * n))
            result["target_item_ids"][i] = torch.tensor(target)
            result["target_timestamp_us"][i] = torch.tensor(target_times)
            result["target_cold_start"][i] = torch.tensor(row.get("target_cold_start", [False] * self.list_length))
        return result


def resolve_sids(batch: dict, catalog_sids: torch.Tensor) -> dict:
    """Return a new mapping with current hard SIDs; retain true item IDs."""
    if catalog_sids.ndim != 2 or catalog_sids.dtype != torch.long or not catalog_sids[0].eq(-1).all():
        raise ValueError("Catalog SIDs must be integer [N+1,n_digit] with PAD=-1")
    result = dict(batch)
    for name in ("history", "target"):
        ids = batch[f"{name}_item_ids"].to(device=catalog_sids.device, dtype=torch.long)
        if ids.numel() and (ids.min() < 0 or ids.max() >= len(catalog_sids)):
            raise ValueError("Batch item ID is outside the current SID catalog")
        result[f"{name}_sids"] = catalog_sids[ids]
    return result


def iter_batches(dataset_dir: Path, split: str = "train", batch_size: int = 64, *,
                 assets: PSEAssets | None = None, alignment=None, device="cpu",
                 shuffle: bool = False, seed: int = 2024, verify: bool = True):
    """Deterministic convenience iterator; trainer may instead use DataLoader.

    No dropped tail, no implicit leave-one-out, and no SID materialization in
    the saved dataset. A new iterator with the same seed repeats sample order.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = KuaiRecListDataset(dataset_dir, split, assets, verify=verify)
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    collate = dataset.collator()
    for start in range(0, len(indices), batch_size):
        batch = collate(dataset[int(index)] for index in indices[start:start + batch_size])
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        if alignment is not None:
            batch = resolve_sids(batch, alignment.hard_codes)
        elif assets is not None:
            batch = resolve_sids(batch, assets.catalog_sids.to(device))
        yield batch
