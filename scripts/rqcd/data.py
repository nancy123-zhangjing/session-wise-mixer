"""Pack Arrow columns once, then gather tensors instead of collating Python rows."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

from scripts.listwise_diffgrm.assets import dataset_binding, catalog_original_ids


def bind_inputs(dataset, content, cfg):
    from .common import sha256
    binding = dataset_binding(Path(dataset), verify=True)
    if binding["history_max"] != cfg["history_max"] or binding["list_length"] != cfg["list_length"]:
        raise ValueError("Dataset geometry differs from protocol")
    ids = catalog_original_ids(Path(dataset))
    if len(ids) != cfg["catalog_size"]:
        raise ValueError("Catalog size mismatch")
    if sha256(content) != cfg["content_sha256"]:
        raise ValueError("Content SHA differs from the approved asset")
    return {"dataset": binding, "content_sha256": sha256(content)}


class PackedSplit:
    def __init__(self, root, split, cfg, device="cpu"):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if cfg.get('dataset_protocol') == 'kuairec_l5o_records_v1':
            self._load_l5o(Path(root), split, cfg, device)
            return
        columns = ["sample_id", "user_id", "history_item_ids", "history_timestamp_us",
                   "target_item_ids", "target_timestamp_us", "target_mask"]
        table = pq.read_table(Path(root)/f"{split}.parquet", columns=columns, memory_map=True).combine_chunks()
        n, h, length = len(table), cfg["history_max"], cfg["list_length"]
        if n != cfg["split_counts"][split]:
            raise ValueError(f"{split} size {n} differs from bound protocol")
        self.sample_ids = table["sample_id"].to_pylist()
        if len(set(self.sample_ids)) != n:
            raise ValueError("Duplicate sample IDs")
        self.users = table["user_id"].to_pylist()
        history = np.zeros((n, h), dtype=np.int64)
        history_times = np.zeros((n, h), dtype=np.int64)
        lengths = None
        for name, destination in (("history_item_ids", history), ("history_timestamp_us", history_times)):
            col = table[name].chunk(0)
            offsets = col.offsets.to_numpy()
            lens = np.diff(offsets)
            if np.any(lens < 1) or np.any(lens > h):
                raise ValueError("Invalid history length")
            if lengths is not None and not np.array_equal(lengths, lens):
                raise ValueError("History timestamps misaligned")
            lengths = lens
            row = np.repeat(np.arange(n), lens)
            positions = np.arange(lens.sum())-np.repeat(offsets[:-1], lens)+np.repeat(h-lens, lens)
            values = col.values.to_numpy()[offsets[0]:offsets[-1]]
            if name == "history_item_ids" and (np.any(values <= 0) or np.any(values > cfg["catalog_size"])):
                raise ValueError("Real historical items must be positive catalog IDs")
            destination[row, positions] = values
        targets, times = [], []
        for name, dest in (("target_item_ids", targets), ("target_timestamp_us", times)):
            col = table[name].chunk(0)
            if not np.all(np.diff(col.offsets.to_numpy()) == length):
                raise ValueError("Invalid target list length")
            dest.append(np.array(col.values.to_numpy(), copy=True).reshape(n, length))
        target, target_times = targets[0], times[0]
        mask_col = table["target_mask"].chunk(0)
        if not np.all(np.diff(mask_col.offsets.to_numpy()) == length) or not all(mask_col.values.to_pylist()):
            raise ValueError("Incomplete targets")
        if (np.any(np.diff(target_times, axis=1) <= 0) or
                np.any(history_times[:, -1] >= target_times[:, 0]) or
                np.any((np.diff(history_times, axis=1) <= 0) & (history[:, :-1] != 0))):
            raise ValueError("History/target chronology violation")
        if (np.any(target <= 0) or np.any(target > cfg["catalog_size"]) or
                np.any(history < 0) or np.any(history > cfg["catalog_size"]) or
                np.any(np.diff(np.sort(target, axis=1), axis=1) == 0)):
            raise ValueError("Invalid/duplicate target or history ID")
        self.history = torch.from_numpy(history).to(device)
        self.target = torch.from_numpy(target).to(device)
        self.first_target_time = target_times[:, 0].copy()
        self.split = split
        # Entire train IDs take ~0.3GiB: small relative to H100 memory. When CPU
        # placement is requested, pin the cache for asynchronous transfer.
        if str(device) == "cpu" and torch.cuda.is_available():
            self.history = self.history.pin_memory()
            self.target = self.target.pin_memory()

    def __len__(self):
        return len(self.sample_ids)

    def _load_l5o(self, root, split, cfg, device):
        from scripts.listwise_diffgrm.data import KuaiRecListDataset
        from scripts.listwise_diffgrm.execution import TrainingTensorCache

        data = KuaiRecListDataset(root, split)
        if (data.config.get('protocol_id') != cfg['dataset_protocol'] or
                data.config['history_max'] != cfg['history_max'] or
                data.config['list_length'] != cfg['list_length'] or
                len(data) != cfg['split_counts'][split] or data.n_items != cfg['catalog_size']):
            raise ValueError('Packed L5O dataset differs from the configured protocol')
        cache = TrainingTensorCache(data, vectorized=True)
        self.history = cache.tensors['history_item_ids'].to(device=device, dtype=torch.long)
        self.target = cache.tensors['target_item_ids'].to(device=device, dtype=torch.long)
        self.sample_ids = data.table['sample_id'].to_pylist()
        if len(set(self.sample_ids)) != len(data):
            raise ValueError('Duplicate L5O sample IDs')
        self.users = data.table['user_id'].to_pylist()
        self.split = split
        table = pq.read_table(root/f'{split}.parquet', columns=['target_timestamp_us'])
        times = table[0].combine_chunks()
        self.first_target_time = times.values.to_numpy()[times.offsets.to_numpy()[:-1]].copy()
        if split != 'train':
            ratios = pq.read_table(root/f'{split}.parquet', columns=['target_watch_ratio'])[0]
            self.watch_ratios = np.asarray(ratios.to_pylist(), dtype=np.float64)

    def batch(self, indices, device):
        idx = indices.to(self.history.device)
        return (self.history[idx].to(device, non_blocking=True),
                self.target[idx].to(device, non_blocking=True))

    def subset_indices(self, sample_ids):
        lookup = {s: i for i, s in enumerate(self.sample_ids)}
        if len(set(sample_ids)) != len(sample_ids) or any(s not in lookup for s in sample_ids):
            raise ValueError("Invalid probe subset identity")
        return torch.tensor([lookup[s] for s in sample_ids])
