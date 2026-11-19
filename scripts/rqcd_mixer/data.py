"""Reuse verified packing, with fully resumable independent R/T permutations."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from scripts.rqcd.data import PackedSplit
from scripts.listwise_diffgrm.assets import dataset_binding, catalog_original_ids
from .common import sha256, seed_for


def bind_inputs(dataset, content, cfg):
    binding = dataset_binding(Path(dataset), verify=True)
    if binding["history_max"] != cfg["history_max"] or binding["list_length"] != cfg["list_length"]:
        raise ValueError("Dataset geometry differs from the locked design")
    original_ids = catalog_original_ids(Path(dataset))
    if len(original_ids) != cfg["catalog_size"]:
        raise ValueError("Catalog count differs from protocol")
    content_hash = sha256(content)
    if content_hash != cfg["content_sha256"]:
        raise ValueError("Content asset differs from the approved PCA256 input")
    return {"dataset": binding, "content_sha256": content_hash}


class PermutationStream:
    def __init__(self, size, seed, *, epoch_size=None):
        if size < 1:
            raise ValueError("Empty stream")
        self.size, self.epoch, self.cursor = size, 0, 0
        self.epoch_size = size if epoch_size is None else epoch_size
        if not 1 <= self.epoch_size <= size:
            raise ValueError('Invalid permutation epoch size')
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)

    def take(self, count):
        if self.at_end:
            self.epoch += 1
            self.cursor = 0
            self.order = torch.randperm(self.size, generator=self.generator)
        stop = min(self.cursor+count, self.epoch_size)
        selected = self.order[self.cursor:stop]
        self.cursor = stop
        return selected

    @property
    def at_end(self):
        return self.cursor == self.epoch_size

    def state(self):
        return {"size": self.size, "epoch": self.epoch, "cursor": self.cursor,
                "order": self.order.clone(), "rng": self.generator.get_state()}

    def restore(self, state):
        if state["size"] != self.size or not 0 <= state["cursor"] <= self.epoch_size:
            raise ValueError("Invalid data cursor")
        order = state["order"].cpu()
        if not torch.equal(order.sort().values, torch.arange(self.size)):
            raise ValueError("Saved order is not a permutation")
        self.epoch, self.cursor, self.order = state["epoch"], state["cursor"], order
        self.generator.set_state(state["rng"].cpu())


def target_counts(data, catalog_size):
    return torch.bincount(data.target.reshape(-1), minlength=catalog_size+1)[1:]


def load_content(path, cfg, device):
    values = np.load(path, allow_pickle=False)
    if values.shape != (cfg["catalog_size"], cfg["rq"]["input_dim"]) or not np.isfinite(values).all():
        raise ValueError("Content shape/nonfinite error")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
        raise ValueError("Expected the existing normalized PCA content")
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.float32, device=device)
