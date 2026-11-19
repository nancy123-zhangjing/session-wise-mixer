"""Versioned GPU trie and exact catalog-constrained RQ landing."""
from __future__ import annotations


import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .common import autocast, sha256, write_json, tensor_digest
from .tokenizer import hard_quantize, distances
from .screen import generation_dim


def distinct_history_ids(history):
    """Keep one copy of each observed item, using zero as padding."""
    ids = history.sort(dim=1).values
    return torch.cat((ids[:, :1], ids[:, 1:].masked_fill(ids[:, 1:] == ids[:, :-1], 0)), dim=1)


class Catalog:
    def __init__(self, asset, device):
        self.device = torch.device(device)
        self.z = asset["z"].detach().float().to(device)
        self.books = asset["books"].detach().float().to(device)
        self.sid = asset["sid"].detach().long().to(device)
        self.original_ids = asset["original_ids"].detach().long().to(device)
        self.stats = {key: asset[key].detach().float().to(device)
                      for key in ("mean", "scale", "distance_scale")}
        self.version = int(asset.get("version", 0))
        self.count, self.dim = self.z.shape
        self.levels, self.codes, _ = self.books.shape
        if self.count < 5 or self.sid.shape != (self.count, self.levels):
            raise ValueError("Invalid catalog geometry")
        if (not torch.isfinite(self.z).all() or not torch.isfinite(self.books).all() or
                (self.sid < 0).any() or (self.sid >= self.codes).any()):
            raise ValueError("Nonfinite catalog or invalid SID")
        if (self.original_ids.shape != (self.count,) or len(self.original_ids.unique()) != self.count or
                self.stats["mean"].shape != (self.dim,) or self.stats["scale"].numel() != 1 or
                self.stats["distance_scale"].shape != (self.levels,) or self.stats["scale"] <= 1e-8 or
                not all(torch.isfinite(v).all() for v in self.stats.values()) or
                (self.stats["distance_scale"] <= 0).any()):
            raise ValueError("Invalid fixed statistics/item mapping")
        actual = hard_quantize(self.z, self.books)
        if not torch.equal(actual["sid"], self.sid):
            raise ValueError("SID/index and effective codebooks belong to different versions")
        self.q = actual["q"]
        if "q" in asset and not torch.equal(asset["q"].to(device), self.q):
            raise ValueError("Cached quantized vectors belong to a different version")
        self.z_pad = torch.cat((self.z.new_zeros(1, self.dim), self.z))
        self.sid_pad = torch.cat((self.sid.new_zeros(1, self.levels), self.sid))
        self.x_pad = (self.z_pad-self.stats["mean"])/self.stats["scale"]
        self._build_trie()
        self.version_hash = tensor_digest({"sid": self.sid, "books": self.books})

    def _build_trie(self):
        """CPU only at a refresh boundary, never in the generation hot loop."""
        sids = self.sid.cpu().tolist()
        originals = self.original_ids.cpu().tolist()
        children, counts, leaves = [{}], [0], {}
        paths = [[0]*(self.levels+1)]
        for item, row in enumerate(sids, 1):
            node, path = 0, [0]
            counts[0] += 1
            for code in row:
                if code not in children[node]:
                    children[node][code] = len(children)
                    children.append({})
                    counts.append(0)
                node = children[node][code]
                counts[node] += 1
                path.append(node)
            paths.append(path)
            leaves.setdefault(node, []).append(item)
        child_array = np.full((len(children), self.codes), -1, dtype=np.int64)
        for n, mapping in enumerate(children):
            for code, child in mapping.items():
                child_array[n, code] = child
        maximum = max(map(len, leaves.values()))
        buckets = np.zeros((len(leaves), maximum), dtype=np.int64)
        node_bucket = np.full(len(children), -1, dtype=np.int64)
        for b, (node, items) in enumerate(leaves.items()):
            items.sort(key=lambda i: originals[i-1])
            buckets[b, :len(items)] = items
            node_bucket[node] = b
        self.children = torch.as_tensor(child_array, device=self.device)
        self.node_count = torch.tensor(counts, device=self.device)
        self.paths = torch.tensor(paths, device=self.device)
        self.node_bucket = torch.as_tensor(node_bucket, device=self.device)
        self.bucket_items = torch.as_tensor(buckets, device=self.device)

    def state(self):
        return {"z": self.z, "books": self.books, "sid": self.sid, "q": self.q, "original_ids": self.original_ids,
                **self.stats, "version": self.version, "version_hash": self.version_hash}

    @torch.no_grad()
    def land(self, generated_z, selected):
        """GPU-only batched traversal; no cpu/item/tolist inside the 4-level loop."""
        z = generated_z.float()
        batch = len(z)
        node = torch.zeros(batch, dtype=torch.long, device=self.device)
        raw_node, raw_valid = node.clone(), torch.ones(batch, dtype=torch.bool, device=self.device)
        residual, raw_residual = z, z
        chosen, raw = [], []
        selected_paths = self.paths[selected]
        for level, book in enumerate(self.books):
            child = self.children[node]
            available = self.node_count[child.clamp_min(0)].masked_fill(child < 0, 0)
            if selected.shape[1]:
                removed = ((child[:, :, None] == selected_paths[:, None, :, level+1]) &
                           (selected[:, None, :] > 0)).sum(-1)
                available = available - removed
            code = distances(residual, book).masked_fill(available <= 0, float("inf")).argmin(-1)
            node = child.gather(1, code[:, None]).squeeze(1)
            residual = residual-book[code]
            chosen.append(code)
            raw_code = distances(raw_residual, book).argmin(-1)
            raw_child = self.children[raw_node.clamp_min(0)].gather(1, raw_code[:, None]).squeeze(1)
            raw_valid = raw_valid & (raw_child >= 0)
            raw_node = raw_child
            raw_residual = raw_residual-book[raw_code]
            raw.append(raw_code)
        candidates = self.bucket_items[self.node_bucket[node]]
        valid = candidates > 0
        if selected.shape[1]:
            valid &= ~(candidates[:, :, None] == selected[:, None, :]).any(-1)
        # Computing all catalog distances avoids a B*max_bucket*32 temporary.
        # Selection is still STRICTLY within the chosen SID bucket, never ANN.
        squared = distances(z, self.z)
        score = squared.gather(1, (candidates-1).clamp_min(0)).masked_fill(~valid, float("inf"))
        item = candidates.gather(1, score.argmin(-1)[:, None]).squeeze(1)
        sid, raw_sid = torch.stack(chosen, -1), torch.stack(raw, -1)
        raw_count = self.node_count[raw_node.clamp_min(0)].masked_fill(~raw_valid, 0)
        if selected.shape[1]:
            raw_count -= ((selected_paths[:, :, -1] == raw_node[:, None]) & (selected > 0)).sum(-1)
        return item, {"finite_vector": torch.isfinite(z).all(-1),
                      "raw_valid": raw_valid, "raw_available": raw_count > 0,
                      "changed": (sid != raw_sid).any(-1), "changed_by_level": sid != raw_sid,
                      "bucket_size": self.node_count[node], "available_bucket_size": valid.sum(-1)}


def evaluation_noise(cfg, sample_ids, split):
    """Stable JSON key, SHA256[:8] little-endian, PCG64 -> FP32; j is 1-based."""
    dim = generation_dim(cfg)
    out = np.empty((len(sample_ids), cfg["list_length"], dim), dtype=np.float32)
    for row, sample_id in enumerate(sample_ids):
        for j in range(cfg["list_length"]):
            key = [cfg["evaluation"]["noise_version"], cfg["evaluation"]["seed"], split, sample_id, j+1]
            seed = int.from_bytes(hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).digest()[:8], "little")
            out[row, j] = np.random.Generator(np.random.PCG64(seed)).standard_normal(dim).astype(np.float32)
    return out


def prepare_noise(path, cfg, sample_ids, split):
    path = Path(path)
    ids_hash = hashlib.sha256(json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    expected = {"noise_version": cfg["evaluation"]["noise_version"], "seed": cfg["evaluation"]["seed"],
                "split": split, "sample_ids_sha256": ids_hash, "numpy": np.__version__,
                "shape": [len(sample_ids), cfg["list_length"], generation_dim(cfg)]}
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(meta.get(k) != v for k, v in expected.items()) or meta["sha256"] != sha256(path):
            raise ValueError("Existing evaluation noise identity/hash mismatch")
        return meta
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial")
    array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=tuple(expected["shape"]))
    for start in range(0, len(sample_ids), 2048):
        array[start:start+2048] = evaluation_noise(cfg, sample_ids[start:start+2048], split)
    array.flush()
    del array
    temp.replace(path)
    meta = {**expected, "sha256": sha256(path)}
    write_json(meta_path, meta)
    return meta


def start_caches(model, catalog, history_ids, runtime, guidance=1.0):
    """Conditional decode cache, plus the null-history cache when guidance != 1."""
    with autocast(runtime, catalog.device):
        memory, valid = model.encode_history(model.lookup(catalog.sid_pad[history_ids]), history_ids != 0)
        caches = [model.start_cache(memory, valid)]
        if guidance != 1.0:
            if model.null_memory is None:
                raise ValueError("Guidance needs a model trained with a null history")
            caches.append(model.start_cache(*model.null_context(len(history_ids), catalog.device, memory.dtype)))
    return caches


def guided_predictor(model, caches, runtime, device, guidance=1.0):
    """x0 prediction; with two caches: x0(null) + w * (x0(history) - x0(null))."""
    def predict(state, times):
        with autocast(runtime, device):
            conditional = model.predict_cached(state, times, caches[0])
            if len(caches) == 1:
                return conditional
            unconditional = model.predict_cached(state, times, caches[1])
        return unconditional.float()+guidance*(conditional.float()-unconditional.float())
    return predict


def append_to_caches(model, catalog, items, caches, runtime):
    with autocast(runtime, catalog.device):
        tokens = model.lookup(catalog.sid_pad[items])
        for cache in caches:
            model.append_prefix(tokens, cache)


@torch.no_grad()
def generate(model, diffusion, catalog, history_ids, initial_noise, runtime, *, filter_history=False, guidance=1.0):
    model.eval()
    caches = start_caches(model, catalog, history_ids, runtime, guidance)
    selected = torch.empty(len(history_ids), 0, device=catalog.device, dtype=torch.long)
    excluded = distinct_history_ids(history_ids) if filter_history else selected
    diagnostics = {}
    predictor = guided_predictor(model, caches, runtime, catalog.device, guidance)
    for j in range(model.length):
        x = diffusion.sample(initial_noise[:, j].float(), predictor)
        z = model.output_to_latent(x, catalog)
        item, diag = catalog.land(z, torch.cat((excluded, selected), 1) if filter_history else selected)
        selected = torch.cat((selected, item[:, None]), 1)
        for key, value in diag.items():
            diagnostics.setdefault(key, []).append(value)
        if j+1 < model.length:
            append_to_caches(model, catalog, item, caches, runtime)
    return selected, {key: torch.stack(value, 1) for key, value in diagnostics.items()}
