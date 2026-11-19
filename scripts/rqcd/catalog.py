"""GPU-resident frozen tables and deterministic constrained residual decoding."""
from __future__ import annotations

import torch
from torch import nn
from .common import derive_seed, tensor_digest


class Catalog(nn.Module):
    def __init__(self, asset):
        super().__init__()
        z, sid, books = asset["z"].float(), asset["sid"].long(), asset["books"].float()
        n, d = z.shape
        if sid.shape != (n, books.shape[0]) or books.shape[-1] != d or n < 5:
            raise ValueError("Invalid frozen catalog geometry")
        if (sid < 0).any() or (sid >= books.shape[1]).any() or not torch.isfinite(z).all():
            raise ValueError("Invalid catalog values")
        mean, scale = asset["mean"].float(), asset["scale"].float()
        if (mean.shape != (d,) or scale.numel() != 1 or
                not torch.isfinite(mean).all() or not torch.isfinite(scale) or scale <= 1e-8):
            raise ValueError("Invalid latent scale")
        if (not torch.isfinite(books).all() or asset["errors"].shape != (n,) or
                not torch.isfinite(asset["errors"]).all() or (asset["errors"] < 0).any() or
                asset["original_ids"].shape != (n,) or len(torch.unique(asset["original_ids"])) != n):
            raise ValueError("Invalid codebooks or collision ordering")
        # Validate the saved assignment before any downstream training. No
        # model is run: this is a deterministic integrity check of numeric tables.
        for start in range(0, n, 1024):
            res = z[start:start+1024]
            for level, book in enumerate(books):
                code = (res[:, None]-book[None]).square().sum(-1).argmin(-1)
                if not torch.equal(code, sid[start:start+1024, level]):
                    raise ValueError("Exported SID does not equal Q(z)")
                res = res-book[code]
        errors, originals = asset["errors"].tolist(), asset["original_ids"].tolist()
        order = sorted(range(n), key=lambda i: (errors[i], originals[i], i+1))
        rank = torch.empty(n, dtype=torch.long)
        rank[torch.tensor(order)] = torch.arange(n)
        self.register_buffer("y", torch.cat((torch.zeros(1, d), (z-mean)/scale)))
        self.register_buffer("sid", sid)
        self.register_buffer("books", books)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.register_buffer("priority", rank)
        self.register_buffer("item_ids", torch.arange(1, n+1))
        # Posting lists replace CPU dictionaries and per-item Python work.
        posting = torch.nn.functional.one_hot(sid, books.shape[1]).permute(1, 0, 2).float()
        self.register_buffer("posting", posting)
        self.frozen_hash = tensor_digest(self.state_dict())

    def verify_frozen(self):
        if tensor_digest(self.state_dict()) != self.frozen_hash:
            raise RuntimeError("Frozen catalog modified")

    @torch.no_grad()
    def quantize(self, z, selected):
        b, n = len(z), len(self.sid)
        eligible = torch.ones(b, n, device=z.device, dtype=torch.bool)
        if selected.numel():
            eligible &= ~(self.item_ids[None, :, None] == selected[:, None, :]).any(-1)
        candidates = eligible.clone()
        raw_res, res = z.float(), z.float()
        raw_codes, codes = [], []
        first_changed = torch.full((b,), -1, device=z.device, dtype=torch.long)
        for l, book in enumerate(self.books):
            # 10k catalog * 256 at B32 is small; use postings GEMM, no CPU sync.
            allowed = candidates.float() @ self.posting[l] > 0
            dist = (res[:, None, :]-book[None]).square().sum(-1)
            raw_dist = (raw_res[:, None, :]-book[None]).square().sum(-1)
            raw = raw_dist.argmin(-1)
            code = dist.masked_fill(~allowed, torch.inf).argmin(-1)
            first_changed = torch.where((first_changed < 0) & (code != raw), l, first_changed)
            candidates &= self.sid[None, :, l] == code[:, None]
            res = res-book[code]
            raw_res = raw_res-book[raw]
            raw_codes.append(raw)
            codes.append(code)
        sid = torch.stack(codes, -1)
        raw_sid = torch.stack(raw_codes, -1)
        raw_bucket = (self.sid[None] == raw_sid[:, None]).all(-1)
        rank = self.priority[None].expand(b, -1).masked_fill(~candidates, n)
        item = rank.argmin(-1)+1
        complete = candidates.any(-1)
        item = torch.where(complete, item, 0)
        return item, {
            "sid": sid, "raw_sid": raw_sid, "raw_valid": raw_bucket.any(-1),
            "raw_available": (raw_bucket & eligible).any(-1),
            "changed": (raw_sid != sid).any(-1), "first_changed_layer": first_changed,
            "raw_residual": raw_res.norm(dim=-1), "residual": res.norm(dim=-1),
            "bucket_size": (self.sid[None] == sid[:, None]).all(-1).sum(-1),
            "complete": complete,
        }


def evaluation_noise(cfg, sample_ids, split, position, dimension, *, tag=None):
    vectors = []
    for sample in sample_ids:
        parts = [cfg["protocol_id"], cfg["evaluation"]["seed"], split, sample, position+1]
        if tag is not None:
            parts.append(tag)
        g = torch.Generator(device="cpu").manual_seed(derive_seed(*parts))
        vectors.append(torch.randn(dimension, generator=g))
    return torch.stack(vectors)


@torch.inference_mode()
def generate(model, diffusion, catalog, history_ids, sample_ids, split, cfg):
    if model.training:
        raise ValueError("Generation requires eval mode")
    history = catalog.y[history_ids]
    mask = history_ids.ne(0)
    memory = model.encode_history(history, mask)
    cache = model.cache_history(memory)
    selected = history_ids.new_empty((len(history_ids), 0))
    diagnostics = []
    for j in range(model.length):
        prefix = catalog.y[selected]
        initial = evaluation_noise(cfg, sample_ids, split, j, catalog.y.shape[1]).to(history.device)

        def predict(y, t):
            times = torch.full((len(y),), t, dtype=torch.long, device=y.device)
            return model.predict(y, times, prefix, j, memory, mask, cache)

        output = diffusion.sample(initial, predict)
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite generated latent; refusing catalog fallback")
        item, diag = catalog.quantize(output*catalog.scale+catalog.mean, selected)
        selected = torch.cat((selected, item[:, None]), -1)
        diagnostics.append(diag)
    if not all(bool(d["complete"].all()) for d in diagnostics):
        raise RuntimeError("Catalog generation failed; no ANN/popularity fallback allowed")
    return selected, {k: torch.stack([d[k] for d in diagnostics], 1) for k in diagnostics[0]}
