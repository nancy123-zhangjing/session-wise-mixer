"""Global RQ codebooks, two masked Mixers and distinct pretrain/joint STE paths."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def distances(x, centers):
    """Squared L2 via GEMM: no [occurrences,256,32] materialization."""
    with torch.autocast(x.device.type, enabled=False):
        x, centers = x.float(), centers.float()
        return (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)
                - 2 * (x @ centers.T)).clamp_min(0)


def make_mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layer = nn.Linear(a, b)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(widths)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CodebookMixer(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        if mode not in ("intra", "causal"):
            raise ValueError("Mixer mode must be intra or causal")
        c = cfg["rq"]
        self.mode, self.levels, self.codes, self.dim = mode, c["levels"], c["codes"], c["latent_dim"]
        self.heads = cfg["mixer"]["heads"]
        self.reference = cfg["mixer"]["reference_level"]
        if self.dim % self.heads:
            raise ValueError("Mixer dimension must divide into attention heads")
        self.norm = nn.LayerNorm(self.dim, eps=cfg["mixer"]["norm_eps"])
        self.layer_embedding = nn.Parameter(torch.empty(self.levels, self.dim))
        nn.init.normal_(self.layer_embedding, std=0.02)
        self.q = nn.Linear(self.dim, self.dim)
        self.k = nn.Linear(self.dim, self.dim)
        self.v = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        levels = torch.arange(self.levels).repeat_interleave(self.codes)
        self.register_buffer("causal_mask", (levels[:, None] >= levels[None, :])[None, None], persistent=False)

    def _heads(self, x):
        return x.view(*x.shape[:-1], self.heads, self.dim//self.heads).transpose(-3, -2)

    def forward(self, books, *, return_raw=False):
        # Small global Mixer and its scale calculation are deliberately FP32.
        with torch.autocast(books.device.type, enabled=False):
            books = books.float()
            x = self.norm(books) + self.layer_embedding[:, None]
            if self.mode == "intra":
                # Four independent 256-token problems, shared Q/K/V; not a
                # 1024-token dense attention with three quarters masked out.
                mask = None
            else:
                x = x.reshape(1, self.levels*self.codes, self.dim)
                mask = self.causal_mask
            q, k, v = (self._heads(layer(x)) for layer in (self.q, self.k, self.v))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            raw = self.out(y.transpose(-3, -2).contiguous().view_as(x)).view_as(books)
            rms = books.square().mean((-2, -1)).sqrt()
            ratio = (rms/rms[self.reference].clamp_min(1e-8)).detach()
            correction = raw * ratio[:, None, None]
            return (correction, raw) if return_raw else correction


class RQTokenizer(nn.Module):
    def __init__(self, cfg, mode="none"):
        super().__init__()
        c = cfg["rq"]
        self.cfg, self.mode = cfg, mode
        self.encoder = make_mlp([c["input_dim"], *c["hidden_dims"], c["latent_dim"]])
        self.decoder = make_mlp([c["latent_dim"], *reversed(c["hidden_dims"]), c["input_dim"]])
        self.books = nn.Parameter(torch.zeros(c["levels"], c["codes"], c["latent_dim"]))
        self.mixer = None if mode == "none" else CodebookMixer(cfg, mode)
        self.register_buffer("initialized", torch.tensor(False))
        self.cf_projection = None
        if cfg.get("target", {}).get("mode") == "hybrid":
            # Pretraining-only head; do not shift shared initialization streams.
            with torch.random.fork_rng(devices=[]):
                self.cf_projection = make_mlp([c["latent_dim"], cfg["target"]["teacher_dim"]])

    def effective_books(self):
        return self.books if self.mixer is None else self.books + self.mixer(self.books)

    @torch.no_grad()
    def books_with_diagnostics(self):
        if self.mixer is None:
            correction, raw = torch.zeros_like(self.books), torch.zeros_like(self.books)
        else:
            correction, raw = self.mixer(self.books, return_raw=True)
        effective = self.books+correction
        rms = lambda x: x.square().mean((-2, -1)).sqrt()
        base_rms = rms(self.books)
        stats = {"base_rms": base_rms.cpu().tolist(), "raw_mixer_rms": rms(raw).cpu().tolist(),
                 "correction_rms": rms(correction).cpu().tolist(), "effective_rms": rms(effective).cpu().tolist(),
                 "correction_relative_rms": (rms(correction)/base_rms.clamp_min(1e-8)).cpu().tolist(),
                 "near_zero_base_scale": (base_rms <= 1e-8).cpu().tolist()}
        if not torch.isfinite(effective).all():
            raise RuntimeError("Nonfinite effective codebooks")
        return effective, stats

    @torch.no_grad()
    def initialize(self, first_content, generator):
        if self.initialized.item():
            raise RuntimeError("Do not initialize an existing checkpoint again")
        residual = self.encoder(first_content.float())
        count = self.books.shape[1]
        if len(residual) < count:
            raise ValueError("First item batch must contain at least K items")
        for book in self.books:
            centers = residual[torch.randperm(len(residual), generator=generator, device=residual.device)[:count]].clone()
            for _ in range(self.cfg["rq"]["kmeans_iterations"]):
                code = distances(residual, centers).argmin(-1)
                member = F.one_hot(code, count).float()
                counts = member.sum(0)
                sums = member.T @ residual
                replacement = residual[torch.randint(len(residual), (count,), generator=generator, device=residual.device)]
                centers = torch.where(counts[:, None] > 0, sums/counts.clamp_min(1)[:, None], replacement)
            book.copy_(centers)
            residual = residual - book[distances(residual, book).argmin(-1)]
        self.initialized.fill_(True)

    def pretrain_forward(self, content, cf=None, groups=None, supported=None):
        with torch.autocast(content.device.type, enabled=False):
            z = self.encoder(content.float())
            books = self.effective_books()
            residual, parts, losses, sids = z, [], [], []
            for book in books:
                code = distances(residual, book).argmin(-1)
                value = book[code]
                losses.append(F.mse_loss(value, residual.detach()) +
                              self.cfg["rq"]["beta"]*F.mse_loss(value.detach(), residual))
                # ETEGRec-style pretraining STE; NOT the soft joint-training STE.
                straight = residual + (value-residual).detach()
                residual = residual-straight
                parts.append(straight)
                sids.append(code)
            q = torch.stack(parts).sum(0)
            reconstruction = F.mse_loss(self.decoder(q), content.float())
            vq = torch.stack(losses).mean()
            align = reconstruction.new_zeros(())
            if self.cf_projection is not None:
                if cf is None or groups is None or supported is None:
                    raise ValueError("Hybrid pretraining requires bound frozen teacher rows")
                from .targets import collaborative_loss
                align = collaborative_loss(q, cf, self.cf_projection, groups, supported,
                                           self.cfg["target"]["cf_temperature"])
            loss = reconstruction+vq+self.cfg.get("target", {}).get("cf_weight", 0)*align
            return {"loss": loss, "reconstruction": reconstruction, "vq": vq, "cf": align,
                    "z": z, "q": q, "sid": torch.stack(sids, -1)}

    def joint_parameters(self):
        return [self.books] + ([] if self.mixer is None else list(self.mixer.parameters()))


@torch.no_grad()
def hard_quantize(z, books):
    residual, sids, values, residuals = z.float(), [], [], []
    for book in books:
        residuals.append(residual)
        code = distances(residual, book).argmin(-1)
        value = book[code]
        sids.append(code)
        values.append(value)
        residual = residual-value
    return {"sid": torch.stack(sids, -1), "q": torch.stack(values, -2).sum(-2),
            "residuals": torch.stack(residuals, -2)}


def joint_quantize(z, books, scale, cached_sid, embedding, noise, sigma, cfg):
    """Per-occurrence p; hard current internal centers, cached-hard input E.

    embedding is [levels,codes,width]; returns only the four losses' needed
    tensors. Hard+zero-soft-difference avoids allocating another one-hot tensor.
    """
    with torch.autocast(z.device.type, enabled=False):
        residual, parts, embeds, layer_losses = z.float(), [], [], []
        for level, book in enumerate(books):
            logits = -distances(residual, book)/(cfg["quantization"]["temperature"]*scale[level])
            p = F.softmax((logits+sigma*noise[:, level])/cfg["quantization"]["softmax_temperature"], dim=-1)
            soft_grad = p-p.detach()
            code = p.argmax(-1)
            value = book[code] + soft_grad @ book
            embeds.append(embedding[level, cached_sid[:, level]] + soft_grad @ embedding[level])
            # Return per occurrence, so caller excludes PAD/history from RQ losses.
            layer_losses.append((value-residual.detach()).square().mean(-1) +
                                cfg["rq"]["beta"]*(value.detach()-residual).square().mean(-1))
            parts.append(value)
            residual = residual-value  # No detach: retain cross-level feedback.
        return {"embedding": torch.stack(embeds, -2), "q": torch.stack(parts, -2).sum(-2),
                "vq": torch.stack(layer_losses, -1).mean(-1)}


@torch.no_grad()
def fixed_statistics(z, books, target_counts):
    """Only training-target occurrence counts; no validation/test contribution."""
    weight, value = target_counts.double(), z.double()
    if weight.sum() <= 0 or (weight < 0).any():
        raise ValueError("Invalid training target counts")
    mean = (weight[:, None]*value).sum(0)/weight.sum()
    scale = ((weight[:, None]*(value-mean).square()).sum()/(weight.sum()*value.shape[-1])).sqrt()
    if not torch.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Degenerate fixed latent space")
    hard = hard_quantize(z, books)
    typical = []
    for level, book in enumerate(books):
        # FP64 statistics, using the FP32 deterministic residual path.
        res = hard["residuals"][:, level].double()
        center = book.double()
        # Mean over every center, algebraically identical without an N*K*D
        # FP64 allocation. Accumulate statistics in FP64, not under autocast.
        d = (res.square().sum(-1) + center.square().sum(-1).mean()
             - 2*(res*center.mean(0)).sum(-1)).clamp_min(0)
        typical.append((weight*d).sum()/weight.sum())
    return {"mean": mean.float(), "scale": scale.float(),
            "distance_scale": torch.stack(typical).clamp_min(1e-8).float(), "target_counts": target_counts}


@torch.no_grad()
def code_health(sid, codes):
    sid = sid.detach().cpu()
    _, count = torch.unique(sid, dim=0, return_counts=True)
    use, perplexity = [], []
    for col in sid.T:
        counts = torch.bincount(col, minlength=codes)
        p = counts[counts > 0].double()/len(sid)
        use.append(int((counts > 0).sum())/codes)
        perplexity.append(float((-(p*p.log()).sum()).exp()))
    return {"collision_redundancy": 1-len(count)/len(sid),
            "collision_items": float(count[count > 1].sum())/len(sid),
            "unique_sids": len(count), "max_bucket": int(count.max()),
            "utilization": use, "perplexity": perplexity}
