"""Standalone, bounded RQ-VAE matching the project's STE/Sinkhorn semantics.

No GenRec import: its top-level text/model dependencies are unnecessary for
precomputed PCA inputs. Source behavior is covered by numerical tests.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b, bias=False))
        if i < len(sizes)-2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


@torch.no_grad()
def kmeans(x, count, iterations, tolerance, generator):
    if len(x) < count:
        raise ValueError("K-means initialization needs at least K input vectors")
    centers = x[torch.randperm(len(x), generator=generator, device=x.device)[:count]].clone()
    used = 0
    for used in range(1, iterations+1):
        ids = (x[:, None] - centers[None]).square().sum(-1).argmin(-1)
        # GPU scatter reductions have nondeterministic atomics. Matrix reduction
        # is small (1024x256), deterministic, and faster than 256 Python masks.
        membership = F.one_hot(ids, count).to(x.dtype)
        counts = membership.sum(0)
        sums = membership.T @ x
        replacements = x[torch.randint(len(x), (count,), device=x.device, generator=generator)]
        next_centers = torch.where(counts[:, None] > 0, sums/counts[:, None].clamp_min(1), replacements)
        delta = (next_centers - centers).norm(dim=-1).max()
        centers = next_centers
        if delta.item() < tolerance:
            break
    return centers, used


@torch.no_grad()
def sinkhorn_ids(distance, iterations=100, epsilon=0.003):
    # Match existing normalization and denominator epsilon, including FP64.
    hi, lo = distance.max(), distance.min()
    cost = ((distance - (hi+lo)/2) / ((hi-lo)/2 + 1e-5)).double()
    kernel = torch.exp(-cost/epsilon)
    b, k = kernel.shape
    u = torch.ones(b, dtype=torch.float64, device=distance.device)
    v = torch.ones(k, dtype=torch.float64, device=distance.device)
    for _ in range(iterations):
        u = (1/b) / (kernel @ v + 1e-8)
        v = (1/k) / (kernel.T @ u + 1e-8)
    return (u[:, None]*kernel*v[None]).argmax(-1)


class RQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg["rq"])
        c = self.cfg
        self.encoder = mlp([c["input_dim"], *c["hidden_dims"], c["latent_dim"]])
        self.decoder = mlp([c["latent_dim"], *reversed(c["hidden_dims"]), c["input_dim"]])
        self.books = nn.Parameter(torch.empty(c["levels"], c["codes"], c["latent_dim"]))
        nn.init.uniform_(self.books)
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, batch, generator):
        if bool(self.initialized):
            raise RuntimeError("Codebooks already initialized; never reinitialize a resume")
        res = self.encoder(batch)
        iters = []
        for book in self.books:
            centers, n = kmeans(res, len(book), self.cfg["kmeans_iterations"],
                                self.cfg["kmeans_tolerance"], generator)
            book.copy_(centers)
            ids = (res[:, None]-book[None]).square().sum(-1).argmin(-1)
            res = res-book[ids]
            iters.append(n)
        self.initialized.fill_(True)
        return iters

    def forward(self, x):
        if not bool(self.initialized):
            raise RuntimeError("Explicit RQ initialization required")
        z = self.encoder(x)
        res, quantized = z, []
        ids, qloss = [], x.new_zeros(len(x))
        for level, book in enumerate(self.books):
            distance = (res[:, None]-book[None]).square().sum(-1)
            code = (sinkhorn_ids(distance.detach(), self.cfg["sinkhorn_iterations"],
                                 self.cfg["sinkhorn_epsilon"])
                    if self.training and level == len(self.books)-1 else distance.detach().argmin(-1))
            value = F.embedding(code, book)
            qloss = qloss + (res.detach()-value).square().sum(-1)
            qloss = qloss + self.cfg["commitment"]*(res-value.detach()).square().sum(-1)
            out = res + (value-res).detach() if self.training else value
            # Deliberately match the project's residual autograd semantics.
            res = res-out
            quantized.append(out)
            ids.append(code)
        q = torch.stack(quantized).sum(0)
        reconstruction = F.normalize(self.decoder(q), dim=-1)
        rec = (reconstruction-x).square().sum(-1)
        return {"loss": (rec+qloss).mean(), "reconstruction_loss": rec.mean(),
                "quantization_loss": qloss.mean(), "z": z, "q": q,
                "sid": torch.stack(ids, -1), "errors": rec}


@torch.no_grad()
def encode_catalog(model, content, batch_size):
    model.eval()
    chunks = [model(x) for x in content.split(batch_size)]
    return {key: torch.cat([r[key] for r in chunks]).cpu()
            for key in ("z", "q", "sid", "errors")}


def health(sids, codes):
    sid = sids.detach().cpu()
    _, counts = torch.unique(sid, dim=0, return_counts=True)
    used, entropy = [], []
    for col in sid.T:
        n = torch.bincount(col, minlength=codes)
        p = n[n > 0].double()/len(sid)
        used.append(int((n > 0).sum()))
        entropy.append(float(-(p*p.log()).sum()))
    return {
        "unique_sids": len(counts), "unique_ratio": len(counts)/len(sid),
        "collision_rate": float(counts[counts > 1].sum())/len(sid),
        "largest_bucket": int(counts.max()), "used_codes": used,
        "min_utilization": min(used)/codes, "entropy_nats": entropy,
        "perplexity": [float(torch.tensor(v, dtype=torch.float64).exp()) for v in entropy],
    }


def latent_scaling(z, target_ids):
    counts = torch.bincount(target_ids.cpu().reshape(-1), minlength=len(z)+1)[1:].double()
    values = z.cpu().double()
    mean = (counts[:, None]*values).sum(0)/counts.sum()
    scale = ((counts[:, None]*(values-mean).square()).sum()/(counts.sum()*z.shape[1])).sqrt()
    if not torch.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Degenerate latent space")
    return mean, scale
