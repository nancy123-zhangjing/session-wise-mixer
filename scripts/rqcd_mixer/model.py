"""T5 semantics with batched five-view training and exact clean-prefix caching.

T5 equations/initialization reference: transformers v4.53.2 modeling_t5.py
(Apache-2.0). This implementation has no LM head or unused vocabulary table.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .screen import generation_dim, generation_space, switches


class T5Norm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)+self.eps)
        return self.weight * y.to(self.weight.dtype)


def relative_buckets(relative, bidirectional, buckets=32, max_distance=128):
    result = torch.zeros_like(relative)
    if bidirectional:
        buckets //= 2
        result = (relative > 0).long()*buckets
        n = relative.abs()
    else:
        n = (-relative).clamp_min(0)
    exact = buckets//2
    large = exact + (torch.log(n.float().clamp_min(exact)/exact) /
                     math.log(max_distance/exact)*(buckets-exact)).long()
    return result + torch.where(n < exact, n, large.clamp_max(buckets-1))


class RelativeBias(nn.Module):
    def __init__(self, cfg, bidirectional):
        super().__init__()
        self.table = nn.Embedding(cfg["relative_buckets"], cfg["heads"])
        nn.init.normal_(self.table.weight, std=cfg["width"]**-0.5)
        self.bidirectional = bidirectional
        self.buckets, self.maximum = cfg["relative_buckets"], cfg["relative_max_distance"]

    def forward(self, qlen, klen, device, offset=0):
        query = torch.arange(offset, offset+qlen, device=device)
        key = torch.arange(klen, device=device)
        indices = relative_buckets(key[None]-query[:, None], self.bidirectional, self.buckets, self.maximum)
        return self.table(indices).permute(2, 0, 1)  # heads,Q,K; one table per stack


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.d_kv = cfg["heads"], cfg["d_kv"]
        width, inner = cfg["width"], cfg["heads"]*cfg["d_kv"]
        self.q = nn.Linear(width, inner, bias=False)
        self.k = nn.Linear(width, inner, bias=False)
        self.v = nn.Linear(width, inner, bias=False)
        self.o = nn.Linear(inner, width, bias=False)
        nn.init.normal_(self.q.weight, std=(width*self.d_kv)**-0.5)
        for layer in (self.k, self.v):
            nn.init.normal_(layer.weight, std=width**-0.5)
        nn.init.normal_(self.o.weight, std=inner**-0.5)
        self.dropout = cfg["dropout"]

    def split(self, x):
        return x.reshape(*x.shape[:-1], self.heads, self.d_kv).transpose(-3, -2)

    def project_kv(self, x):
        return self.split(self.k(x)), self.split(self.v(x))

    def from_qkv(self, q, kv, bias=None, allowed=None):
        # bias remains FP32. SDPA uses the T5 unscaled dot product, not its
        # default 1/sqrt(d_k). Backend dispatch may fall back for relative bias.
        mask = bias
        if allowed is not None:
            padding = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
            mask = padding if bias is None else bias + padding
        view_shape = None
        if q.ndim == 5:
            batch, views, heads, length, width = q.shape
            view_shape = (batch, views, heads, length, width)
            if kv[0].shape[1] == 1:
                # Cross-attention has the SAME history for all five views.
                # Fold views into query length; do not copy history K/V and
                # do not pass 5-D tensors (unsupported by fused CUDA kernels).
                q = q.permute(0, 2, 1, 3, 4).reshape(batch, heads, views*length, width)
                kv = tuple(x[:, 0] for x in kv)
                if mask is not None:
                    mask = mask[:, 0]
                folded_queries = True
            else:
                # Self-attention keeps views independent, including their
                # distinct valid-prefix masks and original query positions.
                q = q.reshape(batch*views, heads, length, width)
                kv = tuple(x.reshape(batch*views, heads, x.shape[-2], width) for x in kv)
                if mask is not None:
                    mask = mask.expand(batch, views, heads, length, kv[0].shape[-2])
                    mask = mask.reshape(batch*views, heads, length, kv[0].shape[-2]).contiguous()
                folded_queries = False
        out = F.scaled_dot_product_attention(q, *kv, attn_mask=mask,
                                            dropout_p=self.dropout if self.training else 0.0, scale=1.0)
        if view_shape is not None:
            batch, views, heads, length, width = view_shape
            if folded_queries:
                out = out.reshape(batch, heads, views, length, width).permute(0, 2, 1, 3, 4)
            else:
                out = out.reshape(view_shape)
        return self.o(out.transpose(-3, -2).contiguous().flatten(-2))

    def forward(self, x, kv=None, bias=None, allowed=None):
        q = self.split(self.q(x))
        return self.from_qkv(q, self.project_kv(x) if kv is None else kv, bias, allowed)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wi = nn.Linear(cfg["width"], cfg["ffn"], bias=False)
        self.wo = nn.Linear(cfg["ffn"], cfg["width"], bias=False)
        nn.init.normal_(self.wi.weight, std=cfg["width"]**-0.5)
        nn.init.normal_(self.wo.weight, std=cfg["ffn"]**-0.5)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.wo(self.drop(F.relu(self.wi(x))))


class Block(nn.Module):
    def __init__(self, cfg, decoder=False):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.self_norm = T5Norm(cfg["width"], cfg["norm_eps"])
        self.ff = FeedForward(cfg)
        self.ff_norm = T5Norm(cfg["width"], cfg["norm_eps"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.cross = Attention(cfg) if decoder else None
        self.cross_norm = T5Norm(cfg["width"], cfg["norm_eps"]) if decoder else None

    def rest(self, x, cross_kv=None, history_allowed=None):
        if self.cross is not None:
            x = x + self.drop(self.cross(self.cross_norm(x), kv=cross_kv, allowed=history_allowed))
        return x + self.drop(self.ff(self.ff_norm(x)))

    def forward(self, x, bias, allowed, cross_kv=None, history_allowed=None):
        x = x + self.drop(self.self_attn(self.self_norm(x), bias=bias, allowed=allowed))
        return self.rest(x, cross_kv, history_allowed)


@dataclass
class DecodeCache:
    owner: int
    cross: list
    prefix: list
    history_allowed: torch.Tensor
    length: int = 0


class Recommender(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c, r = cfg["model"], cfg["rq"]
        self.cfg = cfg
        self.levels, self.codes, self.dim = r["levels"], r["codes"], generation_dim(cfg)
        self.width, self.length = c["width"], cfg["list_length"]
        # Four independent E tables stored contiguously; no tied code IDs.
        self.embedding = nn.Parameter(torch.empty(self.levels, self.codes, self.width))
        nn.init.normal_(self.embedding, std=0.02)
        self.encoder = nn.ModuleList(Block(c) for _ in range(c["encoder_layers"]))
        self.decoder = nn.ModuleList(Block(c, decoder=True) for _ in range(c["decoder_layers"]))
        self.enc_bias, self.dec_bias = RelativeBias(c, True), RelativeBias(c, False)
        self.enc_norm, self.dec_norm = T5Norm(self.width, c["norm_eps"]), T5Norm(self.width, c["norm_eps"])
        self.drop = nn.Dropout(c["dropout"])
        self.time = nn.Sequential(nn.Linear(self.dim, c["time_hidden"]), nn.GELU(),
                                  nn.Linear(c["time_hidden"], self.dim))
        self.input = nn.Linear(self.dim, self.width)
        self.output = nn.Linear(self.width, self.dim)
        for layer in [self.input, self.output, self.time[0], self.time[2]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        half = self.dim//2
        self.register_buffer("time_frequency", torch.exp(-math.log(10000)*torch.arange(half)/max(half-1, 1)))
        self.register_buffer("sid_offsets", torch.arange(self.levels)*self.codes, persistent=False)
        max_len = self.levels*(self.length-1)+1
        positions = torch.arange(max_len)
        queries = self.levels*torch.arange(self.length)
        self.register_buffer("query_positions", queries, persistent=False)
        self.register_buffer("view_prefix", positions[None] < queries[:, None], persistent=False)
        self.register_buffer("view_query", positions[None] == queries[:, None], persistent=False)
        valid = positions[None] <= queries[:, None]
        causal = positions[None, :] <= positions[:, None]
        self.register_buffer("view_allowed", (valid[:, None, :] & causal[None])[:, None], persistent=False)
        # Screening modules are created after every original parameter, so the
        # original ones keep their initialization; without screen switches the
        # module is exactly the original one.
        s = switches(cfg)
        self.generation_space = generation_space(cfg)
        self.history_query = self.history_head = self.null_memory = self.to_latent = None
        if s.get("history_head"):
            self.history_query = nn.Parameter(torch.empty(self.width))
            nn.init.normal_(self.history_query, std=self.width**-0.5)
            self.history_head = nn.Linear(self.width, r["input_dim"])
            nn.init.xavier_uniform_(self.history_head.weight)
            nn.init.zeros_(self.history_head.bias)
        if s.get("guidance"):
            # Same scale as the T5-normalized encoder memory it replaces.
            self.null_memory = nn.Parameter(torch.empty(self.width))
            nn.init.normal_(self.null_memory, std=1.0)
        if self.generation_space == "teacher":
            from .tokenizer import make_mlp
            # Frozen copy of the RQ Encoder (loaded by the trainer): 256-d output -> latent z.
            self.to_latent = make_mlp([r["input_dim"], *r["hidden_dims"], r["latent_dim"]]).requires_grad_(False)
            self.register_buffer("gen_mean", torch.zeros(r["input_dim"]))
            self.register_buffer("gen_scale", torch.ones(()))

    def fixed_parameter_names(self):
        """Parameters that are never optimized (the frozen RQ Encoder copy)."""
        return {n for n, _ in self.named_parameters() if n.startswith("to_latent.")}

    def trainable_parameters(self):
        fixed = self.fixed_parameter_names()
        return [p for n, p in self.named_parameters() if n not in fixed]

    def output_to_latent(self, x, catalog):
        """Generated x0 (standardized) -> RQ latent z used for landing."""
        if self.to_latent is None:
            return x*catalog.stats["scale"]+catalog.stats["mean"]
        return self.to_latent(x.float()*self.gen_scale+self.gen_mean)

    def drop_history(self, memory, valid, drop):
        """Classifier-free guidance training: replace dropped rows by the null memory."""
        null = torch.zeros_like(memory[:1])
        null[0, 0] = self.null_memory.to(memory.dtype)
        first = torch.zeros_like(valid[:1])
        first[0, 0] = True
        memory = torch.where(drop[:, None, None], null.expand_as(memory), memory)
        valid = torch.where(drop[:, None], first.expand_as(valid), valid)
        return memory, valid

    def null_context(self, batch, device, dtype=None):
        """Memory for the unconditional branch: one null token, always valid."""
        memory = self.null_memory.to(device=device, dtype=dtype or self.null_memory.dtype)
        return memory.expand(batch, 1, -1), torch.ones(batch, 1, dtype=torch.bool, device=device)

    def history_logits(self, memory, valid, item_vectors, temperature):
        """Encoder-only next-item scores against fixed item vectors [N, input_dim] (FP32)."""
        with torch.autocast(memory.device.type, enabled=False):
            memory = memory.float()
            score = (memory @ self.history_query.float()) * self.width**-0.5
            weight = torch.softmax(score.masked_fill(~valid, float("-inf")), -1)
            user = self.history_head((weight[..., None]*memory).sum(1))
            return F.normalize(user, dim=-1) @ F.normalize(item_vectors.float(), dim=-1).T / temperature

    def lookup(self, sid):
        return F.embedding(sid+self.sid_offsets, self.embedding.flatten(0, 1))

    def encode_history(self, embeddings, valid_items):
        x = embeddings.flatten(-3, -2)
        valid = valid_items.repeat_interleave(self.levels, -1)
        x = self.drop(x.masked_fill(~valid[..., None], 0))
        bias = self.enc_bias(x.shape[-2], x.shape[-2], x.device)
        allowed = valid[:, None, None, :]
        for block in self.encoder:
            x = block(x, bias, allowed)
        return self.drop(self.enc_norm(x)).masked_fill(~valid[..., None], 0), valid

    def noisy_query(self, noisy, times):
        phase = times.float()[..., None]*self.time_frequency
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        return self.input(noisy+time)

    def training_prediction(self, history_embedding, history_valid, target_embedding, noisy, times,
                            *, uncond=None, return_memory=False):
        """One history encoding; 5 views as a broadcast batch, no repeated H/KV."""
        memory, valid = self.encode_history(history_embedding, history_valid)
        encoded = (memory, valid)
        if uncond is not None:
            memory, valid = self.drop_history(memory, valid, uncond)
        current = self.noisy_query(noisy, times)
        prefix = target_embedding.flatten(-3, -2)[:, :self.view_prefix.shape[-1]]
        x = (prefix[:, None] * self.view_prefix[None, :, :, None] +
             current[:, :, None] * self.view_query[None, :, :, None])
        x = self.drop(x)  # [B,5,17,256], queries stay at 0,4,8,12,16
        bias = self.dec_bias(x.shape[-2], x.shape[-2], x.device)
        for block in self.decoder:
            # Broadcast [B,1,heads,200,64] across views without repeat_interleave.
            kv = tuple(t[:, None] for t in block.cross.project_kv(memory))
            x = block(x, bias, self.view_allowed[None], kv, valid[:, None, None, None, :])
        query = x[:, torch.arange(self.length, device=x.device), self.query_positions]
        prediction = self.output(self.drop(self.dec_norm(query)))
        return (prediction, *encoded) if return_memory else prediction

    def predict_uncached(self, noisy, times, prefix_embedding, memory, history_valid):
        """Full-prefix reference; same computation as cached inference."""
        query = self.noisy_query(noisy, times)[:, None]
        x = self.drop(torch.cat((prefix_embedding.flatten(-3, -2), query), 1))
        n = x.shape[1]
        positions = torch.arange(n, device=x.device)
        allowed = (positions[None] <= positions[:, None])[None, None]
        bias = self.dec_bias(n, n, x.device)
        for block in self.decoder:
            x = block(x, bias, allowed, block.cross.project_kv(memory), history_valid[:, None, None, :])
        return self.output(self.drop(self.dec_norm(x[:, -1])))

    @torch.no_grad()
    def start_cache(self, memory, history_valid):
        if self.training:
            raise RuntimeError("Inference cache requires eval mode")
        return DecodeCache(id(self), [b.cross.project_kv(memory) for b in self.decoder],
                           [None for _ in self.decoder], history_valid[:, None, None, :])

    @torch.no_grad()
    def append_prefix(self, tokens, cache):
        """Append CLEAN actual-item SID tokens; never append a noisy query."""
        if self.training or cache.owner != id(self):
            raise RuntimeError("Invalid inference cache owner/mode")
        x = self.drop(tokens)
        old, n = cache.length, tokens.shape[1]
        device = tokens.device
        bias = self.dec_bias(n, old+n, device, offset=old)
        qpos = torch.arange(old, old+n, device=device)
        allowed = (torch.arange(old+n, device=device)[None] <= qpos[:, None])[None, None]
        for i, block in enumerate(self.decoder):
            normalized = block.self_norm(x)
            new = block.self_attn.project_kv(normalized)
            previous = cache.prefix[i]
            kv = new if previous is None else tuple(torch.cat((a, b), -2) for a, b in zip(previous, new))
            q = block.self_attn.split(block.self_attn.q(normalized))
            x = x + block.drop(block.self_attn.from_qkv(q, kv, bias, allowed))
            cache.prefix[i] = kv
            x = block.rest(x, cache.cross[i], cache.history_allowed)
        cache.length += n

    def predict_cached(self, noisy, times, cache):
        if self.training or cache.owner != id(self):
            raise RuntimeError("Invalid inference cache owner/mode")
        x = self.noisy_query(noisy, times)[:, None]
        bias = self.dec_bias(1, cache.length+1, x.device, offset=cache.length)
        for i, block in enumerate(self.decoder):
            normalized = block.self_norm(x)
            query_kv = block.self_attn.project_kv(normalized)
            previous = cache.prefix[i]
            kv = query_kv if previous is None else tuple(torch.cat((a, b), -2) for a, b in zip(previous, query_kv))
            q = block.self_attn.split(block.self_attn.q(normalized))
            x = x + block.drop(block.self_attn.from_qkv(q, kv, bias))
            x = block.rest(x, cache.cross[i], cache.history_allowed)
        # Query KV is intentionally NOT persisted in cache.prefix.
        return self.output(self.dec_norm(x[:, 0]))


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["diffusion"]
        self.steps, self.sample_steps = c["steps"], c["sample_steps"]
        if not 2 <= self.sample_steps <= self.steps:
            raise ValueError("Invalid DDIM schedule")
        t = torch.arange(self.steps+1, dtype=torch.float64)
        f = torch.cos((t/self.steps+c["cosine_offset"])/(1+c["cosine_offset"])*math.pi/2).square()
        beta = (1-f[1:]/f[:-1]).clamp_max(c["beta_max"])
        alpha = torch.cat((torch.ones(1, dtype=torch.float64), (1-beta).cumprod(0))).float()
        self.register_buffer("alpha", alpha)
        self.register_buffer("root_alpha", alpha.sqrt())
        self.register_buffer("root_noise", (1-alpha).sqrt())
        self.grid = [1+math.floor((self.steps-1)*k/(self.sample_steps-1)+0.5)
                     for k in reversed(range(self.sample_steps))]
        self.register_buffer("grid_times", torch.tensor(self.grid, dtype=torch.long))

    def noise(self, clean, times, epsilon):
        return self.root_alpha[times][..., None]*clean + self.root_noise[times][..., None]*epsilon

    def sample(self, initial, predictor):
        state = initial.float()
        for index, (t, u) in enumerate(zip(self.grid, self.grid[1:]+[0])):
            clean = predictor(state, self.grid_times[index].expand(len(state))).float()
            if u == 0:
                state = clean
            else:
                epsilon = (state-self.root_alpha[t]*clean)/self.root_noise[t]
                state = self.root_alpha[u]*clean + self.root_noise[u]*epsilon
        return state
