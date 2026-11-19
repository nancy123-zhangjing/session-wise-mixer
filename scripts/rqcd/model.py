"""Batched teacher-forced continuous diffusion; shared history, no future labels.

PyTorch SDPA dispatches eligible BF16 CUDA attention to fused kernels. Five
target views are packed into one batch, but each view has its own causal mask.
No torch.compile, approximate precision, or changed loss is required.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def split(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def project_kv(self, memory):
        k, v = self.kv(memory).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def forward(self, x, memory=None, *, allowed=None, causal=False, kv=None):
        q = self.split(self.q(x))
        k, v = self.project_kv(x if memory is None else memory) if kv is None else kv
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(out.transpose(1, 2).flatten(2))


class Block(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__()
        w, p = cfg["width"], cfg["dropout"]
        self.attn = Attention(w, cfg["heads"], p)
        self.n1 = nn.LayerNorm(w, eps=cfg["norm_eps"])
        self.n2 = nn.LayerNorm(w, eps=cfg["norm_eps"])
        self.ffn = nn.Sequential(nn.Linear(w, cfg["ffn"]), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(cfg["ffn"], w))
        self.drop = nn.Dropout(p)
        self.cross = Attention(w, cfg["heads"], p) if cross else None
        self.nc = nn.LayerNorm(w, eps=cfg["norm_eps"]) if cross else None

    def forward(self, x, *, allowed=None, causal=False, memory=None, history_mask=None, kv=None):
        x = x + self.drop(self.attn(self.n1(x), allowed=allowed, causal=causal))
        if self.cross is not None:
            x = x + self.drop(self.cross(self.nc(x), memory, allowed=history_mask, kv=kv))
        return x + self.drop(self.ffn(self.n2(x)))


class ContinuousListModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c, w = cfg["model"], cfg["model"]["width"]
        self.length, self.history_max = cfg["list_length"], cfg["history_max"]
        self.width = w
        d = cfg["rq"]["latent_dim"]
        self.clean = nn.Linear(d, w)
        self.noisy = nn.Linear(d, w)
        self.history_pos = nn.Embedding(self.history_max, w)
        self.target_pos = nn.Embedding(self.length, w)
        self.types = nn.Embedding(3, w)
        self.time = nn.Sequential(nn.Linear(w, 4*w), nn.SiLU(), nn.Linear(4*w, w))
        self.encoder = nn.ModuleList([Block(c) for _ in range(c["encoder_layers"])])
        self.decoder = nn.ModuleList([Block(c, cross=True) for _ in range(c["decoder_layers"])])
        self.enc_norm = nn.LayerNorm(w, eps=c["norm_eps"])
        self.dec_norm = nn.LayerNorm(w, eps=c["norm_eps"])
        self.output = nn.Linear(w, d)
        self.register_buffer("frequency", torch.exp(-math.log(10000) * torch.arange(w//2) / (w//2)))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def time_embedding(self, times):
        phase = times.float().unsqueeze(-1) * self.frequency
        return self.time(torch.cat((phase.cos(), phase.sin()), -1))

    def encode_history(self, history, mask):
        positions = torch.arange(history.shape[1], device=history.device)
        x = self.clean(history) + self.history_pos(positions) + self.types.weight[0]
        x = x.masked_fill(~mask[..., None], 0)
        allowed = mask[:, None, None, :]
        for block in self.encoder:
            x = block(x, allowed=allowed)
        return self.enc_norm(x).masked_fill(~mask[..., None], 0)

    def cache_history(self, memory):
        """Only call once per generation; cross-attention K/V reused 250 times."""
        return [block.cross.project_kv(memory) for block in self.decoder]

    def predict(self, noisy, times, prefix, position, memory, history_mask, cache=None):
        """Target-free inference for one position. prefix contains only earlier items."""
        if prefix.shape[1] != position or not 0 <= position < self.length:
            raise ValueError("Prefix length must equal the zero-based target position")
        pos = torch.arange(position, device=noisy.device)
        clean = self.clean(prefix) + self.target_pos(pos) + self.types.weight[1]
        current = (self.noisy(noisy) + self.target_pos.weight[position] +
                   self.types.weight[2] + self.time_embedding(times))
        x = torch.cat((clean, current[:, None]), dim=1)
        for i, block in enumerate(self.decoder):
            x = block(x, causal=True, memory=memory, history_mask=history_mask[:, None, None, :],
                      kv=None if cache is None else cache[i])
        return self.output(self.dec_norm(x[:, -1]))

    def training_prediction(self, history, mask, clean_targets, noisy_targets, times):
        """All L views in one Decoder batch; future clean tokens are zero/masked.

        Each view is padded to L tokens on the right. Causal attention prevents
        its valid tokens from seeing that padding (or any future target).
        """
        b, length, _ = clean_targets.shape
        if length != self.length:
            raise ValueError("Incorrect list length")
        memory = self.encode_history(history, mask)  # B, not B*L encoder calls
        positions = torch.arange(length, device=history.device)
        token = self.clean(clean_targets) + self.target_pos(positions) + self.types.weight[1]
        before = positions[None, :] < positions[:, None]
        x = token[:, None].expand(-1, length, -1, -1) * before[None, :, :, None]
        eye = torch.eye(length, device=history.device, dtype=x.dtype)
        current = (self.noisy(noisy_targets) + self.target_pos(positions) +
                   self.types.weight[2] + self.time_embedding(times))
        x = (x + current[:, :, None] * eye[None, :, :, None]).reshape(b*length, length, self.width)
        history_mask = mask.repeat_interleave(length, 0)[:, None, None, :]
        for block in self.decoder:
            # Project history K/V at B once, then expand views. This preserves
            # gradients to the shared history and avoids 5 redundant projections.
            kv = tuple(t.repeat_interleave(length, 0) for t in block.cross.project_kv(memory))
            x = block(x, causal=True, history_mask=history_mask, kv=kv)
        x = x.reshape(b, length, length, self.width)
        return self.output(self.dec_norm(x[:, positions, positions]))


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["diffusion"]
        self.steps, self.sample_steps = c["steps"], c["sample_steps"]
        t = torch.arange(self.steps+1, dtype=torch.float64)
        f = torch.cos((t/self.steps+c["cosine_offset"])/(1+c["cosine_offset"])*math.pi/2).square()
        beta = (1 - f[1:]/f[:-1]).clamp(max=c["beta_max"])
        cumulative = torch.cat((torch.ones(1, dtype=torch.float64), (1-beta).cumprod(0)))
        self.register_buffer("alpha64", cumulative)
        self.register_buffer("alpha", cumulative.float())

    def noise(self, clean, times, epsilon):
        a = self.alpha[times][..., None]
        return a.sqrt()*clean + (1-a).sqrt()*epsilon

    def grid(self):
        if not 2 <= self.sample_steps <= self.steps:
            raise ValueError("Invalid DDIM step count")
        return [1 + math.floor((self.steps-1)*k/(self.sample_steps-1)+0.5)
                for k in reversed(range(self.sample_steps))]

    def sample(self, initial, predictor):
        """predictor(y_t, integer_time) returns x0. No hidden clipping or CFG."""
        y = initial
        grid = self.grid()
        for t, u in zip(grid, grid[1:]+[0]):
            clean = predictor(y, t).float()
            if u == 0:
                y = clean
            else:
                epsilon = (y - self.alpha[t].sqrt()*clean) / (1-self.alpha[t]).sqrt()
                y = self.alpha[u].sqrt()*clean + (1-self.alpha[u]).sqrt()*epsilon
        return y
