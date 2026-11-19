"""Session-wise decision rules for the CF line (inference only).

Every rule keeps the CF data flow: history -> encoder -> one DDIM vector per slot ->
RQ landing -> the landed item's SID is appended as the next slot's prefix. Only the
decision made inside a slot changes:

* A, original: one noise per slot, greedy layer-wise landing
  (scripts.rqcd_mixer.catalog.generate, unchanged).
* B, exact landing: the same single vector; the item with the highest landing
  probability. B is C with K = 1.
* C, per-slot mode: K vectors under the same prefix; the item with the highest mean
  landing probability.
* D, list choice: K complete lists from rule A; the one that agrees most with the
  others (minimum Bayes risk with an NDCG-style gain).
* Reference, NOT session-wise and only a diagnostic: the top five items of the slot-1
  mean landing distribution.

The landing probability is the distribution trained by
scripts.rqcd_mixer.losses.landing_loss: per level, a softmax over the legal children
of the item's parent, with logits -||residual - code||^2 / (temperature *
distance_scale[level]); then a softmax over the legal items of the leaf bucket, with
logits -||z - z_item||^2 / (bucket_temperature * 2 * dim * scale^2). A child is legal
when at least one non-blocked item lies below it. Blocked means the filtered history
and the items already chosen for earlier slots.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from scripts.rqcd_mixer.catalog import (append_to_caches, distinct_history_ids, generate, guided_predictor,
                                        start_caches)

DISCOUNT = 1.0 / np.log2(np.arange(5) + 2.0)


def squared_distances(x, y):
    """Float64 squared L2 distances [B, M]. Same expansion as tokenizer.distances."""
    x, y = x.double(), y.double()
    return (x.square().sum(-1, keepdim=True) + y.square().sum(-1) - 2 * (x @ y.T)).clamp_min(0)


def grouped_logsumexp(values, group, groups):
    """values [B, M] (may hold -inf), group [M] in [0, groups) -> [B, groups]; an empty group is -inf."""
    index = group[None].expand(values.shape[0], -1)
    peak = values.new_full((values.shape[0], groups), -math.inf).scatter_reduce(
        1, index, values, 'amax', include_self=True)
    safe = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    total = values.new_zeros((values.shape[0], groups)).scatter_add(1, index, (values - safe.gather(1, index)).exp())
    return total.log() + safe


class LandingScorer:
    """Full-catalog log p(item | z) under the trained landing distribution."""

    def __init__(self, catalog, cfg):
        if cfg['loss'].get('mode') != 'rq_landing':
            raise ValueError('Landing probabilities need a model trained with RQ landing supervision')
        c = self.catalog = catalog
        nodes, paths = len(c.node_count), c.paths[1:]
        parent = torch.zeros(nodes, dtype=torch.long, device=c.device)
        depth = torch.zeros(nodes, dtype=torch.long, device=c.device)
        centre = torch.zeros(nodes, c.dim, dtype=torch.float64, device=c.device)
        cumulative = torch.zeros(c.count, c.dim, dtype=torch.float64, device=c.device)
        for level in range(c.levels):
            child = paths[:, level + 1]
            cumulative = cumulative + c.books[level].double()[c.sid[:, level]]
            parent[child], depth[child], centre[child] = paths[:, level], level + 1, cumulative
        level_temperature = cfg['quantization']['temperature'] * c.stats['distance_scale'].double()
        # A node's logit is -||z - centre||^2 = -||residual at its parent - its code||^2.
        self.node_temperature = torch.cat((level_temperature.new_ones(1), level_temperature))[depth]
        self.parent, self.centre, self.nodes = parent, centre, nodes
        self.item_nodes, self.leaf = paths[:, 1:], paths[:, -1]
        self.z = c.z.double()
        self.bucket_temperature = cfg['loss']['bucket_temperature'] * (
            2 * c.dim * c.stats['scale'].double().square()).clamp_min(1e-8)

    def _blocked_mask(self, blocked):
        mask = torch.zeros(blocked.shape[0], self.catalog.count + 1, dtype=torch.bool, device=blocked.device)
        if blocked.shape[1]:
            mask.scatter_(1, blocked, True)
        return mask[:, 1:]

    def legal_nodes(self, blocked):
        """blocked [B, E] item IDs (0 = padding, duplicates allowed) -> [B, nodes] legality."""
        c = self.catalog
        removed = torch.zeros(blocked.shape[0], self.nodes, dtype=torch.long, device=c.device)
        if blocked.shape[1]:
            ids = distinct_history_ids(blocked)
            on_path = c.paths[ids]                                   # padding maps to the root only
            weight = (ids > 0).long()[:, :, None].expand_as(on_path)
            removed.scatter_add_(1, on_path.flatten(1), weight.flatten(1))
        return (c.node_count[None] - removed) > 0

    def log_prob(self, z, blocked):
        """z [B, dim] in catalog space, blocked [B, E] -> float64 log p(item) [B, N]; -inf when illegal."""
        z, batch = z.double(), len(z)
        legal = self.legal_nodes(blocked)
        legal[:, 0] = False                                          # the root is never a child
        logits = (-squared_distances(z, self.centre) / self.node_temperature).masked_fill(~legal, -math.inf)
        norm = grouped_logsumexp(logits, self.parent, self.nodes)
        step = torch.where(legal, logits - norm.gather(1, self.parent[None].expand(batch, -1)),
                           torch.full_like(logits, -math.inf))
        path = sum(step[:, self.item_nodes[:, level]] for level in range(self.item_nodes.shape[1]))
        allowed = ~self._blocked_mask(blocked)
        bucket = (-squared_distances(z, self.z) / self.bucket_temperature).masked_fill(~allowed, -math.inf)
        bucket_norm = grouped_logsumexp(bucket, self.leaf, self.nodes)
        bucket = torch.where(allowed, bucket - bucket_norm.gather(1, self.leaf[None].expand(batch, -1)),
                             torch.full_like(bucket, -math.inf))
        return path + bucket


def landing_consistency(scorer, z, target, cfg):
    """Mean -log p(true first item | z) from the scorer and from losses.landing_loss (no exclusions)."""
    from scripts.rqcd_mixer.losses import landing_loss
    c = scorer.catalog
    log_p = scorer.log_prob(z, target.new_zeros(len(z), 0))
    ours = -log_p.gather(1, (target - 1)[:, None]).squeeze(1)
    reference = landing_loss(z[:, None].float(), target[:, None], c.books, c, cfg)['landing']
    return float(ours.mean()), float(reference)


@torch.no_grad()
def generate_mode(model, diffusion, catalog, scorer, history_ids, noise, runtime, *,
                  filter_history=True, reference=False, guidance=1.0):
    """Rules B/C. noise [B, K, list, dim]; returns items [B, list] and, if asked, the reference top five."""
    model.eval()
    batch, k = noise.shape[:2]
    history = history_ids.repeat_interleave(k, 0)
    caches = start_caches(model, catalog, history, runtime, guidance)
    predictor = guided_predictor(model, caches, runtime, catalog.device, guidance)
    selected = torch.empty(batch, 0, device=catalog.device, dtype=torch.long)
    excluded = distinct_history_ids(history_ids) if filter_history else selected
    top_five = None
    for j in range(model.length):
        x = diffusion.sample(noise[:, :, j].reshape(batch * k, -1).float(), predictor)
        z = model.output_to_latent(x, catalog)
        blocked = torch.cat((excluded, selected), 1)
        log_p = scorer.log_prob(z, blocked.repeat_interleave(k, 0)).view(batch, k, -1)
        mixture = torch.logsumexp(log_p, 1) - math.log(k)
        item = mixture.argmax(-1) + 1
        if not torch.isfinite(mixture.gather(1, (item - 1)[:, None])).all():
            raise RuntimeError('No legal item left for a slot')
        if j == 0 and reference:
            top_five = mixture.topk(5, -1).indices + 1
        selected = torch.cat((selected, item[:, None]), 1)
        if j + 1 < model.length:
            append_to_caches(model, catalog, item.repeat_interleave(k), caches, runtime)
    return selected, top_five


@torch.no_grad()
def generate_lists(model, diffusion, catalog, history_ids, noise, runtime, *, filter_history=True, guidance=1.0):
    """Rule A on several noise sets at once: noise [B, M, list, dim] -> items [B, M, list]."""
    batch, m = noise.shape[:2]
    items, _ = generate(model, diffusion, catalog, history_ids.repeat_interleave(m, 0),
                        noise.reshape(batch * m, *noise.shape[2:]), runtime, filter_history=filter_history,
                        guidance=guidance)
    return items.view(batch, m, -1)


@torch.no_grad()
def first_slot_vectors(model, diffusion, catalog, history_ids, noise, runtime, guidance=1.0):
    """The slot-1 landing vectors of rule A (empty prefix): noise [B, dim] -> z [B, latent]."""
    model.eval()
    caches = start_caches(model, catalog, history_ids, runtime, guidance)
    x = diffusion.sample(noise.float(), guided_predictor(model, caches, runtime, catalog.device, guidance))
    return model.output_to_latent(x, catalog)


@torch.no_grad()
def history_head_top_five(model, catalog, history_ids, item_vectors, temperature, runtime, *, filter_history=True):
    """Encoder-only diagnostic of the history head: top five items, history filtered like generation."""
    from scripts.rqcd_mixer.common import autocast
    model.eval()
    with autocast(runtime, catalog.device):
        memory, valid = model.encode_history(model.lookup(catalog.sid_pad[history_ids]), history_ids != 0)
    logits = model.history_logits(memory, valid, item_vectors, temperature)
    if filter_history:
        blocked = torch.zeros(len(history_ids), catalog.count + 1, dtype=torch.bool, device=logits.device)
        blocked.scatter_(1, history_ids, True)
        logits = logits.masked_fill(blocked[:, 1:], float('-inf'))
    return logits.topk(5, -1).indices + 1


def mbr_select(lists):
    """lists [B, M, 5] -> index [B] of the list with the largest summed gain against the other lists.

    gain(i, k) = sum over positions p of DISCOUNT[p] * [lists[i, p] in lists[k]]. Ties keep the
    lowest index, so list 0 (the original rule) wins a tie.
    """
    lists = np.asarray(lists)
    member = (lists[:, :, None, :, None] == lists[:, None, :, None, :]).any(-1)
    gain = (member * DISCOUNT).sum(-1)
    m = lists.shape[1]
    gain[:, np.arange(m), np.arange(m)] = 0
    return gain.sum(-1).argmax(-1)
