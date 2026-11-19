"""Full-catalog item metrics with deterministic score-desc/item-id-asc ties."""
import torch


def stable_top_items(scores, k=20):
    """Exact float32 score-desc/ID-asc Top-K without sorting the full catalog.

    IEEE-754 order keys fit in int64 for this catalog. Signed zeros are made
    equal, matching target_ranks' floating-point equality policy.
    """
    scores = scores.float()
    count = scores.shape[1]
    k = min(k, count)
    if k < 1 or count > 1_000_000_000:
        raise ValueError("Invalid Top-K catalog")
    normalized = torch.where(scores == 0, torch.zeros_like(scores), scores).contiguous()
    bits = normalized.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
    tie = torch.arange(count, 0, -1, device=scores.device, dtype=torch.int64)
    keys = ordered * (count + 1) + tie
    return keys.topk(k, dim=1, sorted=True).indices + 1


def target_ranks(scores, targets, item_offset=1):
    target_scores = scores.gather(1, (targets - item_offset)[:, None])
    items = torch.arange(item_offset, item_offset + scores.shape[1], device=scores.device)
    ahead = (scores > target_scores) | ((scores == target_scores) & (items[None] < targets[:, None]))
    return 1 + ahead.sum(1)


def metrics_from_ranks(ranks, ks=(5, 10, 20)):
    ranks = torch.as_tensor(ranks, dtype=torch.float64)
    if ranks.numel() == 0 or torch.any(ranks < 1):
        raise ValueError("Nonempty positive ranks required")
    result = {"samples": ranks.numel()}
    for k in ks:
        hits = ranks <= k
        result[f"HR@{k}"] = hits.double().mean().item()
        result[f"NDCG@{k}"] = (hits / torch.log2(ranks + 1)).mean().item()
    return result
