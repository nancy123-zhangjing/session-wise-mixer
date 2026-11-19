"""Legacy stopped-target KL and opt-in v2 legal RQ path/bucket supervision."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .tokenizer import distances


def landing_loss(predicted_z, target_ids, books, catalog, cfg):
    """True-prefix conditional RQ path NLL + collision-bucket item NLL.

    Trie/labels/z are the frozen block snapshot. Live books remain differentiable.
    Exclude earlier distinct items other than the current observed target.
    L5O can observe the same item twice. Keep both training positions supervised.
    """
    with torch.autocast(predicted_z.device.type, enabled=False):
        if target_ids.ndim != 2 or predicted_z.shape[:2] != target_ids.shape:
            raise ValueError('Landing targets must be [batch,list] aligned with predictions')
        b, length = target_ids.shape
        pred = predicted_z.float().reshape(-1, books.shape[-1])
        ids = target_ids.reshape(-1)
        true_sid = catalog.sid_pad[ids]
        paths = catalog.paths[ids]
        positions = torch.arange(length, device=ids.device)
        earlier = positions[None, :] < positions[:, None]
        previous = target_ids[:, None, :].expand(b, length, length).reshape(-1, length)
        previous_paths = catalog.paths[previous]
        repeated = ((target_ids[:, :, None] == target_ids[:, None, :]) & earlier).any(-1)
        excluded = (earlier[None] & ~repeated[:, None, :]).reshape(-1, length)
        excluded = excluded & (previous != ids[:, None])
        residual, terms = pred, []
        for level, book in enumerate(books):
            child = catalog.children[paths[:, level]]
            available = catalog.node_count[child.clamp_min(0)].masked_fill(child < 0, 0)
            removed = ((child[:, :, None] == previous_paths[:, None, :, level+1]) & excluded[:, None]).sum(-1)
            legal = available-removed > 0
            logits = -distances(residual, book)/(cfg['quantization']['temperature']*catalog.stats['distance_scale'][level])
            terms.append(F.cross_entropy(logits.masked_fill(~legal, -torch.inf), true_sid[:, level], reduction='none'))
            residual = residual-book[true_sid[:, level]]
        path_loss = torch.stack(terms, -1).sum(-1).mean()
        candidates = catalog.bucket_items[catalog.node_bucket[paths[:, -1]]]
        legal = (candidates > 0) & ~((candidates[:, :, None] == previous[:, None, :]) & excluded[:, None]).any(-1)
        # A GEMM [batch*5,catalog] is smaller than [batch*5,max_bucket,latent].
        d = distances(pred, catalog.z).gather(1, (candidates-1).clamp_min(0))
        scale = (2*catalog.dim*catalog.stats['scale'].square()).clamp_min(1e-8)
        logits = (-d/(cfg['loss']['bucket_temperature']*scale)).masked_fill(~legal, -torch.inf)
        label = (candidates == ids[:, None]).long().argmax(-1)
        bucket_loss = F.cross_entropy(logits, label)
        return {'landing': path_loss+bucket_loss, 'path': path_loss, 'bucket': bucket_loss}


def alignment_loss(predicted_z, target_z, sid, books, distance_scale, temperature):
    with torch.autocast(predicted_z.device.type, enabled=False):
        pred, target = predicted_z.float().reshape(-1, books.shape[-1]), target_z.float().reshape(-1, books.shape[-1])
        sid = sid.reshape(-1, books.shape[0])
        prefix = torch.zeros_like(target)
        terms = []
        for level, book in enumerate(books):
            denominator = temperature*distance_scale[level]
            # Entire true branch is stopped, including its current codebook.
            with torch.no_grad():
                log_p = F.log_softmax(-distances(target-prefix, book)/denominator, -1)
            log_q = F.log_softmax(-distances(pred-prefix, book)/denominator, -1)
            terms.append((log_p.exp()*(log_p-log_q)).sum(-1))
            # Prediction branch keeps derivatives through prefix centers.
            prefix = prefix+book[sid[:, level]]
        return torch.stack(terms, -1).mean()


def recommendation_losses(prediction, clean_x, target_z, sid, books, stats, cfg, rq_result=None,
                          *, catalog=None, target_ids=None, to_latent=None, extra=None):
    """Without screen switches, to_latent and extra this is the original CF loss, operation for operation.

    to_latent maps standardized predictions to the RQ latent (teacher-space generation);
    extra is {name: (weight, value)} added to the total and reported (history head).
    """
    screen = cfg.get('screen', {})
    with torch.autocast(prediction.device.type, enabled=False):
        if to_latent is None:
            predicted_z = prediction.float()*stats["scale"] + stats["mean"]
        else:
            predicted_z = to_latent(prediction.float())
        denoise = F.mse_loss(prediction.float(), clean_x.float())
        use_landing = cfg['loss'].get('mode') == 'rq_landing'
        if use_landing:
            if catalog is None or target_ids is None:
                raise ValueError('RQ landing supervision requires the bound catalog and true item IDs')
            terms = landing_loss(predicted_z, target_ids, books, catalog, cfg)
        else:
            align = alignment_loss(predicted_z, target_z, sid, books, stats["distance_scale"],
                                   cfg["quantization"]["temperature"])
            terms = {'align': align}
        latent = denoise.new_zeros(())
        vq = denoise.new_zeros(())
        if rq_result is not None:
            latent = F.mse_loss(rq_result["q"], target_z.reshape_as(rq_result["q"]).detach())
            vq = rq_result["vq"].mean()
        name = 'landing' if use_landing else 'align'
        weight = screen.get(f'{name}_weight', cfg['loss'][name])
        if 'denoise_weight' in screen:
            total = screen['denoise_weight']*denoise + weight*terms[name]
        else:
            total = denoise + weight*terms[name]
        if rq_result is not None:
            total = total + cfg["loss"]["latent"]*latent + cfg["loss"]["vq"]*vq
        reported = {}
        for key, (scale, value) in (extra or {}).items():
            total = total + scale*value
            reported[key] = value
        return {"total": total, "denoise": denoise, **terms, "latent": latent, "vq": vq, **reported}
