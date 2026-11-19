"""Frozen teacher identity, target features and pretraining-only CF alignment."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .common import read_state, sha256


def teacher_spec(cfg):
    return {k: cfg[k] for k in ('seed', 'catalog_size', 'history_max', 'list_length', 'teacher')} | {
        'betas': cfg['training']['betas'], 'eps': cfg['training']['eps'],
        'max_grad_norm': cfg['training']['max_grad_norm']}


def teacher_sources():
    root = Path(__file__).parent
    return {name: sha256(root/name) for name in ('teacher.py', 'targets.py', 'data.py', 'common.py',
                                                '../rqcd/data.py', '../listwise_diffgrm/assets.py')}


def load_teacher(path, cfg, data_binding, original_ids):
    calibrated = cfg.get('calibrated_teacher')
    expected_source = teacher_sources() if calibrated is None else calibrated['source']
    if calibrated is not None and sha256(path) != calibrated['items_sha256']:
        raise ValueError('Teacher file differs from the calibrated item asset')
    state = read_state(path)
    if (state.get('format') != 'rqcd_teacher_items_v1' or state.get('spec') != teacher_spec(cfg)
            or state.get('data_binding') != data_binding or state.get('source') != expected_source or
            (calibrated is not None and state.get('calibration_source') != calibrated['calibration_source'])):
        raise ValueError('Teacher configuration/data/source identity mismatch')
    expected = torch.as_tensor(np.asarray(original_ids).copy(), dtype=torch.long)
    embedding, supported = state['embedding'], state['train_supported']
    if (not torch.equal(state['original_ids'], expected) or
            embedding.shape != (cfg['catalog_size'], cfg['teacher']['width']) or
            not torch.isfinite(embedding).all() or supported.shape != (cfg['catalog_size'],) or
            supported.dtype != torch.bool or not supported.any()):
        raise ValueError('Teacher item mapping, vectors or training support invalid')
    return state


def target_inputs(content, teacher, cfg):
    """All rows retain original catalog order, including unsupported cold items."""
    cf = teacher['embedding'].to(content.device).float().detach()
    scale = cf.double().square().sum(-1).mean().sqrt().float()
    if not torch.isfinite(scale) or scale <= 1e-8:
        raise ValueError('Degenerate collaborative vectors')
    mode = cfg['target']['mode']
    features = cf/scale if mode == 'cf' else content.detach().float()
    # Done once, outside the training loop; exact equal content is not a negative.
    _, groups = np.unique(content.detach().cpu().numpy(), axis=0, return_inverse=True)
    return {'features': features, 'cf': cf, 'supported': teacher['train_supported'].to(content.device),
            'groups': torch.as_tensor(groups, dtype=torch.long, device=content.device),
            'cf_global_scale': scale, 'mode': mode}


def collaborative_loss(q, cf, projection, groups, supported, temperature):
    """Mean over the original item batch; unsupported rows contribute zero."""
    with torch.autocast(q.device.type, enabled=False):
        if temperature <= 0:
            raise ValueError('CF temperature must be positive')
        # One fixed-size NxN matrix; no host-side item loop or variable gather.
        u = F.normalize(projection(q.float()), dim=-1)
        v = F.normalize(cf.detach().float(), dim=-1)
        logits = (u @ v.T)/temperature
        diagonal = torch.eye(len(q), dtype=torch.bool, device=q.device)
        legal = ((groups[:, None] != groups[None, :]) | diagonal) & supported[None, :]
        # An ignored cold query has a singleton diagonal to avoid all -inf/NaN.
        legal = torch.where(supported[:, None], legal, diagonal)
        loss = F.cross_entropy(logits.masked_fill(~legal, -torch.inf),
                               torch.arange(len(q), device=q.device), reduction='none')
        return (loss * supported).mean()
