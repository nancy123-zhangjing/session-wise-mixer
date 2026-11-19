"""The eight round-1 groups. Each differs from B0 in exactly one respect."""
from __future__ import annotations

import copy

from scripts.rqcd_mixer.screen import validate

# name -> (screen switches, RQ geometry override or None, what it tests)
VARIANTS = {
    'B0': ({}, None, 'baseline; must reproduce the first-wave CF-M0 validation record'),
    'L1': ({'landing_weight': 1.0}, None, 'landing cross-entropy weight 0.1 -> 1'),
    'L2': ({'landing_weight': 1.0, 'denoise_weight': 0.1}, None, 'landing 1, denoising MSE 1 -> 0.1'),
    'E128': ({}, {'latent_dim': 128, 'hidden_dims': [512, 256]}, 'RQ Encoder output 32 -> 128'),
    'H1': ({'history_head': {'weight': 0.5, 'temperature': 0.1}}, None, 'history head, hard labels, weight 0.5'),
    'H2': ({'guidance': {'p_uncond': 0.1}}, None, 'classifier-free guidance, 10% null history'),
    'D1': ({'generation_space': 'teacher'}, None, 'generate in the 256-d teacher space, land via frozen RQ Encoder'),
    'E64': ({}, {'latent_dim': 64, 'hidden_dims': [512, 256, 128]}, 'RQ Encoder output 32 -> 64'),
}
JOBS = {'a': ['B0', 'L1', 'L2', 'E128'], 'b': ['H1', 'H2', 'D1', 'E64']}
# Single-slot reruns; a worker resumes or skips finished stages (2026-11-18: D1 smoke evaluation).
RERUNS = {'d1': ['D1']}
STOP_AFTER = 20
EVAL_EPOCHS = (10, 15, 20)
GUIDANCE = (1.0, 1.5, 2.0, 3.0)
SMOKE = {'stop_after': 1, 'eval_epochs': (1,), 'rq_epochs': 50, 'limit': 256, 'ks': (2,), 'guidance': (1.0, 2.0)}


def build(name, base, *, smoke=False):
    """First-wave CF config + this group's change. B0 adds only the screen name."""
    switches, rq, _ = VARIANTS[name]
    cfg = copy.deepcopy(base)
    if 'screen' in cfg or 'protocol_v2' in cfg:
        raise ValueError('The base must be the first-wave CF config')
    cfg['screen'] = {'name': name, **switches}
    if rq is not None:
        cfg['rq'] = {**cfg['rq'], **rq}
        if smoke:
            cfg['rq']['epochs'] = SMOKE['rq_epochs']
    return validate(cfg)


def own_parent(name):
    """Groups that change the RQ geometry need their own preparation and RQ pretraining."""
    return VARIANTS[name][1] is not None
