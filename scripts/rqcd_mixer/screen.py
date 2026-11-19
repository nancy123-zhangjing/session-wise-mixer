"""CF round-1 screening switches (2026-11-18, conversation turns 22-27).

All screening changes live under one top-level config key, ``screen``. A config
without it (or with only ``name``) runs the unchanged CF code path. The switches
are training-time only: they never change data preparation or RQ pretraining, so
prepared parents and RQ assets are compared on ``asset_view(cfg)``, the config
without ``screen``. Changing the RQ geometry (E groups) is NOT a screen switch;
it changes ``cfg['rq']`` and needs its own prepared parent and RQ pretraining.
"""
from __future__ import annotations

import copy
import hashlib
import json

KEY = 'screen'
SWITCHES = {'name', 'landing_weight', 'denoise_weight', 'history_head', 'guidance', 'generation_space'}

# Prepared parents whose recorded source binding may be reused although the CF
# code changed afterwards: the first-wave L5O CF parent (2026-11-17,
# l5o_wave_20261117_v1/parents/cf). Every later change to scripts/rqcd_mixer up
# to the screening code touches joint training, evaluation or loss code only,
# never prepare() or RQ pretraining; the screening B0 run re-checks this by having
# to reproduce the first-wave validation record exactly. Values are SHA-256 of the
# canonical JSON of PREPARED.json["source"].
ACCEPTED_PARENT_SOURCES = {
    'cf32c0047182a18ac543f634b9c80f69361771a30ba18d1648b6e374ce006d97':
        'first-wave L5O CF parent, l5o_wave_20261117_v1/parents/cf (prepared 2026-11-17)',
}


def switches(cfg):
    return cfg.get(KEY, {})


def asset_view(cfg):
    """The config as seen by prepare() and RQ pretraining."""
    return {k: v for k, v in cfg.items() if k != KEY}


def source_digest(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def generation_space(cfg):
    return switches(cfg).get('generation_space', 'latent')


def generation_dim(cfg):
    """Dimension the denoiser generates in: the RQ latent, or the 256-d RQ input space (D1)."""
    return cfg['rq']['input_dim'] if generation_space(cfg) == 'teacher' else cfg['rq']['latent_dim']


def validate(cfg):
    s = switches(cfg)
    if not isinstance(s, dict) or set(s) - SWITCHES:
        raise ValueError(f'Unknown screen switches: {sorted(set(s) - SWITCHES) if isinstance(s, dict) else s}')
    for key in ('landing_weight', 'denoise_weight'):
        if key in s and not (isinstance(s[key], (int, float)) and s[key] > 0):
            raise ValueError(f'{key} must be a positive number')
    if 'history_head' in s:
        h = s['history_head']
        if set(h) != {'weight', 'temperature'} or not h['weight'] > 0 or not h['temperature'] > 0:
            raise ValueError('history_head needs positive weight and temperature only')
    if 'guidance' in s:
        g = s['guidance']
        if set(g) != {'p_uncond'} or not 0 < g['p_uncond'] < 1:
            raise ValueError('guidance needs 0 < p_uncond < 1 only')
    if generation_space(cfg) not in ('latent', 'teacher'):
        raise ValueError('generation_space is latent or teacher')
    if generation_space(cfg) == 'teacher' and cfg.get('target', {}).get('mode') != 'cf':
        raise ValueError('Teacher-space generation needs the CF (teacher-vector) target')
    if s.get('history_head') and cfg.get('target', {}).get('mode') != 'cf':
        raise ValueError('The history head scores against the CF target vectors')
    return cfg


def with_screen(base, **switch):
    cfg = copy.deepcopy(base)
    cfg[KEY] = dict(switch)
    return validate(cfg)
