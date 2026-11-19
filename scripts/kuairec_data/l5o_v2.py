"""L5O protocol v2 (2026-11-18): per-group, per-seed configurations.

Budgets follow the old-protocol schedule in DATA PASSES (not in optimizer
updates): one R unit is one full pass over the L5O training split. See
experiments/09_kuairec_listwise/l5o_v2_20261118/PROTOCOL.md for the rationale
and the pre-registered analysis. Building configs never starts training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .l5o_launch import wave_config
from .selection import RULE

PROTOCOL = 'kuairec_l5o_protocol_v2_20261118'
SEEDS = (2024, 2025, 2026)   # user decision 2026-11-18: same seeds as the previous round
ROOT = Path(__file__).resolve().parents[2]
TRAIN_SLATES = 2_288_120
LIST_LENGTH = 5
# Old global-time protocol (where the Mixer evidence comes from).
OLD_TRAIN_SLATES = 709_443
OLD_TIGER_PASSES = 200_000 * 256 / (OLD_TRAIN_SLATES * LIST_LENGTH)   # 14.434 passes
TIGER_REFERENCE_BATCH, TIGER_REFERENCE_LR, TIGER_REFERENCE_WARMUP = 256, 0.01, 10_000
H_FREEZE_AFTER_PASS = 20      # accepted three-seed H freeze schedule (R20 of 35)
CF_FREEZE_AFTER_PASS = 9      # accepted CF freeze schedule (R9 of 35)
ARMS = {'h': ('H0', 'H1'), 'tiger': ('T0A', 'T1A'), 'cf': ('CF-M0', 'CF-M1'), 'sasrec': ('SASREC',)}


def _common(cfg, seed):
    if seed not in SEEDS:
        raise ValueError(f'Protocol v2 seeds are {SEEDS}')
    cfg['seed'] = seed
    cfg['protocol_v2'] = PROTOCOL
    cfg['evaluation']['filter_history'] = True        # fixed for every method (v2 rule)
    cfg['evaluation']['selection_rule'] = RULE
    return cfg


H_RHYTHMS = ('pass', 'old693')


def h_config(receipt, seed, dataset, assets, rhythm='pass'):
    """rhythm 'pass' is protocol v2. 'old693' (2026-11-19 control, user-approved) keeps the old
    split's 693-update R unit on L5O data: same 35 units, T schedule and R20 freeze, but 3.2x fewer
    recommender updates per codebook block and in total (24,255 instead of 78,225)."""
    if rhythm not in H_RHYTHMS:
        raise ValueError(f'Unknown H rhythm {rhythm}')
    base = json.loads((ROOT/'configs/09_kuairec_listwise/l5o.json').read_text(encoding='utf-8'))
    cfg = _common(wave_config(base, receipt, 'h'), seed)
    cfg.update(dataset=str(dataset), assets=str(assets))
    t = cfg['training']
    if rhythm == 'pass':
        # One R unit = one full pass (ceil(2,288,120/1024) = 2,235 updates); the old
        # 693-update unit was the OLD split's pass and is no longer meaningful.
        t.pop('r_updates_per_unit', None)
    else:
        if t.get('r_updates_per_unit') != 693:
            raise ValueError('old693 expects the 693-update R unit of the base config')
        cfg['h_rhythm_control'] = 'old693'
    t['freeze_after_epoch'] = H_FREEZE_AFTER_PASS
    t['validate_every_r_unit'] = True
    t['checkpoint_every_updates'] = 500
    cfg['execution']['keep_validated_checkpoints'] = True
    return cfg


H_SPLITS = ('l5o', 'old')
STAGE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
OLD_DATASET = f'{STAGE}/goal_20261027_r1/inputs/dataset'
OLD_ASSETS = f'{STAGE}/goal_20261027_r1/assets/pse_v1'


def h_old_reference(seed):
    """The exact configuration of the accepted old-split H freeze runs (base.json, old data paths,
    micro-batch 256, evaluation batch 32); equals multiseed_freeze_20261109_v1/config_seed{seed}.json."""
    cfg = json.loads((ROOT/'configs/09_kuairec_listwise/base.json').read_text(encoding='utf-8'))
    cfg.update(seed=seed, dataset=OLD_DATASET, assets=OLD_ASSETS)
    cfg['training']['microbatch_size'] = 256
    cfg['evaluation']['batch_size'] = 32
    return cfg


def h_oldsplit_config(seed):
    """2026-11-19 control (user-approved step 1): current code with the protocol v2 execution template
    and in-run R20 freeze, on the OLD split with the OLD evaluation (no history masking, exposure
    NDCG@5 selection, validation every 3 R units, one R unit = one old pass of 693 updates).
    Asks whether the current code reproduces the old Mixer gain."""
    if seed not in SEEDS:
        raise ValueError(f'Protocol v2 seeds are {SEEDS}')
    cfg = h_old_reference(seed)
    l5o = json.loads((ROOT/'configs/09_kuairec_listwise/l5o.json').read_text(encoding='utf-8'))
    cfg['execution'] = copy.deepcopy(l5o['execution'])          # same template as the v2 H runs
    cfg['execution']['keep_validated_checkpoints'] = True
    cfg['training']['microbatch_size'] = l5o['training']['microbatch_size']
    cfg['evaluation']['batch_size'] = l5o['evaluation']['batch_size']
    t = cfg['training']
    t['freeze_after_epoch'] = H_FREEZE_AFTER_PASS
    t['checkpoint_every_updates'] = 500
    cfg['h_split_control'] = 'oldsplit_v2code_20261119'
    return cfg


def tiger_config(arm, receipt, seed, *, batch=256):
    from scripts.session_wise_ar.config import load
    cfg = _common(wave_config(load(f'AR-{arm}'), receipt, 'tiger'), seed)
    tasks = TRAIN_SLATES * LIST_LENGTH
    per_pass = math.ceil(tasks / batch)
    scale = batch / TIGER_REFERENCE_BATCH
    t = cfg['training']
    t['batch_size'] = batch
    # Same samples as the old run (14.43 passes), same warmup measured in samples,
    # square-root LR scaling for the adaptive optimizer at the larger batch.
    t['max_steps'] = round(OLD_TIGER_PASSES * tasks / batch)
    t['inverse_sqrt_warmup'] = round(TIGER_REFERENCE_WARMUP / scale)
    t['r_lr'] = TIGER_REFERENCE_LR * math.sqrt(scale)
    t['validation_interval_updates'] = math.ceil(per_pass / 2)
    cfg['execution'].update(precision='tf32', checkpoint_every_updates=1000)
    # H100 gate 2026-11-18 (smoke_tiger_fde70d5): fused SDPA with T5's dense learned
    # bias was SLOWER (1,403 vs 1,703 samples/s at b256; 1,595/s and 73 GiB at b1024,
    # OOM in the full pipeline). Keep eager attention and the original batch 256.
    cfg['model'].update(attention='eager')
    cfg['evaluation']['batch_size'] = 128            # identical lists to 32 (H100 probe), 20% faster
    return cfg


def cf_config(receipt, seed):
    from scripts.rqcd_mixer.config import load_config
    cfg = _common(wave_config(load_config(ROOT/'configs/rqcd_mixer/v2_cf.json'), receipt, 'cf'), seed)
    t = cfg['training']
    t.pop('r_updates_per_unit', None)                # one R unit = one full pass
    t['freeze_after_epoch'] = CF_FREEZE_AFTER_PASS
    cfg['evaluation']['epochs'] = list(range(1, t['epochs'] + 1))
    return cfg


def sasrec_config(seed):
    if seed not in SEEDS:
        raise ValueError(f'Protocol v2 seeds are {SEEDS}')
    cfg = json.loads((ROOT/'configs/rqcd_mixer/l5o_sasrec.json').read_text(encoding='utf-8'))
    cfg['seed'] = seed
    cfg['protocol_v2'] = PROTOCOL
    cfg['evaluation'] = {'filter_history': True, 'selection_rule': RULE}
    return cfg


def smoke(family, cfg):
    """Tiny budgets for the GPU smoke test only: same code paths (warmup sharing,
    in-run freeze, smoothed selection, test), never used for results."""
    cfg['smoke_test'] = True
    t = cfg['training']
    if family == 'h':
        t.update(r_updates_per_unit=60, warmup_epochs=1, cycles=2, r_epochs_per_cycle=1,
                 t_updates_per_cycle=10, freeze_after_epoch=2, decoder_warmup_updates=20,
                 checkpoint_every_updates=50)
        cfg['evaluation'].update(val_beam=4, test_beam=4, candidate_lists=4)
    elif family == 'tiger':
        cfg['rq'].update(epochs=20, eval_interval=1)
        t.update(max_steps=600, validation_interval_updates=200, inverse_sqrt_warmup=100)
        cfg['execution']['checkpoint_every_updates'] = 100
    elif family == 'cf':
        cfg['rq'].update(epochs=5, eval_interval=1)   # a pretraining evaluation must occur to select the RQ
        t.update(epochs=3, preheat_epochs=1, freeze_after_epoch=2, r_updates_per_unit=60, warmup_updates=20)
        cfg['evaluation']['epochs'] = [1, 2, 3]
    elif family == 'sasrec':
        cfg['teacher'].update(epochs=2, patience=2)
    return cfg


def build(family, arm, seed, *, receipt=None, dataset=None, assets=None, tiger_batch=256, smoke_test=False,
          h_rhythm='pass', h_split='l5o'):
    if (h_rhythm != 'pass' or h_split != 'l5o') and family != 'h':
        raise ValueError('h_rhythm / h_split apply to family h only')
    if h_split not in H_SPLITS:
        raise ValueError(f'Unknown H split {h_split}')
    if h_split == 'old' and h_rhythm != 'pass':
        raise ValueError('The old-split control uses the old pass (693 updates) by construction')
    if family == 'h' and h_split == 'old':
        cfg = h_oldsplit_config(seed)
    elif family == 'h':
        cfg = h_config(receipt, seed, dataset, assets, rhythm=h_rhythm)
    elif family == 'tiger':
        cfg = tiger_config(arm, receipt, seed, batch=tiger_batch)
    elif family == 'cf':
        cfg = cf_config(receipt, seed)
    elif family == 'sasrec':
        cfg = sasrec_config(seed)
    else:
        raise ValueError(f'Unknown family {family}')
    return copy.deepcopy(smoke(family, cfg) if smoke_test else cfg)


def budgets(family, cfg):
    """Human-readable budget summary (no torch)."""
    if family == 'h':
        t = cfg['training']
        per = t.get('r_updates_per_unit') or math.ceil(TRAIN_SLATES / t['effective_batch_size'])
        r_units = t['warmup_epochs'] + t['cycles'] * t['r_epochs_per_cycle']
        frozen_blocks = sum(1 for c in range(1, t['cycles'] + 1)
                            if t['warmup_epochs'] + 3 * (c - 1) >= t['freeze_after_epoch'])
        return dict(updates_per_pass=per, passes=r_units, r_updates=per * r_units,
                    t_updates=(t['cycles'] - frozen_blocks) * t['t_updates_per_cycle'],
                    freeze_after_pass=t['freeze_after_epoch'], validations=r_units)
    if family == 'tiger':
        t = cfg['training']
        tasks = TRAIN_SLATES * LIST_LENGTH
        return dict(batch=t['batch_size'], steps=t['max_steps'], passes=t['max_steps'] * t['batch_size'] / tasks,
                    lr=t['r_lr'], warmup=t['inverse_sqrt_warmup'], validation_every=t['validation_interval_updates'],
                    validations=math.ceil(t['max_steps'] / t['validation_interval_updates']))
    if family == 'cf':
        t = cfg['training']
        per = math.ceil(TRAIN_SLATES / t['effective_batch'])
        return dict(updates_per_pass=per, passes=t['epochs'], r_updates=per * t['epochs'],
                    t_updates=(t['freeze_after_epoch'] - t['preheat_epochs']) * t['t_steps_per_cycle'],
                    freeze_after_pass=t['freeze_after_epoch'], validations=len(cfg['evaluation']['epochs']))
    return dict(max_epochs=cfg['teacher']['epochs'], patience=cfg['teacher']['patience'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True, help='first-wave SASRec calibration COMPLETE.json')
    p.add_argument('--dataset', type=Path, default=Path('DATASET'))
    p.add_argument('--assets', type=Path, default=Path('ASSETS'))
    p.add_argument('--output', type=Path, help='write one config JSON per family/arm/seed here')
    p.add_argument('--tiger-batch', type=int, default=256)
    a = p.parse_args()
    summary = {}
    for family, arms in ARMS.items():
        for arm in arms:
            for seed in SEEDS:
                cfg = build(family, arm, seed, receipt=a.receipt, dataset=a.dataset, assets=a.assets,
                            tiger_batch=a.tiger_batch)
                summary[f'{family}/{arm}/{seed}'] = budgets(family, cfg)
                if a.output:
                    a.output.mkdir(parents=True, exist_ok=True)
                    (a.output/f'{family}_{arm}_{seed}.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
