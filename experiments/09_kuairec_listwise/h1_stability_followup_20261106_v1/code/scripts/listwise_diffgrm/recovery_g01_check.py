"""G01 pre-launch checks only: parent binding, start identity, gradient isolation, anchor persistence.

Disposable CPU/GPU inspection; never a formal run. Uses the same authenticated
build_g01 entry as the formal job so parent SHA256/counters/binding checks are
the real ones. Nothing here initializes a formal continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import scripts.listwise_diffgrm.recovery_g01 as recovery_g01
from .recovery import fingerprint
from .recovery_evaluation import write_json
from .recovery_g01 import build_g01
from .run import microbatches
from .training import atomic_save


def _snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.named_parameters()}


def _all_equal(saved, module):
    current = dict(module.named_parameters())
    if set(saved) != set(current):
        return False
    return all(torch.equal(saved[k], current[k].detach().cpu()) for k in saved)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    result = {'device': a.device}

    # 1) Authenticated parent + binding. build_g01 itself raises on any
    # SHA256/counters/config/code mismatch with the authorized R20 parent.
    controller, assets, data = build_g01(a.parent, a.device, a.dataset, a.assets,
                                         inspection=(a.device == 'cpu'))
    t = controller.trainer
    mixer = t.alignment.mixer
    result['parent_bound'] = dict(
        parent_sha256=controller.protocol['parent_sha256'],
        gamma=controller.protocol['gamma'], t_block=controller.protocol['t_block'],
        r_block=controller.protocol['r_block'], t_lr=controller.protocol['t_lr'],
        r_updates=t.r_updates, t_updates=t.t_updates,
        start_phase=controller.start_phase, stop_phase=controller.stop_phase)

    # 2) Start identity: gated effective codebook equals the saved R20 anchor
    # within float tolerance; gamma=1 reduces to the raw crossed codebook; a
    # deterministic SID refresh reproduces the parent's stored hard codes.
    with torch.no_grad():
        raw, _ = mixer._g01_original_crossed_codebooks(False)
        eff, _ = mixer.crossed_codebooks(False)
        c0 = mixer._g01_c0.to(eff)
        anchor_max_abs = float((eff - c0).abs().max())
        raw_vs_c0_max_abs = float((raw - c0).abs().max())
        recovery_g01.GAMMA = 1.0
        eff_one, _ = mixer.crossed_codebooks(False)
        recovery_g01.GAMMA = 0.1
        gamma1_max_abs = float((eff_one - raw).abs().max())
        before = t.alignment.hard_codes.clone()
        refresh = t.alignment.refresh_hard_codes()
        sid_identical = bool(torch.equal(t.alignment.hard_codes, before))
    result['start_identity'] = dict(
        effective_vs_anchor_max_abs=anchor_max_abs,
        raw_vs_anchor_max_abs=raw_vs_c0_max_abs,
        gamma1_vs_raw_max_abs=gamma1_max_abs,
        sid_refresh_identical_to_parent=sid_identical,
        sid_refresh_stats=refresh)
    if not (anchor_max_abs <= 1e-5 and raw_vs_c0_max_abs <= 1e-5
            and gamma1_max_abs <= 1e-5 and sid_identical):
        raise ValueError('Start identity failed: ' + json.dumps(result['start_identity']))

    # 3) Anchor persistence: checkpoint, restore into a fresh authenticated
    # controller, verify anchors and gated output; tampered/ungated rejects.
    state_path = Path(a.output) / 'check_state.pt'
    atomic_save(state_path, controller.checkpoint())
    c2, _, _ = build_g01(a.parent, a.device, a.dataset, a.assets,
                         inspection=(a.device == 'cpu'))
    c2.restore(torch.load(state_path, map_location='cpu', weights_only=False))
    m2 = c2.trainer.alignment.mixer
    anchor_preserved = all(fingerprint(getattr(m2, k)) == fingerprint(getattr(mixer, k))
                           for k in ('_g01_b0', '_g01_m0', '_g01_c0'))
    with torch.no_grad():
        e1, _ = mixer.crossed_codebooks(False)
        e2, _ = m2.crossed_codebooks(False)
    effective_equal = bool(torch.equal(e1, e2))
    tampered = torch.load(state_path, map_location='cpu', weights_only=False)
    tampered['g01']['anchor']['M0'] = tampered['g01']['anchor']['M0'] + 1
    try:
        c2.restore(tampered)
        tamper_rejected = False
    except ValueError:
        tamper_rejected = True
    ungated = torch.load(state_path, map_location='cpu', weights_only=False)
    ungated.pop('g01')
    try:
        c2.restore(ungated)
        ungated_rejected = False
    except ValueError:
        ungated_rejected = True
    result['save_restore'] = dict(
        anchor_preserved=anchor_preserved,
        gated_effective_after_restore_equal=effective_equal,
        tampered_anchor_rejected=tamper_rejected,
        ungated_checkpoint_rejected=ungated_rejected)
    if not (anchor_preserved and effective_equal and tamper_rejected and ungated_rejected):
        raise ValueError('Save/restore failed: ' + json.dumps(result['save_restore']))

    # 4) Gradient connectivity + R/T parameter isolation on one real small
    # update each. The disposable trainer drifts after this; do it last.
    sel = np.arange(64, dtype=np.int64)
    model0 = _snapshot(t.model)
    al0 = _snapshot(t.alignment)
    t.update(microbatches(data, sel, t.config['training']['microbatch_size']), 'T')
    t_grads = {n: float(p.grad.abs().sum().item())
               for n, p in t.alignment.named_parameters() if p.grad is not None}
    model_after_t_unchanged = _all_equal(model0, t.model)
    tokenizer_moved = not _all_equal(al0, t.alignment)
    al_before_r = _snapshot(t.alignment)
    model_before_r = _snapshot(t.model)
    t.update(microbatches(data, sel + 64, t.config['training']['microbatch_size']), 'R')
    alignment_after_r_unchanged = _all_equal(al_before_r, t.alignment)
    model_after_r_changed = not _all_equal(model_before_r, t.model)
    mixer_grad = {k: v for k, v in t_grads.items() if 'mixer.' in k and v > 0}
    result['gradient_isolation'] = dict(
        t_update_model_params_unchanged=model_after_t_unchanged,
        t_update_tokenizer_params_changed=tokenizer_moved,
        t_update_base_codebooks_grad=t_grads.get('mixer.base_codebooks', 0.0),
        t_update_mixer_nonzero_grad_blocks=len(mixer_grad),
        r_update_alignment_params_unchanged=alignment_after_r_unchanged,
        r_update_model_params_changed=model_after_r_changed)
    ok = (model_after_t_unchanged and tokenizer_moved
          and t_grads.get('mixer.base_codebooks', 0.0) > 0 and mixer_grad
          and alignment_after_r_unchanged and model_after_r_changed)
    if not ok:
        raise ValueError('Gradient isolation failed: ' + json.dumps(result['gradient_isolation']))

    result['status'] = 'passed'
    write_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
