"""H1 R14 continuation with a fixed gamma=0.1 Mixer increment gate.

The original Cross-PSE module remains unchanged.  The gate is attached to the
one runtime method used by training, catalog refresh, and evaluation, so the
same effective codebook is used everywhere without changing the checkpoint
state-dict schema of the original model.
"""
from __future__ import annotations

import copy
import math
import types
from pathlib import Path

import torch

from .recovery import RecoveryController, fingerprint
PARENT_SHA256 = '3cd65569e89cc5f8f1d3215cdc403e3ae0d5e6954d7e8818a891c12ed737f51f'
from .run import build_context, phase_indices
from .training import cpu_copy, expected_update_counts, sha256


GAMMA = 0.1
START_PHASE = 17
STOP_PHASE = 25
T_BLOCK = 40
R_BLOCK = 3
NAME = "H1-R14-G01-MixerIncrement-R20"


def g01_plan(plan):
    """Keep the original 45 phase IDs and change only post-R14 T widths."""
    result = copy.deepcopy(plan)
    t_seen = 0
    for index, row in enumerate(result):
        if START_PHASE <= index < STOP_PHASE and row["kind"] == "T":
            t_seen += 1
            row["updates"] = T_BLOCK
    if t_seen != 2:
        raise ValueError(f"Expected two post-R14 T phases, got {t_seen}")
    if [row["kind"] for row in result] != ["R"] * 5 + ["T", "R", "R", "R"] * 10:
        raise ValueError("Original phase ordering changed")
    return result


def _attach_gate(mixer):
    if getattr(mixer, "_g01_attached", False):
        return
    original = mixer.crossed_codebooks
    with torch.no_grad():
        base = mixer.base_codebooks.detach().to(torch.float32).clone()
        crossed, _ = original(False)
        crossed = crossed.detach().to(torch.float32).clone()
        mixer._g01_b0 = base
        mixer._g01_c0 = crossed
        mixer._g01_m0 = crossed - base

    def gated(self, return_attention=False):
        raw, weights = self._g01_original_crossed_codebooks(return_attention)
        base_now = self.base_codebooks
        m0 = self._g01_m0.to(device=raw.device, dtype=raw.dtype)
        effective = base_now + m0 + GAMMA * (raw - base_now - m0)
        return effective, weights if return_attention else None

    mixer._g01_original_crossed_codebooks = original
    mixer.crossed_codebooks = types.MethodType(gated, mixer)
    mixer._g01_attached = True


def gate_metrics(trainer):
    mixer = trainer.alignment.mixer
    _attach_gate(mixer)
    with torch.no_grad():
        raw, _ = mixer._g01_original_crossed_codebooks(False)
        base = mixer.base_codebooks
        effective, _ = mixer.crossed_codebooks(False)
        m0 = mixer._g01_m0.to(device=raw.device, dtype=raw.dtype)
        return {
            "gamma": GAMMA,
            "raw_mixer_norm": float((raw - base).norm().item()),
            "raw_increment_norm": float((raw - base - m0).norm().item()),
            "effective_mixer_increment_norm": float((effective - base - m0).norm().item()),
            "effective_codebook_displacement_from_R14_norm": float((effective - mixer._g01_c0.to(effective)).norm().item()),
            "effective_codebook_norm": float(effective.norm().item()),
        }


class G01Controller(RecoveryController):
    """RecoveryController with unchanged T learning rates and the G01 gate."""

    def __init__(self, trainer, parent_binding, execution_binding, parent_sha256):
        super().__init__(trainer, "decay", parent_binding, execution_binding,
                         parent_sha256, start_phase=START_PHASE, stop_phase=STOP_PHASE)
        self.policy = "g01"
        self.protocol.update(
            policy="g01", name=NAME, gamma=GAMMA, t_block=T_BLOCK,
            r_block=R_BLOCK, t_lr="original_unchanged",
            mixer_gate="C=B+M_R14+gamma*(M_t-M_R14)",
        )
        self.g01_anchor = None

    def set_t_lrs(self):
        # The intervention changes only the effective codebook, never Adam LR.
        return None

    def prepare_boundary(self):
        required = self.row["updates"]
        if self.progress["update"] != required:
            raise ValueError("Phase has unfinished optimizer steps")
        if not self.progress.get("boundary_prepared"):
            if self.row["kind"] == "T":
                self.trainer.refresh_catalog()
            self.progress["boundary_prepared"] = True

    def validate_clocks(self, state):
        counts = self.counts_at(state["progress"])
        for kind, key in (("R", "r_updates"), ("T", "t_updates")):
            if state[key] != counts[kind]:
                raise ValueError(f"{kind} count disagrees with G01 cursor")
            from .training import validate_optimizer_clock
            validate_optimizer_clock(state[f"{kind.lower()}_optimizer"], counts[kind], kind)
        if state["scheduler"]["last_epoch"] != counts["R"]:
            raise ValueError("R scheduler clock changed")
        total_r = sum(row["updates"] for row in self.trainer.plan if row["kind"] == "R")
        warmup = self.trainer.config["training"]["decoder_warmup_updates"]
        step = counts["R"]
        factor = (max(step, 1) / max(warmup, 1) if step < warmup else
                  0.5 * (1 + math.cos(math.pi * min(max(
                      (step - warmup) / max(total_r - warmup, 1), 0), 1))))
        expected_r = self.trainer.config["training"]["decoder_lr"] * factor
        if (state["scheduler"]["base_lrs"] != [self.trainer.config["training"]["decoder_lr"]]
                or state["scheduler"]["_step_count"] != step + 1
                or not abs(state["scheduler"]["_last_lr"][0] - expected_r) <= 1e-14
                or not abs(state["r_optimizer"]["param_groups"][0]["lr"] - expected_r) <= 1e-14):
            raise ValueError("R learning-rate clock changed")
        if self.parent_optimizer_groups:
            for key, original in self.parent_optimizer_groups.items():
                strip = lambda groups: [{k: v for k, v in g.items() if k not in ("lr", "initial_lr")}
                                        for g in groups]
                if strip(state[key]["param_groups"]) != strip(original):
                    raise ValueError(f"{key} topology or hyperparameters changed")
            if [g["lr"] for g in state["t_optimizer"]["param_groups"]] != [
                    g["lr"] for g in self.parent_optimizer_groups["t_optimizer"]]:
                raise ValueError("T learning rate changed")
        if int(state["alignment"]["diger_optimizer_step"]) != counts["T"]:
            raise ValueError("DIGER T clock changed")
        if int(state["alignment"]["diger_total_optimizer_steps"]) != self.total_t:
            raise ValueError("DIGER total T clock changed")

    def fork(self, parent_state, *, exact_rng=True):
        super().fork(parent_state, exact_rng=exact_rng)
        _attach_gate(self.trainer.alignment.mixer)
        self.g01_anchor = {
            "B0": cpu_copy(self.trainer.alignment.mixer._g01_b0),
            "M0": cpu_copy(self.trainer.alignment.mixer._g01_m0),
            "C0": cpu_copy(self.trainer.alignment.mixer._g01_c0),
            "B0_sha256": fingerprint(self.trainer.alignment.mixer._g01_b0),
            "M0_sha256": fingerprint(self.trainer.alignment.mixer._g01_m0),
            "C0_sha256": fingerprint(self.trainer.alignment.mixer._g01_c0),
        }
        return self.checkpoint()

    def checkpoint(self):
        state = super().checkpoint()
        if self.g01_anchor is not None:
            state["g01"] = dict(gamma=GAMMA, anchor=cpu_copy(self.g01_anchor))
        return state

    def restore(self, state, *, exact_rng=True, allow_preflight=False):
        gate = state.get("g01")
        if not isinstance(gate, dict) or gate.get("gamma") != GAMMA:
            raise ValueError("Missing G01 gate state")
        saved = gate.get("anchor")
        if self.g01_anchor is None or not isinstance(saved, dict):
            raise ValueError("G01 anchor is not authenticated")
        for key in ("B0", "M0", "C0"):
            if fingerprint(saved[key]) != fingerprint(self.g01_anchor[key]):
                raise ValueError(f"G01 anchor changed: {key}")
        super().restore(state, exact_rng=exact_rng, allow_preflight=allow_preflight)
        _attach_gate(self.trainer.alignment.mixer)


def build_g01(parent_path, device, dataset=None, assets=None, inspection=False):
    parent_path = Path(parent_path)
    digest = sha256(parent_path)
    if digest != PARENT_SHA256:
        raise ValueError("H1 R14 parent SHA does not match authorized checkpoint")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (parent["r_updates"], parent["t_updates"], parent["progress"]["phase"]) != (9702, 120, 17):
        raise ValueError("Parent counters are not R14/T120")
    config = copy.deepcopy(parent["binding"]["config"])
    if dataset:
        config["dataset"] = str(dataset)
    if assets:
        config["assets"] = str(assets)
    trainer, loaded_assets, data, current = build_context(config, device, parent["precision"], True)
    trainer.plan = g01_plan(trainer.plan)
    controller = G01Controller(trainer, parent["binding"], current, digest)
    controller.fork(parent, exact_rng=not inspection)
    return controller, loaded_assets, data
