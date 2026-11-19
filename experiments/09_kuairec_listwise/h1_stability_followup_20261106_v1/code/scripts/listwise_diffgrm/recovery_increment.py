"""Accept progressively less of each NEW Mixer correction, without LR changes."""
from __future__ import annotations

import copy
import types
from pathlib import Path

import torch

from .recovery import PARENT_SHA256, RecoveryController, fingerprint
from .recovery_g01 import G01Controller, R_BLOCK, T_BLOCK, g01_plan
from .run import build_context
from .training import cpu_copy, sha256

NAME = "H1-MixerIncrement-Linear010to001-R35"
SCHEDULE = dict(kind="linear_per_T_update", start=0.1, end=0.01,
                parent_t=200, updates=200)


def acceptance(completed):
    """Coefficient for the NEXT update; endpoints are T201=.1, T400=.01."""
    if type(completed) is not int or not 0 <= completed <= SCHEDULE["updates"]:
        raise ValueError("Increment gate clock must be in [0, 200]")
    k = min(completed, SCHEDULE["updates"] - 1)
    if k == SCHEDULE["updates"] - 1:
        return SCHEDULE["end"]
    return SCHEDULE["start"] + (
        SCHEDULE["end"] - SCHEDULE["start"]) * k / (SCHEDULE["updates"] - 1)


class IncrementGate:
    """Committed forward state; local derivative alpha*dM, no history graph.

    Commit exactly once AFTER each T optimizer step. All microbatches use the
    same committed codebook and alpha. Reads, R steps and catalog refreshes
    never advance the gate. Base codebooks remain fully trainable.
    """

    def __init__(self, mixer):
        if hasattr(mixer, "_increment_gate"):
            raise ValueError("Increment gate already attached")
        if mixer.attention.dropout != 0:
            raise ValueError("Increment measurements require the existing zero dropout")
        self.mixer = mixer
        self.original = mixer.crossed_codebooks
        with torch.no_grad():
            effective, _ = self.original(False)
            self.b0 = mixer.base_codebooks.detach().clone()
            self.c0 = effective.detach().clone()
        self.previous_raw = self.raw_correction()
        self.accepted_delta = torch.zeros_like(self.c0)
        self.last_raw_delta = torch.zeros_like(self.c0)
        self.completed = 0
        self.anchor = cpu_copy(dict(B0=self.b0, C0=self.c0,
                                    M0=self.c0 - self.b0,
                                    raw_start=self.previous_raw))
        gate = self

        def crossed(module, return_attention=False):
            raw, weights = gate.original(return_attention)
            correction = raw - module.base_codebooks
            # Zero-valued local autograd term: evaluation cannot change codes
            # through train/eval attention-kernel rounding differences.
            local = acceptance(gate.completed) * (correction - correction.detach())
            effective = (gate.c0 + (module.base_codebooks - gate.b0)
                         + gate.accepted_delta + local)
            return effective, weights if return_attention else None

        mixer.crossed_codebooks = types.MethodType(crossed, mixer)
        mixer._increment_gate = self

    @torch.no_grad()
    def raw_correction(self):
        # A fixed kernel path for measurements; no dropout, sampling or RNG.
        # Preserve every submodule's mode, including across R/T boundaries.
        modes = [(m, m.training) for m in self.mixer.modules()]
        try:
            self.mixer.train(True)
            with torch.autocast(device_type=self.mixer.base_codebooks.device.type,
                                enabled=False):
                raw, _ = self.original(False)
                return (raw - self.mixer.base_codebooks).detach().clone()
        finally:
            for module, training in modes:
                module.training = training

    @torch.no_grad()
    def commit(self):
        if self.completed >= SCHEDULE["updates"]:
            raise ValueError("All 200 authorized T increments already committed")
        raw = self.raw_correction()
        delta = raw - self.previous_raw
        accepted = self.accepted_delta + acceptance(self.completed) * delta
        if not torch.isfinite(raw).all() or not torch.isfinite(accepted).all():
            raise FloatingPointError("Non-finite Mixer increment")
        self.last_raw_delta = delta.clone()
        self.accepted_delta = accepted.clone()
        self.previous_raw = raw.clone()
        self.completed += 1

    def state(self):
        payload = cpu_copy(dict(schedule=SCHEDULE, completed=self.completed,
                                anchor=self.anchor, previous_raw=self.previous_raw,
                                accepted_delta=self.accepted_delta,
                                last_raw_delta=self.last_raw_delta))
        return dict(payload=payload, sha256=fingerprint(payload))

    def validate_state(self, state, t_updates):
        if not isinstance(state, dict) or set(state) != {"payload", "sha256"}:
            raise ValueError("Missing incremental gate state")
        payload = state["payload"]
        if fingerprint(payload) != state["sha256"]:
            raise ValueError("Incremental gate state checksum mismatch")
        if payload["schedule"] != SCHEDULE:
            raise ValueError("Increment schedule changed")
        if (type(payload["completed"]) is not int
                or payload["completed"] != t_updates - SCHEDULE["parent_t"]):
            raise ValueError("Increment gate and optimizer clocks disagree")
        acceptance(payload["completed"])
        if fingerprint(payload["anchor"]) != fingerprint(self.anchor):
            raise ValueError("Increment gate R20 anchor changed")
        for key in ("previous_raw", "accepted_delta", "last_raw_delta"):
            value = payload[key]
            if (not torch.is_tensor(value) or value.shape != self.c0.shape
                    or value.dtype != self.c0.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid increment tensor: {key}")
        if payload["completed"] == 0:
            if (not torch.equal(payload["previous_raw"], self.anchor["raw_start"])
                    or torch.count_nonzero(payload["accepted_delta"])
                    or torch.count_nonzero(payload["last_raw_delta"])):
                raise ValueError("Initial gate state differs from R20")

    def restore(self, state, t_updates):
        self.validate_state(state, t_updates)
        payload = state["payload"]
        if not torch.allclose(self.raw_correction().cpu(), payload["previous_raw"],
                              rtol=1e-6, atol=1e-7):
            raise ValueError("Saved raw Mixer correction disagrees with model")
        for key in ("previous_raw", "accepted_delta", "last_raw_delta"):
            setattr(self, key, payload[key].to(self.c0).detach().clone())
        self.completed = payload["completed"]


class IncrementController(G01Controller):
    """Reuse G01 clocks/isolation; replace only its fixed-anchor Mixer gate."""

    def __init__(self, trainer, parent_binding, execution_binding, parent_sha256):
        super().__init__(trainer, parent_binding, execution_binding, parent_sha256)
        self.policy = "increment_linear"
        self.protocol.pop("gamma")
        self.protocol.update(
            policy=self.policy, name=NAME, increment_schedule=copy.deepcopy(SCHEDULE),
            mixer_gate="M_eff[k]=M_eff[k-1]+alpha[k]*(M_raw[k]-M_raw[k-1])",
            gate_commit="after_each_T_optimizer_step",
            gate_gradient="local_alpha_times_raw_Mixer_derivative",
        )
        self.increment_gate = None

    def fork(self, parent_state, *, exact_rng=True):
        # Deliberately bypass G01.fork: never attach the old fixed gate.
        RecoveryController.fork(self, parent_state, exact_rng=exact_rng)
        self.increment_gate = IncrementGate(self.trainer.alignment.mixer)
        return self.checkpoint()

    def step(self, microbatches, kind):
        if self.increment_gate.completed != self.trainer.t_updates - SCHEDULE["parent_t"]:
            raise ValueError("Uncommitted or duplicate T update")
        used = acceptance(self.increment_gate.completed)
        values = super().step(microbatches, kind)
        if kind == "T":
            self.increment_gate.commit()
        values["increment_alpha_used"] = used if kind == "T" else None
        return values

    def checkpoint(self):
        state = RecoveryController.checkpoint(self)
        if self.increment_gate is not None:
            if self.increment_gate.completed != self.trainer.t_updates - SCHEDULE["parent_t"]:
                raise ValueError("Cannot save uncommitted T update")
            state["increment_gate"] = self.increment_gate.state()
        return state

    def restore(self, state, *, exact_rng=True, allow_preflight=False):
        if self.increment_gate is None:
            raise ValueError("Authenticate original R20 parent before restore")
        saved = state.get("increment_gate")
        self.increment_gate.validate_state(saved, state["inner"]["t_updates"])
        RecoveryController.restore(self, state, exact_rng=exact_rng,
                                   allow_preflight=allow_preflight)
        self.increment_gate.restore(saved, self.trainer.t_updates)


def gate_metrics(trainer):
    gate = trainer.alignment.mixer._increment_gate
    with torch.no_grad():
        base_delta = gate.mixer.base_codebooks - gate.b0
        last_alpha = acceptance(gate.completed - 1) if gate.completed else None
        return dict(
            completed_increments=gate.completed,
            last_alpha=last_alpha,
            next_alpha=acceptance(gate.completed) if gate.completed < 200 else None,
            raw_mixer_norm=float(gate.previous_raw.norm()),
            raw_increment_norm=float((gate.previous_raw - gate.anchor["raw_start"].to(gate.c0)).norm()),
            last_raw_delta_norm=float(gate.last_raw_delta.norm()),
            last_accepted_delta_norm=float(
                (gate.last_raw_delta * (last_alpha or 0.0)).norm()),
            effective_mixer_increment_norm=float(gate.accepted_delta.norm()),
            base_codebook_displacement_from_R20_norm=float(base_delta.norm()),
            effective_codebook_displacement_from_R20_norm=float(
                (base_delta + gate.accepted_delta).norm()),
            effective_codebook_norm=float(
                (gate.c0 + base_delta + gate.accepted_delta).norm()),
        )


def build_increment(parent_path, device, dataset=None, assets=None, inspection=False):
    parent_path = Path(parent_path)
    digest = sha256(parent_path)
    if digest != PARENT_SHA256:
        raise ValueError("H1 R20 parent SHA does not match authorized checkpoint")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (parent["r_updates"], parent["t_updates"], parent["progress"]["phase"]) != (13860, 200, 25):
        raise ValueError("Parent counters are not R20/T200")
    config = copy.deepcopy(parent["binding"]["config"])
    if dataset:
        config["dataset"] = str(dataset)
    if assets:
        config["assets"] = str(assets)
    trainer, loaded_assets, data, current = build_context(config, device, parent["precision"], True)
    trainer.plan = g01_plan(trainer.plan)
    controller = IncrementController(trainer, parent["binding"], current, digest)
    controller.fork(parent, exact_rng=not inspection)
    return controller, loaded_assets, data
