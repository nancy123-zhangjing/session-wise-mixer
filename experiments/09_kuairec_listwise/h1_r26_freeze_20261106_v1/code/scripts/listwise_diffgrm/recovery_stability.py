"""Fork committed parent encodings; change only FUTURE Mixer increments."""
from __future__ import annotations

import copy
import json
import types
from pathlib import Path

import torch

from .recovery import RecoveryController, fingerprint
from .recovery_g01 import G01Controller
from .recovery_increment import IncrementGate, SCHEDULE
from .recovery_stability_protocol import VERSION, coefficient, normalize_boundary, specification
from .run import build_context
from .training import cpu_copy, sha256


def validate_parent_gate(parent, spec):
    if parent.get("preflight_only") or parent["inner"].get("preflight_only"):
        raise ValueError("Cannot fork a preflight checkpoint")
    if spec["parent"] == "increment_r26":
        state = parent["increment_gate"]
        payload = state["payload"]
        if fingerprint(payload) != state["sha256"]:
            raise ValueError("Parent incremental state checksum failed")
        if payload["schedule"] != SCHEDULE or payload["completed"] != 80:
            raise ValueError("Requires original IncrementLinear R26/T280")
        if parent["protocol"]["policy"] != "increment_linear":
            raise ValueError("Wrong parent policy")
    else:
        anchor = parent["release1"]
        if (parent["protocol"]["policy"] != "release1"
                or parent["protocol"]["gamma"] != 1.0
                or anchor["parent_sha256"] != "aa565c3bf6daa2809ff1d30230d07aa57468a1286da1fe3eeb177a83380228ea"):
            raise ValueError("Requires the R14-gated R17 Release1 lineage")


@torch.no_grad()
def effective_parent(mixer, parent, spec):
    """Reconstruct the OLD effective codebook before attaching any new gate."""
    validate_parent_gate(parent, spec)
    base = mixer.base_codebooks
    if spec["parent"] == "increment_r26":
        payload = parent["increment_gate"]["payload"]
        anchor = payload["anchor"]
        return (anchor["C0"].to(base) + (base - anchor["B0"].to(base))
                + payload["accepted_delta"].to(base)).detach().clone()
    raw, _ = mixer.crossed_codebooks(False)
    anchor = parent["release1"]
    return (base + anchor["effective_M"].to(raw)
            + (raw - base - anchor["raw_M"].to(raw))).detach().clone()


class FutureGate:
    # Reuse the exact deterministic FP32 raw-increment measurement of the old run.
    raw_correction = IncrementGate.raw_correction

    def __init__(self, mixer, effective, spec):
        if hasattr(mixer, "_future_gate") or hasattr(mixer, "_increment_gate"):
            raise ValueError("Do not nest gates")
        if mixer.attention.dropout != 0:
            raise ValueError("Nonzero Mixer dropout is outside the locked protocol")
        self.mixer, self.original = mixer, mixer.crossed_codebooks
        self.spec = copy.deepcopy(spec)
        self.b0 = mixer.base_codebooks.detach().clone()
        self.c0 = effective.detach().clone()
        self.previous_raw = self.raw_correction()
        self.accepted_delta = torch.zeros_like(self.c0)
        self.last_raw_delta = torch.zeros_like(self.c0)
        self.completed = 0
        self.anchor = cpu_copy(dict(B0=self.b0, C0=self.c0, raw_start=self.previous_raw))
        gate = self

        def crossed(module, return_attention=False):
            raw, weights = gate.original(return_attention)
            correction = raw - module.base_codebooks
            local = coefficient(gate.spec, gate.completed) * (correction - correction.detach())
            result = gate.c0 + (module.base_codebooks - gate.b0) + gate.accepted_delta + local
            return result, weights if return_attention else None

        mixer.crossed_codebooks = types.MethodType(crossed, mixer)
        mixer._future_gate = self

    @torch.no_grad()
    def commit(self):
        if self.completed >= self.spec["updates"]:
            raise ValueError("No more T updates authorized for this arm")
        raw = self.raw_correction()
        delta = raw - self.previous_raw
        accepted = self.accepted_delta + coefficient(self.spec, self.completed) * delta
        if not torch.isfinite(raw).all() or not torch.isfinite(accepted).all():
            raise FloatingPointError("Non-finite Mixer state")
        self.previous_raw, self.last_raw_delta = raw.clone(), delta.clone()
        self.accepted_delta = accepted.clone()
        self.completed += 1

    def state(self):
        payload = cpu_copy(dict(spec=self.spec, anchor=self.anchor, completed=self.completed,
                                previous_raw=self.previous_raw, accepted_delta=self.accepted_delta,
                                last_raw_delta=self.last_raw_delta))
        return dict(payload=payload, sha256=fingerprint(payload))

    def validate(self, state, t_updates):
        if not isinstance(state, dict) or set(state) != {"payload", "sha256"}:
            raise ValueError("Missing gate state")
        payload = state["payload"]
        if fingerprint(payload) != state["sha256"] or payload["spec"] != self.spec:
            raise ValueError("Gate checksum or arm mismatch")
        coefficient(self.spec, payload["completed"])
        if payload["completed"] != t_updates - self.spec["start_t"]:
            raise ValueError("Gate and optimizer clocks disagree")
        if fingerprint(payload["anchor"]) != fingerprint(self.anchor):
            raise ValueError("Fork anchor changed")
        for key in ("previous_raw", "accepted_delta", "last_raw_delta"):
            value = payload[key]
            if (not torch.is_tensor(value) or value.shape != self.c0.shape
                    or value.dtype != self.c0.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid gate tensor: {key}")
        if payload["completed"] == 0 and (
            not torch.equal(payload["previous_raw"], self.anchor["raw_start"])
            or torch.count_nonzero(payload["accepted_delta"])
            or torch.count_nonzero(payload["last_raw_delta"])
        ):
            raise ValueError("Unmodified fork gate was changed")

    def restore(self, state, t_updates):
        self.validate(state, t_updates)
        payload = state["payload"]
        if not torch.allclose(self.raw_correction().cpu(), payload["previous_raw"], rtol=1e-6, atol=1e-7):
            raise ValueError("Raw correction and restored model disagree")
        for key in ("previous_raw", "accepted_delta", "last_raw_delta"):
            setattr(self, key, payload[key].to(self.c0).detach().clone())
        self.completed = payload["completed"]


class FollowupController(G01Controller):
    def __init__(self, trainer, parent_binding, current_binding, digest, spec):
        if spec != specification(spec["arm"]):
            raise ValueError("Unknown scientific configuration")
        self.spec, self.gate = copy.deepcopy(spec), None
        start = next(i + 1 for i, row in enumerate(trainer.plan)
                     if row["kind"] == "R" and row["epoch"] == spec["start_r"])
        policy = "freeze" if spec["mode"] == "freeze" else "decay"
        RecoveryController.__init__(self, trainer, policy, parent_binding, current_binding,
                                    digest, start_phase=start, stop_phase=len(trainer.plan))
        if self.total_t != 400 or trainer.plan[-1]["epoch"] != 35:
            raise ValueError("Original 35R/400T horizon required")
        self.policy = "freeze" if spec["mode"] == "freeze" else "followup"
        self.protocol.update(policy=self.policy, spec=self.spec, t_lr="original_unchanged",
                             gate="parent_effective_plus_new_accepted_increments")

    def fork_parent(self, parent, *, exact_rng=True):
        validate_parent_gate(parent, self.spec)
        inner = copy.deepcopy(parent["inner"])
        inner["progress"] = normalize_boundary(inner, self.trainer.plan, self.spec)
        inner["arm"] = "H1"  # Explicit authenticated fork, not another arm's resume.
        RecoveryController.fork(self, inner, exact_rng=exact_rng)
        mixer = self.trainer.alignment.mixer
        # Match parent inference path, then retain committed forward state in all modes.
        self.trainer.alignment.eval()
        effective = effective_parent(mixer, parent, self.spec)
        self.gate = FutureGate(mixer, effective, self.spec)
        if self.spec["parent"] == "increment_r26" and not torch.allclose(
            self.gate.previous_raw.cpu(), parent["increment_gate"]["payload"]["previous_raw"],
            rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Parent raw increment state does not match model")
        if not torch.equal(mixer.crossed_codebooks(False)[0], effective):
            raise ValueError("Fork changes effective codebook")
        return self.checkpoint()

    def validate_clocks(self, state):
        G01Controller.validate_clocks(self, state)
        if self.policy == "freeze":
            for key in ("alignment", "t_optimizer"):
                if fingerprint(state[key]) != self.parent_hashes[key]:
                    raise ValueError(f"Frozen {key} changed")

    def step(self, microbatches, kind):
        if self.gate.completed != self.trainer.t_updates - self.spec["start_t"]:
            raise ValueError("Uncommitted T update")
        alpha = coefficient(self.spec, self.gate.completed)
        values = RecoveryController.step(self, microbatches, kind)
        if kind == "T":
            self.gate.commit()
        values["increment_alpha_used"] = alpha if kind == "T" else None
        return values

    def prepare_boundary(self):
        skip = self.policy == "freeze" and self.row["kind"] == "T"
        required = 0 if skip else self.row["updates"]
        if self.progress["update"] != required:
            raise ValueError("Incomplete phase")
        if not self.progress["boundary_prepared"]:
            if self.row["kind"] == "T" and not skip:
                self.trainer.refresh_catalog()
            self.progress["boundary_prepared"] = True

    def checkpoint(self):
        state = RecoveryController.checkpoint(self)
        if self.gate is not None:
            if self.gate.completed != self.trainer.t_updates - self.spec["start_t"]:
                raise ValueError("Cannot checkpoint uncommitted increments")
            state["future_gate"] = self.gate.state()
        return state

    def restore(self, state, **kwargs):
        self.gate.validate(state.get("future_gate"), state["inner"]["t_updates"])
        RecoveryController.restore(self, state, **kwargs)
        self.gate.restore(state["future_gate"], self.trainer.t_updates)


def gate_metrics(trainer):
    gate = trainer.alignment.mixer._future_gate
    base_delta = gate.mixer.base_codebooks.detach() - gate.b0
    return dict(completed=gate.completed, next_alpha=coefficient(gate.spec, gate.completed),
                raw_mixer_norm=float(gate.previous_raw.norm()),
                last_raw_delta_norm=float(gate.last_raw_delta.norm()),
                accepted_mixer_displacement=float(gate.accepted_delta.norm()),
                base_displacement=float(base_delta.norm()),
                effective_codebook_displacement=float((base_delta + gate.accepted_delta).norm()))


def build_followup(spec, lock_path, device, dataset, assets, inspection=False):
    locks = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if locks["version"] != VERSION:
        raise ValueError("Parent lock belongs to another suite version")
    locked = locks["parents"][spec["parent"]]
    if locked["status"] != "verified_metadata":
        raise ValueError("Parent file has not been locked")
    path = Path(locked["checkpoint"])
    if sha256(path) != locked["sha256"]:
        raise ValueError("Parent checkpoint bytes changed")
    parent = torch.load(path, map_location="cpu", weights_only=False)
    if fingerprint(parent.get("increment_gate", parent.get("release1"))) != locked["parent_gate"]:
        raise ValueError("Parent gate binding changed")
    inner = parent["inner"]
    if (inner["r_updates"], inner["t_updates"]) != (spec["start_r"] * 693, spec["start_t"]):
        raise ValueError("Wrong formal parent counters")
    config = copy.deepcopy(inner["binding"]["config"])
    config.update(dataset=str(dataset), assets=str(assets))
    trainer, loaded, data, current = build_context(config, device, inner["precision"], True)
    controller = FollowupController(trainer, inner["binding"], current, locked["sha256"], spec)
    controller.fork_parent(parent, exact_rng=not inspection)
    # No inference pass: recompute just the catalog to prove SID continuity.
    before = fingerprint(trainer.alignment.state_dict())
    trainer.refresh_catalog()
    if fingerprint(trainer.alignment.state_dict()) != before:
        raise ValueError("Fork catalog is not identical to the saved parent SID")
    return controller, loaded, data, locked
