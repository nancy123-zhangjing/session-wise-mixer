"""Versioned four-arm protocol. No training, torch import or platform access."""
from __future__ import annotations

import json
from pathlib import Path

VERSION = "h1_stability_followup_v1"
R26_ALPHA = 0.1 + (0.01 - 0.1) * 79 / 199  # Last USED: T280, not next T281.
ARMS = {
    "r26_hold": dict(parent="increment_r26", start_r=26, start_t=280,
                     mode="constant", alpha_start=R26_ALPHA, alpha_end=R26_ALPHA),
    "r26_low": dict(parent="increment_r26", start_r=26, start_t=280,
                    mode="constant", alpha_start=0.01, alpha_end=0.01),
    "release20_freeze": dict(parent="release_r20", start_r=20, start_t=200,
                             mode="freeze", alpha_start=0.0, alpha_end=0.0),
    "release20_linear": dict(parent="release_r20", start_r=20, start_t=200,
                             mode="linear", alpha_start=0.1, alpha_end=0.01),
}


def specification(arm):
    spec = dict(ARMS[arm], version=VERSION, arm=arm, stop_r=35)
    spec["updates"] = 0 if spec["mode"] == "freeze" else 400 - spec["start_t"]
    spec["stop_t"] = spec["start_t"] + spec["updates"]
    spec["validation_epochs"] = list(range(spec["start_r"], 36, 3))
    return spec


def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec != specification(spec["arm"]):
        raise ValueError("Configuration differs from versioned four-arm protocol")
    return spec


def coefficient(spec, completed):
    if type(completed) is not int or not 0 <= completed <= spec["updates"]:
        raise ValueError("Invalid gate clock")
    if spec["mode"] != "linear":
        return spec["alpha_start"]
    index = min(completed, spec["updates"] - 1)
    if index == spec["updates"] - 1:
        return spec["alpha_end"]
    return spec["alpha_start"] + (spec["alpha_end"] - spec["alpha_start"]) * index / (spec["updates"] - 1)


def normalize_boundary(inner, plan, spec):
    """Accept only a completed parent R boundary, never repeat its last R."""
    epoch = spec["start_r"]
    boundary = next(i for i, row in enumerate(plan)
                    if row["kind"] == "R" and row["epoch"] == epoch)
    progress = inner["progress"]
    expected_r = sum(row["updates"] for row in plan[:boundary + 1] if row["kind"] == "R")
    if (inner["r_updates"], inner["t_updates"]) != (expected_r, spec["start_t"]):
        raise ValueError("Wrong parent R/T counts")
    pending = (progress.get("phase") == boundary
               and progress.get("update") == plan[boundary]["updates"]
               and progress.get("boundary_prepared") is True)
    advanced = (progress.get("phase") == boundary + 1
                and progress.get("update") == 0
                and progress.get("boundary_prepared") is False)
    if not (pending or advanced) or progress.get("preflight_only"):
        raise ValueError("Parent is not a completed, catalog-prepared R boundary")
    return dict(phase=boundary + 1, update=0, boundary_prepared=False)
