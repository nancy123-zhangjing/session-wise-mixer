"""Scientific settings are locked separately from execution-only choices."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rqcd_mixer/v1.json"
RUNTIME_TEMPLATE = ROOT / "configs/rqcd_mixer/runtime.example.json"
DESIGN = ROOT / "docs/research/rqcd_mixer/RQCD_MIXER_FULL_DESIGN_20261111.md"
ARMS = {"B0": "none", "B1": "intra", "B2": "causal"}
TARGET_DESIGN = ROOT / "docs/research/rqcd_mixer/RQCD_TARGETS_DESIGN_20261113.md"
TARGET_CONFIGS = {mode: ROOT / f"configs/rqcd_mixer/v2_{mode}.json" for mode in ("cf", "hybrid")}


def design_path(cfg):
    return TARGET_DESIGN if "target" in cfg else DESIGN


def load_config(path=DEFAULT_CONFIG):
    actual = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = actual.get("target", {}).get("mode")
    if mode is not None and mode not in TARGET_CONFIGS:
        raise ValueError("Unknown target protocol")
    expected_path = DEFAULT_CONFIG if mode is None else TARGET_CONFIGS[mode]
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("Scientific configuration differs from the approved protocol")
    if hashlib.sha256(design_path(actual).read_bytes()).hexdigest() != actual["design_sha256"]:
        raise ValueError("The approved design hash changed; create an explicit protocol version")
    return copy.deepcopy(actual)


def load_runtime(path, cfg, *, require_batches=True):
    runtime = json.loads(Path(path).read_text(encoding="utf-8"))
    template = json.loads(RUNTIME_TEMPLATE.read_text(encoding="utf-8"))
    if set(runtime) != set(template):
        raise ValueError("Runtime may contain execution settings only; unexpected/missing keys")
    if runtime["precision"] != "bf16" or runtime["data_device"] not in ("cpu", "cuda"):
        raise ValueError("Formal execution requires BF16; data placement may be cpu or cuda")
    if runtime["attention_backend"] not in ("auto", "math", "efficient", "flash"):
        raise ValueError("Unknown attention backend")
    if not isinstance(runtime["deterministic"], bool) or not runtime["deterministic"]:
        raise ValueError("Do not silently relax deterministic execution")
    if not isinstance(runtime["fused_adamw"], bool):
        raise ValueError("fused_adamw must be boolean")
    for name in ("micro_r", "micro_t", "evaluation_batch"):
        value = runtime[name]
        if value is None and not require_batches:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be selected by the combined short benchmark")
        if name != "evaluation_batch" and value > cfg["training"]["effective_batch"]:
            raise ValueError("A microbatch cannot exceed the fixed effective batch")
    for key in ("checkpoint_every_updates", "evaluation_write_batches"):
        if not isinstance(runtime[key], int) or runtime[key] <= 0:
            raise ValueError(f"Invalid {key}")
    return runtime


def total_t_steps(cfg):
    return (cfg["training"]["epochs"] - cfg["training"]["preheat_epochs"]) * cfg["training"]["t_steps_per_cycle"]
