"""Standard-library preflight and paired budgets for the stage-05 noise pilot.

No model imports, training, cache updates, or platform resource allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json"


def relative_path(root: Path, value: str) -> Path:
    """Only accept portable, project-contained paths (also reject symlink escapes)."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Expected a nonempty POSIX project-relative path: {value!r}")
    parsed = PurePosixPath(value)
    if (parsed.is_absolute() or PureWindowsPath(value).drive
            or ".." in parsed.parts or value in {".", "/"}):
        raise ValueError(f"Unsafe project-relative path: {value!r}")
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes project: {value!r}")
    return path


def load_protocol(config_path: Path) -> dict:
    protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("Pilot configuration must be an object")
    if protocol.get("protocol_id") != "stage05_diger_noise_once_freeze_recover_v1":
        raise ValueError("Unknown pilot protocol")
    for key in ("seed", "gradient_accumulation_steps", "tokenizer_batches",
                "decoder_epochs", "tokenizer_schedule_total_updates", "gumbel_seed",
                "tokenizer_phase_seed", "decoder_seed_base", "data_seed_base"):
        value = protocol.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if protocol.get("expected_best_epoch") != 5 or protocol.get("expected_tokenizer_updates") != 0:
        raise ValueError("The shared start must be Cross v2 epoch 5, before tokenizer updates")
    for key in ("decoder_lr", "assignment_temperature", "softmax_temperature"):
        value = protocol.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("decoder_weight_decay", "noise_start", "noise_end"):
        value = protocol.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if protocol["noise_start"] <= 0:
        raise ValueError("C must be a nonzero-noise intervention")
    if protocol["tokenizer_schedule_total_updates"] < math.ceil(
        protocol["tokenizer_batches"] / protocol["gradient_accumulation_steps"]
    ):
        raise ValueError("Noise schedule horizon cannot be shorter than the tokenizer phase")
    for key in ("shared_checkpoint", "shared_metadata", "asset_lock"):
        relative_path(ROOT, protocol.get(key))
    for key in ("required_assets", "config_layers"):
        values = protocol.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        for value in values:
            relative_path(ROOT, value)
        if len(set(values)) != len(values):
            raise ValueError(f"{key} contains duplicate paths")
    if not re.fullmatch(r"[0-9a-f]{64}", protocol.get("shared_checkpoint_sha256", "")):
        raise ValueError("shared_checkpoint_sha256 must be a lowercase SHA256")
    smoke = protocol.get("smoke", {})
    for key in ("tokenizer_batches", "decoder_batches", "eval_batches", "decoder_epochs"):
        if type(smoke.get(key)) is not int or smoke[key] <= 0:
            raise ValueError(f"smoke.{key} must be a positive integer")


def arm_settings(protocol: dict, arm: str, smoke: bool = False) -> dict:
    validate_protocol(protocol)
    if arm not in {"A", "B", "C"}:
        raise ValueError(f"Unknown arm: {arm}")
    short = protocol["smoke"] if smoke else {}
    return {
        "arm": arm,
        "tokenizer_batches": (0 if arm == "A" else short.get("tokenizer_batches", protocol["tokenizer_batches"])),
        "decoder_epochs": short.get("decoder_epochs", protocol["decoder_epochs"]),
        "decoder_batch_limit": short.get("decoder_batches", 0),
        "eval_batch_limit": short.get("eval_batches", 0),
        "noise_start": protocol["noise_start"] if arm == "C" else 0.0,
        "noise_end": protocol["noise_end"] if arm == "C" else 0.0,
        "noise_total_steps": protocol["tokenizer_schedule_total_updates"],
        "decoder_lr": protocol["decoder_lr"],
        "gradient_accumulation_steps": protocol["gradient_accumulation_steps"],
    }


def phase_seed(seed: int, kind: str, index: int) -> int:
    """Stable independent seeds; arm never enters this mapping."""
    if type(seed) is not int or seed < 0 or type(index) is not int or index < 0:
        raise ValueError("Seed and zero-based phase index must be nonnegative integers")
    if kind not in {"tokenizer", "decoder", "data", "evaluation", "gumbel"}:
        raise ValueError(f"Unknown random stream: {kind}")
    payload = f"{seed}:{kind}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def loss_divisor(batch_index: int, total_batches: int, accumulation: int) -> int:
    """Mean over the actual final accumulation window, not phantom batches."""
    if any(type(x) is not int for x in (batch_index, total_batches, accumulation)):
        raise ValueError("Batch indices and counts must be integers")
    if total_batches <= 0 or accumulation <= 0 or not 0 <= batch_index < total_batches:
        raise ValueError("Invalid batch index or accumulation budget")
    start = (batch_index // accumulation) * accumulation
    return min(accumulation, total_batches - start)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(project_root: Path, protocol: dict) -> list[dict]:
    validate_protocol(protocol)
    root = Path(project_root).resolve()
    lock_path = relative_path(root, protocol["asset_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    artifacts = lock["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("Expected path-keyed stage-05 asset lock")
    requested = [protocol["shared_checkpoint"], *protocol["required_assets"]]
    receipt = []
    for value in requested:
        path = relative_path(root, value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {value}")
        entry = artifacts.get(value)
        if not entry or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"No valid archived input SHA256: {value}")
        expected = entry["sha256"]
        if value == protocol["shared_checkpoint"] and expected != protocol["shared_checkpoint_sha256"]:
            raise RuntimeError("Shared checkpoint binding disagrees with archive lock")
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Input hash mismatch: {value}")
        receipt.append({"path": value, "sha256": actual, "size_bytes": path.stat().st_size,
                        "binding": "stage05_archived_asset_lock"})
    metadata_path = relative_path(root, protocol["shared_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schedule = metadata.get("joint_alignment", {}).get("diger_final_schedule_state", {})
    if (metadata.get("run_id") != protocol["expected_source_run_id"]
            or metadata.get("best_epoch") != protocol["expected_best_epoch"]
            or schedule.get("optimizer_step") != protocol["expected_tokenizer_updates"]
            or schedule.get("total_optimizer_steps") != protocol["tokenizer_schedule_total_updates"]):
        raise RuntimeError("Source metadata is not the pinned pre-tokenizer best state")
    for value in [protocol["asset_lock"], protocol["shared_metadata"], *protocol["config_layers"]]:
        path = relative_path(root, value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing protocol/configuration input: {value}")
        receipt.append({"path": value, "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size, "binding": "current_protocol_source"})
    return receipt


def source_receipt(project_root: Path) -> dict:
    """Capture Git state and source hashes; dirty trees are not called commits."""
    root = Path(project_root).resolve()

    def git(directory: Path, *args: str) -> str | None:
        process = subprocess.run(["git", "-C", str(directory), *args],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        return process.stdout.strip() if process.returncode == 0 else None

    roots = {"project": root, "diffgrm": root / "external/DiffGRM"}
    versions = {key: {"head": git(directory, "rev-parse", "HEAD"),
                      "status": git(directory, "status", "--porcelain"),
                      "diff": git(directory, "diff", "HEAD", "--", ".")}
                for key, directory in roots.items()}
    paths = ["scripts/train/run_diger_noise_pilot.py", "scripts/checks/diger_noise_pilot_protocol.py"]
    engine = root / "external/DiffGRM/genrec"
    if engine.is_dir():
        paths.extend(path.relative_to(root).as_posix() for path in engine.rglob("*")
                     if path.suffix in {".py", ".yaml"} and path.is_file())
    return {"git": versions, "source_sha256": {
        path: file_sha256(root / path) for path in sorted(set(paths)) if (root / path).is_file()
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--plan-only", action="store_true", help="Validate budgets without reading large input assets")
    args = parser.parse_args()
    root = args.project_root.resolve()
    path = args.config if args.config.is_absolute() else root / args.config
    protocol = load_protocol(path)
    result = {"protocol_id": protocol["protocol_id"], "shared_checkpoint": protocol["shared_checkpoint"],
              "arms": [arm_settings(protocol, arm) for arm in "ABC"],
              "optimizer_start": "fresh, identical per arm; not exact resume",
              "evaluation": "validation only; SID-level; no early stopping",
              "asset_verification": "not_requested" if args.plan_only else "passed"}
    if not args.plan_only:
        result["inputs"] = verify_assets(root, protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
