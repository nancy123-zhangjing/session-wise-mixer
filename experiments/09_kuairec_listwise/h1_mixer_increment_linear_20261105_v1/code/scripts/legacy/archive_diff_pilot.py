"""Archive the first DiffGRM seed-2024 run with the misaligned warmup schedule.

The operation is non-destructive: source artifacts remain where training wrote
them, while an immutable-looking evidence bundle is copied under outputs/archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
OLD_RUN_ID = "diff_formal_pse_4x256"
ARCHIVE_NAME = "diff_pilot_seed2024_warmup_misaligned_20260810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (DIFF_ROOT / path).resolve()


def latest_old_metadata(metadata_root: Path) -> Path:
    candidates = []
    for path in metadata_root.rglob("*_run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") == OLD_RUN_ID and int(data.get("seed", -1)) == 2024:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No seed-2024 metadata with run_id={OLD_RUN_ID!r} under {metadata_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "archive" / ARCHIVE_NAME,
    )
    args = parser.parse_args()

    metadata_path = (
        args.metadata.resolve()
        if args.metadata
        else latest_old_metadata(PROJECT_ROOT / "outputs" / "run_metadata")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != OLD_RUN_ID or int(metadata.get("seed", -1)) != 2024:
        raise ValueError(
            "Refusing to archive a run other than the original DiffGRM seed-2024 pilot"
        )

    if args.log:
        log_path = args.log.resolve()
    else:
        logs = list(
            (PROJECT_ROOT / "outputs" / "console").glob(
                "diff_formal_pse_seed2024_*.log"
            )
        )
        if not logs:
            raise FileNotFoundError("No original seed-2024 formal console log found")
        log_path = max(logs, key=lambda path: path.stat().st_mtime_ns)

    sources = {
        "console_log": log_path,
        "run_metadata": metadata_path,
        "position_metrics": resolve_run_path(metadata["position_metrics_csv"]),
        "order_metrics": resolve_run_path(metadata["order_metrics_csv"]),
        "checkpoint": resolve_run_path(metadata["checkpoint"]),
    }
    missing = {name: str(path) for name, path in sources.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"Pilot artifacts are incomplete: {missing}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, source in sources.items():
        suffix = source.suffix
        destination = output_dir / f"{name}{suffix}"
        shutil.copy2(source, destination)
        copied[name] = {
            "source": str(source),
            "archive": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    manifest = {
        "archive_name": ARCHIVE_NAME,
        "status": "pilot_excluded_from_formal_summary",
        "reason": (
            "gradient_accumulation_steps=16 yielded about 10,400 optimizer "
            "updates over 100 epochs, so warmup_steps=10,000 occupied roughly "
            "96% of training; best validation occurred at the final epoch"
        ),
        "original_run_id": metadata.get("run_id"),
        "seed": metadata.get("seed"),
        "best_epoch": metadata.get("best_epoch"),
        "fit_elapsed_seconds": metadata.get("fit_elapsed_seconds"),
        "test_results": metadata.get("test_results"),
        "sid_collision_stats": metadata.get("sid_collision_stats"),
        "known_schedule": {
            "physical_batch": 64,
            "gradient_accumulation_steps": 16,
            "effective_batch": 1024,
            "micro_batches_per_epoch": 1652,
            "optimizer_updates_per_epoch": 104,
            "total_optimizer_updates": 10400,
            "warmup_steps": 10000,
            "eval_interval": 5,
        },
        "artifacts": copied,
    }
    manifest_path = output_dir / "archive_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Archived pilot to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    for name, details in copied.items():
        print(f"  {name}: {details['archive']}")


if __name__ == "__main__":
    main()
