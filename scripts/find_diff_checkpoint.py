"""Resolve the newest completed DiffGRM checkpoint for one run and seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIFF_ROOT = ROOT / "external" / "DiffGRM"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    matches = []
    metadata_root = ROOT / "outputs" / "run_metadata"
    for path in metadata_root.rglob("*_run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("run_id") == args.run_id
            and int(data.get("seed", -1)) == args.seed
        ):
            matches.append((path.stat().st_mtime_ns, path, data))
    if not matches:
        raise SystemExit(
            f"No completed run found: run_id={args.run_id}, seed={args.seed}"
        )
    _, metadata_path, data = max(matches)
    checkpoint = Path(data["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (DIFF_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint recorded by {metadata_path} is missing: {checkpoint}"
        )
    print(checkpoint)


if __name__ == "__main__":
    main()
