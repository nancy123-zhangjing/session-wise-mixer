"""Verify that a completed DIGER smoke run exercised both gradient routes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime_paths import resolve_project_path  # noqa: E402

DEFAULT_RUN_ID = "diff_smoke_crosspse_joint_diger_st_4x256_seed2024"


def resolve_runtime_path(value: str) -> Path:
    return resolve_project_path(value, project_root=ROOT, relative_to=ROOT / "external" / "DiffGRM")


def number(row, key):
    value = row.get(key, "")
    return None if value in (None, "") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    candidates = []
    metadata_root = ROOT / "outputs" / "run_metadata"
    for path in metadata_root.rglob("*_run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("run_id") == args.run_id
            and int(data.get("seed", -1)) == args.seed
        ):
            candidates.append((path.stat().st_mtime_ns, path, data))
    if not candidates:
        raise FileNotFoundError(
            f"No completed smoke metadata for {args.run_id} seed {args.seed}"
        )
    _, metadata_path, metadata = max(candidates)
    metrics_path = resolve_runtime_path(metadata["joint_metrics_csv"])
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    tokenizer_rows = [row for row in rows if row["phase"] == "tokenizer"]
    recommender_rows = [row for row in rows if row["phase"] == "recommender"]
    assert tokenizer_rows and recommender_rows, "both alternating phases must run"

    direct = [
        number(row, "diger_direct_gradient_norm") for row in tokenizer_rows
    ]
    direct = [value for value in direct if value is not None]
    assert direct and max(direct) > 0, (
        "recommendation loss did not directly update Cross-PSE"
    )
    for row in tokenizer_rows:
        assert number(row, "recommendation_loss") is not None
        assert number(row, "recommender_gradient_norm") == 0.0
        assert number(row, "tokenizer_gradient_norm") > 0.0
        assert math.isfinite(number(row, "diger_assignment_entropy"))
    for row in recommender_rows:
        assert number(row, "recommender_gradient_norm") > 0.0
        assert number(row, "tokenizer_gradient_norm") == 0.0

    print("DIGER smoke evidence: PASS")
    print(f"metadata: {metadata_path}")
    print(f"joint metrics: {metrics_path}")
    print(f"direct recommendation gradient norm: {max(direct):.6g}")


if __name__ == "__main__":
    main()
