"""Verify that a frozen-control smoke run never updated Cross-PSE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = (
    "diff_smoke_crosspse_frozen_decoder_continuation_4x256_seed2024"
)


def resolve_runtime_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (
        ROOT / "external" / "DiffGRM" / path
    ).resolve()


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value in (None, "") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    candidates = []
    for path in (ROOT / "outputs/run_metadata").rglob("*_run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("run_id") == args.run_id
            and int(data.get("seed", -1)) == args.seed
        ):
            candidates.append((path.stat().st_mtime_ns, path, data))
    if not candidates:
        raise FileNotFoundError(
            f"No completed frozen smoke metadata for {args.run_id}"
        )
    _, metadata_path, metadata = max(candidates)

    assert metadata.get("joint_frozen_tokenizer_control") is True
    assert metadata.get("joint_frozen_tokenizer_verified") is True
    alignment = metadata.get("joint_alignment") or {}
    assert int(alignment.get("max_tokenizer_phases", -1)) == 0
    assert (
        metadata.get("joint_initial_semantic_ids_sha256")
        == metadata.get("joint_final_semantic_ids_sha256")
    ), "hard SID hash changed in the frozen control"

    metrics_path = resolve_runtime_path(metadata["joint_metrics_csv"])
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert rows, "frozen smoke produced no training rows"
    assert {row["phase"] for row in rows} == {"recommender"}
    for row in rows:
        assert number(row, "recommender_gradient_norm") > 0.0
        assert number(row, "tokenizer_gradient_norm") == 0.0
        assert number(row, "digit_sid_change_rate") is None
        assert number(row, "item_sid_change_rate") is None

    print("Frozen Cross-PSE smoke evidence: PASS")
    print(f"metadata: {metadata_path}")
    print(f"joint metrics: {metrics_path}")
    print("phases: recommender only")
    print("tokenizer gradient: 0")
    print("initial/final hard SID hash: identical")


if __name__ == "__main__":
    main()
