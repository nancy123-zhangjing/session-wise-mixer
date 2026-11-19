"""Audit and compare the paired Refined-PSE and Cross-PSE v3 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def compact(summary: dict) -> dict:
    return {
        "protocol": summary.get("reproduction_protocol"),
        "best_step": summary.get("best_step"),
        "initial_validation_reconstruction": summary.get(
            "initial_validation_reconstruction"
        ),
        "best_validation_reconstruction": summary.get(
            "best_validation_reconstruction"
        ),
        "validation_reconstruction_improvement": summary.get(
            "validation_reconstruction_improvement"
        ),
        "best_improves_initial": summary.get("best_improves_initial"),
        "selection_protocol": summary.get("selection_protocol"),
        "selection_equals_training_catalog": summary.get(
            "selection_equals_training_catalog"
        ),
        "relative_codebook_delta_rms": summary.get(
            "relative_codebook_delta_rms"
        ),
        "base_parameter_delta_rms": summary.get("base_parameter_delta_rms"),
        "mixer_delta_rms": summary.get("mixer_delta_rms"),
        "digit_assignment_change_rate_from_base": summary.get(
            "digit_assignment_change_rate_from_base"
        ),
        "item_sid_change_rate_from_base": summary.get(
            "item_sid_change_rate_from_base"
        ),
        "collision_rate": summary.get("collision_rate"),
        "utilization": [
            row.get("utilization_rate") for row in summary.get("per_position", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refined-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tokenizers" / "diff_refinedpse"
        / "Beauty" / "v1_seed2024" / "summary.json",
    )
    parser.add_argument(
        "--cross-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tokenizers" / "diff_crosspse"
        / "Beauty" / "v3_seed2024" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports"
        / "diff_pse_refinement_tokenizers" / "summary.json",
    )
    args = parser.parse_args()
    refined = load(args.refined_summary)
    cross = load(args.cross_summary)

    paired_fields = (
        "n_digit", "codebook_size", "seed", "validation_items",
        "validation_fraction", "validation_split_seed", "validation_source",
        "selection_protocol", "selection_items",
        "codebook_lr", "embedding_sha256", "items_sha256", "fit_mask_sha256",
        "base_pse_semantic_ids_sha256", "base_pse_codebooks_sha256",
    )
    mismatches = {
        field: {"refined_pse": refined.get(field), "cross_pse_v3": cross.get(field)}
        for field in paired_fields
        if refined.get(field) != cross.get(field)
    }
    if refined.get("mixer_enabled") is not False:
        mismatches["refined_mixer_disabled"] = refined.get("mixer_enabled")
    if cross.get("mixer_enabled") is not True:
        mismatches["cross_v3_mixer_enabled"] = cross.get("mixer_enabled")
    if not refined.get("base_codebooks_trainable"):
        mismatches["refined_base_trainable"] = False
    if not cross.get("base_codebooks_trainable"):
        mismatches["cross_v3_base_trainable"] = False

    result = {
        "paired_control_pass": not mismatches,
        "mismatches": mismatches,
        "refined_pse": compact(refined),
        "cross_pse_v3": compact(cross),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved paired tokenizer audit to {args.output.resolve()}")
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
