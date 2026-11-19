"""Print a compact Cross-PSE tokenizer readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-protocol")
    parser.add_argument("--expected-selection-protocol")
    parser.add_argument("--min-item-change", type=float, default=0.005)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    base = summary["base_pse"]
    utilizations = [row["utilization_rate"] for row in summary["per_position"]]
    item_change = float(summary["item_sid_change_rate_from_base"])
    digit_change = float(summary["digit_assignment_change_rate_from_base"])
    relative_delta = float(summary.get("relative_codebook_delta_rms") or 0.0)
    validation_items = int(summary.get("validation_items") or 0)
    reconstruction_delta = float(summary["catalog_reconstruction_mse_sum"]) - float(
        summary["base_catalog_reconstruction_mse_sum"]
    )
    collision_delta = float(summary["collision_rate"]) - float(
        base["collision_rate"]
    )
    initial_validation = float(
        summary.get("initial_validation_reconstruction")
        or summary["best_validation_reconstruction"]
    )
    best_validation = float(summary["best_validation_reconstruction"])
    validation_improvement = initial_validation - best_validation
    expected_protocol = args.expected_protocol or summary.get(
        "reproduction_protocol"
    )
    selection_protocol = summary.get("selection_protocol")
    selection_items = int(
        summary.get("selection_items") or summary.get("validation_items") or 0
    )
    train_items = int(summary.get("train_items") or 0)

    checks = {
        "expected_protocol": summary.get("reproduction_protocol")
        == expected_protocol,
        "expected_selection_protocol": (
            args.expected_selection_protocol is None
            or selection_protocol == args.expected_selection_protocol
        ),
        "catalog_selection_invariant": (
            selection_protocol != "fit_catalog_reconstruction"
            or (
                bool(summary.get("selection_equals_training_catalog"))
                and selection_items == train_items
                and float(summary.get("validation_fraction") or 0.0) == 0.0
            )
        ),
        "stable_validation_size": validation_items >= 1000,
        "nontrivial_codebook_delta": relative_delta >= 1e-4,
        "nontrivial_sid_change": item_change >= args.min_item_change,
        "validation_improves_step0_pse": bool(
            summary.get("best_improves_initial", validation_improvement > 1e-6)
        ),
        "no_codebook_collapse": min(utilizations) >= 0.8,
        "no_large_collision_regression": collision_delta <= 0.10,
    }
    print(f"summary: {args.summary.resolve()}")
    print(f"protocol: {summary.get('reproduction_protocol')}")
    print(f"selection protocol: {selection_protocol}")
    print(f"selection source: {summary.get('validation_source')}")
    print(f"train/selection items: {train_items}/{selection_items}")
    print(f"tokenizer type: {summary.get('tokenizer_type')}")
    print(f"base codebooks trainable: {summary.get('base_codebooks_trainable')}")
    print(f"mixer enabled: {summary.get('mixer_enabled', True)}")
    print(f"best step: {summary.get('best_step')}")
    print(f"validation items: {validation_items}")
    print(f"step-0 validation reconstruction: {initial_validation:.8f}")
    print(f"best validation reconstruction: {best_validation:.8f}")
    print(f"validation improvement: {validation_improvement:+.8f}")
    print(f"relative codebook delta RMS: {relative_delta:.8f}")
    print(
        "base/mixer delta RMS: "
        f"{float(summary.get('base_parameter_delta_rms') or 0.0):.8f} / "
        f"{float(summary.get('mixer_delta_rms') or 0.0):.8f}"
    )
    print(
        "codebook/mixer LR: "
        f"{summary.get('codebook_lr')} / {summary.get('mixer_lr')}"
    )
    print(f"digit assignment change: {digit_change:.4%}")
    print(f"item SID change: {item_change:.4%}")
    print(f"reconstruction delta vs PSE: {reconstruction_delta:+.8f}")
    print(f"collision delta vs PSE: {collision_delta:+.4%}")
    print("utilization: " + ", ".join(f"{value:.2%}" for value in utilizations))
    for name, passed in checks.items():
        print(f"[{('PASS' if passed else 'WARN')}] {name}")
    ready = all(checks.values())
    print(f"READINESS: {'PASS' if ready else 'REVIEW'}")
    if args.strict and not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
