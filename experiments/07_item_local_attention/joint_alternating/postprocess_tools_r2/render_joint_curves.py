"""Plot completed formal joint runs from a packer-verified local backup on CPU.

Three panels: recommender-phase recommendation loss and validation HR/NDCG@10.
Tokenizer loss and test observations are never used for plotting/selection.
This is a visualization, not the paired experiment or recovery audit.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

CAMPAIGN = "stage07_item_local_alternating_joint_seed2024_20261020_r1"
FORMAT = "stage07_item_local_joint_delivery_v1"
ARMS = ("control", "item_local")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def number(value):
    need(not isinstance(value, bool), "Boolean numeric observation")
    result = float(value)
    need(math.isfinite(result), "Nonfinite curve observation")
    return result


def integer(value):
    result = number(value)
    need(result.is_integer() and result >= 0, "Invalid logical epoch")
    return int(result)


def table(path, columns):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        need(columns <= set(reader.fieldnames or []), "Missing curve CSV columns: " + str(path))
        return list(reader)


def phase(epoch):
    return "tokenizer" if epoch > 5 and (epoch - 6) % 4 == 0 else "recommender"


def parse_joint(path, completed):
    rows = table(path, {"epoch", "phase", "recommendation_loss"})
    need([integer(row["epoch"]) for row in rows] == list(range(1, completed + 1)),
         "Joint curve must contain each committed logical epoch once")
    loss = []
    for row in rows:
        epoch = integer(row["epoch"])
        need(row["phase"] == phase(epoch), "Joint phase differs from the alternating recipe")
        if row["phase"] == "recommender":
            value = number(row["recommendation_loss"])
            need(value >= 0, "Negative recommendation loss")
            loss.append(value)
        else:
            loss.append(None)  # Break the line instead of mixing tokenizer loss.
    return {"epochs": list(range(1, completed + 1)), "recommendation_loss": loss,
            "recommender_epochs": sum(value is not None for value in loss),
            "tokenizer_phases_excluded": sum(value is None for value in loss)}


def parse_validation(path, completed):
    observations, ignored_test_rows = {}, 0
    for row in table(path, {"epoch", "split", "metric", "value"}):
        if row["split"] == "test":
            ignored_test_rows += 1
            continue
        need(row["split"] == "val", "Unexpected evaluation split")
        epoch = integer(row["epoch"])
        group = observations.setdefault(epoch, {})
        need(row["metric"] not in group, "Duplicated validation metric")
        group[row["metric"]] = number(row["value"])
    expected, rec_count = [0], 0
    for epoch in range(1, completed + 1):
        if phase(epoch) == "recommender":
            rec_count += 1
            if rec_count % 5 == 0:
                expected.append(epoch)
    need(sorted(observations) == expected, "Missing/unplanned committed validation epochs")
    for group in observations.values():
        if "recall@10" in group:
            need("hit_rate@10" not in group or abs(group["hit_rate@10"] - group["recall@10"]) < 1e-12,
                 "HR/recall aliases disagree")
            group.setdefault("hit_rate@10", group["recall@10"])
        need(all(key in group and 0 <= group[key] <= 1 for key in ("hit_rate@10", "ndcg@10", "weighted_score")),
             "Missing/out-of-range validation observation")
        need(abs(group["weighted_score"] - .8 * group["ndcg@10"] - .2 * group["hit_rate@10"]) < 1e-7,
             "Validation weighted-score formula differs")
    return observations, ignored_test_rows


def selection(observations, fit):
    best = integer(fit["best_epoch"])
    need(best in observations, "Selected epoch has no validation observation")
    # Strict improvement keeps the first maximum; epoch zero is a valid candidate.
    chosen = max(sorted(observations), key=lambda epoch: observations[epoch]["weighted_score"])
    need(best == chosen and abs(number(fit["best_val_score"]) - observations[best]["weighted_score"]) < 1e-7,
         "Selected checkpoint differs from validation weighted-score maximum")
    return {"best_epoch": best, "best_val_score": number(fit["best_val_score"]),
            "rule": "First maximum of validation 0.8*NDCG@10 + 0.2*HR@10, including epoch 0",
            "best_validation": observations[best]}


def collect(root):
    root = Path(root).resolve(strict=True)
    manifest, byte_receipt = read(root / "MANIFEST.json"), read(root / "BYTE_VERIFICATION.json")
    records = {row["member"]: row for row in manifest["files"]}
    need(manifest["format"] == FORMAT and manifest["campaign_id"] == CAMPAIGN
         and byte_receipt["status"] == "BYTE_RESTORATION_VERIFIED"
         and len(records) == len(manifest["files"]) == byte_receipt["files"],
         "Input is not a packer-verified joint backup")
    used = {}
    def checked(member):
        name = PurePosixPath(member)
        need(member in records and not name.is_absolute() and ".." not in name.parts
             and "\\" not in member and ":" not in member, "Unsafe/unlisted plot input")
        path = root.joinpath(*name.parts)
        need(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
             and path.stat().st_size == records[member]["size_bytes"] and sha(path) == records[member]["sha256"],
             "Plot input differs from verified manifest: " + member)
        used[member] = records[member]
        return path
    arms = {}
    for arm in ARMS:
        run_id = "stage07_item_local_joint_" + arm + "_seed2024_20261020_r1"
        prefix = "runs/" + run_id + "/"
        complete = read(checked(prefix + "COMPLETE.json"))
        result_path = checked(prefix + "results.json")
        result = read(result_path)
        need(complete["mode"] == "formal" and complete["status"] == result["status"] == "formal_completed"
             and complete["result_file"] == "results.json" and complete["result_sha256"] == sha(result_path)
             and result["test_data_evaluated"] is True and result["evaluation_level"] == "sid",
             "Both real formal COMPLETE/results are required; smoke cannot substitute")
        for value in (complete, result):
            need(value["campaign_id"] == CAMPAIGN and value["variant"] == arm and value["run_id"] == run_id,
                 "Wrong formal campaign/arm identity")
        best_path = checked(prefix + "training_state/best_model.bin")
        need(sha(best_path) == result["checkpoint_sha256"] == complete["checkpoint_sha256"], "Selected checkpoint SHA differs")
        paths = {}
        for kind in ("joint_metrics", "evaluation_metrics"):
            candidates = [name for name in records if name.startswith(prefix + "diagnostics/") and name.endswith("_" + kind + ".csv")]
            need(len(candidates) == 1, "Expected one formal diagnostic CSV: " + kind)
            paths[kind] = checked(candidates[0])
        completed = integer(result["fit"]["completed_epochs"])
        need(1 <= completed <= 45, "Logical epoch count outside the full joint recipe")
        joint = parse_joint(paths["joint_metrics"], completed)
        validation, ignored = parse_validation(paths["evaluation_metrics"], completed)
        arms[arm] = {"run_id": run_id, "completed_epochs": completed, "joint": joint,
                     "validation": validation, "selection": selection(validation, result["fit"]),
                     "test_rows_excluded_from_plot_and_selection": ignored}
    return arms, {"manifest_sha256": sha(root / "MANIFEST.json"),
                  "byte_verification_receipt_sha256": sha(root / "BYTE_VERIFICATION.json"),
                  "source_identity": manifest["source_identity"], "files": used}


def draw(arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.5), sharex=True, layout="constrained")
    for arm, color, label in (("control", "#2874A6", "Control"), ("item_local", "#D47721", "Item-local attention")):
        data = arms[arm]
        best = data["selection"]["best_epoch"]
        joint, validation = data["joint"], data["validation"]
        axes[0].plot(joint["epochs"], [math.nan if value is None else value for value in joint["recommendation_loss"]],
                     color=color, marker=".", markersize=3, linewidth=1.2, label=label)
        for axis, metric in zip(axes[1:], ("hit_rate@10", "ndcg@10")):
            epochs = sorted(validation)
            axis.plot(epochs, [validation[epoch][metric] for epoch in epochs], color=color,
                      marker="o", markersize=3, linewidth=1.2, label=label)
            axis.scatter([best], [validation[best][metric]], marker="*", s=85, color=color, edgecolors="white", linewidths=.5, zorder=4)
        for axis in axes:
            axis.axvline(best, color=color, alpha=.3, linewidth=.9, linestyle="--")
        if best > 0:
            value = joint["recommendation_loss"][best - 1]
            need(value is not None, "Selected validation epoch belongs to a tokenizer phase")
            axes[0].scatter([best], [value], marker="*", s=85, color=color, edgecolors="white", linewidths=.5, zorder=4)
    for axis, title, ylabel in zip(axes, ("Recommender phases only", "Validation HR@10", "Validation NDCG@10"),
                                    ("Recommendation loss", "HR@10", "NDCG@10")):
        axis.set(title=title, xlabel="Logical epoch", ylabel=ylabel)
        axis.grid(alpha=.16)
        axis.set_xlim(left=-.6)
    axes[0].legend(frameon=False, fontsize=8)
    marks = ", ".join(f"{arm}={arms[arm]['selection']['best_epoch']}" for arm in ARMS)
    figure.suptitle("Alternating joint training  |  validation-selected best epoch: " + marks, fontsize=10)
    for suffix in ("png", "pdf"):
        figure.savefig(output / ("joint_training_curves." + suffix), dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are preserved")
    args = parser.parse_args()
    need(not os.path.lexists(args.output), "Choose a new plot output directory")
    arms, inputs = collect(args.backup_root)
    args.output.mkdir(parents=True, exist_ok=False)
    draw(arms, args.output)
    metadata = {"status": "FORMAL_JOINT_CURVES_RENDERED", "campaign_id": CAMPAIGN,
                "created_utc": datetime.now(timezone.utc).isoformat(), "tool_sha256": sha(__file__),
                "input_identity": inputs, "arms": arms,
                "outputs": {name: sha(args.output / name) for name in ("joint_training_curves.png", "joint_training_curves.pdf")},
                "goal_complete": False, "scope": "Visualization of bound formal CSVs; tokenizer loss and test points excluded. No recovery, research-success or significance audit."}
    with (args.output / "CURVE_INPUTS_AND_SELECTION.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": metadata["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
