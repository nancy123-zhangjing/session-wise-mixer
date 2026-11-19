"""Independent stdlib audit of the 89-run budget, curves and validation selections."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def parameter_count(config):
    d, m = config["dim"], config["ffn_multiplier"]
    embedding = (config["n_items"] + 1 + config["max_length"]) * d
    block = ((6 + 3*m) * d*d + 10*d if config["model"] == "esasrec"
             else (4 + 2*m) * d*d + (9+m)*d)
    return embedding + config["layers"] * block


def audit_curves(config, result, rows, n_train, n_validation):
    if len(rows) != result["epochs"] or not 1 <= len(rows) <= config["max_epochs"]:
        raise ValueError("Curve/epoch budget mismatch")
    batch_updates = math.ceil(n_train / config["batch_size"])
    for epoch, row in enumerate(rows, 1):
        if (int(row["epoch"]) != epoch or int(row["train_samples"]) != n_train
                or int(row["validation_samples"]) != n_validation
                or int(row["updates"]) != epoch * batch_updates):
            raise ValueError("Partial epoch, missing validation or update count mismatch")
        for name in ("train_loss", "validation_NLL", "validation_NDCG@10", "validation_HR@10"):
            if not math.isfinite(float(row[name])) or float(row[name]) < 0:
                raise ValueError("Nonfinite/negative curve value")
    best = max(rows, key=lambda r: (float(r["validation_NDCG@10"]), float(r["validation_HR@10"])))
    if int(best["epoch"]) != result["best_epoch"]:
        raise ValueError("Wrong best validation epoch")
    if result["updates"] != len(rows) * batch_updates or result["parameter_count"] != parameter_count(config):
        raise ValueError("Optimizer budget or parameter count differs")
    if len(rows) < config["max_epochs"] and len(rows) - result["best_epoch"] < config["patience"]:
        raise ValueError("Premature early stopping")
    for key, value in result["best_validation"].items():
        if abs(float(best["validation_" + key]) - value) > 1e-12:
            raise ValueError("Reported best validation differs from raw curves")
    return {"epochs": len(rows), "best_epoch": result["best_epoch"], "updates": result["updates"]}


def selection_from_results(tasks):
    selected, top3 = {}, {}
    for model in ("sasrec_fullce", "esasrec"):
        searches = [t for t in tasks if t["phase"] == "search" and t["config"]["model"] == model]
        if len(searches) != 32 or {t["seed"] for t in searches} != {2024}:
            raise ValueError("Search population/seed changed")
        ordered = sorted(searches, key=lambda t: (-t["result"]["best_validation"]["NDCG@10"],
            -t["result"]["best_validation"]["HR@10"], t["result"]["parameter_count"], t["trial_id"]))[:3]
        top3[model] = [t["trial_id"] for t in ordered]
        candidates = []
        confirmations = [t for t in tasks if t["phase"] == "confirmation" and t["config"]["model"] == model]
        if len(confirmations) != 6 or {t["trial_id"] for t in confirmations} != set(top3[model]):
            raise ValueError("Confirmation did not follow complete Top-3")
        for base in ordered:
            group = [base] + [t for t in confirmations if t["trial_id"] == base["trial_id"]]
            if len(group) != 3 or {t["seed"] for t in group} != {2024, 2025, 2026}:
                raise ValueError("Confirmation seeds incomplete")
            if any(t["config_sha256"] != base["config_sha256"] for t in group):
                raise ValueError("Confirmation silently changed config")
            mean_ndcg = statistics.mean(t["result"]["best_validation"]["NDCG@10"] for t in group)
            mean_hr = statistics.mean(t["result"]["best_validation"]["HR@10"] for t in group)
            candidates.append((-mean_ndcg, -mean_hr, parameter_count(base["config"]), base["trial_id"], base))
        base = min(candidates, key=lambda c: c[:4])[-1]
        formal = [t for t in tasks if t["phase"] == "formal" and t["config"]["model"] == model]
        if len(formal) != 5 or {t["seed"] for t in formal} != set(range(3031, 3036)):
            raise ValueError("Formal fresh seeds incomplete")
        if any(t["config_sha256"] != base["config_sha256"] for t in formal):
            raise ValueError("Formal configuration not locked by confirmation")
        selected[model] = {"config_sha256": base["config_sha256"], "trial_id": base["trial_id"],
            "mean_ndcg": statistics.mean(t["result"]["best_validation"]["NDCG@10"] for t in formal),
            "mean_hr": statistics.mean(t["result"]["best_validation"]["HR@10"] for t in formal),
            "parameters": parameter_count(base["config"])}
    choice = min(selected, key=lambda m: (-selected[m]["mean_ndcg"], -selected[m]["mean_hr"],
                                          selected[m]["parameters"], m))
    return selected, top3, choice


def audit_campaign(campaign, manifest_path):
    campaign = Path(campaign)
    state, selection, manifest = read(campaign / "queue.json"), read(campaign / "FROZEN_SELECTION.json"), read(manifest_path)
    tasks = state["tasks"]
    if len(tasks) != 89 or len({t["task_id"] for t in tasks}) != 89 or state["workers"]:
        raise ValueError("Budget not exactly 89 terminal distinct tasks")
    if {phase: sum(t["phase"] == phase for t in tasks) for phase in ("search", "confirmation", "formal", "legacy")} != {
            "search": 64, "confirmation": 12, "formal": 10, "legacy": 3}:
        raise ValueError("Wrong phase budgets")
    expected = {t["trial_id"]: t for t in manifest["trials"]}
    for task in tasks:
        base = expected[task["trial_id"]]
        if (task["status"] != "succeeded" or task["config"] != base["config"]
                or task["config_sha256"] != fingerprint(task["config"])):
            raise ValueError("Unverified task or config changed from fixed search manifest")
        output = Path(task["output"])
        result = read(output / "COMPLETE.json")
        if result != task["result"] or result["status"] != "SUCCEEDED" or result["smoke"]:
            raise ValueError("Queue result not a real completed full training run")
        if result["binding"]["source_sha256"] != state["source_sha256"]:
            raise ValueError("Mixed execution source versions")
        n_train = 105668 if task["phase"] == "legacy" else 131413
        expected_protocol = "legacy_prefix_v1" if task["phase"] == "legacy" else "full_train_prefix_v1"
        if task["protocol"] != expected_protocol or result["binding"]["protocol"] != expected_protocol:
            raise ValueError("Training protocol changed")
        with (output / "curves.csv").open(newline="", encoding="utf-8") as stream:
            audit_curves(task["config"], result, list(csv.DictReader(stream)), n_train, 22363)
    selected, top3, choice = selection_from_results(tasks)
    if top3 != state["top3"] or choice != selection["selected_model"]:
        raise ValueError("Independent validation selection differs")
    for model, picked in selected.items():
        if picked["config_sha256"] != selection["models"][model]["config_sha256"]:
            raise ValueError("Sealed config differs from independent selection")
    legacy = [t for t in tasks if t["phase"] == "legacy"]
    if ({t["seed"] for t in legacy} != {3031, 3032, 3033}
            or any(t["config"]["model"] != "sasrec_fullce" or t["config_sha256"] !=
                   selected["sasrec_fullce"]["config_sha256"] for t in legacy)):
        raise ValueError("Legacy paired configuration/seed changed")
    if (selection["validation_only"] is not True or selection["test_unlocked"] is not True
            or selection["training_results_sha256"] != fingerprint([
                {"task_id": t["task_id"], "result": t["result"]} for t in tasks])):
        raise ValueError("Frozen validation-only selection seal differs")
    return {"status": "PASS", "verified_training_runs": 89, "selected_model": choice,
            "top3": top3, "independent_selected_configs": selected,
            "source_sha256": state["source_sha256"], "raw_curve_audits": 89}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = audit_campaign(args.campaign, args.manifest)
    with Path(args.report).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
