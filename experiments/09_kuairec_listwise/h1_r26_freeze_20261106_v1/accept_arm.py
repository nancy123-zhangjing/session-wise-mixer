"""CPU-only independent result acceptance. Never train or decode predictions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
os.environ["DIFFGRM_ENGINE_ROOT"] = str(ROOT / "code/external/DiffGRM")
from audit_analysis_core import audit_predictions
from audit_recovery_support import audit_recovery_predictions
from scripts.listwise_diffgrm.recovery import fingerprint
from scripts.listwise_diffgrm.recovery_stability_protocol import ARMS, coefficient, load_spec
from scripts.listwise_diffgrm.run import phase_indices
from scripts.listwise_diffgrm.training import phase_plan


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(path).with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def audit_log(output, spec, config):
    plan = phase_plan(config, 709443)
    if len(plan) != 45 or sum(row["updates"] for row in plan if row["kind"] == "T") != 400:
        raise ValueError("Original full horizon changed")
    start = next(i + 1 for i, row in enumerate(plan) if row["kind"] == "R" and row["epoch"] == spec["start_r"])
    logged, repeats = {}, 0
    with (output / "training.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["phase_index"], row["update"])
            normalized = {k: v for k, v in row.items() if k != "seconds"}
            if key in logged:
                if normalized != logged[key]:
                    raise ValueError("A replayed optimizer step changed its logged values")
                repeats += 1
            logged[key] = normalized
    r, t, seen = spec["start_r"] * 693, spec["start_t"], set()
    for phase in range(start, len(plan)):
        row = plan[phase]
        if row["kind"] == "T" and spec["mode"] == "freeze":
            continue
        order = phase_indices(config, 709443, phase, row)
        for update in range(1, row["updates"] + 1):
            key = (phase, update)
            event = logged[key]
            seen.add(key)
            used_alpha = coefficient(spec, t - spec["start_t"]) if row["kind"] == "T" else None
            r += row["kind"] == "R"
            t += row["kind"] == "T"
            if (event["kind"], event["epoch"], event["r_updates"], event["t_updates"]) != (row["kind"], row["epoch"], r, t):
                raise ValueError("Logged phase/update budget mismatch")
            batch = config["training"]["effective_batch_size"]
            ids = order[(update - 1) * batch:update * batch]
            if hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() != event["sample_indices_sha256"]:
                raise ValueError("Training sample order changed")
            if event["increment_alpha_used"] != used_alpha:
                raise ValueError("Mixer coefficient schedule changed")
            if not math.isfinite(event["loss"]):
                raise ValueError("Nonfinite training loss")
            for key_lr in ("codebook", "mixer"):
                if event["learning_rates_used"][key_lr] != config["training"][key_lr + "_lr"]:
                    raise ValueError("Tokenizer LR changed")
            if event["mixer_gate"]["completed"] != t - spec["start_t"]:
                raise ValueError("Gate clock changed")
            if spec["mode"] == "freeze" and any(event["mixer_gate"][k] != 0 for k in (
                "base_displacement", "accepted_mixer_displacement", "effective_codebook_displacement")):
                raise ValueError("Frozen encoding drifted")
    if seen != set(logged) or (r, t) != (35 * 693, spec["stop_t"]):
        raise ValueError("Missing or extra optimizer steps")
    return dict(R_added=r - spec["start_r"] * 693, T_added=t - spec["start_t"],
                R_total=r, T_total=t, unique_updates=len(seen), replayed_log_rows=repeats,
                sample_order_verified=True, coefficient_verified=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    spec = load_spec(ROOT / "configs" / f"{args.arm}.json")
    output, dest = ROOT / "runs" / args.arm, ROOT / "acceptance" / args.arm
    dest.mkdir(parents=True, exist_ok=True)
    result_path = dest / "RESULT.json"
    if result_path.exists():
        raise FileExistsError("Already accepted; verify existing receipts instead of repeating audits")
    jobs = read(ROOT / "jobs" / args.arm / "result.json")
    if jobs["status"] != "training_test_restore_complete_pending_result_audit" or any(x["exit_code"] for x in jobs["stages"]):
        raise ValueError("Train/test/restore stages have not all succeeded")
    request = read(output / "request.json")
    if request["spec"] != spec:
        raise ValueError("Wrong requested arm")
    inputs = read(ROOT / "INPUTS.json")
    source = read(ROOT / "SOURCE_MANIFEST.json")
    for name, expected in source["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Versioned source changed: {name}")
    completion = read(output / "complete.json")
    if completion["spec"] != spec or (completion["r_updates"], completion["t_updates"]) != (35 * 693, spec["stop_t"]):
        raise ValueError("Completion counters changed")
    if digest(output / "last.pt") != completion["last_sha256"]:
        raise ValueError("Final checkpoint changed")
    updates = audit_log(output, spec, request["execution_binding"]["config"])
    subset = read(ROOT / "validation_subset_8192.json")
    expected_labels = {f"R{epoch}" for epoch in range(spec["start_r"] + 1, 36)}
    if spec["mode"] != "freeze":
        expected_labels |= {f"after_T{t}" for t in range(spec["start_t"] + 40, 401, 40)}
    actual_labels = {p.name for p in (output / "evaluations").iterdir() if (p / "complete.json").is_file()}
    if actual_labels != expected_labels:
        raise ValueError("Missing/extra validation nodes")
    validation = {}
    for label in sorted(expected_labels):
        receipt = read(output / "evaluations" / label / "complete.json")
        path = output / "evaluations" / label / receipt["attempt"] / "per_sample.jsonl.gz"
        if digest(path) != receipt["predictions_sha256"] or digest(path.parent / "summary.json") != receipt["summary_sha256"]:
            raise ValueError("Validation artifact hashes changed")
        if receipt["binding"]["source"] != fingerprint(request["execution_binding"]):
            raise ValueError("Validation source changed")
        full = label.startswith("R") and int(label[1:]) in spec["validation_epochs"]
        if receipt["binding"]["full"] != full:
            raise ValueError("Full validation grid changed")
        gate_binding = read(output / "evaluation_gate_bindings" / f"{label}.json")
        if gate_binding["protocol"] != request["protocol"]:
            raise ValueError("Evaluation arm/gate protocol changed")
        audit_path = dest / f"{label}_audit.json"
        if audit_path.exists():
            audit = read(audit_path)
            if audit["predictions_sha256"] != digest(path) or audit["summary_sha256"] != digest(path.parent / "summary.json"):
                raise ValueError("Existing audit input changed")
        else:
            audit = audit_recovery_predictions(path, dataset_dir=inputs["dataset"], subset_manifest=subset,
                                               require_extracted_subset=full)
            write(audit_path, audit)
        if not audit["passed"] or not audit["source_labels_verified"] or audit["examples"] != (71657 if full else 8192):
            raise ValueError(f"Independent validation audit failed: {label}")
        validation[label] = dict(examples=audit["examples"], metrics=audit["metrics"], full=full)
        print(json.dumps(dict(arm=args.arm, accepted_validation=label, ndcg5=audit["metrics"]["list/ndcg@5"])), flush=True)
    selection = read(output / "selection.json")
    records = {str(e): read(output / "validation_records" / f"R{e}.json") for e in spec["validation_epochs"]}
    best = min(spec["validation_epochs"], key=lambda e: (-records[str(e)]["ndcg5"], e))
    if selection["candidates"] != records or selection["selected"] != records[str(best)]:
        raise ValueError("Best checkpoint selection differs from prespecified validation grid")
    restore = read(output / "restore_verified.json")
    if restore["status"] != "passed" or restore["spec"] != spec or restore["updates_performed"] != 0:
        raise ValueError("Full checkpoint restoration failed")
    if {x["epoch"] for x in restore["checkpoints"]} != set(spec["validation_epochs"]):
        raise ValueError("Restore receipt omitted checkpoints")
    for epoch, record in records.items():
        if digest(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError("Selected/retained checkpoint hash changed")
        restored = next(x for x in restore["checkpoints"] if x["epoch"] == int(epoch))
        if restored["sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Restore receipt checkpoint mismatch")
        checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        payload = checkpoint["future_gate"]["payload"]
        if (checkpoint["protocol"] != request["protocol"] or payload["spec"] != spec
                or fingerprint(payload) != checkpoint["future_gate"]["sha256"]
                or payload["completed"] != checkpoint["inner"]["t_updates"] - spec["start_t"]):
            raise ValueError("Retained checkpoint gate/protocol/clock mismatch")
        if int(epoch) == spec["start_r"]:
            initial = checkpoint
            parent = torch.load(request["parent"]["checkpoint"], map_location="cpu", weights_only=False)
            if digest(request["parent"]["checkpoint"]) != request["parent"]["sha256"]:
                raise ValueError("Authenticated parent file changed")
            for key in ("model", "alignment", "r_optimizer", "t_optimizer", "scheduler", "rng"):
                if fingerprint(initial["inner"][key]) != fingerprint(parent["inner"][key]):
                    raise ValueError(f"Fork did not preserve parent {key}")
            initial_anchor = fingerprint(payload["anchor"])
            initial_frozen = {key: fingerprint(initial["inner"][key]) for key in ("alignment", "t_optimizer")}
            del parent, initial
        else:
            if fingerprint(payload["anchor"]) != initial_anchor:
                raise ValueError("Fork anchor changed during training")
            gate_binding = read(output / "evaluation_gate_bindings" / f"R{epoch}.json")
            if gate_binding["gate"] != fingerprint(checkpoint["future_gate"]):
                raise ValueError("Full validation was not bound to this gate state")
        if spec["mode"] == "freeze" and any(
            fingerprint(checkpoint["inner"][key]) != expected for key, expected in initial_frozen.items()
        ):
            raise ValueError("Frozen checkpoint changed tokenizer or optimizer")
        del checkpoint
    test_receipt = read(output / "test_predictions_complete.json")
    if test_receipt["selection_sha256"] != digest(output / "selection.json"):
        raise ValueError("Test selection changed")
    if {row["epoch"] for row in test_receipt["results"]} != {best, 35}:
        raise ValueError("Wrong test endpoints")
    tests = {}
    for row in test_receipt["results"]:
        path = Path(row["summary"]).parent / "per_sample.jsonl.gz"
        for name, expected in row["files"].items():
            if digest(path.parent / name) != expected:
                raise ValueError("Test prediction/summary checksum changed")
        audit_path = dest / f"test_R{row['epoch']}_audit.json"
        if audit_path.exists():
            audit = read(audit_path)
            if audit["predictions_sha256"] != digest(path):
                raise ValueError("Existing test audit input changed")
        else:
            audit = audit_predictions(path, dataset_dir=inputs["dataset"])
            write(audit_path, audit)
        if not audit["passed"] or not audit["source_labels_verified"] or audit["examples"] != 53194:
            raise ValueError("Independent test audit failed")
        tests[str(row["epoch"])] = dict(metrics=audit["metrics"], examples=audit["examples"])
        print(json.dumps(dict(arm=args.arm, accepted_test=row["epoch"], ndcg5=audit["metrics"]["list/ndcg@5"])), flush=True)
    artifacts = {}
    for directory in (output, ROOT / "jobs" / args.arm):
        for path in directory.rglob("*"):
            if path.is_file():
                artifacts[path.relative_to(ROOT).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
    write(dest / "ARTIFACT_MANIFEST.json", artifacts)
    result = dict(status="passed", arm=args.arm, spec=spec, updates=updates, source_unchanged=True,
                  full_checkpoint_restore_verified=True, best_epoch=best, validation=validation, tests=tests,
                  parent=request["parent"], artifact_manifest_sha256=digest(dest / "ARTIFACT_MANIFEST.json"),
                  limitation="Single seed. Platform terminal state and GPU release are checked separately.")
    write(result_path, result)
    print(json.dumps(dict(status="passed", arm=args.arm, best_epoch=best,
                         best_test=tests[str(best)]["metrics"]["list/ndcg@5"],
                         final_test=tests["35"]["metrics"]["list/ndcg@5"])), flush=True)


if __name__ == "__main__":
    main()
