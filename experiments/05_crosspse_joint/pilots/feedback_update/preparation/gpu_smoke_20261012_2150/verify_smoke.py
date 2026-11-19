"""Read-only acceptance gate for the three-arm stage-05 GPU feedback smoke.

Uses only Python's standard library. A passing receipt is created exclusively
outside the smoke directory; training artifacts are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


PROTOCOL = "stage05_diger_feedback_once_freeze_recover_v1"
ARMS = ("frozen", "cross_recon", "cross_rec")
WEIGHTS = {"frozen": 1.0, "cross_recon": 0.0, "cross_rec": 1.0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def check_hash(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid SHA256: {label}")


def check_files(directory, hashes):
    require(isinstance(hashes, dict) and hashes, f"No bound files in {directory}")
    for name, expected in hashes.items():
        require(isinstance(name, str) and Path(name).name == name, "Artifact file name must be a single path component")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"Missing or linked artifact: {path}")
        check_hash(expected, name)
        require(sha256(path) == expected, f"Artifact hash mismatch: {path}")


def check_routing(row, phase):
    sides = row["last_gradient_sides"]
    active = "tokenizer" if phase == "tokenizer" else "recommender"
    inactive = "recommender" if phase == "tokenizer" else "tokenizer"
    require(math.isfinite(sides[active]) and sides[active] > 0, f"Missing active gradient: {phase}")
    require(sides[inactive] == 0.0, f"Unexpected frozen-side gradient: {phase}")


def verify(output, expected_commit, expected_gpu="H100"):
    output = Path(output).resolve(strict=True)
    require(not (output / "FAILED.json").exists(), "Smoke has FAILED.json")
    summary = read_json(output / "summary.json")
    complete = read_json(output / "COMPLETE.json")
    provenance = read_json(output / "provenance.json")
    protocol = read_json(output / "protocol.json")
    post = read_json(output / "post_input_verification.json")
    summary_hash = sha256(output / "summary.json")
    require(complete["summary_sha256"] == summary_hash, "COMPLETE does not bind the current summary")
    require(complete["arms"] == list(ARMS), "COMPLETE does not list all three ordered arms")
    require(complete["all_protocol_arms_complete"] is True, "COMPLETE is a partial run")
    require(summary["protocol_id"] == protocol["protocol_id"] == PROTOCOL, "Unexpected protocol")
    require(summary["mode"] == provenance["mode"] == "smoke", "This gate accepts smoke only")
    require(summary["status"] == "complete" and summary["all_protocol_arms_complete"] is True, "Incomplete smoke")
    require(summary["no_test_evaluation"] is True and summary["no_historical_artifact_changes"] is True, "Evidence preservation/evaluation guard failed")
    require(provenance["smoke_not_for_metric_comparison"] is True, "Smoke must not be treated as a metric result")
    require(str(provenance["device"]).startswith("cuda"), "Smoke did not use CUDA")
    require(isinstance(provenance["gpu"], str) and expected_gpu.lower() in provenance["gpu"].lower(), "Smoke used an unexpected GPU")
    require(provenance["tokenizer_variant"] == "cross_pse", "Smoke did not retain Cross-PSE interaction")
    require(provenance["tokenizer_forward_precision"] == "float32_tf32_disabled", "Unexpected tokenizer precision")
    require(provenance["decoder_forward_precision"] == "bf16", "Unexpected decoder precision")
    versions = provenance["source"]["git"]
    require(versions["project"]["head"] == expected_commit, "Smoke did not execute the expected project commit")
    for name in ("project", "diffgrm"):
        require(versions[name]["status"] == "" and versions[name]["diff"] == "", f"Dirty source at execution: {name}")
    require(len(provenance["assets"]) == 18, "Expected all 18 bound input receipts")
    require(post["verified"] is True and post["inputs"] == provenance["assets"], "Input receipts changed or post-check failed")
    require(protocol["arms"] == list(ARMS), "Protocol arm ordering changed")
    require(protocol["noise_start"] == protocol["noise_end"] == 0.0, "Feedback smoke must have no Gumbel noise")
    require(protocol["recommendation_weights"] == WEIGHTS, "Feedback weights changed")
    require(protocol["smoke"]["tokenizer_batches"] == protocol["smoke"]["decoder_batches"] == 16, "Unexpected smoke microbatch budget")
    require(protocol["smoke"]["decoder_epochs"] == 1 and protocol["smoke"]["eval_batches"] == 2, "Unexpected recovery/evaluation smoke budget")
    require(protocol["smoke"]["tokenizer_probe_updates"] == [1], "Unexpected smoke probe budget")
    require(protocol["gradient_accumulation_steps"] == 16, "Unexpected accumulation")
    arms = summary["arms"]
    require([arm["arm"] for arm in arms] == list(ARMS), "Summary arm ordering changed")
    source_hash = provenance["shared_model_tensor_sha256"]
    check_hash(source_hash, "shared_model_tensor_sha256")
    decoder_rows, tokenizer_rows, initial_sids, initial_books = [], [], [], []
    for arm in arms:
        name = arm["arm"]
        arm_dir = output / name
        require(read_json(arm_dir / "metrics.json") == arm, f"Per-arm metrics differ from summary: {name}")
        settings = arm["settings"]
        require(read_json(arm_dir / "settings.json") == settings, f"Settings differ: {name}")
        expected_updates = 0 if name == "frozen" else 1
        require(settings["recommendation_weight"] == WEIGHTS[name] and settings["audit_recommendation_feedback"] is True, f"Feedback weight/audit changed: {name}")
        require(settings["tokenizer_batches"] == 16 * expected_updates and settings["decoder_epochs"] == 1, f"Wrong arm budget: {name}")
        require(settings["decoder_batch_limit"] == 16 and settings["eval_batch_limit"] == 2, f"Wrong smoke limits: {name}")
        require(settings["noise_start"] == settings["noise_end"] == 0.0, f"Wrong noise: {name}")
        require(settings["tokenizer_probe_updates"] == ([] if name == "frozen" else [1]), f"Wrong probes: {name}")
        training = arm["training"]
        require([row["phase"] for row in training] == (["recommender"] if name == "frozen" else ["tokenizer", "recommender"]), f"Unexpected phase order: {name}")
        for row in training:
            require(row["batches"] == 16 and row["optimizer_updates"] == 1 and row["accumulation"] == 16, f"Incomplete optimizer budget: {name}/{row['phase']}")
            check_routing(row, row["phase"])
            if row["phase"] == "tokenizer":
                tokenizer_rows.append(row)
                require(row["noise_applied_calls"] == 0 and row["gumbel_calls"] == row["assignment_training_enabled_calls"] == 32, f"Assignment mode/noise failed: {name}")
                audit = row["feedback_gradient_audit"]
                raw, weighted = audit["raw_direct_gradient_norm"], audit["weighted_direct_gradient_norm"]
                require(audit["recommendation_weight"] == WEIGHTS[name] and audit["measured_at_microbatch"] == 1, f"Missing gradient audit: {name}")
                require(math.isfinite(raw) and math.isfinite(weighted) and raw > 0, f"Invalid direct gradient: {name}")
                require((weighted <= 1e-8 if name == "cross_recon" else weighted > 0 and math.isclose(weighted, raw, rel_tol=1e-4, abs_tol=1e-7)), f"Wrong weighted recommendation gradient: {name}")
            else:
                require(row["logical_index"] == 1, f"Unexpected recovery index: {name}")
                decoder_rows.append(row)
        snaps = arm["snapshots"]
        require([snap["label"] for snap in snaps] == ["initial", "post_tokenizer", "decoder_1"], f"Incomplete snapshots: {name}")
        for snap in snaps:
            require(read_json(arm_dir / "snapshots" / snap["label"] / "summary.json") == snap, f"Snapshot summary differs: {name}/{snap['label']}")
            require(set(snap["files"]) == {"model.pt", "training_state.pt", "semantic_ids.npy", "crossed_codebooks.npy"}, f"Incomplete snapshot files: {name}/{snap['label']}")
            check_files(arm_dir / "snapshots" / snap["label"], snap["files"])
        initial, updated, recovered = snaps
        require(initial["model_tensor_state_sha256"] == source_hash, f"Non-shared initial model: {name}")
        require(initial["tokenizer_updates"] == initial["recommender_updates"] == initial["recommender_epochs"] == 0, f"Initial counters changed: {name}")
        require(updated["tokenizer_updates"] == expected_updates and updated["recommender_updates"] == updated["recommender_epochs"] == 0, f"Wrong tokenizer counters: {name}")
        require(recovered["tokenizer_updates"] == expected_updates and recovered["recommender_updates"] == recovered["recommender_epochs"] == 1, f"Wrong recovery counters: {name}")
        require(recovered["semantic_ids_array_sha256"] == updated["semantic_ids_array_sha256"], f"SID changed during frozen recovery: {name}")
        require(recovered["files"]["crossed_codebooks.npy"] == updated["files"]["crossed_codebooks.npy"], f"Codebooks changed during frozen recovery: {name}")
        initial_sids.append(initial["semantic_ids_array_sha256"])
        initial_books.append(initial["files"]["crossed_codebooks.npy"])
        probes = arm["tokenizer_probes"]
        require(len(probes) == expected_updates, f"Missing probe: {name}")
        if expected_updates:
            probe = probes[0]
            require(probe["optimizer_updates"] == 1 and probe["microbatches"] == 16, f"Wrong probe timing: {name}")
            require(probe["diagnostic_only_not_resume_checkpoint"] is True and probe["no_mapping_refresh_no_rng_or_model_state_change"] is True, f"Probe side effects: {name}")
            require(probe["semantic_ids_array_sha256"] == updated["semantic_ids_array_sha256"], f"Fresh probe/deployed SID mismatch: {name}")
            require(set(probe["files"]) == {"fresh_probe_codes.npy", "crossed_codebooks.npy"}, f"Incomplete probe files: {name}")
            require(probe["files"]["fresh_probe_codes.npy"] == updated["files"]["semantic_ids.npy"]
                    and probe["files"]["crossed_codebooks.npy"] == updated["files"]["crossed_codebooks.npy"], f"Probe/snapshot file mismatch: {name}")
            probe_dir = arm_dir / "tokenizer_probes" / "update0001"
            require(read_json(probe_dir / "summary.json") == probe, f"Probe summary differs: {name}")
            check_files(probe_dir, probe["files"])
        else:
            require(updated["model_tensor_state_sha256"] == initial["model_tensor_state_sha256"] and updated["semantic_ids_array_sha256"] == initial["semantic_ids_array_sha256"], "Frozen control changed before recovery")
        require([row["label"] for row in arm["validation"]] == ["initial", "post_tokenizer", "decoder_1"], f"Validation coverage incomplete: {name}")
        require(arm["final_snapshot"] == "snapshots/decoder_1", f"Wrong final snapshot: {name}")
    require(len(set(initial_sids)) == len(set(initial_books)) == 1, "Initial SIDs/codebooks differ across arms")
    for rows, keys in ((decoder_rows, ("logical_index", "batches", "optimizer_updates", "phase_seed", "data_seed", "data_order_sha256")),
                       (tokenizer_rows, ("batches", "optimizer_updates", "phase_seed", "data_seed", "data_order_sha256"))):
        for row in rows[1:]:
            for key in keys:
                require(row[key] == rows[0][key], f"Unpaired phase control: {key}")
    return {"gate": "stage05_feedback_gpu_smoke_v1", "passed": True,
            "status": "passed", "commit": expected_commit,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "smoke_output": str(output), "summary_sha256": summary_hash,
            "provenance_sha256": sha256(output / "provenance.json"),
            "complete_sha256": sha256(output / "COMPLETE.json"),
            "expected_commit": expected_commit, "gpu": provenance["gpu"],
            "arms": list(ARMS), "purpose": "plumbing acceptance only; not a recommendation metric comparison"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu", default="H100")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.expected_commit), "--expected-commit must be the full Git SHA")
    receipt = args.receipt.resolve()
    output = args.output.resolve()
    require(not receipt.is_relative_to(output), "Receipt must be outside training artifacts")
    require(not receipt.exists() and not args.receipt.is_symlink(), "Refusing to overwrite a receipt")
    result = verify(output, args.expected_commit, args.expected_gpu)
    with receipt.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
