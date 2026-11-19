"""CPU readback of every archived .pt/.bin plus all 28 mandatory joint states.

Use after safe extraction. Reads actual model/Adam/scheduler/RNG/SID tensors;
does not construct a model, evaluate data, or execute a relocated resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import multiseed_backup as backup
from checkpoint_payload_checks import inspect_boundary, tensor_digest, tensor_inventory

need, read, sha, write = backup.need, backup.read, backup.sha, backup.write
NECESSARY = ("training_state/last.pt", "training_state/best_model.bin",
             "smoke/training_state/last.pt", "smoke/training_state/best_model.bin",
             "smoke/smoke_tokenizer_boundary.pt", "smoke/recovery_probe/source_last.pt",
             "smoke/recovery_probe/best_model.bin")


def run_id(seed, variant):
    return f"stage07_item_local_joint_{variant}_seed{seed}_20261021_r1"


def verify(root, package):
    os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import torch
    torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(), "CUDA must not be initialized for CPU readback")
    root, package = root.resolve(strict=True), package.resolve(strict=True)
    manifest, entries = backup.verified_manifest(root, package)
    need(manifest["source_identity"]["engine"]["commit"] == backup.ENGINE_COMMIT, "Wrong frozen engine")

    def checked(member):
        need(member in entries, "Required archived member missing: " + member)
        path = backup.byte.inside(root, backup.byte.safe_name(member))
        need(backup.byte.file_record(member, path) == entries[member], "Archived bytes changed: " + member)
        return path

    required = {f"seeds/seed{seed}/runs/{run_id(seed, variant)}/{name}"
                for seed in (2025, 2026) for variant in ("control", "item_local") for name in NECESSARY}
    checkpoints = {name for name in entries if PurePosixPath(name).suffix.lower() in (".pt", ".bin")}
    need(required <= checkpoints and len(required) == 28, "Missing mandatory joint checkpoint files")
    actual = {path.relative_to(root).as_posix() for path in backup.regular_tree(root)
              if path.suffix.lower() in (".pt", ".bin")}
    need(actual == checkpoints, "Unlisted or missing local checkpoint file")
    reports, formal = [], {}
    for member in sorted(checkpoints):
        path = checked(member)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensors, elements = tensor_inventory(payload, member)
        need(tensors > 0, "Checkpoint has no actual tensors: " + member)
        report = {**entries[member], "tensor_count": tensors, "tensor_elements": elements, "all_tensors_finite": True}
        if member in required and member.endswith(".pt"):
            if path.name == "smoke_tokenizer_boundary.pt":
                best = path.parent / "recovery_probe/best_model.bin"
            else:
                best = path.parent / "best_model.bin"
            report["joint_boundary"] = inspect_boundary(payload, checked(best.relative_to(root).as_posix()),
                                                         terminal=path.name == "last.pt")
            if path.name == "last.pt" and path.parents[1].name != "smoke":
                # Retain summaries only, not four full optimizer payloads in RAM.
                formal[path.parents[1].name] = {"report": report, "progress": payload["progress"],
                                               "model_keys": set(payload["model"])}
        elif member in required:
            need(isinstance(payload, dict) and payload and all(torch.is_tensor(x) for x in payload.values()),
                 "Best checkpoint is not a model tensor dictionary")
        reports.append(report)
        del payload
        gc.collect()
    need(len(formal) == 4, "Missing four formal terminal states")
    arms = {}
    for seed in (2025, 2026):
        for variant in ("control", "item_local"):
            name = run_id(seed, variant)
            prefix = f"seeds/seed{seed}/runs/{name}/"
            terminal = formal[name]
            progress, last_report = terminal["progress"], terminal["report"]
            result_path = checked(prefix + "results.json")
            result, complete, reload, fit = [read(checked(prefix + filename)) for filename in
                                            ("results.json", "COMPLETE.json", "checkpoint_reload.json", "FIT_COMPLETE.json")]
            best_path = checked(prefix + "training_state/best_model.bin")
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            selected_sid, best_sha = tensor_digest(best["joint_alignment.hard_codes"][1:]), sha(best_path)
            campaign = f"stage07_item_local_alternating_joint_seed{seed}_20261021_r1"
            need(result["campaign_id"] == complete["campaign_id"] == campaign
                 and result["seed"] == seed and result["parent_campaign_id"] == backup.CAMPAIGN
                 and result["variant"] == complete["variant"] == variant
                 and result["run_id"] == complete["run_id"] == name and complete["mode"] == "formal"
                 and result["status"] == complete["status"] == "formal_training_completed_validation_pending"
                 and result["test_data_evaluated"] is False, "Formal result identity/status differs")
            jp = progress["joint_progress"]
            need(result["fit"] == fit and fit["training_state_sha256"] == last_report["sha256"]
                 and fit["completed_epochs"] == progress["next_epoch"] == 45
                 and progress["decoder_epochs_completed"] == 35
                 and fit["best_epoch"] == progress["best_epoch"]
                 and fit["best_val_score"] == progress["best_val_score"]
                 and fit["joint_progress"] == jp and fit["training_stop_reason"] == progress["stop_reason"] == "max_epochs"
                 and fit["cumulative_recommender_optimizer_updates"] == progress["cumulative_optimizer_updates"] == 3640
                 and jp["tokenizer_epochs_completed"] == jp["sid_refresh_count"] == 10
                 and jp["tokenizer_optimizer_updates"] == 60, "Formal terminal/result fixed-budget counters differ")
            need(result["checkpoint_sha256"] == reload["checkpoint_sha256"] == complete["checkpoint_sha256"] == best_sha
                 and complete["result_file"] == "results.json" and complete["result_sha256"] == sha(result_path)
                 and reload["strict_all_tensor_match"] is True and reload["mapping_synchronized_without_requantization"] is True
                 and reload["best_epoch"] == progress["best_epoch"]
                 and result["selected_SID_sha256"] == reload["matching_committed_SID_sha256"] == selected_sid
                 and set(best) == terminal["model_keys"], "Selected checkpoint/reload/SID binding differs")
            evaluation = read(checked(prefix + "validation/EVALUATION_COMPLETE.json"))
            need(evaluation["status"] == "selected_validation_completed" and evaluation["seed"] == seed
                 and evaluation["variant"] == variant and evaluation["test_data_evaluated"] is False
                 and evaluation["count_per_cell"] == 22363 and set(evaluation["cells"]) == {"beam32", "beam256"}
                 and evaluation["checkpoint_sha256"] == best_sha and evaluation["selected_SID_sha256"] == selected_sid,
                 "Final validation references different selected bytes")
            arms[name] = {**last_report["joint_boundary"], "seed": seed, "variant": variant,
                          "best_checkpoint_sha256": best_sha, "selected_SID_sha256": selected_sid,
                          "results_sha256": sha(result_path)}
            del best
    need(not torch.cuda.is_initialized(), "CPU check unexpectedly initialized CUDA")
    return {"status": "LOCAL_CPU_CHECKPOINT_READBACK_PASS", "campaign_id": backup.CAMPAIGN,
            "verified_utc": datetime.now(timezone.utc).isoformat(), "backup_root": str(root),
            "package_receipt_sha256": sha(package / "PACKAGE.json"), "manifest_sha256": sha(root / "MANIFEST.json"),
            "source_identity": manifest["source_identity"], "checkpoints": reports, "checkpoint_count": len(reports),
            "required_joint_checkpoint_count": len(required), "formal_arms": arms, "all_actual_tensors_finite": True,
            "load_method": "torch.load(map_location='cpu', weights_only=False)", "GPU_initialized": False,
            "strict_model_reload_performed_here": False, "full_relocated_resume_exercised": False,
            "platform_gpu_release_audited_here": False, "goal_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not os.path.lexists(args.output) and not args.output.resolve().is_relative_to(args.verified_root.resolve()),
         "Use a new receipt outside the extracted evidence")
    try:
        result, code = verify(args.verified_root, args.package_dir), 0
    except Exception as error:
        result, code = {"status": "LOCAL_CPU_CHECKPOINT_READBACK_FAILED", "error_type": type(error).__name__,
                        "error": str(error), "goal_complete": False}, 1
    result["tool_sha256"] = sha(Path(__file__))
    write(args.output, result)
    print(json.dumps({"status": result["status"], "receipt": str(args.output), "sha256": sha(args.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
