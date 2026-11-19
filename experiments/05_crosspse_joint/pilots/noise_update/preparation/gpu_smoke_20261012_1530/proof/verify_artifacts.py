"""Read-only, stdlib acceptance for this SHA-bound GPU smoke, not a pilot."""
import hashlib
import json
import math
from pathlib import Path
import sys

out, root = [Path(p).resolve() for p in sys.argv[1:3]]
SHARED = "dcdd5ecbe41f5356494d8efb4f082edf440db0200120cfdb99603e64d22e51a5"
EXPECTED = {
    "runner_sha256": "655a843616d4311a046f9cbd5a5bf2a57364256c043254a1f1727681ba4c0bd7",
    "executed_helper_sha256": "879716431edc4bc918e545b703d167b80d19172f0003f761381b4ccc77837336",
    "config_sha256": "9cd123e0fe763947eb49c4002b17d458a870e9ac6bd1273fb0478423fd726d92",
}


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(name):
    return json.loads((out / name).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(base, relative):
    rel = Path(relative)
    check(not rel.is_absolute() and ".." not in rel.parts, "Unsafe receipt path")
    path = (base / rel).resolve()
    check(path.is_relative_to(base), "Receipt escaped base")
    return path


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


check(not (out / "FAILED.json").exists(), "FAILED receipt present")
done, summary, prov, proto, post = [read(p) for p in (
    "COMPLETE.json", "summary.json", "provenance.json", "protocol.json",
    "post_input_verification.json")]
check(done["summary_sha256"] == sha(out / "summary.json"), "COMPLETE hash mismatch")
check(done["arms"] == ["A", "B", "C"], "Incomplete arms")
check(summary["status"] == "complete" and summary["mode"] == "smoke", "Not completed smoke")
check(summary["protocol_id"] == proto["protocol_id"], "Protocol mismatch")
check(summary["no_test_evaluation"] and summary["no_historical_artifact_changes"], "Scope mismatch")
check(finite(summary), "Non-finite summary")
check(prov["device"] == "cuda:0" and "H100" in prov["gpu"], "Not single H100 run")
check(prov["decoder_forward_precision"] == "bf16", "Decoder precision mismatch")
check(prov["tokenizer_forward_precision"] == "float32_tf32_disabled", "Tokenizer precision mismatch")
check(prov["smoke_not_for_metric_comparison"] and prov["fresh_optimizer_pilot_not_historical_exact_resume"], "Missing boundary")
check(prov["shared_model_tensor_sha256"] == SHARED, "Wrong shared tensor")
check(proto["shared_checkpoint_sha256"] == "1855c8d22a134525e3af6d8990f6816b43bca01ae692931a46b9266ddc0c5715", "Wrong checkpoint")
check(proto["expected_tokenizer_updates"] == 0 and proto["tokenizer_schedule_total_updates"] == 60, "Wrong start or schedule")
for key, expected in EXPECTED.items():
    check(prov[key] == expected, "Source/config hash mismatch: " + key)
for path_key, hash_key in (("executed_runner", "runner_sha256"),
                           ("executed_helper", "executed_helper_sha256"),
                           ("executed_config", "config_sha256")):
    check(sha(prov[path_key]) == prov[hash_key], "Executed source changed")
check(post["verified"] and post["inputs"] == prov["assets"], "Pre/post inputs mismatch")
check(len(prov["assets"]) == 18 and len({r["path"] for r in prov["assets"]}) == 18, "Wrong input count")
for receipt in prov["assets"]:
    path = safe(root, receipt["path"])
    check(path.stat().st_size == receipt["size_bytes"] and sha(path) == receipt["sha256"], "Input changed: " + str(path))
for name, expected in prov["source"]["source_sha256"].items():
    check(sha(safe(root, name)) == expected, "Engine source changed: " + name)

arms = {a["arm"]: a for a in summary["arms"]}
check(set(arms) == {"A", "B", "C"} and len(summary["arms"]) == 3, "Wrong/duplicate arms")
rec_pairs, tok_pairs, noise_calls, gradients = [], [], {}, {}
for name in "ABC":
    arm, settings = arms[name], arms[name]["settings"]
    check(settings["decoder_epochs"] == 1 and settings["decoder_batch_limit"] == 16 and settings["eval_batch_limit"] == 2, "Wrong smoke budget")
    check(settings["gradient_accumulation_steps"] == 16 and settings["decoder_lr"] == 0.0001 and settings["noise_total_steps"] == 60, "Wrong optimizer settings")
    check(settings["tokenizer_batches"] == (0 if name == "A" else 16), "Wrong Tok budget")
    check(settings["noise_start"] == (1.0 if name == "C" else 0.0) and settings["noise_end"] == (0.2 if name == "C" else 0.0), "Wrong noise config")
    rows = arm["training"]
    rec = [r for r in rows if r["phase"] == "recommender"]
    tok = [r for r in rows if r["phase"] == "tokenizer"]
    check(len(rec) == 1 and len(tok) == (0 if name == "A" else 1) and len(rows) == len(rec) + len(tok), "Wrong phases")
    gradients[name] = {}
    for row in rows:
        check(row["batches"] == 16 and row["optimizer_updates"] == 1 and row["accumulation"] == 16 and row["final_window_batches"] == 16, "Wrong actual update budget")
        active, sides = row["phase"], row["last_gradient_sides"]
        inactive = "tokenizer" if active == "recommender" else "recommender"
        check(math.isfinite(sides[active]) and sides[active] > 0 and sides[inactive] == 0, "Gradient isolation failure")
        gradients[name][active] = sides
    pair_keys = ("logical_index", "batches", "optimizer_updates", "phase_seed", "data_seed", "data_order_sha256")
    rec_pairs.append(tuple(rec[0][k] for k in pair_keys))
    noise_calls[name] = 0
    if tok:
        row = tok[0]
        check(row["gumbel_calls"] == 32 and row["assignment_training_enabled_calls"] == 32, "Assignment count/mode failure")
        noise_calls[name] = row["noise_applied_calls"]
        check(noise_calls[name] == (32 if name == "C" else 0), "Noise gate count failure")
        check(row["loss_means"]["noise_scale"] == (1.0 if name == "C" else 0.0), "Noise scale failure")
        tok_pairs.append(tuple(row[k] for k in pair_keys))
    snaps = {s["label"]: s for s in arm["snapshots"]}
    check(set(snaps) == {"initial", "post_tokenizer", "decoder_1"} and len(arm["snapshots"]) == 3, "Snapshot set failure")
    initial, after, final = [snaps[k] for k in ("initial", "post_tokenizer", "decoder_1")]
    check(initial["model_tensor_state_sha256"] == SHARED and initial["deterministic_sid_change"]["changed_items"] == 0 and initial["tokenizer_updates"] == 0, "Unpaired start")
    check(final["recommender_updates"] == 1 and final["tokenizer_updates"] == (0 if name == "A" else 1), "Final update count failure")
    for snap in snaps.values():
        check(snap["schedule"]["total_optimizer_steps"] == 60 and snap["schedule"]["optimizer_step"] == snap["tokenizer_updates"], "Schedule failure")
        snap_dir = safe(out, f"{name}/snapshots/{snap['label']}")
        check(set(snap["files"]) == {"model.pt", "training_state.pt", "semantic_ids.npy", "crossed_codebooks.npy"}, "Missing snapshot files")
        for filename, expected in snap["files"].items():
            check(sha(safe(snap_dir, filename)) == expected, "Snapshot file hash failure")
        check(json.loads((snap_dir / "summary.json").read_text(encoding="utf-8")) == snap, "Snapshot summary mismatch")
    for filename in ("semantic_ids.npy", "crossed_codebooks.npy"):
        check(after["files"][filename] == final["files"][filename], "Frozen Tokenizer artifact changed")
    check(after["semantic_ids_array_sha256"] == final["semantic_ids_array_sha256"], "Frozen SID changed")
    if name == "A":
        check(initial["model_tensor_state_sha256"] == after["model_tensor_state_sha256"], "A changed before recovery")
    check(arm["final_snapshot"] == "snapshots/decoder_1" and [v["label"] for v in arm["validation"]] == ["initial", "post_tokenizer", "decoder_1"], "Recovery/validation labels failure")
    check(all(v["metrics"] and finite(v["metrics"]) for v in arm["validation"]), "Validation finite check failure")
    check(read(f"{name}/metrics.json") == arm, "Per-arm receipt mismatch")
check(rec_pairs[0] == rec_pairs[1] == rec_pairs[2], "Decoder budget/RNG/data mismatch")
check(tok_pairs[0] == tok_pairs[1], "B/C Tok budget/RNG/data mismatch")
print(json.dumps({
    "status": "accepted_gpu_smoke", "gpu": prov["gpu"],
    "elapsed_seconds": summary["elapsed_seconds"], "verified_inputs": 18,
    "summary_sha256": done["summary_sha256"], "noise_applied_calls": noise_calls,
    "gradient_sides": gradients, "paired_decoder_data_order_sha256": rec_pairs[0][-1],
    "research_metric_result": False,
    "frozen_full_tokenizer_state": "locked runner assertion passed before COMPLETE; not independently deserialized",
    "frozen_sid_and_codebook_files": "independently SHA-verified",
}, ensure_ascii=False, indent=2))
