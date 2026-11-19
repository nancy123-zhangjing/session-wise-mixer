"""Read-only final audit of the two remote target experiments."""
import json
import shlex

import control

REMOTE_SCRIPT = r'''
import hashlib
import json
from pathlib import Path

root = Path.cwd()
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

out = {"teacher": read(root / "teacher/COMPLETE.json"), "arms": {}}
out["teacher_sha_ok"] = sha(root / "teacher/items.pt") == out["teacher"]["items_sha256"]
for mode in ("cf", "hybrid"):
    run = root / mode
    prepared = root / (mode + "_prepared")
    rq = root / (mode + "_rq")
    prep = read(prepared / "PREPARED.json")
    train = read(run / "TRAINING_COMPLETE.json")
    best = read(run / "BEST.json")
    test = read(run / "TEST_COMPLETE.json")
    pre = read(rq / "COMPLETE.json")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    vals = {e["R"]: {"T": e["T"], "ndcg@5": e["metrics"]["ndcg@5"]} for e in events if e["kind"] == "validation"}
    val_summaries = [read(p) for p in sorted((run / "validation").glob("R*/summary.json"))]
    t_blocks = [e for e in events if e["kind"] == "T_block"]
    r_epochs = [e for e in events if e["kind"] == "R_epoch"]
    test_compact = {}
    for label, result in test.items():
        ident = result["identity"]
        test_compact[label] = {"R": ident["R"], "T": ident["T"], "count": result["count"],
            "metrics": result["metrics"], "checkpoint_sha256": ident["checkpoint_sha256"],
            "prepared_sha256": ident["binding"]["prepared_sha256"],
            "teacher_sha256": ident["binding"].get("teacher_sha256"),
            "target_mode": ident["binding"].get("target_mode"),
            "noise_sha256": result["noise_sha256"], "prediction_sha256": result["prediction_sha256"]}
    out["arms"][mode] = {
        "prepared": {"sha256": sha(prepared / "PREPARED.json"),
            "teacher_sha256": prep.get("teacher_sha256"),
            "target_mode": prep.get("target_mode"),
            "initial_sha256": prep.get("initial_sha256"),
            "common_recommender_hash": prep.get("common_recommender_hash"),
            "common_Mixer_hash": prep.get("common_Mixer_hash"),
            "noise_sha256": {s: sha(prepared / (s + "_noise.npy")) for s in ("validation", "test")}},
        "pretraining": pre,
        "training": train,
        "best": best,
        "validation": vals,
        "validation_count": len(vals),
        "validation_files": len(val_summaries),
        "validation_summary_counts": sorted(set(v["count"] for v in val_summaries)),
        "validation_summary_match": all(
            v["metrics"]["ndcg@5"] == vals[v["identity"]["R"]]["ndcg@5"]
            and v["identity"]["binding"]["prepared_sha256"] == sha(prepared / "PREPARED.json")
            for v in val_summaries),
        "T_blocks": len(t_blocks), "T_updates": sum(e["updates_in_block"] for e in t_blocks),
        "R_epochs": len(r_epochs),
        "checkpoint_count": len(list((run / "checkpoints").glob("R*_T*.pt"))),
        "best_checkpoint_sha_ok": sha(run / best["checkpoint"]) == best["sha256"],
        "final_checkpoint_sha_ok": sha(run / train["final_checkpoint"]) == train["final_sha256"],
        "test": test_compact,
        "state": read(root / (mode + "_STATE.json")),
    }
print(json.dumps(out, ensure_ascii=False))
'''

if __name__ == "__main__":
    command = f"cd {control.REMOTE} && {control.PYTHON} -B -c {shlex.quote(REMOTE_SCRIPT)}"
    data = json.loads(control.remote(command, "FINAL_AUDIT_REMOTE.json", timeout=120))
    control.write(control.HERE / "FINAL_AUDIT.json", data)
    errors = []
    if not data["teacher_sha_ok"]:
        errors.append("shared teacher SHA")
    arms = data["arms"]
    for mode, arm in arms.items():
        status = control.cli(["job", "status", control.JOBS[mode], "--workspace", control.WORKSPACE],
                             mode + "_FINAL_STATUS.json")
        if status["status"] != "SUCCEEDED" or arm["state"]["phase"] != "complete":
            errors.append(mode + " terminal status")
        if arm["training"]["R"] != 35 or arm["training"]["T"] != 300 or arm["training"]["R_updates"] != 24255:
            errors.append(mode + " training budget")
        if (arm["validation_count"] != 31 or arm["validation_files"] != 31
                or sorted(map(int, arm["validation"])) != list(range(5, 36))
                or arm["validation_summary_counts"] != [71657] or not arm["validation_summary_match"]):
            errors.append(mode + " full validation")
        if (arm["T_blocks"] != 30 or arm["T_updates"] != 300 or arm["R_epochs"] != 35
                or arm["checkpoint_count"] != 31 or not arm["best_checkpoint_sha_ok"]
                or not arm["final_checkpoint_sha_ok"]):
            errors.append(mode + " checkpoints/schedule")
        if (arm["best"]["R"] != arm["training"]["best_R"]
                or arm["best"]["validation_ndcg@5"] != max(v["ndcg@5"] for v in arm["validation"].values())):
            errors.append(mode + " best selection")
        for label in ("best", "final"):
            test = arm["test"][label]
            expected = arm["best"]["sha256"] if label == "best" else arm["training"]["final_sha256"]
            if (test["count"] != 53194 or test["metrics"]["completion_rate"] != 1.0
                    or test["checkpoint_sha256"] != expected
                    or test["prepared_sha256"] != arm["prepared"]["sha256"]
                    or test["teacher_sha256"] != data["teacher"]["items_sha256"]
                    or test["target_mode"] != mode
                    or test["noise_sha256"] != arm["prepared"]["noise_sha256"]["test"]):
                errors.append(mode + " " + label + " test identity")
    for key in ("common_recommender_hash", "common_Mixer_hash", "teacher_sha256"):
        if arms["cf"]["prepared"][key] != arms["hybrid"]["prepared"][key]:
            errors.append("shared " + key)
    for split in ("validation", "test"):
        if arms["cf"]["prepared"]["noise_sha256"][split] != arms["hybrid"]["prepared"]["noise_sha256"][split]:
            errors.append("shared " + split + " noise")
    control.write(control.HERE / "ACCEPTANCE.json", {"passed": not errors, "errors": errors,
        "remote": control.REMOTE, "jobs": control.JOBS,
        "source_commit": json.loads((control.HERE / "DEPLOYMENT.json").read_text(encoding="utf-8"))["code_commit"]})
    for mode, arm in data["arms"].items():
        print(mode, json.dumps({
            "training": arm["training"], "best": arm["best"],
            "validation_count": arm["validation_count"],
            "validation_files": arm["validation_files"],
            "validation_summary_counts": arm["validation_summary_counts"],
            "validation_summary_match": arm["validation_summary_match"],
            "T_blocks": arm["T_blocks"], "T_updates": arm["T_updates"],
            "R_epochs": arm["R_epochs"], "checkpoint_count": arm["checkpoint_count"],
            "best_checkpoint_sha_ok": arm["best_checkpoint_sha_ok"],
            "final_checkpoint_sha_ok": arm["final_checkpoint_sha_ok"],
            "test": arm["test"], "state": arm["state"],
        }, ensure_ascii=False))
    print(json.dumps({"acceptance_passed": not errors, "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)
