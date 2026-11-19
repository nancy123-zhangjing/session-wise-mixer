"""Read-only CPU inspection of two existing parents; writes only a small lock.

Run on the shared CPU file host. Does not train, infer, submit, or download weights.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import torch
from scripts.listwise_diffgrm.recovery import fingerprint, verify_parent_binding
from scripts.listwise_diffgrm.recovery_stability import validate_parent_gate
from scripts.listwise_diffgrm.recovery_stability_protocol import specification, normalize_boundary, VERSION
from scripts.listwise_diffgrm.training import sha256, phase_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "PARENTS.lock.json"))
    args = parser.parse_args()
    inputs = json.loads((ROOT / "INPUTS.json").read_text(encoding="utf-8"))
    rows = {}
    for key, arm in (("increment_r26", "r26_freeze"),):
        item, spec = inputs["parents"][key], specification(arm)
        run = Path(item["run"])
        record_path = run / "validation_records" / f"R{item['epoch']}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        path, summary_path = Path(record["checkpoint"]), Path(record["summary"])
        if not path.resolve().is_relative_to(run.resolve()) or not summary_path.resolve().is_relative_to(run.resolve()):
            raise ValueError("Parent record points outside its approved run")
        digest = sha256(path)
        if digest != record["checkpoint_sha256"] or sha256(summary_path) != record["summary_sha256"]:
            raise ValueError("Parent validation-record hashes failed")
        parent = torch.load(path, map_location="cpu", weights_only=False)
        validate_parent_gate(parent, spec)
        inner = parent["inner"]
        if (inner["r_updates"], inner["t_updates"]) != (item["epoch"] * 693, item["t"]):
            raise ValueError("Parent counters differ")
        # Use the same plan constructor, with formal training data count from its binding.
        cfg = inner["binding"]["config"]
        plan = phase_plan(cfg, 693 * cfg["training"]["effective_batch_size"])
        normalize_boundary(inner, plan, spec)
        source = {p.relative_to(ROOT / "code").as_posix(): sha256(p)
                  for p in (ROOT / "code/scripts/listwise_diffgrm").glob("*.py")}
        current = dict(inner["binding"], code=source)
        current["engine"] = {p.relative_to(ROOT / "code/external/DiffGRM").as_posix(): sha256(p)
                             for p in list((ROOT / "code/external/DiffGRM/genrec").glob("*.py"))
                             + list((ROOT / "code/external/DiffGRM/genrec/models/DIFF_GRM").glob("*.py"))}
        verify_parent_binding(inner["binding"], current)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt_path = run / "evaluations" / f"R{item['epoch']}" / "complete.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        binding = receipt["binding"]
        if binding["state"] != fingerprint(dict(model=inner["model"], alignment=inner["alignment"])):
            raise ValueError("Validation does not belong to parent weights")
        if (not binding["full"] or not record["full_split"] or summary["examples"] != 71657
                or summary["split"] != "validation" or summary["beam"] != 32
                or summary["r_updates"] != inner["r_updates"] or summary["t_updates"] != inner["t_updates"]
                or record["ndcg5"] != summary["metrics"]["list/ndcg@5"]
                or receipt["summary_sha256"] != sha256(summary_path)):
            raise ValueError("Parent validation metadata mismatch")
        if key == "increment_r26":
            gate_binding = json.loads((run / "evaluation_gate_bindings" / "R26.json").read_text(encoding="utf-8"))
            if gate_binding["gate_sha256"] != fingerprint(parent["increment_gate"]):
                raise ValueError("Parent evaluation gate mismatch")
        rows[key] = dict(status="verified_metadata", checkpoint=str(path), sha256=digest,
                         bytes=path.stat().st_size, summary=str(summary_path),
                         summary_sha256=sha256(summary_path), ndcg5=record["ndcg5"],
                         r_updates=inner["r_updates"], t_updates=inner["t_updates"],
                         record=str(record_path), record_sha256=sha256(record_path),
                         parent_gate=fingerprint(parent.get("increment_gate", parent.get("release1"))))
    result = dict(version=VERSION, parents=rows, updates_performed=0,
                  note="CPU metadata lock; CUDA SID continuity and state restoration checked by runner")
    out = Path(args.output)
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != result:
        raise ValueError("Existing parent lock differs; do not overwrite a bound suite")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
