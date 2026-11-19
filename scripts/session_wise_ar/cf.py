"""CF-M0 delegates unchanged to the accepted CF engine, selecting B0 only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiments/rqcd_mixer/targets_formal_20261113_v1"
CONFIG = ROOT / "configs/rqcd_mixer/v2_cf.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    manifest = json.loads((REFERENCE / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if name.startswith(("scripts/", "configs/", "docs/")):
            if sha(ROOT / name) != expected:
                raise ValueError(f"CF reference source changed: {name}")


def verify_parent(prepared, runtime):
    verify_source()
    audit = json.loads((REFERENCE / "FINAL_AUDIT.json").read_text())
    expected = audit["arms"]["cf"]["prepared"]
    prepared = Path(prepared)
    for name, digest in (("PREPARED.json", expected["sha256"]),
                         ("initial.pt", expected["initial_sha256"])):
        if sha(prepared / name) != digest:
            raise ValueError(f"CF common parent mismatch: {name}")
    meta = json.loads((prepared / "PREPARED.json").read_text())
    if meta["teacher_sha256"] != expected["teacher_sha256"]:
        raise ValueError("Do not retrain or replace the CF teacher")
    if json.loads(Path(runtime).read_text()) != json.loads((REFERENCE / "runtime.locked.json").read_text()):
        raise ValueError("CF-M0 must retain accepted runtime settings")
    for split, digest in expected["noise_sha256"].items():
        if sha(prepared / f"{split}_noise.npy") != digest:
            raise ValueError(f"CF evaluation noise mismatch: {split}")


def commands(dataset, prepared, output, runtime, python=sys.executable, resume=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output == prepared or prepared in output.parents or output in prepared.parents:
        raise ValueError("Use an independent output directory, never the common prepared tree")
    base = [str(python), "-m", "scripts.rqcd_mixer.run", "--config", str(CONFIG)]
    common = ["--runtime", str(runtime), "--dataset", str(dataset),
              "--prepared", str(prepared), "--arm", "B0", "--device", "cuda"]
    tail = ["--resume"] if resume else []
    pre = base + ["pretrain", *common, "--output", str(output / "pretrain"), *tail]
    train = base + ["train", *common, "--pretrained", str(output / "pretrain"),
                    "--output", str(output / "train"), *tail]
    test = base + ["test", *common, "--pretrained", str(output / "pretrain"),
                   "--run", str(output / "train")]
    return {"pretrain": pre, "train": train, "test": test}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "pretrain", "train", "test"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime", default=str(REFERENCE / "runtime.locked.json"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    jobs = commands(args.dataset, args.prepared, args.output, args.runtime, resume=args.resume)
    if args.action == "plan":
        verify_source()
        print(json.dumps({"arm": "CF-M0", "scientific_change": "B1 -> B0 (no Mixer)",
                          "commands": jobs, "training_started": False}, indent=2))
        return
    verify_parent(args.prepared, args.runtime)
    subprocess.run(jobs[args.action], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
