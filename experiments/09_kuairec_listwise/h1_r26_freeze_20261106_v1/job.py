"""Explicit single-arm entry; no platform submission. Default is a plan preview."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from scripts.listwise_diffgrm.recovery_stability_protocol import ARMS, load_spec


def check_sources():
    lock = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in lock["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Versioned source changed: {name}")
    actual_modules = {p.relative_to(ROOT).as_posix() for p in (ROOT / "code/scripts/listwise_diffgrm").glob("*.py")}
    expected_modules = {name for name in lock["sha256"] if name.startswith("code/scripts/listwise_diffgrm/") and name.endswith(".py")}
    if actual_modules != expected_modules:
        raise ValueError("Unexpected runtime module added or removed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--execute", action="store_true", help="Run train, test, restore on this machine")
    parser.add_argument("--resume", action="store_true", help="Resume this arm only, after previous job is terminal")
    args = parser.parse_args()
    spec = load_spec(ROOT / "configs" / f"{args.arm}.json")
    if not args.execute:
        print(json.dumps(dict(spec=spec, submitted=False, training_started=False), indent=2))
        return
    check_sources()
    if not (ROOT / "PARENTS.lock.json").is_file():
        raise RuntimeError("Run lock_parents.py on the CPU file host first; parents are not yet verified")
    inputs = json.loads((ROOT / "INPUTS.json").read_text(encoding="utf-8"))
    output = ROOT / "runs" / args.arm
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires this arm's last.pt")
    if not args.resume and output.exists():
        raise FileExistsError("Fresh run may not reuse existing output")
    env = dict(os.environ)
    for key in list(env):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"} or key.startswith(("TORCHELASTIC_", "OMPI_", "PMI_", "PMIX_", "MV2_")):
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               DIFFGRM_ENGINE_ROOT=str(ROOT / "code/external/DiffGRM"))
    common = ["--config", str(ROOT / "configs" / f"{args.arm}.json"), "--parents", str(ROOT / "PARENTS.lock.json"),
              "--dataset", inputs["dataset"], "--assets", inputs["assets"],
              "--subset", str(ROOT / "validation_subset_8192.json"), "--output", str(output), "--device", "cuda:0"]
    jobs = ROOT / "jobs" / args.arm
    jobs.mkdir(parents=True, exist_ok=True)
    started, stages, status = time.monotonic(), [], "incomplete"
    try:
        for stage in ("train", "test", "verify"):
            command = [sys.executable, "-B", "-u", "-m", "scripts.listwise_diffgrm.recovery_stability_run", stage] + common
            if stage == "train" and args.resume:
                command += ["--resume", str(output / "last.pt")]
            with (jobs / f"{stage}.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
            stages.append(dict(stage=stage, exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError(f"{args.arm} {stage} failed; inspect its own log")
        status = "training_test_restore_complete_pending_result_audit"
    finally:
        receipt = dict(status=status, arm=args.arm, stages=stages, elapsed_seconds=time.monotonic() - started)
        (jobs / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
