"""One GPU worker, same execution environment as original H1."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = args.root
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in list(env):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"} or \
                key.startswith(("TORCHELASTIC_", "OMPI_", "PMI_", "PMIX_", "MV2_")):
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", TMPDIR=str(root / "tmp"),
               HF_HOME=str(root / "cache/huggingface"))
    (root / "tmp").mkdir(exist_ok=True)
    started = time.time()
    with (jobs / "training.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            [args.python, "-B", "-u", str(root / "replay.py"), "train", "--root", str(root)],
            env=env, cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(exit_code=result.returncode, elapsed_seconds=time.time()-started,
                   status="verified_complete" if result.returncode == 0 else "needs_inspection")
    (jobs / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
