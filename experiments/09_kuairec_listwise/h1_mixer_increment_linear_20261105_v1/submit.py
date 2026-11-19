"""One explicitly authorized GPU submission, with an exclusive attempt receipt."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
REMOTE = "/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_mixer_increment_linear_20261105_v1"
PYTHON = "/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python"
JOB = "nancy-s09-minc-linear-1105-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = shlex.join([PYTHON, REMOTE + "/job.py", "--root", REMOTE, "--python", PYTHON])
    argv = [str(Path.home() / ".local/bin/inspire.exe"), "--json", "--account", "sii-shared",
            "job", "create", "--name", JOB, "--workspace", "分布式训练空间",
            "--project", "公共科研项目", "--group", "开发区-H100-cuda12.8版本-183核",
            "--quota", "1,20,200", "--image", "ngc-pytorch:25.02-cuda12.8.0-py3",
            "--nodes", "1", "--shm-size", "32", "--max-time", "12",
            "--no-auto-fault-tolerance", "--no-enable-notification", "--command", command]
    if args.dry_run:
        argv.append("--dry-run")
    else:
        check = json.loads((HERE / "PREFLIGHT_RESULT.json").read_text(encoding="utf-8"))
        if check["status"] != "passed" or not check["disk_restore_next_step_exact"]:
            raise RuntimeError("Real-parent preflight has not passed")
        # Persist BEFORE the request; unknown outcomes must be queried, never retried.
        with (HERE / "SUBMISSION.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(job=JOB, unix=time.time(), gpus=1, max_hours=12,
                           command=command, remote_root=REMOTE), stream, indent=2)
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(MSYS_NO_PATHCONV="1", PYTHONUTF8="1")
    result = subprocess.run(argv, capture_output=True, encoding="utf-8", env=env, timeout=180)
    data = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (HERE / ("DRY_RUN.json" if args.dry_run else "SUBMISSION_RESULT.json")).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
