"""Explicit prepare/check/submit/status operations with durable submission intent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CLI = Path.home() / ".local/bin/inspire.exe"
REMOTE = "/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_replay_r5_r20_20261105_r1"
PYTHON = "/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python"
JOB = "nancy-s09-replay-r5-r20-1105-r1"
GROUP = "开发区-H100-cuda12.8版本-183核"
WORKSPACE = "分布式训练空间"
ENV = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONUTF8="1")
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    ENV.pop(key, None)


def cli(args, receipt, timeout=180):
    result = subprocess.run([str(CLI), "--json", "--account", "sii-shared", *args],
                            env=ENV, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    data = json.loads(result.stdout)
    if isinstance(data.get("data"), dict) and "output" in data["data"]:
        data["data"]["output"] = data["data"]["output"].split("Starting client")[0]
    (HERE / receipt).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if result.returncode or not data.get("success"):
        raise RuntimeError(f"{args[:2]} failed; inspect {receipt}")
    if "returncode" in data.get("data", {}) and data["data"]["returncode"]:
        raise RuntimeError(f"Remote command failed; inspect {receipt}")
    return data["data"]


def remote(command, receipt, timeout=180):
    return cli(["notebook", "exec", "env-zzl", "--workspace", "CPU资源空间",
                "--timeout", str(timeout), command + " #"], receipt, timeout + 30)["output"].strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "check", "dry-run", "submit", "status"))
    args = parser.parse_args()
    if args.mode == "prepare":
        remote(f"test ! -e {REMOTE} && mkdir -p {REMOTE}", "PREPARE_DIRECTORY.json")
        files = ("replay.py", "job.py", "README.md", "test_replay.py")
        manifest = {}
        for name in files:
            local = HERE / name
            cli(["notebook", "scp", "env-zzl", local.as_posix(), REMOTE + "/" + name,
                 "--workspace", "CPU资源空间", "-t", "180"], f"UPLOAD_{name}.json", 210)
            manifest[name] = hashlib.sha256(local.read_bytes()).hexdigest()
        values = remote("sha256sum " + " ".join(REMOTE + "/" + name for name in files), "REMOTE_SOURCE_SHA.json")
        for name, expected in manifest.items():
            if expected + "  " + REMOTE + "/" + name not in values:
                raise ValueError(f"Remote source digest mismatch: {name}")
        (HERE / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Remote sources verified; original code and data remain referenced in place.")
        return
    if args.mode == "check":
        command = (f"env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                   f"CUBLAS_WORKSPACE_CONFIG=:4096:8 {PYTHON} -B -u {REMOTE}/replay.py check --root {REMOTE}")
        output = remote(command, "CPU_PREFLIGHT_COMMAND.json", 600)
        print(output)
        check = json.loads(remote(f"cat {REMOTE}/CPU_PREFLIGHT.json", "CPU_PREFLIGHT_READBACK.json"))
        (HERE / "CPU_PREFLIGHT.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
        return
    if args.mode == "status":
        print(json.dumps(cli(["job", "status", JOB, "--workspace", WORKSPACE], "STATUS.json"),
                         indent=2, ensure_ascii=False))
        return
    check = json.loads((HERE / "CPU_PREFLIGHT.json").read_text(encoding="utf-8"))
    if check["status"] != "passed" or not check["model_optimizer_scheduler_restore_exact"]:
        raise RuntimeError("Real-parent CPU preflight not passed")
    if check["wrapper_sha256"] != hashlib.sha256((HERE / "replay.py").read_bytes()).hexdigest():
        raise RuntimeError("Wrapper changed since preflight")
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Local source changed since deployment: {name}")
    if args.mode == "dry-run":
        cli(["job", "quota", "--workspace", WORKSPACE, "--group", GROUP, "--all"], "LIVE_QUOTA.json")
        cli(["image", "list", "--workspace", WORKSPACE, "--keyword", "ngc-pytorch:25.02", "--limit", "5"],
            "LIVE_IMAGE.json")
        cli(["resources", "availability", "--workspace", WORKSPACE, "--group", GROUP], "LIVE_AVAILABILITY.json")
        cli(["resources", "policy", "--workspace", WORKSPACE], "LIVE_POLICY.json")
        cli(["project", "detail", "公共科研项目"], "LIVE_PROJECT.json")
    existing = cli(["job", "list", "--workspace", WORKSPACE, "--keyword", JOB, "--all"], "JOB_ABSENCE.json")
    if existing.get("items"):
        raise RuntimeError("Matching job already exists; query it, do not duplicate")
    command = shlex.join(["bash", "-c", shlex.join(
        [PYTHON, REMOTE + "/job.py", "--root", REMOTE, "--python", PYTHON])])
    request = ["job", "create", "--name", JOB, "--workspace", WORKSPACE,
               "--project", "公共科研项目", "--group", GROUP, "--quota", "1,20,200",
               "--image", "ngc-pytorch:25.02-cuda12.8.0-py3", "--nodes", "1",
               "--shm-size", "32", "--max-time", "12",
               "--no-auto-fault-tolerance", "--no-enable-notification", "--command", command]
    if args.mode == "dry-run":
        print(json.dumps(cli(request + ["--dry-run"], "DRY_RUN.json"), indent=2, ensure_ascii=False))
        return
    dry = json.loads((HERE / "DRY_RUN.json").read_text(encoding="utf-8"))
    if not dry.get("success"):
        raise RuntimeError("Dry-run failed")
    # Check deployed code again at the submission boundary.
    values = remote("sha256sum " + " ".join(REMOTE + "/" + name for name in manifest),
                    "PRE_SUBMIT_SOURCE_SHA.json")
    for name, expected in manifest.items():
        if expected + "  " + REMOTE + "/" + name not in values:
            raise ValueError(f"Remote source changed since preflight: {name}")
    with (HERE / "SUBMISSION_INTENT.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(job=JOB, root=REMOTE, command=command, gpus=1, max_hours=12,
                       forecast_hours=[6, 9], unix=time.time(),
                       authorization="Explicit user instruction to submit automatically after checks"),
                  stream, indent=2, ensure_ascii=False)
    print(json.dumps(cli(request, "SUBMISSION_RESULT.json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
