"""Bounded local monitoring of the four EXISTING jobs. Never submits or restarts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CLI = Path.home() / "AppData/Roaming/uv/tools/inspire-skill/Scripts/inspire.exe"
REMOTE = "/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_r26_freeze_20261106_v1"
PYTHON = "/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python"


def call(arguments):
    env = dict(os.environ, NO_PROXY=".sii.edu.cn", HTTP_PROXY="", HTTPS_PROXY="", ALL_PROXY="", PYTHONUTF8="1")
    result = subprocess.run([str(CLI), "--json", "--account", "sii-shared"] + arguments,
                            env=env, capture_output=True, encoding="utf-8", timeout=120)
    value = json.loads(result.stdout)
    if result.returncode or not value.get("success"):
        raise RuntimeError(value.get("error", {"exit_code": result.returncode}))
    return value


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def full_nodes(progress):
    return {(arm, row["epoch"]) for arm, state in progress.get("arms", {}).items()
            for row in state["validation"] if row["kind"] != "reference"}


def poll():
    names = json.loads((ROOT / "JOB_NAMES.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        platform_future = pool.submit(call, ["job", "status", *names.values(), "--workspace", "分布式训练空间"])
        remote_future = pool.submit(call, ["notebook", "exec", "env-zzl", "--workspace", "CPU资源空间",
            f"{PYTHON} -B {REMOTE}/collect_status.py # ", "--timeout", "30"])
        platform, envelope = platform_future.result(), remote_future.result()
    if envelope["data"]["returncode"]:
        raise RuntimeError("Remote progress reader failed")
    progress = next(json.loads(line) for line in envelope["data"]["output"].splitlines() if line.startswith("{"))
    write(ROOT / "PLATFORM_LATEST.json", platform)
    write(ROOT / "PROGRESS_LATEST.json", progress)
    statuses = {row["name"]: row["status"] for row in platform["data"]["items"]}
    brief = dict(observed_utc=progress["observed_utc"], arms={})
    for arm, name in names.items():
        row = progress["arms"][arm]
        update = row.get("latest_update", {})
        brief["arms"][arm] = dict(status=statuses.get(name, "UNRESOLVED"),
            phase=update.get("kind"), R=update.get("epoch"), update=update.get("update"), T=update.get("t_updates"),
            validation=[v for v in row["validation"] if v["kind"] != "reference"],
            job_result=row["job_result"], tests=row["tests"])
    with (ROOT / "MONITOR_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(brief, ensure_ascii=False) + "\n")
    print(json.dumps(brief, ensure_ascii=False), flush=True)
    return progress, statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60 or not 0 <= args.seconds <= 1800:
        raise ValueError("Use a bounded watch up to 30 minutes and sleep no more than 60 seconds")
    initial = (json.loads((ROOT / "PROGRESS_LATEST.json").read_text(encoding="utf-8"))
               if (ROOT / "PROGRESS_LATEST.json").exists() else {})
    baseline = full_nodes(initial)
    deadline = time.monotonic() + args.seconds
    while True:
        progress, statuses = poll()
        if (full_nodes(progress) - baseline or any(state not in {"RUNNING", "QUEUING", "PENDING", "CREATING"}
                for state in statuses.values()) or any(row["job_result"] for row in progress["arms"].values())
                or time.monotonic() >= deadline):
            return
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
