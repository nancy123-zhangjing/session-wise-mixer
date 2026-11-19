"""Read existing single job only; no submission or restart."""
import json
from concurrent.futures import ThreadPoolExecutor
import poll_progress as p


def poll():
    name = "nancy-s09-r26freeze-1106-v1"
    with ThreadPoolExecutor(max_workers=2) as pool:
        job = pool.submit(p.call, ["job", "status", name, "--workspace", "分布式训练空间"])
        remote = pool.submit(p.call, ["notebook", "exec", "env-zzl", "--workspace", "CPU资源空间",
            f"{p.PYTHON} -B {p.REMOTE}/collect_status.py # ", "--timeout", "30"])
        platform, envelope = job.result(), remote.result()
    if envelope["data"]["returncode"]:
        raise RuntimeError("Progress reader failed")
    progress = next(json.loads(line) for line in envelope["data"]["output"].splitlines() if line.startswith("{"))
    p.write(p.ROOT / "PLATFORM_LATEST.json", platform)
    p.write(p.ROOT / "PROGRESS_LATEST.json", progress)
    brief = dict(time=progress["observed_utc"], platform=platform["data"], progress=progress["arms"]["r26_freeze"])
    with (p.ROOT / "MONITOR_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(brief, ensure_ascii=False) + "\n")
    print(json.dumps(brief, ensure_ascii=False), flush=True)
    return brief


if __name__ == "__main__":
    poll()
