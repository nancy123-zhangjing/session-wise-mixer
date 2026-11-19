"""Low-frequency real resource observations; never creates load to affect policy."""
import json
import os
import shutil
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path

from .protocol import atomic_json


class ResourceMonitor:
    def __init__(self, directory, interval=30, points_per_node_hour=8.0):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.interval = interval
        self.rate = points_per_node_hour
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = []

    def observe(self):
        command = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
                   "--format=csv,noheader,nounits"]
        observed = {"time": time.time(), "host": socket.gethostname()}
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
            observed["gpus"] = [dict(zip(("index", "uuid", "utilization_percent", "memory_mib", "power_watts"),
                                          [v.strip() for v in line.split(",")]))
                                 for line in result.stdout.strip().splitlines()]
        except (OSError, subprocess.SubprocessError) as error:
            observed["gpu_observation_error"] = type(error).__name__
        observed["disk"] = dict(zip(("total_bytes", "used_bytes", "free_bytes"),
                                      shutil.disk_usage(self.directory)))
        for name in ("cpu.stat", "memory.current", "memory.max", "io.stat"):
            path = Path("/sys/fs/cgroup") / name
            if path.is_file():
                observed["cgroup_" + name.replace(".", "_")] = path.read_text()[:16384]
        observed["cpu_scope"] = "container cgroup where available; disk filesystem space is not personal quota"
        self.samples.append(observed)
        with (self.directory / "resources.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observed) + "\n")

    def loop(self):
        while not self.stop_event.wait(self.interval):
            self.observe()

    def start(self):
        self.started = time.time()
        self.observe()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
        self.observe()
        elapsed = time.time() - self.started
        values = {}
        for sample in self.samples:
            for gpu in sample.get("gpus", []):
                try:
                    values.setdefault(gpu["uuid"], []).append(float(gpu["utilization_percent"]))
                except ValueError:
                    pass
        result = {"business_started_unix": self.started, "business_finished_unix": time.time(),
                  "business_elapsed_seconds": elapsed, "allocated_gpus": 8, "nodes": 1,
                  "business_allocated_gpu_hours": 8 * elapsed / 3600,
                  "points_per_node_hour": self.rate,
                  "business_runtime_points_calculated": self.rate * elapsed / 3600,
                  "billing_scope": "Calculated from observed in-container runtime and live quota rate; platform allocation/billing reconciliation pending",
                  "samples": len(self.samples), "interval_seconds": self.interval,
                  "mean_observed_gpu_utilization_percent": {uuid: statistics.mean(v) for uuid, v in values.items()},
                  "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}
        atomic_json(self.directory / "usage_summary.json", result)
        return result
