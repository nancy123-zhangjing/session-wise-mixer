"""Read-only CPU preflight before the single H100 Job is requested."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cpu-gate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=65)
    args = parser.parse_args()
    import matplotlib
    import numpy as np
    import torch
    from scripts.continuous.data import DATA_MANIFEST_SHA256, sha256
    from scripts.continuous.trainer import source_identity
    identity = source_identity(ROOT)
    gate = json.loads(Path(args.cpu_gate).read_text(encoding="utf-8"))
    if gate["status"] != "PASS" or gate["skipped"] != 0 or gate["source_sha256"] != identity["source_sha256"]:
        raise ValueError("CPU gate is stale or incomplete")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode:
        raise ValueError("Remote tracked execution code is dirty")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        raise ValueError("Remote staged execution code is dirty")
    if sha256(Path(args.data) / "manifest.json") != DATA_MANIFEST_SHA256:
        raise ValueError("Remote prepared dataset differs from accepted data")
    cache = json.loads((Path(args.cache) / "CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    if cache["data_manifest_sha256"] != DATA_MANIFEST_SHA256:
        raise ValueError("Remote cache dataset identity changed")
    if gate.get("cache_manifest_sha256") != cache["cache_sha256"]:
        raise ValueError("Remote cache differs from CPU-validated cache")
    if not torch.__version__.startswith("2.6.0") or np.__version__ != "2.2.6":
        raise ValueError("Unvalidated Torch/NumPy environment; do not upgrade a shared environment")
    disk = shutil.disk_usage(ROOT / "outputs")
    if disk.free < args.minimum_free_gib * 1024**3:
        raise ValueError("Insufficient filesystem free space for conservative checkpoint budget")
    result = {"status": "PASS", "source": identity, "source_sha256": identity["source_sha256"],
              "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
              "matplotlib": matplotlib.__version__, "data_manifest_sha256": DATA_MANIFEST_SHA256,
              "cache_manifest_sha256": cache["cache_sha256"], "filesystem_free_bytes": disk.free,
              "space_scope": "Filesystem free space, not proof of an administrator user/project storage quota",
              "gpu_requested": False, "gpu_compatibility_pending": True}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
