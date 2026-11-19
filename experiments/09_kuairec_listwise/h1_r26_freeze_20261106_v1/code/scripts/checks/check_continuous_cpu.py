"""Auditable CPU model/campaign gate plus independent full-row cache validation."""
import argparse
import contextlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.continuous.protocol import atomic_json
from scripts.continuous.trainer import source_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-data-tests", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity(ROOT)
    start = time.perf_counter()
    summaries = []
    paths = ["tests/test_continuous_models.py", "tests/test_continuous_campaign.py",
             "tests/test_continuous_delivery.py"]
    if args.include_data_tests:
        paths.append("tests/test_continuous_data.py")
    for index, name in enumerate(paths):
        spec = importlib.util.spec_from_file_location(f"continuous_gate_suite_{index}", ROOT / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
        (output / (Path(name).stem + ".log")).write_text(buffer.getvalue(), encoding="utf-8")
        summary = {"suite": name, "tests": result.testsRun, "failures": len(result.failures),
                   "errors": len(result.errors), "skipped": len(result.skipped),
                   "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL"}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    command = [sys.executable, "-B", str(ROOT / "scripts/checks/verify_continuous_cache.py"),
               "--data", str(Path(args.data).resolve()), "--cache", str(Path(args.cache).resolve()),
               "--report", str(output / "cache_verification.json")]
    checked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / "cache_verifier.stdout.log").write_text(checked.stdout, encoding="utf-8")
    (output / "cache_verifier.stderr.log").write_text(checked.stderr, encoding="utf-8")
    unchanged = source_identity(ROOT)["source_sha256"] == identity["source_sha256"]
    cache_proof = json.loads((output / "cache_verification.json").read_text(encoding="utf-8")) if checked.returncode == 0 else {}
    passed = (all(s["status"] == "PASS" for s in summaries) and checked.returncode == 0 and unchanged)
    proof = {"status": "PASS" if passed else "FAIL", "source": identity,
             "source_sha256": identity["source_sha256"], "source_unchanged": unchanged,
             "suites": summaries, "tests": sum(s["tests"] for s in summaries),
             "skipped": sum(s["skipped"] for s in summaries),
             "cache_verifier_command": command, "cache_verifier_exit_code": checked.returncode,
             "cache_manifest_sha256": cache_proof.get("cache_sha256"),
             "data_manifest_sha256": cache_proof.get("data_manifest_sha256"),
             "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
             "platform": platform.platform(), "cpu_only": True,
             "elapsed_seconds": time.perf_counter() - start}
    atomic_json(output / "CPU_GATE.json", proof)
    print(json.dumps({"status": proof["status"], "tests": proof["tests"], "skipped": proof["skipped"],
                      "source_sha256": proof["source_sha256"]}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
