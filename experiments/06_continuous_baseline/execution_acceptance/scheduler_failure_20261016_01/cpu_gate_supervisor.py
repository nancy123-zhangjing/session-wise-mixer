"""Persist CPU-only supervisor regression evidence, including real Linux cleanup."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "deployment", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root, deployment, output = (Path(v).resolve() for v in (args.root, args.deployment, args.output))
    sys.path.insert(0, str(root))
    from scripts.continuous.trainer import source_identity
    identity = source_identity(root)
    assert identity["source_sha256"] == "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
    output.mkdir(parents=True, exist_ok=False)
    test_path = deployment / "tests/test_queue_supervisor.py"
    spec = importlib.util.spec_from_file_location("supervisor_gate_suite", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer, started = io.StringIO(), time.monotonic()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    log = output / "tests.log"
    log.write_text(buffer.getvalue(), encoding="utf8", newline="\n")
    unchanged = source_identity(root)["source_sha256"] == identity["source_sha256"]
    passed = result.wasSuccessful() and result.testsRun == 11 and not result.skipped and unchanged
    proof = {"status": "PASS" if passed else "FAIL", "cpu_only": True, "tests": result.testsRun,
        "skipped": len(result.skipped), "failures": len(result.failures), "errors": len(result.errors),
        "platform": os.name, "real_linux_process_group_cleanup_executed": os.name == "posix",
        "training_source_sha256": identity["source_sha256"], "training_source_unchanged": unchanged,
        "supervisor_sha256": sha(deployment / "scripts/train/continuous_queue_supervisor.py"),
        "tests_sha256": sha(test_path), "gate_script_sha256": sha(__file__), "test_log_sha256": sha(log),
        "elapsed_seconds": time.monotonic()-started, "python": sys.version,
        "scope": "CPU scheduling logic fixtures and actual child processes only; no GPU, optimizer updates, validation or test metric evaluation"}
    (output / "CPU_GATE.json").write_text(json.dumps(proof, indent=2)+"\n", encoding="utf8", newline="\n")
    print(json.dumps(proof), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
