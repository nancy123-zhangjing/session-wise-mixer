"""Run the six stdlib reproduction suites without importing model dependencies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "test_asset_locations.py",
    "test_daily_repro.py",
    "test_legacy_import.py",
    "test_project_layout.py",
    "test_project_repro.py",
    "test_stage05_repro.py",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true",
        help="list the six suites without importing or running any test",
    )
    args = parser.parse_args(argv)
    if args.list:
        for name in TESTS:
            print(name)
        return 0

    paths = [ROOT / "tests" / name for name in TESTS]
    for path in paths:
        if not path.is_file():
            print(f"Missing reproduction test: {path}", file=sys.stderr)
            return 1

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-B", str(path)], cwd=ROOT, env=environment,
                check=False,
            )
        except OSError as error:
            print(f"Cannot run {path.name}: {error}", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print(
                f"Reproduction checks failed: {path.name} "
                f"(exit {result.returncode})",
                file=sys.stderr,
            )
            return result.returncode if result.returncode > 0 else 1
    print("All six stdlib reproduction suites passed. No GPU replay was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
