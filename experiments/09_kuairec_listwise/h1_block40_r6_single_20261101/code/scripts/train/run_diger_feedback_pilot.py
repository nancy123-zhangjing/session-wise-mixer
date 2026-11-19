"""Validation-only Cross-PSE reconstruction vs recommendation-feedback pilot.

Run all three shared-start arms in a NEW outputs/pilots/stage05_feedback/<name>
directory. First use --smoke; it is a plumbing test, never a research result.
This entrypoint reuses the matched noise-pilot runtime without changing its
legacy arm definitions or selecting checkpoints on the test set.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_adjacent_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load executed module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("all", "frozen", "cross_recon", "cross_rec"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Permitted only with explicit --smoke")
    return parser.parse_args(argv)


def validate_output(root, candidate):
    """Protect historical outputs and source paths before importing the ML stack."""
    root = Path(root).resolve()
    parent = root / "outputs" / "pilots" / "stage05_feedback"
    # Do not follow symlinks into an unexpected shared or historical directory.
    for path in (root / "outputs", root / "outputs" / "pilots", parent):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Feedback output parent must be a real project directory: {path}")
    raw = Path(candidate) if Path(candidate).is_absolute() else root / candidate
    resolved = raw.resolve()
    if resolved.parent != parent or resolved.name in ("", ".", "..") or raw.is_symlink():
        raise ValueError("Use a new single child: outputs/pilots/stage05_feedback/<unique_name>")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {resolved}")
    return resolved


def main(argv=None):
    args = parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.output = validate_output(args.project_root, args.output)
    if args.allow_cpu and not args.smoke:
        raise ValueError("--allow-cpu is only permitted with --smoke")
    helper = load_adjacent_module("_executed_feedback_protocol", ROOT / "scripts/checks/diger_feedback_pilot_protocol.py")
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    helper.load_protocol(config_path)
    runtime = load_adjacent_module("_executed_feedback_runtime", Path(__file__).with_name("run_diger_noise_pilot.py"))
    args.entrypoint = Path(__file__).resolve()
    args.output_owned = False
    try:
        runtime.run(args, protocol_helper=helper)
    except Exception as exc:
        if args.output_owned and not (args.output / "FAILED.json").exists():
            runtime.write_json(args.output / "FAILED.json", {
                "error": type(exc).__name__, "message": str(exc), "complete": False})
        raise


if __name__ == "__main__":
    main()
