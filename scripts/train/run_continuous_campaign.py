"""Run the validated 89-training campaign inside one already allocated H100 node."""
import argparse
import json
import sys
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.continuous.campaign import initialize_state, run_queue
from scripts.continuous.trainer import source_identity
from scripts.train.run_continuous import guarded_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--gpu-gate")
    parser.add_argument("--cpu-gate")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--job-name")
    parser.add_argument("--points-per-node-hour", type=float, default=8.0)
    parser.add_argument("--terminal-platform-evidence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = guarded_output(args.output)
    if sum(bool(v) for v in (args.dry_run, args.smoke_only, args.pipeline, args.finalize_only,
                            args.terminal_platform_evidence)) > 1:
        parser.error("Choose one execution mode")
    if args.dry_run:
        state = initialize_state(source_identity(ROOT)["source_sha256"])
        print(json.dumps({"status": "DRY_RUN_ONLY", "initial_search_tasks": len(state["tasks"]),
              "maximum_valid_training_runs": 89, "workers": 8, "nodes": 1,
              "output": str(output), "source_sha256": state["source_sha256"]}, indent=2))
        return
    if args.terminal_platform_evidence:
        evidence = json.loads(Path(args.terminal_platform_evidence).read_text(encoding="utf-8"))
        if (output / "PIPELINE.json").is_file():
            from scripts.continuous.workflow import reconcile_workflow
            state = reconcile_workflow(output, evidence)
        else:
            from scripts.continuous.campaign import reconcile_terminal_job
            state = reconcile_terminal_job(output, evidence)
        print(json.dumps({"status": state["status"]}))
        return
    if args.finalize_only:
        if not args.data:
            parser.error("--data required for independent CPU delivery")
        from scripts.continuous.workflow import finalize_on_cpu
        result = finalize_on_cpu(ROOT, output, args.data)
        print(json.dumps({"status": result["status"], "training_runs": result["verified_training_runs"]}))
        return
    if args.pipeline:
        if not args.cache or not args.cpu_gate or not args.job_name:
            parser.error("--cache, --cpu-gate and --job-name required for GPU pipeline")
        from scripts.continuous.workflow import run_gpu_workflow
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Platform termination signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        result = run_gpu_workflow(ROOT, output, args.cache, args.cpu_gate, args.job_name,
                                  args.points_per_node_hour)
        print(json.dumps(result))
        return
    if args.smoke_only:
        if not args.cache or not args.cpu_gate:
            parser.error("--cache and --cpu-gate required for GPU smoke")
        from scripts.continuous.gpu_smoke import run_gpu_gate
        result = run_gpu_gate(ROOT, output, args.cache, args.cpu_gate)
        print(json.dumps({"status": result["status"], "source_sha256": result["source_sha256"]}))
        return
    if not args.cache or not args.gpu_gate:
        parser.error("--cache and --gpu-gate required for execution")
    state = run_queue(ROOT, output, args.cache, args.gpu_gate)
    print(json.dumps({"status": state["status"], "verified_training_runs": len(state["tasks"])}))


if __name__ == "__main__":
    main()
