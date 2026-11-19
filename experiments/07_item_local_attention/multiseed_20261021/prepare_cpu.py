"""CPU acceptance and immutable preparation of the four new Stage07 arms.

Run from the isolated campaign checkout with the owned grm-exp Python:
    python -B prepare_cpu.py --goal-root /.../goal_20261021_multiseed_r1
This script never submits a Job, imports torch in its parent, or trains a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


BASE_CODE_COMMIT = "842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1"
PROJECT = Path("/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized")
SELF_RELATIVE = "experiments/07_item_local_attention/multiseed_20261021/prepare_cpu.py"
TEST_FILES = (
    "tests/test_item_local_joint_multiseed.py",
    "tests/test_item_local_multiseed_evaluation.py",
    "tests/test_item_local_multiseed_gpu_job.py",
    "tests/test_item_local_alternating_joint.py",
    "tests/test_item_local_joint_training_state.py",
    "tests/test_ocn_autocast_gradient.py",
)
EXTRA_FILES = (SELF_RELATIVE, *TEST_FILES, "tests/test_item_local_attention.py",
               "tests/test_item_local_training_state.py",
               "experiments/05_crosspse_joint/reproducibility/recipes/refined_joint.json",
               "experiments/05_crosspse_joint/reproducibility/assets.lock.json",
               "experiments/05_crosspse_joint/evidence/refined_joint/run_metadata.json",
               "experiments/05_crosspse_joint/evidence/configs/diff_formal_common.yaml",
               "experiments/05_crosspse_joint/evidence/configs/diff_formal_repo_v6.yaml",
               "experiments/05_crosspse_joint/evidence/configs/diff_formal_refinedpse_joint_diger_v1.yaml")


TEST_HARNESS = r"""
import hashlib, importlib.util, json, os, pathlib, sys, traceback, unittest
path, receipt = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
row = {'test_file': sys.argv[3], 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
       'python_executable': sys.executable, 'status': 'failed'}
try:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(2)
    row.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
               cuda_initialized_before=torch.cuda.is_initialized())
    assert not row['cuda_available'] and not row['cuda_initialized_before']
    def forbidden_cuda_init(*args, **kwargs):
        raise RuntimeError('CUDA initialization is forbidden in CPU acceptance')
    torch.cuda._lazy_init = forbidden_cuda_init
    spec = importlib.util.spec_from_file_location('isolated_cpu_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    row.update(tests_run=result.testsRun, skipped=len(result.skipped), failures=len(result.failures),
               errors=len(result.errors), cuda_initialized_after=torch.cuda.is_initialized())
    assert result.testsRun > 0 and result.wasSuccessful() and not result.skipped
    assert row['cuda_initialized_after'] is False
    row['status'] = 'passed'
except BaseException as error:
    row.update(error_type=type(error).__name__, error=str(error))
    traceback.print_exc()
finally:
    with receipt.open('x', encoding='utf-8') as handle:
        json.dump(row, handle, indent=2, allow_nan=False); handle.write('\n')
sys.exit(0 if row['status'] == 'passed' else 1)
"""


PREPARE_HARNESS = r"""
import importlib.abc, json, pathlib, runpy, sys, traceback
adapter, receipt = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
arguments = sys.argv[3:]
row = {'status': 'failed', 'cuda_initialized': False, 'training_started': False,
       'test_evaluated': False, 'python_executable': sys.executable}
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('CPU preparation must not import torch')
sys.meta_path.insert(0, NoTorch())
try:
    assert 'torch' not in sys.modules
    sys.argv = [str(adapter), *arguments]
    runpy.run_path(str(adapter), run_name='__main__')
    assert 'torch' not in sys.modules
    row.update(status='passed', torch_imported=False)
except BaseException as error:
    row.update(error_type=type(error).__name__, error=str(error))
    traceback.print_exc()
finally:
    with receipt.open('x', encoding='utf-8') as handle:
        json.dump(row, handle, indent=2, allow_nan=False); handle.write('\n')
sys.exit(0 if row['status'] == 'passed' else 1)
"""


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def git(path, *arguments):
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def load_module(path):
    spec = importlib.util.spec_from_file_location("multiseed_cpu_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    result = os.environ.copy()
    result.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", OMP_NUM_THREADS="2",
                  MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
                  PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        result.pop(name, None)
    return result


def source_snapshot(goal, wrapper):
    code, engine = goal / "code", goal / "engine"
    need(Path(__file__).resolve() == code / SELF_RELATIVE, "Run CPU acceptance from this campaign's frozen checkout")
    need((code / "external/DiffGRM").resolve() == engine, "CPU tests must resolve to this campaign's engine")
    commits = {name: git(goal / name, "rev-parse", "HEAD") for name in ("code", "engine")}
    need(commits["engine"] == wrapper.ENGINE_COMMIT, "Frozen engine commit differs")
    need(git(code, "merge-base", BASE_CODE_COMMIT, commits["code"]) == BASE_CODE_COMMIT,
         "New code must descend from the frozen base commit")
    need(all(not git(goal / name, "status", "--porcelain") for name in commits), "Source checkout is not clean")
    names = {"code/" + name for name in (*wrapper.SOURCE_FILES, *EXTRA_FILES)}
    names |= {"engine/" + name for name in wrapper.ENGINE_FILES}
    records = []
    for name in sorted(names):
        target = wrapper.under(goal, name)
        need(target.is_file(), "Required source is missing: " + name)
        records.append({"path": name, "sha256": wrapper.sha(target), "size_bytes": target.stat().st_size})
    return {"schema": "stage07_multiseed_deployment_v1", "campaign_id": wrapper.CAMPAIGN,
            "code_commit": commits["code"], "engine_commit": commits["engine"],
            "base_code_commit": BASE_CODE_COMMIT, "files": records}


def no_training_markers(work):
    forbidden = ("STARTED.json", "TEST_STARTED.json", "TEST_RESULT.json", "FIT_COMPLETE.json", "COMPLETE.json",
                 "training_state/last.pt", "training_state/best_model.bin", "initial_state_identity.json")
    need(not any((work / name).exists() for name in forbidden), "Preparation found a started/completed GPU run: " + str(work))
    need(not any((work / "attempts").glob("*/STARTED.json")), "A preparation attempt contains a GPU-start marker")


def prepared_row(goal, seed, variant, wrapper):
    work = wrapper.run_path(goal, seed, variant)
    no_training_markers(work)
    files = {name: wrapper.record(work / filename) for name, filename in {
        "identity": "identity.json", "resolved_config": "resolved_config.json",
        "input_identities": "input_identities.json", "code_identity": "code_identity.json",
        "study": "study.json", "recipe": "recipe.json", "dataset_audit": "dataset_audit.json",
        "formal_configuration_audit": "formal_configuration_audit.json"}.items()}
    identity, config, inputs, sources = [wrapper.load(files[key]["path"])
                                        for key in ("identity", "resolved_config", "input_identities", "code_identity")]
    need(identity["seed"] == seed and identity["variant"] == variant
         and identity["run_id"] == wrapper.run_id(seed, variant)
         and identity["study_sha256"] == wrapper.sha(goal / "code" / wrapper.CONFIG)
         and identity["engine_commit"] == wrapper.ENGINE_COMMIT, "Prepared seed/source/study differs")
    bound = identity["multiseed_adapter"]
    need(bound["sha256"] == wrapper.sha(goal / "code" / wrapper.ADAPTER)
         and bound["base_runner_sha256"] == wrapper.sha(goal / "code/scripts/train/run_item_local_joint.py")
         and sources["multiseed_adapter"] == bound, "Prepared adapter/base source differs")
    need(config["rand_seed"] == seed and config["patience"] == 0 and config["epochs"] == 45
         and config["max_train_batches_per_epoch"] is None and config["max_eval_batches"] is None
         and not config["resume_training_state"], "Prepared configuration is truncated, resumed, or changed")
    lock = wrapper.load(goal / "code/experiments/05_crosspse_joint/reproducibility/assets.lock.json")
    recipe = wrapper.load(files["recipe"]["path"])
    need({row["path"] for row in inputs["artifacts"]} == set(recipe["required_artifacts"])
         and len(inputs["artifacts"]) == 13, "Prepared artifacts do not match the exact recipe")
    for row in inputs["artifacts"]:
        path = wrapper.under(work / "workspace", row["path"])
        need(row["sha256"] == lock["artifacts"][row["path"]]["sha256"]
             and path.is_file() and not path.is_symlink() and wrapper.sha(path) == row["sha256"]
             and path.stat().st_size == row["size_bytes"], "Prepared input bytes differ: " + row["path"])
    return {"seed": seed, "variant": variant, "run_id": wrapper.run_id(seed, variant),
            "output": str(work), "files": files}


def reservation(wrapper):
    rows = []
    for seed in wrapper.SEEDS:
        rows.append({"key": wrapper.job_key("smoke", seed, None, 1), "mode": "smoke", "seed": seed,
                     "variant": None, "attempt": 1, "gpu_count": 1, "max_process_seconds": 750,
                     "platform_max_seconds": 900})
        for variant in wrapper.VARIANTS:
            rows.append({"key": wrapper.job_key("formal", seed, variant, 1), "mode": "formal", "seed": seed,
                         "variant": variant, "attempt": 1, "gpu_count": 1, "max_process_seconds": 10440,
                         "platform_max_seconds": 10620})
    return {"schema": "stage07_multiseed_budget_v1", "campaign_id": wrapper.CAMPAIGN, "status": "passed",
            "execution_cap_gpu_hours": 14, "budget_scope": "execution_safety_cap_not_user_total_budget",
            "automatic_retry": False, "allocations": rows, "spent_allocations": [],
            "reserved_gpu_seconds": sum(row["platform_max_seconds"] for row in rows),
            "reserved_gpu_hours": 12.3}


def run_child(command, log, *, code, timeout):
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=code, env=environment(), stdout=handle,
                                stderr=subprocess.STDOUT, timeout=timeout)
    need(result.returncode == 0, "CPU subprocess failed; inspect preserved log: " + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", type=Path, required=True)
    args = parser.parse_args()
    goal = args.goal_root.resolve()
    code, delivery = goal / "code", goal / "delivery"
    wrapper = load_module(code / "scripts/train/item_local_multiseed_gpu_job.py")
    before = source_snapshot(goal, wrapper)
    delivery.mkdir(exist_ok=True)
    deployment_path, gate_path, budget_path = [delivery / name for name in
                                             ("DEPLOYMENT_LOCK.json", "CPU_GATE.json", "BUDGET_RESERVATION.json")]
    if gate_path.exists():
        wrapper.source_gate(goal, deployment_path)
        wrapper.cpu_gate(goal, gate_path, deployment_path)
        budget = wrapper.load(budget_path)
        for row in budget["allocations"]:
            wrapper.budget_gate(budget_path, row["mode"], row["seed"], row["variant"], row["attempt"],
                                row["max_process_seconds"], row["platform_max_seconds"], delivery / "claims")
        print(json.dumps({"status": "existing_cpu_acceptance_verified", "cpu_gate": wrapper.record(gate_path)}, indent=2))
        return
    if deployment_path.exists():
        need(wrapper.load(deployment_path) == before, "Existing deployment lock differs; inspect instead of overwriting")
    if budget_path.exists():
        need(wrapper.load(budget_path) == reservation(wrapper), "Existing budget receipt differs; inspect instead of overwriting")
    attempt = delivery / "cpu_attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}")
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        tests = []
        for relative in TEST_FILES:
            stem = Path(relative).stem
            receipt, log = attempt / (stem + ".json"), attempt / (stem + ".log")
            print("CPU test: " + relative, flush=True)
            run_child([sys.executable, "-B", "-c", TEST_HARNESS, str(code / relative), str(receipt), relative],
                      log, code=code, timeout=600)
            row = wrapper.load(receipt)
            need(row["status"] == "passed" and row["tests_run"] > 0 and row["skipped"] == row["failures"] == row["errors"] == 0
                 and row["cuda_initialized_before"] is False and row["cuda_initialized_after"] is False,
                 "CPU test did not pass without skips/CUDA: " + relative)
            tests.append({**row, "receipt": wrapper.record(receipt), "log": wrapper.record(log)})
        prepared = []
        for seed in wrapper.SEEDS:
            for variant in wrapper.VARIANTS:
                work = wrapper.run_path(goal, seed, variant)
                no_training_markers(work)
                log, receipt = attempt / f"prepare_{seed}_{variant}.log", attempt / f"prepare_{seed}_{variant}.json"
                if work.exists() and any(work.iterdir()):
                    row = prepared_row(goal, seed, variant, wrapper)
                    log.write_text("Reused existing CPU preparation after full identity/input verification.\n", encoding="utf-8")
                    write_new(receipt, {"status": "passed", "mode": "verified_existing_prepare",
                                       "cuda_initialized": False, "training_started": False, "test_evaluated": False})
                else:
                    print(f"CPU prepare: seed={seed} arm={variant}", flush=True)
                    command = [sys.executable, "-B", "-c", PREPARE_HARNESS, str(code / wrapper.ADAPTER), str(receipt),
                               "--seed", str(seed), "--mode", "prepare", "--config", str(code / wrapper.CONFIG),
                               "--variant", variant, "--output", str(work), "--project-root", str(PROJECT),
                               "--engine-root", str(goal / "engine"), "--expected-engine-commit", wrapper.ENGINE_COMMIT]
                    run_child(command, log, code=code, timeout=900)
                    row = prepared_row(goal, seed, variant, wrapper)
                row["preparation"] = {"receipt": wrapper.record(receipt), "log": wrapper.record(log)}
                prepared.append(row)
        need(before == source_snapshot(goal, wrapper), "Sources changed during CPU acceptance")
        if not deployment_path.exists():
            write_new(deployment_path, before)
        if not budget_path.exists():
            write_new(budget_path, reservation(wrapper))
        gate = {"schema": "stage07_multiseed_cpu_gate_v1", "campaign_id": wrapper.CAMPAIGN, "status": "passed",
                "tests_passed": True, "cuda_initialized": False, "test_data_evaluated": False, "gpu_started": False,
                "deployment_lock_sha256": wrapper.sha(deployment_path), "config_sha256": wrapper.sha(code / wrapper.CONFIG),
                "tests": tests, "tests_run": sum(row["tests_run"] for row in tests), "skipped": 0,
                "prepared_runs": prepared, "cpu_environment": {"python_executable": sys.executable, "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                "accepted_utc": datetime.now(timezone.utc).isoformat()}
        # Validate the exact prospective gate before the final immutable receipt.
        candidate = attempt / "CPU_GATE_CANDIDATE.json"
        write_new(candidate, gate)
        wrapper.source_gate(goal, deployment_path)
        wrapper.cpu_gate(goal, candidate, deployment_path)
        for row in reservation(wrapper)["allocations"]:
            wrapper.budget_gate(budget_path, row["mode"], row["seed"], row["variant"], row["attempt"],
                                row["max_process_seconds"], row["platform_max_seconds"], delivery / "claims")
        write_new(gate_path, gate)
        print(json.dumps({"status": "passed", "tests_run": gate["tests_run"], "skipped": 0,
                          "prepared_runs": 4, "gpu_started": False, "cpu_gate": wrapper.record(gate_path)}, indent=2), flush=True)
    except BaseException as error:
        write_new(attempt / "CPU_PREP_FAILED.json", {"status": "failed", "error_type": type(error).__name__,
                                                    "error": str(error), "gpu_started": False})
        raise


if __name__ == "__main__":
    main()
