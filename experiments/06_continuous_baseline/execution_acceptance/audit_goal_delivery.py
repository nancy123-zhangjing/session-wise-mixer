"""Build source-bound delivery status without relabeling unverified settlement.

The accompanying requirements review explains coverage of the original Goal.
This consolidates actual artifacts, not a replacement for their independent audits.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FINAL = HERE / "final_delivery_20261016"
SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    objective = Path("C:/Users/nancy/.codex/attachments/98b43b1e-1197-4638-b7bc-28346ea4988f/goal-objective.md")
    assert sha(objective) == "44ae1b2a62f9fe628ef8f81c483a72138c82745eaee759d7cf253d4f6f780678"
    proofs = {}

    def proof(name, path, status="PASS"):
        value = read(path)
        assert value["status"] == status, (name, value.get("status"))
        proofs[name] = {"path": str(Path(path).relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)}
        return value

    data = proof("data_identity_and_protocols", ROOT / "experiments/06_continuous_baseline/acceptance/data_verification.json")
    assert data["independent_outputs"] == 2 and data["all_content_hashes_equal"]
    for run in data["runs"]:
        assert run["verified_rows"] == 326533
        assert run["counts"] == {"full_train_prefix_v1": {"train": 131413, "validation": 22363, "test": 22363},
                                 "legacy_prefix_v1": {"train": 105668, "validation": 22363, "test": 22363}}
    for name, expected in (("local", 73), ("remote", 36)):
        gate = proof(name + "_cpu_gate", HERE / (name + "_cpu_20261016_02/CPU_GATE.json"))
        assert gate["tests"] == expected and gate["skipped"] == 0 and gate["source_sha256"] == SOURCE
        assert all(s["failures"] == s["errors"] == s["skipped"] == 0 for s in gate["suites"])
    gate = proof("real_gpu_gate", HERE / "scheduler_failure_20261016_01/gpu_gate_supervisor_v3/GPU_GATE_AUDIT.json")
    assert gate["distinct_h100s"] == 8 and gate["parallel_tasks"] == 10 and gate["actual_resume_updates"] == [20, 21]
    assert gate["stress_full_epochs"] == 35 and gate["actual_resumed_tasks"] == 8
    metadata = proof("all_training_metadata", HERE / "completed_metadata_precision_v2_batch20/AUDIT.json")
    assert metadata["completed_trials"] == 89 and metadata["verified_metadata_files"] == 740
    assert Counter(t["phase"] for t in metadata["trials"]) == {"search": 64, "confirmation": 12, "formal": 10, "legacy": 3}
    payload = proof("all_full_state_payloads", HERE / "checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_22.json")
    assert payload["completed_full_payload_trials"] == 89 and payload["complete_states"] == 178
    assert not payload["metadata_only_task_ids"] and not payload["payload_only_task_ids"]
    selection = proof("validation_only_selection_and_order", FINAL / "FINAL_SELECTION_AND_ORDER_AUDIT.json")
    assert selection["raw_curves"] == 89 and selection["validation_only_selected_model"] == "sasrec_fullce"
    delivery = proof("independent_test_and_export_delivery", FINAL / "CPU_DELIVERY_PASS.json")
    assert (delivery["verified_training_runs"], delivery["test_evaluations"], delivery["independent_metric_audits"], delivery["formal_exports"]) == (89, 13, 13, 10)
    for relative, expected in delivery["files"].items():
        assert sha(FINAL / relative) == expected
    for name in FINAL.glob("*_audit.json"):
        if name.name.startswith(("formal__", "legacy__")):
            audit = read(name)
            assert audit["status"] == "PASS"
    metric = proof("local_metric_aggregation", FINAL / "LOCAL_FINAL_METRIC_AUDIT.json")
    assert metric["formal_test_rows_recomputed"] == 10 and metric["metrics_recomputed"] == 6 and metric["standard_deviation_ddof"] == 1
    with (FINAL / "test_results.csv").open(encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 13 and all(int(r["samples"]) == 22363 for r in rows)
    assert Counter(r["phase"] for r in rows) == {"formal": 10, "legacy": 3}
    exports = proof("all_formal_exports", FINAL / "FINAL_EXPORT_AUDIT.json")
    assert exports["formal_exports_audited"] == len(exports["records"]) == 10
    assert all(r["all_model_tensors_exact"] and r["input_output_tables_exact"] and r["portable_query_and_catalog_scores_exact"] for r in exports["records"].values())
    speed = proof("final_raw_benchmark", FINAL / "FINAL_BENCHMARK_AUDIT.json")
    assert speed["raw_measurement_groups"] == 12 and speed["timed_samples"] == 2400 and speed["representative_seed"] == 3031
    proof("curve_visual_review", FINAL / "FIGURE_VISUAL_AUDIT.json")
    recovery = proof("usable_recovery_transaction", FINAL / "LOCAL_RECOVERY_TRANSACTION_AUDIT.json")
    assert recovery["original_repair_committed_passed"] and recovery["strict_model_restore"] and recovery["optimizer_state_restore"]
    assert recovery["scheduler_state_restore"] and recovery["cpu_random_state_restore"] and recovery["completed_marker_preserved"]
    assert not recovery["training_started"]
    preserved = proof("workspace_and_training_source_preservation", FINAL / "FINAL_PRESERVATION_AUDIT.json")
    assert preserved["original_tracked_files_checked"] == 908 and preserved["training_source_files_checked"] == 27
    assert preserved["training_source_sha256"] == SOURCE and preserved["protected_root_readmes_unchanged"]
    retention = proof("prior_attempts_retained", HERE / "resource_accounting_20261016/OLD_ATTEMPTS_RETENTION_20261016_02.json", "ALL_PRIOR_STARTED_ATTEMPTS_EXIST")
    assert [len(r["saved_runs"]) for r in retention["records"]] == [36, 46]
    backup = proof("complete_local_backup", FINAL / "FINAL_BACKUP_LOCAL_AUDIT.json")
    assert (backup["training_runs"], backup["checkpoint_states"], backup["formal_exports"], backup["test_evaluations"], backup["verified_archive_parts"], backup["verified_files"]) == (89, 178, 10, 13, 9, 1300)
    resource = proof("resource_usage_and_release", HERE / "resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json", "RESOURCE_WINDOWS_AND_RATES_PASS")
    assert resource["goal_jobs"] == 5 and resource["maximum_goal_concurrent_gpus"] == 8 and resource["goal_gpu_resources_released"]
    accepted = False
    acceptance_path = FINAL / "FEE_ACCEPTANCE_20261016.json"
    if acceptance_path.is_file():
        acceptance = proof("direct_human_fee_delivery_acceptance", acceptance_path, "USER_ACCEPTED_BILLING_DELIVERY_BOUNDARY")
        assert acceptance["source"] == "Direct human reply to request_user_input_async in this conversation"
        assert acceptance["answer"] == "接受，以此完成Goal" and acceptance["fee_delivery_requirement_accepted"]
        assert acceptance["original_objective_sha256"] == sha(objective)
        assert acceptance["resource_audit_sha256"] == sha(HERE / "resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json")
        assert not acceptance["actual_goal_settlement_verified"]
        accepted = True
    unresolved = [] if resource["actual_goal_settlement_verified"] or accepted else ["Goal-specific actual settlement cannot be attributed from available personal-project aggregate billing; calculated runtime times rate is not verified settlement."]
    documents = [ROOT / "docs/guides/PROJECT_HANDOFF.md", ROOT / "experiments/06_continuous_baseline/README.md",
        ROOT / "experiments/06_continuous_baseline/REPORT.md", ROOT / "experiments/06_continuous_baseline/IMPLEMENTATION_STATUS.md",
        FINAL / "GOAL_REQUIREMENTS_REVIEW.md"]
    document_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in documents}
    result = {"status": "INCOMPLETE" if unresolved else ("ALL_REQUIRED_EVIDENCE_PASS_WITH_USER_ACCEPTED_FEE_BOUNDARY" if accepted else "ALL_REQUIRED_EVIDENCE_PASS"), "observed_unix": time.time(),
        "objective_sha256": sha(objective), "source_sha256": SOURCE, "training_runs": 89, "test_evaluations": 13,
        "formal_exports": 10, "backup_parts": 9, "backup_files": 1300,
        "goal_gpu_resources_released": True, "actual_goal_settlement_verified": resource["actual_goal_settlement_verified"],
        "fee_delivery_boundary_accepted_by_user": accepted,
        "unresolved_required_items": unresolved, "proofs": proofs, "document_sha256": document_hashes,
        "audit_script_sha256": sha(__file__),
        "scope": "Consolidates counts, hashes and artifact-specific independent audits. GOAL_REQUIREMENTS_REVIEW.md covers the original eight sections and their measurement limits. Does not turn calculated cost into settlement or authorize extra training; explicit Git staging is checked separately."}
    (FINAL / "FINAL_DELIVERY_STATUS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"proofs", "document_sha256"}}), flush=True)


if __name__ == "__main__":
    main()
