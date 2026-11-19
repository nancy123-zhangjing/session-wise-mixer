"""Aggregate the real Stage05 long-factorial delivery proofs without rerunning ML.

Require the completed local-backup binder first. Rehash small evidence/source
files, not the already verified multi-gigabyte payload. Write a new aggregate
receipt only after every requirement passes. This does not change Goal state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]
CAMPAIGN = "goal_20261020_long_mixer_v1"
ARMS = ("no_mixer_frozen", "no_mixer_joint", "mixer_frozen", "mixer_joint")
CODE = "69682698c15ba796b98e6329c950061a2f68a05b"
ENGINE = "1c879252f6395d6e64462511823c3929b9c42c00"
ANALYSIS = "4b40c90b56490868e5972e75bafbf8e50ca8a3de"
PACKAGE_SHA = "23ca79aa50e980c8f85be4447715c665a965e5a79aabecc3e361d0f58683e300"
MANIFEST_SHA = "14bc3502ee5a207dd64ccf29657099080e982650418d1f9016e9ca633c4b1ce8"
ARCHIVE_SHA = "c2ad35b0c87b219e6435a970dfd62bf5677cf7e7c2924b34c53e48c6fec47af7"
DOC_MARKER = "<!-- stage05-long-delivery-verified -->"


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def sha(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
         "Input changed while hashing: " + str(path))
    return h.hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}
        self.requirements = []

    def file(self, path, expected=None):
        path = Path(path).resolve()
        need(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),
             "Missing/nonregular/outside-workspace evidence: " + str(path))
        digest = sha(path)
        need(expected is None or digest == expected, "Evidence SHA changed: " + str(path))
        relative = path.relative_to(ROOT).as_posix()
        self.files[relative] = dict(path=relative, size_bytes=path.stat().st_size, sha256=digest)
        return digest

    def json(self, path, expected=None):
        self.file(path, expected)
        return read(path)

    def passed(self, name, paths, **details):
        self.requirements.append(dict(requirement=name, passed=True,
            evidence=[Path(p).resolve().relative_to(ROOT).as_posix() for p in paths],
            details=details))


def section(text, start, end=None):
    need(text.count(start) == 1, "Expected one current-document section: " + start)
    result = text.split(start, 1)[1]
    if end is not None:
        need(result.count(end) == 1, "Missing current-document section end: " + end)
        result = result.split(end, 1)[0]
    return result


def local_links(path, text, output):
    checked = []
    for value in re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", text):
        value = value.strip()
        if value.startswith("<") and ">" in value:
            value = value[1:value.index(">")]
        else:
            value = value.split(' "', 1)[0]
        if value.startswith(("#", "https://", "http://", "mailto:", "codex://", "plugin://")):
            continue
        value = re.sub(r":\d+$", "", value.split("#", 1)[0])
        target = Path(value)
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve()
        need(target == output or target.exists(), "Broken current-delivery link: " + value)
        checked.append(str(target))
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=EVIDENCE / "FINAL_BACKUP_BINDING_AUDIT.json")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "FINAL_DELIVERY_AUDIT.json")
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.is_relative_to(EVIDENCE) and not output.exists(), "Require a new aggregate receipt in this experiment")
    ev = Evidence()
    binding_path = args.binding.resolve()
    b = ev.json(binding_path)
    need(b["status"] == "FINAL_FULL_BACKUP_RECEIPTS_BOUND_NOT_GOAL_COMPLETION"
         and b["campaign_id"] == CAMPAIGN and b["all_checks_passed"] is True
         and b["local_full_backup_verified"] is True
         and b["local_final_downloaded_package_git_recovery_verified"] is True
         and b["remote_cpu_consumer_restoration_bound"] is True,
         "Full local archive/Git/remote-model binding must pass first")
    ev.file(EVIDENCE / "delivery_tools/bind_final_backup_receipts.py", b["binder_sha256"])
    paths = {k: Path(v).resolve() for k, v in b["paths"].items()}
    need(all(p.is_relative_to(ROOT) for p in paths.values()), "Binding points outside the local workspace")
    backup, recovered, remote = (paths[k] for k in ("verified_backup_root", "git_restore_root", "remote_evidence"))
    package = ev.json(paths["package_dir"] / "PACKAGE.json", PACKAGE_SHA)
    manifest = ev.json(backup / "MANIFEST.json", MANIFEST_SHA)
    need(b["package_raw_sha256"] == PACKAGE_SHA and b["manifest_raw_sha256"] == MANIFEST_SHA
         and b["archive_sha256"] == package["archive_sha256"] == ARCHIVE_SHA
         and b["all_local_archive_parts_and_concatenation_rehashed"] is True
         and b["all_manifest_members_rehashed"] is True
         and b["verified_manifest_member_count"] == len(manifest["files"]) == package["files"] == 1871
         and b["locked_input_count"] == 14 and b["formal_complete_state_count"] == 9,
         "Final archive identity/full-payload proof differs")
    members = {r["member"]: r for r in manifest["files"]}
    need(len(members) == 1871 and "MANIFEST.json" not in members, "Manifest member inventory differs")
    for part in package["parts"]:
        actual = paths["package_dir"] / part["member"]
        need(actual.is_file() and actual.stat().st_size == part["size_bytes"], "A verified archive part is missing or resized")
    local_byte = ev.json(backup / "BYTE_VERIFICATION.json", b["local_byte_verification_raw_sha256"])
    git_receipt = ev.json(recovered / "FINAL_GIT_RESTORE_PASS.json", b["local_git_restore_raw_sha256"])
    need(local_byte["all_member_bytes_verified"] is True and local_byte["all_14_locked_inputs_included"] is True
         and git_receipt["actual_final_downloaded_package_used"] is True
         and git_receipt["all_backup_members_rehashed"] is True
         and b["actual_recovered_git_tracked_count"] == 1493
         and b["formal_execution_source_hashes_checked"] == 19,
         "Local byte/Git recovery proof is incomplete")
    ev.passed("Complete final local archive, all 14 inputs, nine states and real Git recovery",
              [binding_path, paths["package_dir"] / "PACKAGE.json", backup / "MANIFEST.json",
               backup / "BYTE_VERIFICATION.json", recovered / "FINAL_GIT_RESTORE_PASS.json"],
              archive_sha256=ARCHIVE_SHA, manifest_records=1871, manifest_itself_additional=True,
              archive_size_bytes=package["archive_size_bytes"], restored_tracked_sources=1493,
              verification_scope="Payload rehash/restore reused from the bound successful verifier; aggregate rehashes evidence and source inputs.")

    audit_path = EVIDENCE / "final_results_r1/delivery/final_audit_r1/AUDIT.json"
    a = ev.json(audit_path, b["scientific_audit_raw_sha256"])
    ev.file(paths["scientific_audit"], b["scientific_audit_raw_sha256"])
    need(a["status"] == "FORMAL_ARTIFACT_AUDIT_PASS_NOT_DELIVERY"
         and a["campaign_id"] == CAMPAIGN and a["tests_checked"] is True
         and a["checkpoints_checked"] is True and set(a["summaries"]) == set(ARMS),
         "Require actual complete four-arm model-and-test audit")
    review_path = EVIDENCE / "INDEPENDENT_SCIENTIFIC_DELIVERY_REVIEW.json"
    review = ev.json(review_path)
    need(review["status"] == "SCIENTIFIC_DELIVERY_INDEPENDENT_REVIEW_PASS_NOT_LOCAL_BACKUP"
         and review["campaign_id"] == CAMPAIGN, "Missing independent actual scientific review")
    for row in review["source_files"]:
        ev.file(ROOT / row["path"], row["sha256"])
    formal = EVIDENCE / "final_results_r1/formal"
    progress, tests = {}, {}
    for arm in ARMS:
        summary = a["summaries"][arm]
        joint = arm.endswith("_joint")
        need(summary["complete"] is True and summary["recommender_epochs"] == 35
             and summary["recommender_updates"] == 3640
             and summary["tokenizer_updates"] == (60 if joint else 0)
             and summary["noisy_calls"] == (1920 if joint else 0), "Literal arm budget differs: " + arm)
        for name, digest in a["source_receipt_sha256"][arm].items():
            ev.file(formal / arm / name, digest)
        progress[arm] = read(formal / arm / "progress.json")
        need(progress[arm]["decoder_epochs"] == 35 and progress[arm]["decoder_updates"] == 3640
             and progress[arm]["tokenizer_updates"] == summary["tokenizer_updates"]
             and progress[arm]["noisy_calls"] == summary["noisy_calls"], "Actual final progress differs")
        counts = review["training_counts_and_phase_controls"][arm]
        need(counts["recommender_phases"] == 35 and counts["recommender_batches_per_phase"] == 1652
             and counts["recommender_updates"] == 3640
             and counts["tokenizer_phases"] == (10 if joint else 0)
             and counts["tokenizer_updates"] == summary["tokenizer_updates"]
             and counts["actual_noise_calls"] == summary["noisy_calls"]
             and counts["exact_phase_order_verified"] is True
             and counts["all_recorded_gradient_Adam_and_frozen_side_checks_pass"] is True,
             "Independent complete phase/gradient review differs")
    need(a["controls"]["common_recommender_epochs"] == list(range(1, 36))
         and a["controls"]["frozen_loss_epochs"] == list(range(1, 36))
         and a["controls"]["common_tokenizer_cycles"] == list(range(10)), "Frozen or paired scope incomplete")
    paired_boolean_keys = (
        "first_five_recommender_epoch_loss_dicts_all_four_exact",
        "frozen_all_35_epoch_loss_dicts_exact",
        "frozen_all_eight_validation_metric_dicts_exact",
        "frozen_best_and_last_model_tensor_and_SID_hashes_exact_in_actual_checkpoint_audit",
        "frozen_formal_test_complete_metric_dicts_exact")
    need(all(review["paired_controls"][k] is True for k in paired_boolean_keys),
         "Independent Frozen/paired checks failed")
    ev.passed("Full 5R plus ten alternating cycles, actual noise, gradients, optimizer/freeze and paired controls",
              [audit_path, review_path], per_arm_R_epochs=35, per_arm_R_updates=3640,
              joint_T_updates=60, joint_actual_noise_calls=1920,
              frozen_tokenizer_updates=0, frozen_identity_negative_control=True)

    metric_names = tuple(a["Mixer_increment"])
    final = a["fixed_epoch35_validation"]
    did = {k: (final["mixer_joint"][k] - final["mixer_frozen"][k])
           - (final["no_mixer_joint"][k] - final["no_mixer_frozen"][k]) for k in metric_names}
    need(did == a["Mixer_increment"] == review["fixed_R35_primary_analysis"]["Mixer_increment"],
         "Fixed-R35 unrounded DID differs")
    selection_path = formal / "SELECTION_SEALED.json"
    seal = ev.json(selection_path)
    seal_sha = ev.file(selection_path)
    for arm in ARMS:
        test_path = formal / arm / "TEST_RESULT.json"
        test = tests[arm] = ev.json(test_path)
        need(test["split"] == test["actual_dataset_split"] == "test"
             and test["beam_profile_not_test"] is False and test["selection_sha256"] == seal_sha
             and test["best_sha256"] == seal["arms"][arm]["best_sha256"]
             and test["started_sha256"] == ev.file(formal / arm / "TEST_STARTED.json")
             and test["decoder_epoch"] == a["summaries"][arm]["best_epoch"]
             and test["selected_tokenizer_updates"] == a["summaries"][arm]["best_tokenizer_updates"]
             and all(test["metrics"][k] == v for k, v in a["summaries"][arm]["selected_test"].items()),
             "One-time sealed test/checkpoint chain differs: " + arm)
    need(tests["no_mixer_frozen"]["metrics"] == tests["mixer_frozen"]["metrics"],
         "Frozen full test dictionaries differ")
    coverage_path = EVIDENCE / "TEST_COVERAGE_AUDIT.json"
    coverage = ev.json(coverage_path)
    need(coverage["status"] == "FOUR_FORMAL_TEST_LOG_COVERAGE_PASS_NOT_DELIVERY"
         and coverage["full_artifact_audit_sha256"] == ev.file(audit_path)
         and coverage["selection_sha256"] == seal_sha
         and coverage["compute_complete_sha256"] == ev.file(formal / "COMPUTE_COMPLETE.json")
         and coverage["resolved_config_sha256"] == ev.file(formal / "resolved_config_evaluate_all.json")
         and coverage["environment_sha256"] == ev.file(formal / "environment_evaluate_all.json")
         and coverage["one_completed_LONG_EVAL_per_arm_in_this_log"] is True
         and coverage["four_result_records_and_LONG_DONE_match_compute_complete"] is True
         and coverage["runtime_single_accelerator_process_assertion_passed"] is True
         and coverage["runtime_test_dataset_length_assertion_passed"] is True
         and coverage["test_beam_act"] == coverage["test_beam_max"] == 256
         and coverage["eval_batch_size"] == 8 and coverage["max_eval_batches"] is None,
         "Actual complete-test coverage/source chain is incomplete")
    ev.file(EVIDENCE / coverage["log_file"], coverage["log_sha256"])
    for arm in ARMS:
        c = coverage["arms"][arm]
        need(c["progress_records"] == 2798 and c["logged_complete_batches"] == 2796
             and c["all_counts_0_through_2796_present"] is True and c["progress_nondecreasing"] is True
             and c["dataset_samples_asserted_at_runtime"] == 22363
             and c["test_result_sha256"] == ev.file(formal / arm / "TEST_RESULT.json"),
             "A full test log/receipt chain differs")
    ev.passed("Fixed-R35 primary DID and separate validation-selected one-time complete formal tests",
              [audit_path, review_path, selection_path, coverage_path, formal / "COMPUTE_COMPLETE.json"],
              fixed_R35_Mixer_increment=did, test_models={arm: dict(R=v["decoder_epoch"], T=v["selected_tokenizer_updates"])
              for arm, v in tests.items()}, complete_batches_per_test=2796, dataset_samples=22363,
              boundary="Coverage is full execution-path/log evidence, not per-sample-ID prediction replay.")

    figdir = EVIDENCE / "final_results_r1/delivery/final_figures_r1"
    figure_path = figdir / "FIGURES.json"
    figures = ev.json(figure_path)
    need(figures["artifact_audit_sha256"] == ev.file(audit_path) and len(figures["files"]) == 6
         and sum(r["file"].endswith(".png") for r in figures["files"]) == 3
         and sum(r["file"].endswith(".pdf") for r in figures["files"]) == 3, "Six audit-bound figures required")
    for row in figures["files"]:
        ev.file(figdir / row["file"], row["sha256"])
    raw_csv_names = ("fixed_epoch35_validation.csv", "best_selected_test.csv", "phases.csv", "validation.csv")
    for name in raw_csv_names:
        ev.file(audit_path.parent / name, members["delivery/final_audit_r1/" + name]["sha256"])
    ev.passed("Raw results and six audit-bound plot files", [figure_path],
              raw_CSV_files=list(raw_csv_names), raw_CSV_hashes_equal_verified_archive=True,
              files=figures["files"], visual_inspection="Three PNGs actually viewed in prior formal review; aggregate checks their unchanged hashes.")

    resource_path = EVIDENCE / "RESOURCE_RELEASE_FINAL.json"
    resource = ev.json(resource_path, manifest["resource_release_sha256"])
    need(resource["status"] == "RESOURCE_RELEASE_AND_ALLOCATION_AUDIT_PASS"
         and resource["all_goal_jobs_released"] is True and len(resource["jobs"]) == 7
         and all(j["released"] is True for j in resource["jobs"])
         and sum(j["allocated_gpu_seconds"] for j in resource["jobs"]) == resource["actual_allocated_gpu_seconds"] == 35549
         and resource["acceptance_allocated_gpu_seconds"] == 636
         and resource["actual_allocated_gpu_seconds"] <= 43200
         and resource["acceptance_allocated_gpu_seconds"] <= 1800
         and resource["actual_allocated_gpu_hours"] == 35549 / 3600
         and resource["rate_points_per_allocated_gpu_hour"] == 1.0
         and resource["estimated_points_from_allocated_time"] == 35549 / 3600
         and resource["actual_goal_invoice_settlement_verified"] is False, "Budget/release/fee evidence differs")
    for job in resource["jobs"]:
        for name, digest in job["platform_evidence_sha256"].items():
            ev.file(EVIDENCE / name, digest)
    ev.file(EVIDENCE / "PRE_GPU_QUOTA.json", resource["quota_evidence_sha256"])
    ev.file(EVIDENCE / "FINAL_JOB_LIST_R1.json", resource["final_job_list_sha256"])
    ev.passed("All seven Goal GPU allocations released within total and acceptance budgets",
              [resource_path], actual_allocated_gpu_seconds=35549, total_budget_seconds=43200,
              acceptance_seconds=636, acceptance_budget_seconds=1800,
              estimated_points=resource["estimated_points_from_allocated_time"], actual_invoice_verified=False)

    cpu_path = remote / ("delivery/restore_check_cpu_r1/" + CAMPAIGN + "/RESTORE_PASS.json")
    cpu = ev.json(cpu_path, b["remote_cpu_restore_raw_sha256"])
    remote_summary_path = remote / "delivery/restore_check_execution_r1/REMOTE_RESTORE_SUMMARY.json"
    remote_summary = ev.json(remote_summary_path, b["remote_summary_raw_sha256"])
    ev.file(remote / "REMOTE_RESTORE_EVIDENCE_AUDIT.json", b["independent_remote_evidence_audit_sha256"])
    remote_byte_path = remote / ("delivery/package_extract_r1/" + CAMPAIGN + "/BYTE_VERIFICATION.json")
    ev.file(remote_byte_path, b["remote_byte_verification_raw_sha256"])
    need(cpu["status"] == "FORMAL_FULL_BACKUP_CPU_RESTORE_PASS" and len(cpu["checkpoints"]) == 8
         and cpu["all_14_inputs_copied_from_verified_backup"] is True
         and cpu["original_checkpoints_unchanged"] is True
         and cpu["execution_backend"] == "cpu" and cpu["cuda_execution_verified_here"] is False
         and cpu["byte_verification_sha256"] == b["remote_byte_verification_raw_sha256"]
         and remote_summary["formal_states_restored"] == 8
         and b["eight_real_cpu_restores_bound_to_actual_local_states_and_scientific_audit"] is True
         and b["common_initial_reconstruction_enforced_by_locked_cpu_restorer"] is True,
         "Full final-archive CPU model recovery binding incomplete")
    for row in cpu["checkpoints"]:
        name = "formal/" + row["arm"] + "/" + row["checkpoint"]
        need(row["sha256"] == members[name]["sha256"] == b["formal_complete_state_hashes"][name],
             "Actual CPU-restored raw checkpoint differs from final local archive")
    cuda_paths = []
    for arm in ("no_mixer_joint", "mixer_joint"):
        path = EVIDENCE / "h100_preflight_r1/smoke" / arm / "RECOVERY_PASS.json"
        value = ev.json(path, members["smoke/" + arm + "/RECOVERY_PASS.json"]["sha256"])
        need(value["status"] == "passed" and value["device"] == "cuda:0"
             and value["model_optimizer_scheduler_RNG_restored"] is True
             and value["repeated_real_next_phase_identical"] is True
             and value["original_checkpoints_unchanged"] is True
             and value["source_sha256"] == members["smoke/" + arm + "/tokenizer_boundary.pt"]["sha256"],
             "Actual H100 next-phase recovery proof missing or source boundary differs")
        cuda_paths.append(path)
    ev.passed("Actual eight-state final-backup CPU recovery and both H100 boundary replay proofs",
              [binding_path, cpu_path, remote_summary_path, *cuda_paths], common_initial_reconstructed=True,
              final_complete_R35_is_last_pt=True, CPU_restore_executes_CUDA=False,
              local_Pipeline_restore_executed=False, remote_same_downloaded_bytes_CPU_restore=True)

    lock_path = EVIDENCE / "ANALYSIS_SOURCE_LOCK_R3.json"
    tools_path = EVIDENCE / "ANALYSIS_TOOLING_V3R2.json"
    lock, tooling = ev.json(lock_path), ev.json(tools_path)
    need(lock["training_code_commit"] == tooling["training_code_commit"] == CODE
         and lock["engine_commit"] == tooling["engine_commit"] == ENGINE
         and lock["analysis_commit"] == ANALYSIS
         and b["recovered_git_repositories"]["code"]["head"] == CODE
         and b["recovered_git_repositories"]["engine"]["head"] == ENGINE
         and manifest["source_locks"]["code"]["commit"] == CODE
         and manifest["source_locks"]["engine"]["commit"] == ENGINE, "Execution/analysis code versions differ")
    for name, digest in tooling["files"].items():
        ev.file(ROOT / name, digest)
        if name.startswith("scripts/analysis/"):
            blob = subprocess.run(["git", "-C", str(ROOT), "show", ANALYSIS + ":" + name], capture_output=True)
            need(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == digest,
                 "Analysis source not in its declared frozen commit: " + name)
    need(a["auditor_sha256"] == tooling["files"]["scripts/analysis/audit_mixer_long_campaign.py"]
         and resource["auditor_sha256"] == tooling["files"]["scripts/analysis/audit_mixer_long_resources.py"]
         and cpu["restorer_sha256"] == tooling["files"]["scripts/analysis/restore_mixer_long_backup.py"]
         and figures["plotter_sha256"] == tooling["files"]["scripts/analysis/plot_mixer_long_campaign.py"]
         and manifest["packer_sha256"] == tooling["files"]["scripts/analysis/pack_mixer_long_delivery.py"],
         "Executed audit/restore/plot/pack tools differ from analysis lock")
    for directory, field in (("code", "root_sources"), ("engine", "engine_sources")):
        for name, digest in a["identity"][field].items():
            ev.file(recovered / directory / name, digest)
            need(members[directory + "/" + name]["sha256"] == digest, "Recovered execution source not bound to archive")
    protocol_path = ROOT / "configs/05_crosspse_joint/long/mixer_long_seed2024.json"
    protocol = ev.json(protocol_path, a["identity"]["protocol_sha256"])
    need(protocol["seed"] == 2024 and protocol["mixer_scope"] == "full"
         and "AmazonReviews2014_Aug-28-2026_19-01-02" in protocol["initial_decoder"]
         and protocol["mixer_zero_output"] is True and protocol["mixer_residual_scale"] == 1
         and protocol["mixer_learn_residual_scale"] is False
         and protocol["max_allocated_gpu_seconds"] == 43200
         and protocol["max_acceptance_gpu_seconds"] == 1800, "Shared Aug-28/E0 identity or protocol boundary differs")
    for arm in ARMS:
        env_path = formal / ("environment_train_" + arm + ".json")
        env = ev.json(env_path)
        need(env["code_commit"] == CODE and env["engine_commit"] == ENGINE
             and env["initial_model_tensor_sha256"] == a["initial_model_tensor_sha256"],
             "Actual arm initial/code environment differs")
    ev.passed("Frozen training versions, actual analysis sources and common Aug-28/E0 protocol identity",
              [lock_path, tools_path, protocol_path], training_commit=CODE, engine_commit=ENGINE,
              analysis_commit=ANALYSIS, actual_execution_source_hashes=19,
              analysis_tool_and_fixture_hashes=len(tooling["files"]))

    report_path, readme_path = EVIDENCE / "REPORT.md", EVIDENCE / "README.md"
    entry_path, handoff_path = ROOT / "experiments/05_crosspse_joint/README.md", ROOT / "docs/guides/PROJECT_HANDOFF.md"
    document_sections = {
        report_path: report_path.read_text(encoding="utf-8"),
        readme_path: readme_path.read_text(encoding="utf-8"),
        entry_path: section(entry_path.read_text(encoding="utf-8"), "<!-- stage05-long-factorial-20261020 -->"),
        handoff_path: section(handoff_path.read_text(encoding="utf-8"),
            "<!-- stage05-long-factorial-handoff-begin -->", "<!-- stage05-long-factorial-handoff-end -->")}
    document_links = {}
    for path, text in document_sections.items():
        ev.file(path)
        need(DOC_MARKER in text, "Current delivery document lacks final reviewed marker: " + str(path))
        need(not re.search(r"\]\([^)\n]*REPORT_DRAFT\.md", text), "Current entry still links the draft")
        need("FINAL_DELIVERY_AUDIT.json" in text, "Current document must link the aggregate audit")
        if path != report_path:
            need("REPORT.md" in text, "Current entry must link final REPORT.md")
        if path in (report_path, readme_path):
            need("FINAL_BACKUP_BINDING_AUDIT.json" in text, "Final report/entry must link actual backup binding")
        document_links[str(path.relative_to(ROOT))] = local_links(path, text, output)
    report = document_sections[report_path]
    need(all(token in report for token in ("R35", "1920", "3640", "seed2024", "SID-level", "CPU", "CUDA")),
         "Final report omits core protocol/interpretation boundaries")
    need(("实际结算" in report or "实际账单" in report or "实际发票" in report)
         and ("参数" in report and "优化路径" in report)
         and ("码本内部" in report and "码本之间" in report), "Final report omits cost or Full-interaction limitations")
    status_path = EVIDENCE / "EXECUTION_STATUS.json"
    status = ev.json(status_path)
    need(status["remote_full_archive_byte_and_actual_cpu_model_restore_passed"] is True
         and status["local_full_archive_byte_and_git_restore_passed"] is True,
         "Current execution entry still reports pending complete backup")
    ev.file(EVIDENCE / "RESTORE_GUIDE.md")
    ev.file(EVIDENCE / "ACCEPTANCE_MATRIX.md")
    ev.file(EVIDENCE / "GOAL_COMMAND.txt")
    ev.passed("Final report, experiment entries, project handoff and usable local evidence links",
              [report_path, readme_path, entry_path, handoff_path, status_path,
               EVIDENCE / "RESTORE_GUIDE.md", EVIDENCE / "ACCEPTANCE_MATRIX.md"],
              checked_current_section_links=document_links,
              current_source_branch_not_used_as_training_identity=True)
    ev.file(Path(__file__))
    result = dict(status="ALL_REQUIRED_DELIVERY_CHECKS_PASS", all_required_delivery_checks_passed=True,
        campaign_id=CAMPAIGN, utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(Path(__file__)),
        requirements=ev.requirements, evidence_files=sorted(ev.files.values(), key=lambda x: x["path"]),
        local_full_backup_verified=True, actual_final_archive_model_recovery_bound=True,
        scientific_model_and_test_audit_passed=True, all_goal_GPU_allocations_released=True,
        limitations=dict(actual_invoice_settlement_verified=False, CPU_restore_executed_CUDA=False,
            local_full_Pipeline_model_restore_executed=False, remote_same_archive_CPU_restore_executed=True,
            seed_count=1, seed=2024, scope="Full Mixer overall effect, including within/between interaction",
            additional_parameters_and_optimization_path_not_isolated=True,
            test_best_models_have_different_R_T_budgets=True, SID_level_not_item_level=True,
            per_sample_ID_prediction_replay_executed=False),
        goal_tool_state_changed_here=False, goal_complete=False,
        goal_state_note="All required delivery evidence passes. The root agent must separately update Goal tool state.")
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(output), sha256=sha(output),
        requirements=len(ev.requirements), evidence_files=len(ev.files), goal_tool_state_changed_here=False),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps(dict(status="REQUIRED_DELIVERY_CHECK_FAILED", error=str(exc),
                              success_receipt_written=False), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
