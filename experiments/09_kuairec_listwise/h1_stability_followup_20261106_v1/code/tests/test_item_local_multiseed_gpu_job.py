"""CPU-only rejection tests for the new bounded multiseed Job wrapper."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("multiseed_job", ROOT / "scripts/train/item_local_multiseed_gpu_job.py")
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class MultiSeedJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="multiseed_job_test_")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def reservation(self):
        allocations = []
        for seed in job.SEEDS:
            allocations.append(dict(key=job.job_key("smoke", seed, None, 1), seed=seed, mode="smoke", variant=None,
                                    attempt=1, gpu_count=1, platform_max_seconds=900, max_process_seconds=750))
            for variant in job.VARIANTS:
                allocations.append(dict(key=job.job_key("formal", seed, variant, 1), seed=seed, mode="formal", variant=variant,
                                        attempt=1, gpu_count=1, platform_max_seconds=10620, max_process_seconds=10440))
        value = dict(schema="stage07_multiseed_budget_v1", campaign_id=job.CAMPAIGN, status="passed",
                     execution_cap_gpu_hours=14, budget_scope="execution_safety_cap_not_user_total_budget",
                     allocations=allocations, automatic_retry=False)
        return value

    def gate_budget(self, value):
        path = self.root / "budget.json"
        put(path, value)
        return job.budget_gate(path, "formal", 2025, "control", 1, 10440, 10620, self.root / "claims")

    def test_initial_budget_reserves_all_six_jobs(self):
        result = self.gate_budget(self.reservation())
        self.assertEqual(result["reserved_and_spent_gpu_seconds"], 12.3 * 3600)

    def test_budget_rejects_missing_and_duplicate_jobs(self):
        for kind in ("missing", "duplicate"):
            value = self.reservation()
            if kind == "missing":
                value["allocations"].pop()
            else:
                value["allocations"].append(copy.deepcopy(value["allocations"][0]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.gate_budget(value)

    def test_budget_rejects_overspend_and_automatic_retry(self):
        value = self.reservation()
        extra = copy.deepcopy(value["allocations"][1]); extra.update(key="formal_seed2025_control_attempt2", attempt=2)
        value["allocations"].append(extra)
        with self.assertRaises(ValueError):
            self.gate_budget(value)
        value = self.reservation(); value["automatic_retry"] = True
        with self.assertRaises(ValueError):
            self.gate_budget(value)

    def test_budget_cannot_forget_already_claimed_attempt(self):
        put(self.root / "claims/formal_seed2025_control_attempt2.json", {})
        with self.assertRaises(ValueError):
            self.gate_budget(self.reservation())

    def test_run_output_isolated_across_all_seeds_and_arms(self):
        paths = {job.run_path(self.root, seed, variant) for seed in job.SEEDS for variant in job.VARIANTS}
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(path.is_relative_to(self.root / "seeds") for path in paths))
        with self.assertRaises(ValueError):
            job.run_id(2024, "control")

    def test_record_tampering_and_path_escape_rejected(self):
        path = self.root / "evidence.json"; put(path, {"ok": True})
        evidence = job.record(path)
        job.verify_record(evidence, self.root)
        put(path, {"ok": False})
        with self.assertRaises(ValueError):
            job.verify_record(evidence, self.root)
        with self.assertRaises(ValueError):
            job.under(self.root, "../outside")

    def test_cpu_gate_requires_real_nonskipped_source_bound_test_receipts(self):
        deployment=self.root / "deployment.json";put(deployment,{"locked":True})
        put(self.root / "code" / job.CONFIG,{"config":True})
        gate=dict(schema="stage07_multiseed_cpu_gate_v1",campaign_id=job.CAMPAIGN,status="passed",
                  tests_passed=True,cuda_initialized=False,deployment_lock_sha256=job.sha(deployment),
                  config_sha256=job.sha(self.root / "code" / job.CONFIG),tests=[],prepared_runs=[])
        for index,name in enumerate(job.CPU_TESTS):
            put(self.root / "code" / name,{"test":index})
            proof=dict(status="passed",test_file=name,source_sha256=job.sha(self.root / "code" / name),
                       tests_run=1,skipped=0,failures=0,errors=0,cuda_initialized_before=False,
                       cuda_initialized_after=False,cuda_available=False)
            receipt=self.root / f"test{index}.json";log=self.root / f"test{index}.log"
            put(receipt,proof);put(log,{"passed":1})
            gate["tests"].append(dict(proof,receipt=job.record(receipt),log=job.record(log)))
        path=self.root / "cpu_gate.json"
        put(path,gate)
        # All valid test proofs reach the following preparation guard.
        with self.assertRaisesRegex(ValueError,"four unique"):
            job.cpu_gate(self.root,path,deployment)
        for field,bad in (("skipped",1),("tests_run",0),("cuda_initialized_after",True)):
            modified=copy.deepcopy(gate)
            modified["tests"][0][field]=bad
            proof=job.load(modified["tests"][0]["receipt"]["path"]);proof[field]=bad
            altered=self.root / f"altered_{field}.json";put(altered,proof)
            modified["tests"][0]["receipt"]=job.record(altered)
            put(path,modified)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"skipped, failed or used CUDA"):
                job.cpu_gate(self.root,path,deployment)
        put(self.root / "test0.log",{"changed":True});put(path,gate)
        with self.assertRaisesRegex(ValueError,"Evidence bytes differ"):
            job.cpu_gate(self.root,path,deployment)

    def evaluation_fixture(self):
        run=self.root / "run"
        training=dict(checkpoint={"sha256":"best"},selected_SID_sha256="sid",fit={"best_epoch":25})
        cells={}
        for beam in (32,256):
            cell=run / "validation" / f"beam{beam}"
            metrics=dict(seed=2025,variant="control",count=22363,split="val",beam_act=beam,beam_max=beam,
                         model_state_unchanged=True,selected_SID_unchanged=True,training_gradients_created=False,
                         test_data_evaluated=False,checkpoint_sha256="best",selected_SID_sha256="sid",best_epoch=25,
                         model_state_before_sha256="state",model_state_after_sha256="state",
                         user_target_order_sha256="order",original_val32_reproduction={"status":"matched"})
            put(cell / "metrics.json",metrics);put(cell / "CELL_STARTED.json",{})
            files={"metrics":job.record(cell / "metrics.json")}
            for name in ("per_user","parsed_config","environment"):
                put(cell / (name+".json"),{});files[name]=job.record(cell / (name+".json"))
            receipt=dict(status="cell_completed",identity_sha256="identity",beam=beam,
                         started_receipt=job.record(cell / "CELL_STARTED.json"),files=files)
            put(cell / "CELL_COMPLETE.json",receipt)
            cells[f"beam{beam}"]=dict(receipt=job.record(cell / "CELL_COMPLETE.json"),files=files,metrics=metrics)
        value=dict(status="selected_validation_completed",seed=2025,variant="control",identity_sha256="identity",
                   count_per_cell=22363,model_state_unchanged=True,selected_SID_unchanged=True,
                   training_gradients_created=False,test_data_evaluated=False,training_performed=False,
                   checkpoint_sha256="best",selected_SID_sha256="sid",best_epoch=25,cells=cells,
                   original_val32_reproduction={"status":"matched"})
        path=run / "validation/EVALUATION_COMPLETE.json";put(path,value)
        return run,training,value,path

    def test_final_beam_pair_requires_matching_actual_cell_evidence(self):
        run,training,value,path=self.evaluation_fixture()
        job.evaluation_result(run,2025,"control",training)
        value["cells"]["beam256"]["metrics"]["beam_act"]=32;put(path,value)
        with self.assertRaisesRegex(ValueError,"actual cell evidence"):
            job.evaluation_result(run,2025,"control",training)

    def test_final_beam_pair_rejects_other_best_or_test(self):
        run,training,value,path=self.evaluation_fixture()
        training["checkpoint"]["sha256"]="different"
        with self.assertRaises(ValueError):job.evaluation_result(run,2025,"control",training)
        training["checkpoint"]["sha256"]="best";value["test_data_evaluated"]=True;put(path,value)
        with self.assertRaises(ValueError):job.evaluation_result(run,2025,"control",training)

    def training_fixture(self, mode="formal", variant="control"):
        seed = 2025
        run = job.run_path(self.root, seed, variant)
        work = run / "smoke" if mode == "smoke" else run
        for relative in (job.CONFIG, job.ADAPTER, "scripts/train/run_item_local_joint.py"):
            path = self.root / "code" / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"frozen source")
        state_dir = work / "training_state"; state_dir.mkdir(parents=True)
        (state_dir / "best_model.bin").write_bytes(b"selected best")
        (state_dir / "last.pt").write_bytes(b"committed last")
        initial = dict(initial_base_state_sha256="base", initial_rng_sha256="rng", initial_SID_sha256="initial sid")
        put(work / "initial_state_identity.json", initial)
        identifier = job.run_id(seed, variant) + ("_smoke" if mode == "smoke" else "")
        identity = dict(campaign_id=job.seed_campaign(seed), parent_campaign_id=job.CAMPAIGN, seed=seed, variant=variant,
                        run_id=identifier, study_sha256=job.sha(self.root / "code" / job.CONFIG), engine_commit=job.ENGINE_COMMIT,
                        multiseed_adapter={"path":str(self.root / "code" / job.ADAPTER),"sha256":job.sha(self.root / "code" / job.ADAPTER),
                                           "base_runner_sha256":job.sha(self.root / "code/scripts/train/run_item_local_joint.py")})
        put(work / "identity.json", identity)
        put(work / "parsed_config.json", dict(rand_seed=seed,patience=0,vectorized_beam_search={"val":{"beam_act":32,"beam_max":32}}))
        put(work / "input_identities.json", {"artifacts":[{"path":f"file{i}","sha256":str(i),"size_bytes":1} for i in range(13)]})
        evaluations = [0,1,3] if mode == "smoke" else [0,5,12,19,25,32,39,45]
        put(work / "evaluation_timings.json", [{"split":"val","epoch":epoch} for epoch in evaluations])
        r_updates, t_updates, count, refreshes = (4,2,3,1) if mode == "smoke" else (3640,60,45,10)
        covered = ["decoder.weight", "joint_alignment.mixer.base_codebooks"]
        if variant == "item_local": covered.append("item_local_attention.weight")
        phases = ["recommender","tokenizer","recommender"] if mode == "smoke" else ["recommender"]*5 + ["tokenizer","recommender","recommender","recommender"]*10
        records=[]
        for epoch,phase in enumerate(phases,1):
            names=[name for name in covered if name.startswith("joint_alignment.") == (phase=="tokenizer")]
            records.append(dict(epoch=epoch,phase=phase,optimizer_updates=2 if mode=="smoke" else 6 if phase=="tokenizer" else 104,
                                changed_parameters=names,active_gradient_hook_calls={name:1 for name in names},active_parameter_tensors=len(names),
                                frozen_parameters_unchanged=True,all_active_gradients_finite=True,direct_recommendation_gradient_norm=1.0,
                                SID_integrity={"nearest_assignments_verified":True,"catalogue_mapping_matches":True}))
        put(work / "phase_integrity.json", records)
        integrity=dict(status="passed",global_codebook_mixer_enabled=False,local_block_parameter_side="recommender",
                       phase_records_sha256=job.sha(work / "phase_integrity.json"),
                       optimizer=dict(status="passed",all_active_optimizer_steps_match=True,recommender_optimizer_updates=r_updates,
                                      tokenizer_optimizer_updates=t_updates,covered_parameter_names=covered,optimizer_state_count=len(covered)))
        put(work / "training_integrity.json", integrity)
        fit=dict(completed_epochs=count,training_stop_reason="max_epochs",cumulative_recommender_optimizer_updates=r_updates,
                 joint_progress=dict(tokenizer_optimizer_updates=t_updates,sid_refresh_count=refreshes,tokenizer_epochs_completed=refreshes),
                 training_state_sha256=job.sha(state_dir / "last.pt"))
        result=dict(campaign_id=job.seed_campaign(seed),parent_campaign_id=job.CAMPAIGN,seed=seed,variant=variant,run_id=identifier,
                    status="smoke_passed_not_formal" if mode=="smoke" else "formal_training_completed_validation_pending",
                    test_data_evaluated=False,checkpoint_sha256=job.sha(state_dir / "best_model.bin"),selected_SID_sha256="selected sid",
                    fit=fit,training_integrity=integrity,training_budget_policy="fixed_45",final_validation_pending=[32,256],**initial)
        if mode=="smoke":
            result["recovery_check"]=dict(status="passed",strict_model_optimizer_scheduler_RNG_joint_runtime_restore=True,
                                         optimizer_updates_executed=1,tokenizer_parameters_and_SID_unchanged=True)
            put(work / "recovery_probe/recovery_check.json",result["recovery_check"])
        put(work / "checkpoint_reload.json", dict(strict_all_tensor_match=True,mapping_synchronized_without_requantization=True,
                                                  checkpoint_sha256=result["checkpoint_sha256"],matching_committed_SID_sha256="selected sid"))
        filename="smoke_results.json" if mode=="smoke" else "results.json"
        put(work / filename,result)
        complete=dict(mode=mode,status=result["status"],campaign_id=result["campaign_id"],variant=variant,run_id=identifier,
                      result_file=filename,result_sha256=job.sha(work / filename),checkpoint_sha256=result["checkpoint_sha256"])
        put(work / "COMPLETE.json",complete)
        return work,result,complete

    def test_formal_full_budget_and_smoke_evidence_accepted(self):
        self.training_fixture("formal")
        result=job.training_result(self.root,2025,"control","formal")
        self.assertEqual(result["fit"]["cumulative_recommender_optimizer_updates"],3640)
        self.training_fixture("smoke","item_local")
        result=job.training_result(self.root,2025,"item_local","smoke")
        self.assertEqual(result["fit"]["joint_progress"]["tokenizer_optimizer_updates"],2)

    def test_early_stop_is_not_fixed_budget_completion(self):
        work,result,complete=self.training_fixture()
        result["fit"]["training_stop_reason"]="early_stopping"
        put(work / "results.json",result);complete["result_sha256"]=job.sha(work / "results.json");put(work / "COMPLETE.json",complete)
        with self.assertRaises(ValueError): job.training_result(self.root,2025,"control","formal")

    def test_no_test_markers_or_missing_gradient_evidence(self):
        work,_,_=self.training_fixture()
        put(work / "TEST_STARTED.json",{})
        with self.assertRaises(ValueError): job.training_result(self.root,2025,"control","formal")
        (work / "TEST_STARTED.json").unlink()
        phases=job.load(work / "phase_integrity.json");phases[0]["active_gradient_hook_calls"]={};put(work / "phase_integrity.json",phases)
        with self.assertRaises(ValueError): job.training_result(self.root,2025,"control","formal")

    def test_recovery_requires_terminal_release_and_latest_owner(self):
        old=self.root / "previous.json"
        put(old,dict(seed=2025,variant="control",status="failed_or_incomplete",owned_process_groups_alive_after_cleanup=[]))
        owner=self.root / "owner.json";put(owner,dict(attempt=1,receipt=str(old)))
        release=self.root / "release.json"
        value=dict(campaign_id=job.CAMPAIGN,status="passed",seed=2025,variant="control",job_terminal=True,gpu_released=True,
                   previous_attempt=1,previous_receipt=job.record(old))
        put(release,value);job.release_gate(release,owner,2025,"control",2)
        value["gpu_released"]=False;put(release,value)
        with self.assertRaises(ValueError):job.release_gate(release,owner,2025,"control",2)
        value["gpu_released"]=True;put(release,value);put(owner,dict(attempt=2,receipt=str(old)))
        with self.assertRaises(ValueError):job.release_gate(release,owner,2025,"control",3)


if __name__ == "__main__":
    unittest.main()
