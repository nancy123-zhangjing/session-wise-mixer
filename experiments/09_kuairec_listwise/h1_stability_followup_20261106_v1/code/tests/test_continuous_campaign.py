"""Dependency and selection gates; no GPU allocation or fake training claims."""
import copy
import sys
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.continuous.campaign import (
    expand_dependencies, freeze_validation_selection, initialize_state, next_task,
    reconcile_terminal_job,
)


def finish_logic_fixture(task, ndcg=None):
    """Only selection-logic fixture data; never written to run artifacts."""
    task["status"] = "succeeded"
    index = int(task["trial_id"].split("_")[-1])
    task["result"] = {"best_validation": {
        "NDCG@10": index / 100 if ndcg is None else ndcg, "HR@10": index / 90},
        "parameter_count": 1000 + index}


class CampaignTests(unittest.TestCase):
    def infrastructure_fixture(self, parent):
        state = initialize_state("logic-test")
        task = state["tasks"][0]
        output = parent/"runs"/task["task_id"]/"attempt_0"
        output.mkdir(parents=True)
        task.update(status="interrupted", output=str(output))
        (parent/"queue.json").write_text(json.dumps(state))
        evidence = {"terminal_status": "FAILED", "reason": "node_lost", "platform_job_name": "unit-fixture",
                    "platform_job_id": "unit-fixture-id", "observed_unix": 1,
                    "raw_platform_observations": {"unit_test_fixture": True}}
        return state, evidence

    def test_no_retry_from_local_pid_or_nonterminal_platform_state(self):
        parent = ROOT/"tmp/continuous_campaign_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            directory = Path(tmp)
            _, evidence = self.infrastructure_fixture(directory)
            for invalid in ({}, {**evidence, "terminal_status": "RUNNING"},
                            {**evidence, "reason": "implementation_error"},
                            {**evidence, "raw_platform_observations": {}}):
                with self.assertRaises(ValueError):
                    reconcile_terminal_job(directory, invalid)

    def test_at_most_two_infrastructure_retries(self):
        parent = ROOT/"tmp/continuous_campaign_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            directory = Path(tmp)
            _, evidence = self.infrastructure_fixture(directory)
            for expected_attempt in (1, 2):
                state = reconcile_terminal_job(directory, evidence)
                self.assertEqual(state["tasks"][0]["attempt"], expected_attempt)
                self.assertEqual(state["tasks"][0]["status"], "pending")
                state["tasks"][0]["status"] = "interrupted"
                (directory/"queue.json").write_text(json.dumps(state))
            before = (directory/"queue.json").read_bytes()
            with self.assertRaises(RuntimeError):
                reconcile_terminal_job(directory, evidence)
            self.assertEqual((directory/"queue.json").read_bytes(), before)

    def test_live_original_worker_blocks_reconciliation(self):
        parent = ROOT/"tmp/continuous_campaign_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            directory = Path(tmp)
            state, evidence = self.infrastructure_fixture(directory)
            identity = {"pid": 99, "host": "unit-fixture", "start_ticks": "1", "command": "unit-fixture"}
            state["workers"]["0"] = {"task_id": state["tasks"][0]["task_id"], "identity": identity,
                                      "output": state["tasks"][0]["output"]}
            (directory/"queue.json").write_text(json.dumps(state))
            with mock.patch("scripts.continuous.campaign.process_identity", return_value=identity):
                with self.assertRaises(RuntimeError):
                    reconcile_terminal_job(directory, evidence)

    def test_initial_four_four_assignment_and_stealing(self):
        state = initialize_state("test")
        self.assertEqual(len(state["tasks"]), 64)
        self.assertEqual(next_task(state, 0)["config"]["model"], "sasrec_fullce")
        self.assertEqual(next_task(state, 7)["config"]["model"], "esasrec")
        for task in state["tasks"]:
            if task["config"]["model"] == "sasrec_fullce":
                task["status"] = "running"
        self.assertEqual(next_task(state, 0)["config"]["model"], "esasrec")

    def test_no_selection_from_31_completed_or_failed_searches(self):
        state = initialize_state("test")
        for task in state["tasks"][:31]:
            finish_logic_fixture(task)
        self.assertEqual(expand_dependencies(state), [])
        state["tasks"][31]["status"] = "failed"
        self.assertEqual(expand_dependencies(state), [])
        self.assertEqual(state["top3"], {})

    def test_exact_89_task_graph_and_seed_sets(self):
        state = initialize_state("test")
        for task in state["tasks"]:
            finish_logic_fixture(task)
        self.assertEqual(len(expand_dependencies(state)), 12)
        self.assertEqual(len(state["tasks"]), 76)
        self.assertEqual(expand_dependencies(state), [])
        for task in state["tasks"]:
            if task["phase"] == "confirmation":
                finish_logic_fixture(task)
        self.assertEqual(len(expand_dependencies(state)), 13)
        self.assertEqual(len(state["tasks"]), 89)
        self.assertEqual(expand_dependencies(state), [])
        self.assertEqual(sum(t["phase"] == "formal" for t in state["tasks"]), 10)
        self.assertEqual(sum(t["phase"] == "legacy" for t in state["tasks"]), 3)
        self.assertTrue(all(t["protocol"] == "legacy_prefix_v1" for t in state["tasks"]
                            if t["phase"] == "legacy"))

    def test_completion_order_does_not_change_top3_or_graph(self):
        first = initialize_state("test")
        second = copy.deepcopy(first)
        for index in range(64):
            finish_logic_fixture(first["tasks"][index])
            expand_dependencies(first)
            finish_logic_fixture(second["tasks"][63-index])
            expand_dependencies(second)
        self.assertEqual(first["top3"], second["top3"])
        self.assertEqual({t["task_id"] for t in first["tasks"]},
                         {t["task_id"] for t in second["tasks"]})

    def test_confirmation_waits_for_both_new_seeds(self):
        state = initialize_state("test")
        for task in state["tasks"]:
            finish_logic_fixture(task)
        expand_dependencies(state)
        confirmations = [t for t in state["tasks"] if t["phase"] == "confirmation"
                         and t["config"]["model"] == "sasrec_fullce"]
        for task in confirmations[:-1]:
            finish_logic_fixture(task)
        self.assertEqual(expand_dependencies(state), [])
        self.assertEqual(state["selected_configs"], {})
        finish_logic_fixture(confirmations[-1])
        self.assertEqual(len(expand_dependencies(state)), 8)
        self.assertEqual(state["selected_configs"]["sasrec_fullce"]["trial_id"], "sasrec_fullce_31")

    def test_test_gate_requires_all_89_and_uses_validation_only(self):
        state = initialize_state("test")
        with self.assertRaises(ValueError):
            freeze_validation_selection(state)
        for _ in range(3):
            for task in state["tasks"]:
                finish_logic_fixture(task)
            expand_dependencies(state)
        lock = freeze_validation_selection(state)
        self.assertTrue(lock["validation_only"])
        self.assertTrue(lock["test_unlocked"])
        state["tasks"][-1]["status"] = "running"
        with self.assertRaises(ValueError):
            freeze_validation_selection(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
