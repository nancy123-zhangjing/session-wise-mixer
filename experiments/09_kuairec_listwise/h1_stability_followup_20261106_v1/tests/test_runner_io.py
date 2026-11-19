"""Fast regression for evaluation cache/record directories; no model loading."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from scripts.listwise_diffgrm.recovery_stability_run import evaluation_event, select_best, initial_reference
from scripts.listwise_diffgrm.recovery_stability_protocol import specification
from scripts.listwise_diffgrm.training import sha256


class RunnerIOTests(unittest.TestCase):
    def test_new_directory_cache_binding_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new_output"
            state = {"accepted": 1}
            ctrl = SimpleNamespace(gate=SimpleNamespace(state=lambda: dict(state)), protocol={"arm": "r26_low"})
            with patch("scripts.listwise_diffgrm.recovery_stability_run.original_evaluation") as evaluate:
                evaluation_event(ctrl, None, None, None, out, "R29", True)
                evaluation_event(ctrl, None, None, None, out, "R29", True)
                self.assertEqual(evaluate.call_count, 2)
                state["accepted"] = 2
                with self.assertRaises(ValueError): evaluation_event(ctrl, None, None, None, out, "R29", True)

    def test_initial_reference_and_selection_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            spec = specification("r26_hold")
            (out / "initial_R26.pt").write_bytes(b"fixture checkpoint, never loaded")
            summary = out / "parent_summary.json"
            summary.write_text(json.dumps(dict(metrics={"list/ndcg@5": 0.1}, r_updates=26 * 693, t_updates=280)))
            initial_reference(out, spec, dict(summary=str(summary), summary_sha256=sha256(summary)))
            for epoch in (29, 32, 35):
                (out / "validation_records" / f"R{epoch}.json").write_text(json.dumps(dict(epoch=epoch, ndcg5=0.1)))
            selected = select_best(out, spec)
            self.assertEqual(selected["selected"]["epoch"], 26)
            self.assertEqual(set(selected["candidates"]), {"26", "29", "32", "35"})


if __name__ == "__main__":
    unittest.main()
