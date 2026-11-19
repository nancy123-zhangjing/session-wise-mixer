"""Read-only CF routing contract; no real training/data access."""
import json
import unittest
from scripts.session_wise_ar import cf


class CFRoutingTests(unittest.TestCase):
    def test_original_source_is_unchanged(self):
        cf.verify_source()

    def test_only_existing_b0_path_is_selected(self):
        jobs = cf.commands("dataset", "parent", "new_run", "runtime.json")
        self.assertEqual(set(jobs), {"pretrain", "train", "test"})
        for command in jobs.values():
            self.assertEqual(command[command.index("--arm")+1], "B0")
            self.assertEqual(command[command.index("--config")+1], str(cf.CONFIG))
        cfg = json.loads(cf.CONFIG.read_text())
        self.assertEqual(cfg["training"]["epochs"], 35)
        self.assertEqual(cfg["training"]["t_steps_per_cycle"], 10)

    def test_cannot_write_into_parent(self):
        with self.assertRaises(ValueError):
            cf.commands("data", "parent", "parent/output", "runtime.json")


if __name__ == "__main__":
    unittest.main()
