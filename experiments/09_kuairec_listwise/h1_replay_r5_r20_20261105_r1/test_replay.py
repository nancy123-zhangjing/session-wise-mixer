"""Checkpoint publication and stopping gates, without GPU or scientific changes."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from replay import BOUNDARIES, BoundaryPublisher, StopAtR20, compare_predictions, sha256


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.published = []

        def original(output, state, *, save_best):
            self.published.append(copy.deepcopy(state))
            (output / "last.pt").write_bytes(json.dumps(state, sort_keys=True).encode())

        self.publish = BoundaryPublisher(original, self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def state(self, phase):
        epoch, t_updates = BOUNDARIES[phase]
        validation = self.output / f"validation_R{epoch:02d}"
        validation.mkdir()
        (validation / "summary.json").write_text(json.dumps(dict(
            r_updates=epoch * 693, t_updates=t_updates, examples=71657,
            full_split=True, beam=32, metrics={"list/ndcg@5": .01})))
        return dict(progress=dict(phase=phase, update=0, validations=[str(validation)]),
                    r_updates=epoch * 693, t_updates=t_updates,
                    model={"weight": [1, 2]}, rng={"state": [5, 6]})

    def test_every_validation_boundary_retained_then_stop_exactly_at_r20(self):
        for phase, (epoch, t_updates) in BOUNDARIES.items():
            state = self.state(phase)
            before = copy.deepcopy(state)
            if epoch == 20:
                with self.assertRaises(StopAtR20):
                    self.publish(self.output, state, save_best=True)
            else:
                self.publish(self.output, state, save_best=False)
            self.assertEqual(state, before)
            path = self.output / "checkpoints" / f"R{epoch:02d}_T{t_updates:03d}.pt"
            self.assertEqual(sha256(path), sha256(self.output / "last.pt"))
        self.assertEqual(len(list((self.output / "checkpoints").glob("*.pt"))), 5)
        self.assertEqual([s["progress"]["phase"] for s in self.published], [9, 13, 17, 21, 25])

    def test_incomplete_validation_is_not_archived(self):
        state = self.state(21)
        p = Path(state["progress"]["validations"][-1]) / "summary.json"
        d = json.loads(p.read_text())
        d["examples"] = 8192
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            self.publish(self.output, state, save_best=False)
        self.assertFalse((self.output / "checkpoints").exists())

    def test_no_overwrite_and_no_post_r20_training(self):
        state = self.state(21)
        self.publish(self.output, state, save_best=False)
        with self.assertRaises(FileExistsError):
            self.publish(self.output, state, save_best=False)
        count = len(self.published)
        state["progress"]["phase"] = 26
        with self.assertRaises(RuntimeError):
            self.publish(self.output, state, save_best=False)
        self.assertEqual(len(self.published), count)

    def test_predictions_comparison_detects_changed_and_missing_rows(self):
        paths = [self.root / f"{name}.gz" for name in ("original", "same", "changed")]
        for index, path in enumerate(paths):
            with gzip.open(path, "wt") as stream:
                for i in range(71657):
                    stream.write(json.dumps({"id": i, "prediction": 2 if index == 2 and i == 7 else 1}) + "\n")
        self.assertTrue(compare_predictions(paths[0], paths[1])["exact"])
        difference = compare_predictions(paths[0], paths[2])
        self.assertFalse(difference["exact"])
        self.assertEqual(difference["differing_rows"], 1)


if __name__ == "__main__":
    unittest.main()
