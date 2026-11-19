"""Real CPU Torch fixtures for full four-arm serialized-state readback."""
from __future__ import annotations

import copy
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import audit_multiseed_checkpoints as audit
import multiseed_backup as backup
from checkpoint_payload_checks import tensor_digest
from test_multiseed_backup import mark_complete

byte = backup.byte


def model_state(t_updates=0):
    return {"weight": torch.tensor([1.25]), "joint_alignment.mixer.base_codebooks": torch.tensor([0.5]),
            "joint_alignment.hard_codes": torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
            "joint_alignment.diger_optimizer_step": torch.tensor(t_updates),
            "joint_alignment.diger_total_optimizer_steps": torch.tensor(60)}


def boundary(best_bytes, *, terminal, formal):
    logical, r_epochs, r_updates, t_updates, refreshes = (45, 35, 3640, 60, 10) if formal else (3, 2, 4, 2, 1)
    state = model_state(t_updates)
    progress = {"next_epoch": logical, "best_epoch": 0, "best_val_score": 0.125, "eval_count": 8,
                "no_improve_count": 7, "cumulative_optimizer_updates": r_updates, "decoder_epochs_completed": r_epochs,
                "fit_elapsed_seconds": 1.0, "training_complete": terminal, "stop_reason": "max_epochs",
                "epoch_timing_records": [{} for _ in range(logical)], "output_paths": {},
                "joint_progress": {"boundary_phase": "recommender", "last_committed_phase": "recommender",
                    "next_phase": "recommender", "next_phase_index": logical, "tokenizer_epochs_completed": refreshes,
                    "tokenizer_optimizer_updates": t_updates, "sid_refresh_count": refreshes,
                    "last_refresh_stats": {}, "initial_validation_complete": True}}
    layout = [{"group_name": "recommender", "parameters": ["weight"]},
              {"group_name": "joint_codebook", "parameters": ["joint_alignment.mixer.base_codebooks"]}]
    runtime = {"version": 1, "phase": "recommender", "optimizer_groups": layout,
               "initial_codebooks": torch.ones(1), "recommender_codebook_cache": torch.ones(1),
               "requires_grad": {"weight": True}, "module_training": {"model": True}, "alignment_settings": {},
               "diger_schedule": {"optimizer_step": t_updates, "total_optimizer_steps": 60},
               "hard_codes_sha256": tensor_digest(state["joint_alignment.hard_codes"][1:]),
               "generate_w_decoding_graph": False, "mapping_restore": {}}
    optimizer = {"param_groups": [{"group_name": row["group_name"], "params": [i]} for i, row in enumerate(layout)],
                 "state": {i: {"step": torch.tensor(float(n)), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.ones(1)}
                           for i, n in enumerate((r_updates, t_updates))}}
    return {"format": "diffgrm_joint_phase_boundary_v1", "model": state, "optimizer": optimizer,
            "scheduler": {"last_epoch": r_updates, "_last_lr": [0.01, 0.01]}, "scaler": None,
            "rng": {"python": random.Random(2025).getstate(), "numpy": np.random.RandomState(2025).get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda_all": [torch.get_rng_state()],
                    "loader_counters": {"train": 0}, "loader_generators": {"train": torch.get_rng_state()}},
            "progress": progress, "joint_runtime": runtime, "configuration_fingerprint": "a" * 64,
            "best_checkpoint_bytes": best_bytes, "best_checkpoint_sha256": __import__("hashlib").sha256(best_bytes).hexdigest(),
            "diagnostic_offsets": {str(i): 0 for i in range(5)}}


class CheckpointReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="multiseed_payload_test_")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, corrupt=None):
        source, package = self.root / "source", self.root / "package"
        source.mkdir()
        package.mkdir()
        best_stream = io.BytesIO()
        torch.save(model_state(), best_stream)
        best_bytes = best_stream.getvalue()
        for seed in (2025, 2026):
            for variant in ("control", "item_local"):
                name = audit.run_id(seed, variant)
                work = source / f"seeds/seed{seed}/runs/{name}"
                for relative in audit.NECESSARY:
                    path = work / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".bin":
                        path.write_bytes(best_bytes)
                    else:
                        formal = relative == "training_state/last.pt"
                        payload = boundary(best_bytes, terminal=path.name == "last.pt", formal=formal)
                        if corrupt and seed == 2025 and variant == "control" and formal:
                            corrupt(payload)
                        torch.save(payload, path)
                terminal = torch.load(work / "training_state/last.pt", map_location="cpu", weights_only=False)
                p = terminal["progress"]
                fit = {"best_epoch": 0, "best_val_score": p["best_val_score"], "completed_epochs": 45,
                       "training_stop_reason": "max_epochs", "joint_progress": p["joint_progress"],
                       "cumulative_recommender_optimizer_updates": 3640,
                       "training_state_sha256": byte.sha(work / "training_state/last.pt")}
                common = {"campaign_id": f"stage07_item_local_alternating_joint_seed{seed}_20261021_r1",
                          "variant": variant, "run_id": name, "checkpoint_sha256": byte.sha(work / "training_state/best_model.bin"),
                          "status": "formal_training_completed_validation_pending"}
                sid = tensor_digest(model_state()["joint_alignment.hard_codes"][1:])
                byte.write(work / "FIT_COMPLETE.json", fit)
                byte.write(work / "results.json", {**common, "seed": seed, "parent_campaign_id": backup.CAMPAIGN,
                           "fit": fit, "test_data_evaluated": False, "selected_SID_sha256": sid})
                byte.write(work / "COMPLETE.json", {**common, "mode": "formal", "result_file": "results.json",
                           "result_sha256": byte.sha(work / "results.json")})
                byte.write(work / "checkpoint_reload.json", {"strict_all_tensor_match": True,
                           "mapping_synchronized_without_requantization": True, "checkpoint_sha256": common["checkpoint_sha256"],
                           "best_epoch": 0, "matching_committed_SID_sha256": sid})
                (work / "validation").mkdir()
                byte.write(work / "validation/EVALUATION_COMPLETE.json", {"status": "selected_validation_completed", "seed": seed,
                           "variant": variant, "test_data_evaluated": False, "count_per_cell": 22363,
                           "cells": {"beam32": {}, "beam256": {}}, "checkpoint_sha256": common["checkpoint_sha256"], "selected_SID_sha256": sid})
        sources = {path.relative_to(source).as_posix(): path for path in backup.regular_tree(source)}
        receipt = byte.archive_files(sources, package, {"source_identity": {"code": {"commit": "fixture"},
                                     "engine": {"commit": backup.ENGINE_COMMIT}}}, 1024 * 1024)
        mark_complete(package, receipt)
        extracted = self.root / "extracted"
        byte.verify(package, extracted)
        return extracted, package

    def test_four_arm_28_state_cpu_readback(self):
        root, package = self.fixture()
        result = audit.verify(root, package)
        self.assertEqual(result["status"], "LOCAL_CPU_CHECKPOINT_READBACK_PASS")
        self.assertEqual(result["checkpoint_count"], 28)
        self.assertEqual(len(result["formal_arms"]), 4)
        self.assertFalse(result["GPU_initialized"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_adam_clock_rejected_even_if_bytes_valid(self):
        root, package = self.fixture(lambda state: state["optimizer"]["state"][0].update(step=torch.tensor(3641.0)))
        with self.assertRaisesRegex(ValueError, "Adam state clock"):
            audit.verify(root, package)

    def test_nonfinite_state_rejected_even_if_bytes_valid(self):
        root, package = self.fixture(lambda state: state["model"]["weight"].fill_(float("nan")))
        with self.assertRaisesRegex(ValueError, "Nonfinite/non-CPU"):
            audit.verify(root, package)


if __name__ == "__main__":
    unittest.main()
