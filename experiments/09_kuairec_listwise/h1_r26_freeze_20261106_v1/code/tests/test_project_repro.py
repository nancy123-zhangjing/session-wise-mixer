"""CPU-only tests for stage 01-04 provenance and non-destructive packaging.

These tests never build/seal a real package or start a training process. All
fixture writes use temporary directories, leaving historical evidence intact.
"""
from __future__ import annotations

import hashlib
import argparse
import builtins
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repro"))

import project_repro as project
import project_replay as replay


class ProjectReproTests(unittest.TestCase):
    def test_specs_have_14_runs_and_17_stage_references(self):
        items = project.specs()
        self.assertEqual(len(items), 17)
        self.assertEqual(len({r["stamp"] for r in items}), 14)
        self.assertEqual(len({(r["stage"], r["id"]) for r in items}), 17)
        self.assertEqual({s: sum(r["stage"] == s for r in items) for s in project.STAGES},
                         {"01": 6, "02": 5, "03": 3, "04": 3})

    def test_references_preserve_exact_run_and_initialization(self):
        items = project.specs()
        lookup = {(r["stage"], r["id"]): r for r in items}
        references = [r for r in items if r["reference"]]
        self.assertEqual(len(references), 3)
        self.assertEqual({(r["stage"], r["reference"]["stage"]) for r in references},
                         {("02", "01"), ("04", "03")})
        for row in references:
            original = lookup[(row["reference"]["stage"], row["reference"]["id"])]
            for key in ("run_id", "seed", "stamp", "config", "artifact_dir", "model", "semantic_codebook_sizes"):
                self.assertEqual(row[key], original[key], key)

    def test_seed_counts_are_not_promoted_by_baseline_reuse(self):
        items = project.specs()
        for method in ("RQ-KMeans", "RQ-VAE"):
            self.assertEqual({r["seed"] for r in items if r["stage"] == "01" and r["method"] == method},
                             {2024, 2025, 2026})
        self.assertEqual({r["seed"] for r in items if r["stage"] != "01"}, {2024})

    def test_model_and_semantic_dimensions_are_stage_specific(self):
        for row in project.specs():
            expected = ("AR_GRM", 3) if row["stage"] in ("01", "02") else ("DIFF_GRM", 4)
            self.assertEqual((row["model"], len(row["semantic_codebook_sizes"])), expected)
        layouts = [r["semantic_codebook_sizes"] for r in project.specs()
                   if r["stage"] == "02" and r["method"].startswith("RQ-KMeans")]
        self.assertEqual({a * b * c for a, b, c in layouts}, {256 ** 3})

    def test_all_ar_stamps_have_exact_console_binding(self):
        stamps = {r["stamp"].removeprefix("AmazonReviews2014_") for r in project.specs()
                  if r["model"] == "AR_GRM"}
        self.assertEqual(stamps, set(project.AR_LOGS))
        self.assertEqual(len(set(project.AR_LOGS.values())), 9)
        self.assertNotIn("ar_formal_rqkmeans_seed2024_20260809_031510.log", project.AR_LOGS.values())

    def test_immutable_copy_is_idempotent_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.csv"
            original.write_bytes(b"a,b\r\n1,2\r\n")
            copy = root / "package/evidence/copy.csv"
            digest = project.immutable_copy(original, copy)
            self.assertEqual(project.immutable_copy(original, copy), digest)
            self.assertEqual(original.read_bytes(), copy.read_bytes())
            self.assertEqual(digest, hashlib.sha256(original.read_bytes()).hexdigest())

    def test_immutable_copy_refuses_different_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, destination = root / "old", root / "new"
            original.write_bytes(b"source")
            destination.write_bytes(b"user content")
            with self.assertRaisesRegex(ValueError, "Preserved differing"):
                project.immutable_copy(original, destination)
            self.assertEqual(destination.read_bytes(), b"user content")
            self.assertEqual(original.read_bytes(), b"source")

    def test_immutable_bytes_refuses_different_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "copy"
            project.immutable_bytes(b"old", destination)
            project.immutable_bytes(b"old", destination)
            with self.assertRaisesRegex(ValueError, "Preserved differing"):
                project.immutable_bytes(b"new", destination)
            self.assertEqual(destination.read_bytes(), b"old")

    def test_untrusted_relative_paths_cannot_escape(self):
        for value in ("", "../file", "a/../../outside", "/etc/passwd", "C:/file", "a\\..\\outside"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                project.relative(value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(project.under(root, "safe/file"), root / "safe/file")

    def test_recipe_required_inputs_include_identity_and_no_initial_decoder(self):
        shared = {project.DATA + "raw/" + name for name in ("meta_Beauty.json.gz", "reviews_Beauty_5.json.gz")}
        shared |= {project.DATA + "processed/" + name
                   for name in ("all_item_seqs.json", "id_mapping.json", "metadata.sentence.json")}
        for item in project.specs():
            with self.subTest(stage=item["stage"], experiment=item["id"]):
                result = project.recipe_inputs(item, {}, {})
                self.assertTrue(shared.issubset(result["required_artifacts"]))
                self.assertIn(result["initial_sid"], result["required_artifacts"])
                self.assertEqual(len(result["required_artifacts"]), len(set(result["required_artifacts"])))
                self.assertEqual(result["decoder_initialization"],
                                 {"type": "random_initialization_with_seed", "seed": item["seed"], "checkpoint": None})
                self.assertNotIn(result["historical_final_checkpoint"], result["required_artifacts"])
                self.assertIn(result["historical_final_checkpoint"], result["evaluation_artifacts"])

    def test_ar_rqk_recipe_binds_pca_and_quantizer_cache_per_seed(self):
        for item in (r for r in project.specs() if r["model"] == "AR_GRM" and r["method"].startswith("RQ-KMeans")):
            result = project.recipe_inputs(item, {}, {})
            cache = project.DATA + f"processed/sentence-t5-xl_pca32_fitseed{item['seed']}.sent_emb"
            self.assertIn(cache, result["required_artifacts"])
            self.assertIn(cache + ".pkl", result["required_artifacts"])
            for filename in ("reconstructed_embeddings.npy", "quantizer_artifacts.npz", "semantic_ids.npy", "semantic_ids.json"):
                self.assertIn(item["artifact_dir"] + "/" + filename, result["required_artifacts"])

    def test_historically_recorded_sid_hash_must_match(self):
        item = next(r for r in project.specs() if r["stage"] == "04" and not r["reference"])
        sid = project.recipe_inputs(item, {}, {})["initial_sid"]
        with self.assertRaisesRegex(ValueError, "Historical SID hash mismatch"):
            project.recipe_inputs(item, {"semantic_ids_sha256": "a" * 64}, {sid: {"sha256": "b" * 64}})

    def test_check_inputs_rejects_missing_unbound_or_changed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack, inputs = root / "pack", root / "inputs"
            inputs.mkdir()
            expected = hashlib.sha256(b"expected").hexdigest()
            project.write_json(pack / "reproducibility/assets.lock.json", {"artifacts": {"input.bin": {"sha256": expected}}})
            recipe = {"id": "fixture", "required_artifacts": ["input.bin"]}
            with patch.object(project, "package", return_value=pack):
                with self.assertRaisesRegex(ValueError, "Missing"):
                    project.check_inputs("01", recipe, inputs)
                (inputs / "input.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    project.check_inputs("01", recipe, inputs)
                (inputs / "input.bin").write_bytes(b"expected")
                project.check_inputs("01", recipe, inputs)
                with self.assertRaisesRegex(ValueError, "Not bound"):
                    project.check_inputs("01", {"id": "fixture", "required_artifacts": ["other.bin"]}, inputs)

    def make_analysis_fixture(self, pack, stage="01", multiple=True):
        items = [dict(r) for r in project.specs() if r["stage"] == stage]
        if multiple:
            items = [r for r in items if r["method"] == "RQ-KMeans"]
        else:
            items = items[:1]
        for index, item in enumerate(items):
            metrics = {name: (.05 + index * .01 if name.startswith("hit_rate") else .03 + index * .005)
                       for name in project.METRICS}
            metrics.update({"legal@10": 1.0, "duplicate@10": 0.0})
            meta = {"run_id": item["run_id"], "seed": item["seed"], "best_epoch": 55, "best_val_score": .04,
                    "fit_elapsed_seconds": 3600, "test_results": metrics,
                    "checkpoint": "saved/" + item["stamp"] + "/pytorch_model.bin"}
            meta_path = f"evidence/{item['id']}/run_metadata.json"
            report_path = f"evidence/{item['id']}/per_seed.csv"
            project.write_json(pack / meta_path, meta)
            (pack / report_path).parent.mkdir(parents=True, exist_ok=True)
            project.write_csv(pack / report_path, [{"seed": item["seed"], "best_epoch": 55, "fit_hours": 1,
                              "metadata": item["stamp"] + "_run_metadata.json", **metrics}])
            project.write_json(pack / "reproducibility/recipes" / (item["id"] + ".json"),
                               {**item, "metadata": meta_path, "report_per_seed": report_path,
                                "evaluation_level": "item", "selection_metric": "ndcg@10",
                                "replay_mode": "frozen_historical_sid_random_decoder"})
        return items

    def test_analyze_recomputes_sample_std_for_three_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            items = self.make_analysis_fixture(pack)
            with patch.object(project, "package", return_value=pack), patch.object(project, "specs", return_value=items):
                result = project.analyze("01")
            summary = project.rows(pack / "summary.csv")[0]
            self.assertEqual(len(result), 3)
            self.assertEqual(summary["n_seeds"], "3")
            self.assertAlmostEqual(float(summary["hit_rate@10_mean"]), statistics.mean([.05, .06, .07]))
            self.assertAlmostEqual(float(summary["hit_rate@10_std"]), statistics.stdev([.05, .06, .07]))
            self.assertEqual(summary["std_ddof"], "1")

    def test_analyze_single_seed_std_is_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            items = self.make_analysis_fixture(pack, "02", multiple=False)
            with patch.object(project, "package", return_value=pack), patch.object(project, "specs", return_value=items):
                project.analyze("02")
            summary = project.rows(pack / "summary.csv")[0]
            self.assertEqual(summary["n_seeds"], "1")
            for metric in project.METRICS:
                self.assertEqual(summary[metric + "_std"], "")

    def test_analyze_rejects_metric_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            items = self.make_analysis_fixture(pack, multiple=False)
            path = pack / f"evidence/{items[0]['id']}/run_metadata.json"
            meta = project.read_json(path)
            meta["test_results"]["ndcg@10"] = .9
            project.write_json(path, meta)
            with patch.object(project, "package", return_value=pack), patch.object(project, "specs", return_value=items):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    project.analyze("01")

    def test_analyze_rejects_metadata_identity_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            items = self.make_analysis_fixture(pack, multiple=False)
            path = pack / f"evidence/{items[0]['id']}/run_metadata.json"
            meta = project.read_json(path)
            meta["run_id"] = "different_experiment"
            project.write_json(path, meta)
            with patch.object(project, "package", return_value=pack), patch.object(project, "specs", return_value=items):
                with self.assertRaises(ValueError):
                    project.analyze("01")

    def test_analyze_rejects_best_epoch_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            items = self.make_analysis_fixture(pack, multiple=False)
            path = pack / f"evidence/{items[0]['id']}/run_metadata.json"
            meta = project.read_json(path)
            meta["best_epoch"] = 70
            project.write_json(path, meta)
            with patch.object(project, "package", return_value=pack), patch.object(project, "specs", return_value=items):
                with self.assertRaises(ValueError):
                    project.analyze("01")

    def test_load_recipe_rejects_unknown_id_before_read(self):
        with self.assertRaises(ValueError):
            project.load_recipe("01", "../../other")

    def test_console_parser_handles_numpy_plain_and_scientific_numbers(self):
        text = (
            "Training: 50%\rTraining: 100%\r\n"
            "[Epoch 1] Train Loss: 3.125000\n"
            "[Epoch 2] Train Loss: 2.9e+00; extra diagnostics\n"
            "[Epoch 5] Val Results: OrderedDict([('ndcg@10', np.float64(0.041)), "
            "('weighted_score', np.float64(4.8e-2))])\n"
            "[Epoch 10] Val Results: OrderedDict([('ndcg@10', 0.042), ('weighted_score', 0.049)])\n"
            "Best epoch: 10, Best val score: 0.0420\n"
        )
        training, evaluation = project.parse_ar_console(text)
        self.assertEqual([(int(r["epoch"]), float(r["train_loss"])) for r in training], [(1, 3.125), (2, 2.9)])
        self.assertEqual(len(evaluation), 4)
        keyed = {(int(r["epoch"]), r["metric"]): float(r["value"]) for r in evaluation}
        self.assertEqual(keyed, {(5, "ndcg@10"): .041, (5, "weighted_score"): .048,
                                 (10, "ndcg@10"): .042, (10, "weighted_score"): .049})
        self.assertEqual({r["split"] for r in evaluation}, {"val"})

    def test_console_parser_refuses_duplicate_training_epoch(self):
        text = "[Epoch 1] Train Loss: 3.0\n[Epoch 1] Train Loss: 3.0\n"
        with self.assertRaises(ValueError):
            project.parse_ar_console(text)

    def test_console_parser_refuses_duplicate_validation_record(self):
        line = "[Epoch 5] Val Results: OrderedDict([('ndcg@10', np.float64(0.04))])\n"
        with self.assertRaises(ValueError):
            project.parse_ar_console(line + line)

    def test_console_parser_empty_or_untrusted_text_never_uses_eval(self):
        with patch("builtins.eval", side_effect=AssertionError("Logs must never be evaluated")):
            self.assertEqual(project.parse_ar_console("no training data"), ([], []))
            text = "[Epoch 1] Val Results: __import__('os').system('this must not run')\n"
            self.assertEqual(project.parse_ar_console(text), ([], []))

    def test_runtime_sid_names_follow_ar_capacity_and_diff_conventions(self):
        ar = {"model": "AR_GRM", "semantic_codebook_sizes": [256, 256, 256]}
        config = {"sid_quantizer": "rq_kmeans", "sent_emb_model": "sentence-transformers/sentence-t5-xl",
                  "sent_emb_pca": 32, "sent_emb_dim": 768, "rq_kmeans_use_pca": True,
                  "rq_kmeans_seed": 2025, "rq_kmeans_niters": 20}
        self.assertEqual(replay.expected_runtime_sid(config, ar), project.DATA +
                         "processed/sentence-t5-xl_pca32_RQKMEANS_256x256x256_seed2025_it20_inputpca32.sem_ids")
        ragged = {**ar, "semantic_codebook_sizes": [128, 256, 512]}
        self.assertIn("RQKMEANS_128x256x512_seed2025", replay.expected_runtime_sid(config, ragged))
        diff = {"model": "DIFF_GRM", "semantic_codebook_sizes": [256] * 4}
        config.update(sent_emb_model="sentence-transformers/sentence-t5-base", sent_emb_pca=256,
                      rq_kmeans_seed=2024)
        self.assertEqual(replay.expected_runtime_sid(config, diff), project.DATA +
                         "processed/sentence-t5-base_pca256_RQKMEANS4x8_seed2024_it20_PCA.sem_ids")
        config["sid_quantizer"] = "opq_pq"
        self.assertEqual(replay.expected_runtime_sid(config, diff), project.DATA +
                         "processed/sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8.sem_ids")

    def test_rqk_alias_is_byte_identical_and_preserves_historical_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "original.json"
            payload = b'{"item-z": [1, 2, 3], "item-a": [1, 2, 3]}\r\n'
            source.write_bytes(payload)
            workspace = root / "workspace"
            initial = project.DATA + "processed/sentence-t5-xl_pca32_RQKMEANS3x8_seed2024_it20_inputpca32.sem_ids"
            item = {"model": "AR_GRM", "semantic_codebook_sizes": [256] * 3, "initial_sid": initial}
            config = {"sid_quantizer": "rq_kmeans", "sent_emb_model": "sentence-transformers/sentence-t5-xl",
                      "sent_emb_pca": 32, "sent_emb_dim": 768, "rq_kmeans_use_pca": True,
                      "rq_kmeans_seed": 2024, "rq_kmeans_niters": 20}
            project.immutable_copy(source, project.under(workspace, initial))
            records = [{"path": initial, "sha256": project.sha(source)}]
            runtime, record = replay.bind_runtime_sid(workspace, config, item, records,
                                                      "self.index_factory = f'RQKMEANS_{layout}'")
            self.assertNotEqual(runtime, initial)
            self.assertEqual(project.under(workspace, runtime).read_bytes(), payload)
            self.assertEqual(project.under(workspace, initial).read_bytes(), payload)
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(record["byte_identical_alias_of"], initial)
            self.assertEqual(records[-1], record)
            replay.verify_copies(workspace, records)
            with self.assertRaisesRegex(ValueError, "Unexpected existing"):
                replay.bind_runtime_sid(workspace, config, item, records,
                                        "self.index_factory = f'RQKMEANS_{layout}'")

    def test_smoke_support_rejects_ar_and_unrecognized_diff_limits(self):
        tokens = "max_train_batches_per_epoch max_eval_batches batch_index >= epoch_steps batch_index >= eval_limit"
        self.assertFalse(replay.smoke_support("AR_GRM", tokens)["supported"])
        self.assertFalse(replay.smoke_support("DIFF_GRM", "max_train_batches_per_epoch")["supported"])
        self.assertTrue(replay.smoke_support("DIFF_GRM", tokens)["supported"])

    def test_diff_smoke_is_two_batches_without_mutating_formal_config(self):
        formal = {"epochs": 200, "gradient_accumulation_steps": 16, "train_batch_size": 64,
                  "vectorized_beam_search": {"beam_act": 32, "beam_max": 256,
                                              "val": {"beam_act": 32}, "test": {"beam_act": 256}}}
        smoke = replay.apply_smoke(formal)
        self.assertEqual(smoke["epochs"], 1)
        self.assertEqual(smoke["max_train_batches_per_epoch"], 2)
        self.assertEqual(smoke["max_eval_batches"], 2)
        self.assertEqual(smoke["gradient_accumulation_steps"], 1)
        self.assertEqual(smoke["vectorized_beam_search"]["test"]["beam_act"], 16)
        self.assertEqual(formal["epochs"], 200)
        self.assertEqual(formal["vectorized_beam_search"]["test"]["beam_act"], 256)
        self.assertNotIn("max_train_batches_per_epoch", formal)

    def test_configure_paths_points_only_to_isolated_work_and_never_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "new_run"
            workspace = work / "workspace"
            artifact_rel = "outputs/tokenizers/rqkmeans/Beauty/seed2024"
            artifact = project.under(workspace, artifact_rel)
            artifact.mkdir(parents=True)
            (artifact / "reconstructed_embeddings.npy").write_bytes(b"fixed reconstruction")
            config = {"sid_quantizer": "rq_kmeans", "cache_dir": "/old/data/cache",
                      "tokenizer_artifact_dir": "/old/outputs/tokenizers",
                      "sent_emb_path": "/old/embeddings.npy", "run_id": "historical"}
            recipe = {"model": "AR_GRM", "artifact_dir": artifact_rel}
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch.object(Path, "mkdir", side_effect=AssertionError("configure_paths must not write")):
                result = replay.configure_paths(config, workspace, work, recipe, "new_run")
            self.assertEqual(result["tokenizer_artifact_dir"], str(artifact))
            for key in ("cache_dir", "log_dir", "tensorboard_log_dir", "ckpt_dir", "run_metadata_dir",
                        "position_loss_log_dir", "diffusion_diagnostics_log_dir", "sent_emb_path", "tokenizer_artifact_dir"):
                self.assertTrue(Path(result[key]).resolve().is_relative_to(work.resolve()), key)
            self.assertEqual(config["cache_dir"], "/old/data/cache")
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_configure_paths_refuses_implicit_rqk_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = {"model": "AR_GRM", "artifact_dir": "outputs/tokenizers/rqkmeans/Beauty/seed2024"}
            with self.assertRaisesRegex(ValueError, "refusing implicit SID regeneration"):
                replay.configure_paths({"sid_quantizer": "rq_kmeans"}, root / "workspace", root, recipe, "new")

    def replay_plan_fixture(self):
        return {"id": "pse_seed2024", "model": "AR_GRM",
                "decoder_initialization": {"type": "random_initialization_with_seed", "seed": 2024},
                "initial_sid": "data/sid.json", "config_layers": []}

    def test_default_replay_plan_never_imports_gpu_dependencies_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = self.replay_plan_fixture()
            versions = {"repositories": [{"name": "DiffGRM", "snapshot": "code/fixed.tar", "commit": "fixed"}]}
            arguments = argparse.Namespace(stage="02", experiment=recipe["id"], mode="smoke", execute=False,
                                           project_root=str(root))
            original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name.split(".")[0] in ("torch", "yaml"):
                    raise AssertionError("A plan must not import GPU/YAML dependencies")
                return original_import(name, *args, **kwargs)
            with patch.object(project, "ROOT", root), patch.object(project, "verify"), \
                    patch.object(project, "package", return_value=root / "package"), \
                    patch.object(project, "load_recipe", return_value=recipe), \
                    patch.object(project, "read_json", return_value=versions), \
                    patch.object(replay, "snapshot_text", return_value="trainer text"), \
                    patch.object(project, "check_inputs") as check, \
                    patch.object(replay.subprocess, "Popen") as launch, \
                    patch("builtins.__import__", side_effect=guarded_import), \
                    patch.object(Path, "mkdir", side_effect=AssertionError("A plan must not create directories")):
                replay.run_replay(arguments)
                check.assert_not_called()
                launch.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_execute_ar_smoke_is_rejected_before_input_check_or_launch(self):
        recipe = self.replay_plan_fixture()
        versions = {"repositories": [{"name": "DiffGRM", "snapshot": "code/fixed.tar", "commit": "fixed"}]}
        arguments = argparse.Namespace(stage="02", experiment=recipe["id"], mode="smoke", execute=True,
                                       project_root="unused")
        with patch.object(project, "verify"), patch.object(project, "load_recipe", return_value=recipe), \
                patch.object(project, "read_json", return_value=versions), \
                patch.object(replay, "snapshot_text", return_value="trainer text"), \
                patch.object(project, "check_inputs") as check, patch.object(replay.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "AR trainer has no train/eval batch limits"):
                replay.run_replay(arguments)
            check.assert_not_called()
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
