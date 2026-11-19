"""Verify all ten final portable models and item tables on CPU.

Run only after the frozen CPU delivery entrypoint has finished. This separate
auditor does not alter training, choose a model, or read test predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

SOURCE = "b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_one(checkpoint, directory, histories, expected_binding, expected_sha):
    import numpy as np
    import torch
    from scripts.continuous.model import ContinuousRecommender, ModelConfig
    from scripts.continuous.trainer import restore_model
    directory = Path(directory)
    result = read(directory / "result.json")
    assert sha(checkpoint) == result["checkpoint_sha256"] == expected_sha
    assert result["binding"] == expected_binding
    assert result["input_output_tied"] is True and result["pad_output_included"] is False
    assert result["output_item_ids"] == [1, 12101]
    assert set(result["files"]) == {"recommender.pt", "input_item_embeddings_with_pad.npy",
                                    "output_item_embeddings.npy"}
    for filename, expected in result["files"].items():
        assert sha(directory / filename) == expected
    original, state = restore_model(checkpoint, "cpu")
    assert state["binding"] == expected_binding and result["config"] == state["config"]
    portable = torch.load(directory / "recommender.pt", map_location="cpu", weights_only=False)
    assert portable["training_binding"] == expected_binding
    assert portable["training_checkpoint_sha256"] == expected_sha
    assert portable["model_config"] == original.config.to_dict()
    assert portable["model"].keys() == state["model"].keys()
    for name, tensor in portable["model"].items():
        assert torch.isfinite(tensor).all().item() and torch.equal(tensor, state["model"][name])
    loaded = ContinuousRecommender(ModelConfig(**portable["model_config"])).eval()
    loaded.load_state_dict(portable["model"], strict=True)
    inputs = np.load(directory / "input_item_embeddings_with_pad.npy", allow_pickle=False)
    outputs = np.load(directory / "output_item_embeddings.npy", allow_pickle=False)
    assert inputs.shape == (12102, original.config.dim) and outputs.shape == (12101, original.config.dim)
    assert inputs.dtype == outputs.dtype == np.float32
    assert np.isfinite(inputs).all() and np.isfinite(outputs).all() and not inputs[0].any()
    assert np.array_equal(inputs, original.item_embedding.weight.detach().numpy())
    assert np.array_equal(outputs, inputs[1:])
    with torch.inference_mode():
        first, second = original(histories), loaded(histories)
        assert torch.isfinite(first).all().item() and torch.equal(first, second)
        assert torch.equal(original.score_catalog(first), loaded.score_catalog(second))
    return {"status": "PASS", "checkpoint_sha256": expected_sha,
            "export_files_sha256": result["files"], "all_model_tensors_exact": True,
            "input_output_tables_exact": True, "portable_query_and_catalog_scores_exact": True,
            "cpu_validation_history_rows": len(histories), "dim": original.config.dim}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root, campaign = Path(args.root).resolve(), Path(args.campaign).resolve()
    sys.path.insert(0, str(root))
    import torch
    from scripts.continuous.data import load_cached_split
    from scripts.continuous.trainer import source_identity
    torch.set_num_threads(2)
    assert source_identity(root)["source_sha256"] == SOURCE
    delivered = read(campaign.parent / "CPU_DELIVERY_PASS.json")
    state, selection, exports = (read(campaign / name) for name in
                                  ("queue.json", "FROZEN_SELECTION.json", "export_index.json"))
    assert delivered["status"] == "PASS" and delivered["source_sha256"] == SOURCE
    assert state["source_sha256"] == selection["source_sha256"] == SOURCE
    assert selection["validation_only"] is True and selection["test_unlocked"] is True
    assert len(state["tasks"]) == 89 and not state["workers"]
    assert all(task["status"] == "succeeded" for task in state["tasks"])
    formal = [task for task in state["tasks"] if task["phase"] == "formal"]
    assert len(formal) == len(exports) == 10 and {t["task_id"] for t in formal} == exports.keys()
    assert {(t["config"]["model"], t["seed"]) for t in formal} == {
        (model, seed) for model in ("sasrec_fullce", "esasrec") for seed in range(3031, 3036)}
    validation = load_cached_split(args.cache, "full_train_prefix_v1", "validation")
    histories = validation["histories"][:8].cpu()
    records = {}
    for task in formal:
        assert task["config_sha256"] == selection["models"][task["config"]["model"]]["config_sha256"]
        directory = campaign / "exports" / task["task_id"]
        assert exports[task["task_id"]] == read(directory / "result.json")
        records[task["task_id"]] = audit_one(Path(task["output"]) / "best.pt", directory, histories,
                                             task["result"]["binding"], task["result"]["best_sha256"])
    assert source_identity(root)["source_sha256"] == SOURCE
    result = {"status": "PASS", "source_sha256": SOURCE, "formal_exports_audited": 10,
              "records": records, "test_metrics_read": False,
              "scope": "CPU exact weights, all item vectors, and eight validation-history queries per export; no test inference or GPU performance claim",
              "audit_script_sha256": sha(__file__)}
    with Path(args.report).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
