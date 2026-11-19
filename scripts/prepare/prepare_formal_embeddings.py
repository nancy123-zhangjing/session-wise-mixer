"""Prepare one frozen item-embedding matrix shared by all SID tokenizers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFGRM_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
sys.path.insert(0, str(DIFFGRM_ROOT))

from genrec.utils import get_config, get_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    items_output = args.items_output.resolve()
    manifest_output = args.manifest_output.resolve()
    # DiffGRM's YAML paths are intentionally relative to its runnable root.
    os.chdir(DIFFGRM_ROOT)
    config = get_config(
        model_name="AR_GRM",
        dataset_name="AmazonReviews2014",
        config_file=args.config,
        config_dict=None,
    )
    accelerator = Accelerator()
    config["accelerator"] = accelerator
    config["device"] = str(accelerator.device)
    config["use_ddp"] = False

    dataset = get_dataset("AmazonReviews2014")(config)
    item_keys = [dataset.id_mapping["id2item"][i] for i in range(1, dataset.n_items)]
    sentences = [dataset.item2meta[item] for item in item_keys]
    expected_shape = (len(item_keys), int(config["sent_emb_dim"]))

    if output.exists() and items_output.exists() and not args.force:
        embeddings = np.load(output, mmap_mode="r")
        stored_items = json.loads(items_output.read_text(encoding="utf-8"))
        if embeddings.shape != expected_shape or stored_items != item_keys:
            raise ValueError(
                "Existing shared embedding cache does not match the current catalog/config. "
                "Use --force only after checking the dataset and model settings."
            )
        print(f"Shared embeddings already valid: {output} {embeddings.shape}")
        return

    started = time.perf_counter()
    model = SentenceTransformer(
        config["sent_emb_model"],
        trust_remote_code=True,
        cache_folder=str(Path(config["hf_home"]).resolve()) if config.get("hf_home") else None,
        # Load on CPU first so a 10 GB GPU never briefly holds both FP32 and
        # FP16 copies of Sentence-T5-XL.
        device="cpu",
    )
    use_fp16 = bool(config.get("sent_emb_fp16", True)) and accelerator.device.type == "cuda"
    if use_fp16:
        model.half()
    model.to(accelerator.device)

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        batch_size=int(config["sent_emb_batch_size"]),
        show_progress_bar=True,
        device=str(accelerator.device),
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    if embeddings.shape != expected_shape:
        raise ValueError(f"Encoded shape {embeddings.shape} != expected {expected_shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    items_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    items_output.write_text(json.dumps(item_keys, ensure_ascii=False) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    manifest = {
        "embedding_path": str(output),
        "items_path": str(items_output),
        "model": config["sent_emb_model"],
        "shape": list(embeddings.shape),
        "normalized": True,
        "dtype": str(embeddings.dtype),
        "elapsed_seconds": elapsed,
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared shared embeddings in {elapsed / 60:.1f} min: {output}")


if __name__ == "__main__":
    main()
