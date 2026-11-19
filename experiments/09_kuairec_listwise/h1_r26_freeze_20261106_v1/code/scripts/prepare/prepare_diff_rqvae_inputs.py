"""Export the exact PCA item embeddings consumed by DiffGRM tokenizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from accelerate import Accelerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
sys.path.insert(0, str(DIFF_ROOT))

from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer  # noqa: E402
from genrec.utils import get_config, get_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    items_output = args.items_output.resolve()
    manifest_output = args.manifest_output.resolve()
    os.chdir(DIFF_ROOT)

    config = get_config(
        model_name="DIFF_GRM",
        dataset_name="AmazonReviews2014",
        config_file=args.config,
        config_dict=None,
    )
    accelerator = Accelerator()
    config["accelerator"] = accelerator
    config["device"] = str(accelerator.device)
    config["use_ddp"] = False
    dataset = get_dataset("AmazonReviews2014")(config)

    item_keys = [
        dataset.id_mapping["id2item"][index]
        for index in range(1, dataset.n_items)
    ]
    expected_shape = (len(item_keys), int(config["sent_emb_pca"]))
    if (
        output.exists()
        and items_output.exists()
        and manifest_output.exists()
        and not args.force_export
    ):
        embeddings = np.load(output, mmap_mode="r")
        stored_items = json.loads(items_output.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
        if embeddings.shape != expected_shape or stored_items != item_keys:
            raise ValueError(
                "Existing DiffGRM RQ-VAE input artifact does not match the "
                "catalog/config; inspect it before using --force-export."
            )
        if manifest.get("embedding_sha256") != sha256(output):
            raise ValueError("Existing DiffGRM RQ-VAE embedding SHA256 mismatch")
        if manifest.get("items_sha256") != sha256(items_output):
            raise ValueError("Existing DiffGRM RQ-VAE item-list SHA256 mismatch")
        print(f"DiffGRM RQ-VAE inputs already valid: {output} {embeddings.shape}")
        return

    cache_dir = Path(dataset.cache_dir) / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_basename = os.path.basename(config["sent_emb_model"])
    raw_path = cache_dir / (
        f"{model_basename}_raw_d{config['sent_emb_dim']}.sent_emb"
    )
    pca_path = cache_dir / (
        f"{model_basename}_pca{config['sent_emb_pca']}.sent_emb"
    )

    # Reuse the same implementation and cache as PSE/RQ-KMeans. This avoids
    # fitting a second PCA basis specifically for RQ-VAE.
    helper = DIFF_GRMTokenizer.__new__(DIFF_GRMTokenizer)
    helper.config = config
    helper.log = lambda message, level="info": print(message, flush=True)
    started = time.perf_counter()
    embeddings = helper._prepare_pca_embeddings(
        dataset, str(raw_path), str(pca_path)
    ).astype(np.float32, copy=False)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Prepared embedding shape {embeddings.shape} != {expected_shape}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    items_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    items_output.write_text(
        json.dumps(item_keys, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "embedding_path": str(output),
        "items_path": str(items_output),
        "source_pca_cache": str(pca_path.resolve()),
        "model": config["sent_emb_model"],
        "shape": list(embeddings.shape),
        "pca_dim": int(config["sent_emb_pca"]),
        "pca_whiten": True,
        "normalized_after_pca": bool(config.get("normalize_after_pca", True)),
        "dtype": str(embeddings.dtype),
        "embedding_sha256": sha256(output),
        "items_sha256": sha256(items_output),
        "elapsed_seconds": elapsed,
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared DiffGRM RQ-VAE inputs in {elapsed / 60:.1f} min: {output}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
