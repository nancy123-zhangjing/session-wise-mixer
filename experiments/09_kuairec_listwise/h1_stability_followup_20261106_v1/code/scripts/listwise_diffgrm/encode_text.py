"""Explicit, resumable BGE-M3 dense encoding; importing this module does no work.

Default execution is CPU and offline (local_files_only). Downloading a pinned
model requires the explicit --allow-download flag. The 512-token cap is a
locked engineering choice for this experiment, not BGE-M3's maximum context
or a hyperparameter claimed by the DiffGRM paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

from .assets import (TEXT_TEMPLATE_ID, TEXT_TEMPLATE, catalog_original_ids,
                     dataset_binding, sha256)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(_canonical(value) + b"\n")
    os.replace(temp, path)


def catalog_texts(dataset_dir: Path) -> tuple[np.ndarray, list[str], str]:
    root = Path(dataset_dir)
    original = catalog_original_ids(root)
    source = {}
    with (root / "item_text.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = int(row["video_id"])
            if key in source:
                raise ValueError("Duplicate video ID in static text")
            if not isinstance(row["text"], str):
                raise ValueError("Static text must be a string")
            source[key] = row["text"]
    # Missing catalog text remains empty; never invent a description or drop it.
    texts = [source.get(int(item), "") for item in original]
    text_hash = hashlib.sha256(_canonical(list(zip(original.tolist(), texts)))).hexdigest()
    return original, texts, text_hash


def _make_encoder(revision: str, device: str, allow_download: bool, model_path: Path | None = None):
    from sentence_transformers import SentenceTransformer
    source = str(model_path) if model_path is not None else "BAAI/bge-m3"
    model = SentenceTransformer(source, revision=revision, device=device,
                                local_files_only=model_path is not None or not allow_download,
                                trust_remote_code=False)
    model.max_seq_length = 512
    if model.get_sentence_embedding_dimension() != 1024:
        raise ValueError("Pinned encoder does not have BGE-M3's 1024-dimensional output")
    pooling = [module for module in model.modules() if hasattr(module, "pooling_mode_cls_token")]
    if len(pooling) != 1 or not pooling[0].pooling_mode_cls_token:
        raise ValueError("Pinned BGE-M3 must use CLS pooling")
    if any(getattr(pooling[0], name, False) for name in (
            "pooling_mode_mean_tokens", "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken")):
        raise ValueError("Pinned BGE-M3 pooling differs from the locked CLS-only dense representation")
    return model


def untruncated_token_lengths(encoder, texts: list[str]) -> list[int]:
    """Count actual pinned-tokenizer input, including its special tokens.

    SentenceTransformer's Transformer module strips outer whitespace and may
    lowercase before calling its tokenizer. Mirror that preprocessing, while
    explicitly disabling truncation/padding. Encoding uses an empty prompt.
    """
    if not hasattr(encoder, "tokenizer"):
        raise ValueError("Encoder must expose its actual tokenizer for truncation audit")
    first = encoder._first_module() if hasattr(encoder, "_first_module") else None
    prepared = [text.strip() for text in texts]
    if bool(getattr(first, "do_lower_case", False)):
        prepared = [text.lower() for text in prepared]
    tokens = encoder.tokenizer(prepared, truncation=False, padding=False,
                               add_special_tokens=True, return_length=True,
                               return_attention_mask=False, return_token_type_ids=False,
                               verbose=False)
    lengths = [len(ids) for ids in tokens["input_ids"]]
    if len(lengths) != len(texts) or any(length < 1 for length in lengths):
        raise ValueError("Invalid untruncated tokenizer lengths")
    return lengths


def encode_catalog(dataset_dir: Path, output_dir: Path, *, revision: str,
                   batch_size: int = 8, chunk_size: int = 128, device: str = "cpu",
                   allow_download: bool = False, resume: bool = False,
                   model_path: Path | None = None, model_manifest: Path | None = None,
                   num_threads: int = 4):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be an explicit 40-character BGE-M3 commit hash, not main")
    if batch_size < 1 or chunk_size < 1 or num_threads < 1:
        raise ValueError("Batch/chunk/thread counts must be positive")
    torch.set_num_threads(num_threads)
    if (model_path is None) != (model_manifest is None):
        raise ValueError("A local model requires both model_path and model_manifest")
    model_identity = None
    loading_provenance = {"source": "huggingface_cache_or_explicit_download", "model_id": "BAAI/bge-m3"}
    if model_path is not None:
        from .model_manifest import DENSE_FILES, verify_snapshot
        manifest = json.loads(Path(model_manifest).read_text(encoding="utf-8"))
        if manifest.get("model_id") != "BAAI/bge-m3" or manifest.get("revision") != revision.lower():
            raise ValueError("Local model manifest revision/model differs from requested BGE-M3")
        files = manifest.get("files", [])
        if len(files) != len(DENSE_FILES) or {row["path"] for row in files} != set(DENSE_FILES):
            raise ValueError("Local model manifest lacks the complete dense file allowlist")
        checked = verify_snapshot(Path(model_path), files)
        # Content identity excludes machine-specific paths and observation times.
        model_identity = {name: {"sha256": row["sha256"], "bytes": row["bytes"]}
                          for name, row in sorted(checked.items())}
        loading_provenance = {"source": "verified_local_snapshot", "path": str(Path(model_path).resolve()),
                              "manifest_path": str(Path(model_manifest).resolve()),
                              "manifest_sha256": sha256(Path(model_manifest))}
    binding = dataset_binding(dataset_dir)
    original, texts, texts_hash = catalog_texts(dataset_dir)
    ordered_text_bytes = b"".join(_canonical({"original_item_id": int(item_id), "text": text}) + b"\n"
                                   for item_id, text in zip(original, texts))
    ordered_file_hash = hashlib.sha256(ordered_text_bytes).hexdigest()
    specification = {
        "schema_version": 1, "model_id": "BAAI/bge-m3", "revision": revision.lower(),
        "dimension": 1024, "pooling": "cls", "normalize_embeddings": True,
        "prompt": "", "truncation_audit": "actual_tokenizer_including_special_tokens_v1",
        "model_artifact_identity": model_identity,
        "max_length": 512, "max_length_reason": "fixed engineering bound for short static catalog text",
        "text_template_id": TEXT_TEMPLATE_ID, "text_template": TEXT_TEMPLATE,
        "item_order": "internal_item_id_ascending", "n_items": len(original),
        "item_map_sha256": binding["files"]["item_map.json"]["sha256"],
        "item_text_sha256": binding["files"]["item_text.jsonl"]["sha256"],
        "ordered_text_sha256": texts_hash, "dataset_binding": binding,
        "ordered_text_file_sha256": ordered_file_hash,
        "empty_text_count": sum(not text for text in texts),
        "batch_size": int(batch_size), "chunk_size": int(chunk_size), "device": device,
        "torch_num_threads": int(num_threads),
    }
    fingerprint = hashlib.sha256(_canonical(specification)).hexdigest()
    root = Path(output_dir)
    if root.exists() and not resume:
        raise FileExistsError(f"Encoding output exists; pass --resume to verify/reuse: {root}")
    if not root.exists():
        root.mkdir(parents=True)
    progress_path = root / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise ValueError("Resume encoder/data/text/settings identity differs")
    elif any(root.iterdir()):
        raise ValueError("Nonempty encoding output has no valid progress manifest")
    else:
        progress = {"fingerprint": fingerprint, "specification": specification, "chunks": {}}
        _atomic_json(progress_path, progress)
    # This can be recreated after a crash immediately following progress init,
    # but an existing, changed snapshot is an error even for incomplete runs.
    text_path = root / "ordered_text.jsonl"
    if text_path.exists():
        if sha256(text_path) != ordered_file_hash:
            raise ValueError("Ordered encoder input snapshot failed its hash check")
    else:
        temporary_text = text_path.with_suffix(".tmp")
        temporary_text.write_bytes(ordered_text_bytes)
        os.replace(temporary_text, text_path)
    if (root / "metadata.json").exists():
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError("Completed encoding metadata identity differs")
        for name, key in (("dense_embeddings.npy", "embeddings_sha256"),
                          ("original_item_ids.npy", "original_item_ids_sha256"),
                          ("untruncated_token_lengths.npy", "token_lengths_sha256"),
                          ("ordered_text.jsonl", "ordered_text_file_sha256")):
            if not (root / name).is_file() or sha256(root / name) != metadata[key]:
                raise ValueError(f"Completed encoding file failed hash check: {name}")
        return metadata
    # Validate already-finished chunks before constructing/loading the encoder.
    for start, record in progress["chunks"].items():
        expected_name = f"chunk_{int(start):08d}.npy"
        path = root / expected_name
        if record["file"] != expected_name or not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Resume chunk failed hash check: {start}")
        if len(record.get("untruncated_token_lengths", [])) != record["rows"]:
            raise ValueError(f"Resume chunk lacks complete tokenizer length audit: {start}")
    encoder = None
    for start in range(0, len(texts), chunk_size):
        stop, key = min(start + chunk_size, len(texts)), str(start)
        if key in progress["chunks"]:
            continue
        if encoder is None:
            encoder = (_make_encoder(revision.lower(), device, allow_download)
                       if model_path is None else _make_encoder(revision.lower(), device, False, Path(model_path)))
        token_lengths = untruncated_token_lengths(encoder, texts[start:stop])
        values = np.asarray(encoder.encode(texts[start:stop], batch_size=batch_size,
                                          convert_to_numpy=True, normalize_embeddings=True,
                                          show_progress_bar=False, prompt=""), dtype=np.float32)
        if values.shape != (stop - start, 1024) or not np.isfinite(values).all():
            raise ValueError("BGE-M3 returned invalid dense vectors")
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
            raise ValueError("BGE-M3 dense vectors are not normalized")
        path = root / f"chunk_{start:08d}.npy"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temp, path)
        progress["chunks"][key] = {"file": path.name, "rows": stop - start, "sha256": sha256(path),
                                   "untruncated_token_lengths": token_lengths}
        _atomic_json(progress_path, progress)
    temp = root / "dense_embeddings.tmp.npy"
    values = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(len(original), 1024))
    token_lengths = np.empty(len(original), dtype=np.int64)
    for start in range(0, len(original), chunk_size):
        stop = min(start + chunk_size, len(original))
        chunk = np.load(root / progress["chunks"][str(start)]["file"], allow_pickle=False)
        if chunk.shape != (stop - start, 1024):
            raise ValueError("Resume chunk shape differs")
        values[start:stop] = chunk
        token_lengths[start:stop] = progress["chunks"][str(start)]["untruncated_token_lengths"]
    values.flush()
    del values
    os.replace(temp, root / "dense_embeddings.npy")
    np.save(root / "original_item_ids.npy", original, allow_pickle=False)
    np.save(root / "untruncated_token_lengths.npy", token_lengths, allow_pickle=False)
    metadata = {**specification, "fingerprint": fingerprint,
                "loading_provenance": loading_provenance,
                "embeddings_sha256": sha256(root / "dense_embeddings.npy"),
                "original_item_ids_sha256": sha256(root / "original_item_ids.npy"),
                "token_lengths_sha256": sha256(root / "untruncated_token_lengths.npy"),
                "truncation_statistics": {
                    "items": len(original), "truncated_items": int((token_lengths > 512).sum()),
                    "truncated_fraction": float((token_lengths > 512).mean()),
                    "discarded_token_count": int(np.maximum(token_lengths - 512, 0).sum()),
                    "max_untruncated_tokens": int(token_lengths.max()),
                    "mean_untruncated_tokens": float(token_lengths.mean()),
                    "p50_p95_p99_untruncated_tokens": np.quantile(token_lengths, [.5, .95, .99]).tolist(),
                },
                "ordered_text_file_sha256": sha256(root / "ordered_text.jsonl")}
    _atomic_json(root / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Pinned BGE-M3 40-hex Hugging Face commit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model-path", type=Path, help="Use a verified local BGE-M3 snapshot without network")
    parser.add_argument("--model-manifest", type=Path, help="Fixed-revision manifest for --model-path, including all file hashes")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    metadata = encode_catalog(args.dataset, args.output, revision=args.revision,
                              batch_size=args.batch_size, chunk_size=args.chunk_size,
                              device=args.device, allow_download=args.allow_download,
                              resume=args.resume, model_path=args.model_path,
                              model_manifest=args.model_manifest, num_threads=args.num_threads)
    print(json.dumps({"output": str(args.output), "items": metadata["n_items"],
                      "revision": metadata["revision"]}))


if __name__ == "__main__":
    main()
