"""Pin and audit the public BGE-M3 dense-only download, without implicit weights.

Default execution fetches public metadata and small config files only. Weight
download requires --download and an explicit owned --cache-dir. Existing cache
entries are reused by huggingface_hub and every resulting file is verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


MODEL_ID = "BAAI/bge-m3"
VERIFIED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DENSE_FILES = (
    "1_Pooling/config.json", "README.md", "config.json",
    "config_sentence_transformers.json", "modules.json", "pytorch_model.bin",
    "sentence_bert_config.json", "sentencepiece.bpe.model",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
)
SMALL_CONFIGS = ("config.json", "modules.json", "1_Pooling/config.json",
                 "config_sentence_transformers.json", "sentence_bert_config.json",
                 "tokenizer_config.json", "special_tokens_map.json")


def hash_file(path: Path, *, git_blob: bool = False) -> str:
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_dense_files(info: dict, revision: str) -> list[dict]:
    if info.get("sha") != revision:
        raise ValueError("HF response revision differs from the locked commit")
    by_name = {entry["rfilename"]: entry for entry in info["siblings"]}
    files = []
    for name in DENSE_FILES:
        row = by_name.get(name)
        if row is None or "size" not in row or "blobId" not in row:
            raise ValueError(f"Pinned model lacks complete file metadata: {name}")
        lfs = row.get("lfs")
        files.append({"path": name, "bytes": int(row["size"]), "git_blob_id": row["blobId"],
                      "sha256": lfs["sha256"] if lfs else None,
                      "resolve_url": f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}"})
    return files


def verify_snapshot(snapshot: Path, records: list[dict]) -> dict:
    checks = {}
    for record in records:
        name = record["path"]
        if name not in DENSE_FILES:
            raise ValueError(f"Unexpected file outside dense-only allowlist: {name}")
        path = Path(snapshot) / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing or size-mismatched model file: {name}")
        actual_sha256 = hash_file(path)
        if record["sha256"]:
            if actual_sha256 != record["sha256"]:
                raise ValueError(f"LFS SHA256 mismatch: {name}")
        elif hash_file(path, git_blob=True) != record["git_blob_id"]:
            raise ValueError(f"Git blob hash mismatch: {name}")
        checks[name] = {"bytes": path.stat().st_size, "sha256": actual_sha256, "passed": True}
    return checks


def collect_manifest(revision: str = VERIFIED_REVISION) -> dict:
    import requests
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Revision must be a lowercase 40-hex commit")
    url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{revision}?blobs=true"
    response = requests.get(url, timeout=(10, 45))
    response.raise_for_status()
    info = response.json()
    records = select_dense_files(info, revision)
    configs = {}
    for name in SMALL_CONFIGS:
        record = next(row for row in records if row["path"] == name)
        if record["bytes"] > 16384:
            raise ValueError(f"Refusing an unexpectedly large config fetch: {name}")
        result = requests.get(record["resolve_url"], timeout=(10, 45))
        result.raise_for_status()
        data = result.content
        if len(data) != record["bytes"]:
            raise ValueError(f"Config size mismatch: {name}")
        git_digest = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if git_digest != record["git_blob_id"]:
            raise ValueError(f"Config content does not match pinned Git blob: {name}")
        configs[name] = {"sha256": hashlib.sha256(data).hexdigest(), "content": result.json()}
    pooling = configs["1_Pooling/config.json"]["content"]
    if (configs["config.json"]["content"]["hidden_size"] != 1024
            or not pooling.get("pooling_mode_cls_token")
            or pooling.get("pooling_mode_mean_tokens")):
        raise ValueError("Pinned revision is not the expected BGE-M3 CLS dense model")
    return {
        "schema_version": 1, "model_id": MODEL_ID, "revision": revision,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "source": url,
        "public": not info.get("private", True), "gated": info.get("gated", False),
        "files": records, "total_bytes": sum(row["bytes"] for row in records),
        "excluded": ["onnx/*", "colbert_linear.pt", "sparse_linear.pt", "imgs/*", "long.jpg"],
        "verified_small_configs": configs, "weight_downloaded_by_this_command": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=VERIFIED_REVISION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-dir", type=Path,
                        help="Optional explicit materialized snapshot; recommended to end with the revision")
    parser.add_argument("--verify-snapshot", type=Path)
    args = parser.parse_args()
    if args.download and args.cache_dir is None:
        parser.error("--download requires an explicit owned --cache-dir")
    if args.download and args.verify_snapshot:
        parser.error("Select either --download or --verify-snapshot")
    if args.local_dir is not None and not args.download:
        parser.error("--local-dir only applies with --download")
    if args.output.exists():
        parser.error("Manifest output already exists; choose a new evidence path")
    manifest = collect_manifest(args.revision)
    snapshot = args.verify_snapshot
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=args.revision,
                                         allow_patterns=list(DENSE_FILES),
                                         cache_dir=str(args.cache_dir),
                                         local_dir=str(args.local_dir) if args.local_dir else None,
                                         max_workers=4))
        manifest["weight_downloaded_by_this_command"] = True
    if snapshot is not None:
        manifest["snapshot"] = str(snapshot.resolve())
        manifest["verified_downloads"] = verify_snapshot(snapshot, manifest["files"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"revision": manifest["revision"], "file_count": len(manifest["files"]),
                      "total_bytes": manifest["total_bytes"],
                      "downloaded": manifest["weight_downloaded_by_this_command"],
                      "manifest": str(args.output)}))


if __name__ == "__main__":
    main()
