"""Dataset-bound, offline PSE assets for the KuaiRec list experiment.

PAD is internal item zero. Arrays on disk have no PAD row; loaded tensors do.
The original FAISS index encodes OPQ vectors *minus its IVF1 coarse center*.
Never substitute ordinary PQ of the rotated content vectors for those slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from types import SimpleNamespace

import numpy as np
import torch


SCHEMA_VERSION = 1
TEXT_TEMPLATE_ID = "kuairec_static_text_v1"
TEXT_TEMPLATE = (
    "Use item_text.jsonl text verbatim: nonmissing caption as 标题, "
    "manual_cover_text as 封面文字, topic_tag as 主题, category names as "
    "类别 joined by ' / '; fields separated by newline; empty text stays empty."
)
BOUND_FILES = (
    "config.json", "item_map.json", "item_catalog.parquet", "item_text.jsonl",
    "train_seen.json", "train.parquet", "validation.parquet", "test.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2) + "\n", encoding="utf-8")


def dataset_binding(dataset_dir: Path, *, verify: bool = True) -> dict:
    """Bind the split, item mapping and static content; optionally hash bytes.

    ``verify=False`` trusts the dataset manifest, but still checks sizes and
    small mapping/config bytes. Use only after a complete verification at run
    startup, never as a replacement for that check.
    """
    root = Path(dataset_dir)
    manifest = _json(root / "manifest.json")
    entries = {entry["path"]: entry for entry in manifest["artifacts"]}
    binding = {"manifest_sha256": sha256(root / "manifest.json"), "files": {}}
    for name in BOUND_FILES:
        if name not in entries:
            raise ValueError(f"Dataset manifest does not bind {name}")
        entry, path = entries[name], root / name
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Dataset file missing or size differs: {name}")
        if verify or name in {"config.json", "item_map.json"}:
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {name}")
        binding["files"][name] = {"bytes": int(entry["bytes"]),
                                    "sha256": entry["sha256"]}
    config = _json(root / "config.json")
    binding["history_max"] = int(config["history_max"])
    binding["list_length"] = int(config["list_length"])
    binding["pad_id"] = 0
    return binding


def catalog_original_ids(dataset_dir: Path) -> np.ndarray:
    mapping = _json(Path(dataset_dir) / "item_map.json")
    entries = sorted(mapping["entries"], key=lambda row: int(row["id"]))
    if mapping["pad_id"] != 0 or [int(row["id"]) for row in entries] != list(range(1, len(entries) + 1)):
        raise ValueError("Catalog IDs must be contiguous 1..N with PAD=0")
    original = np.asarray([int(row["original_id"]) for row in entries], dtype=np.int64)
    if len(original) == 0 or len(np.unique(original)) != len(original) or np.any(original < 0):
        raise ValueError("Original KuaiRec IDs must be distinct nonnegative integers")
    return original


def validate_embedding_metadata(metadata: dict, binding: dict, dimension: int) -> None:
    expected = {
        "model_id": "BAAI/bge-m3", "pooling": "cls", "normalize_embeddings": True,
        "max_length": 512, "text_template_id": TEXT_TEMPLATE_ID,
        "item_order": "internal_item_id_ascending",
        "item_map_sha256": binding["files"]["item_map.json"]["sha256"],
        "item_text_sha256": binding["files"]["item_text.jsonl"]["sha256"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Embedding metadata {key} must equal {value!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(metadata.get("revision", ""))):
        raise ValueError("BGE-M3 revision must be a pinned 40-character commit hash")
    if dimension != 1024 or metadata.get("dimension") != 1024:
        raise ValueError("BGE-M3 dense embeddings must have dimension 1024")


def normalized_pca(embeddings: np.ndarray, dimensions: int = 256, seed: int = 2024):
    """Original normalized dense -> whitened PCA -> L2-normalized content."""
    from sklearn.decomposition import PCA

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite [N,D] matrix")
    if dimensions < 1 or min(values.shape) < dimensions:
        raise ValueError("PCA dimensions exceed available items/features")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero embeddings are not valid dense content (including missing text)")
    values = values / norms
    pca = PCA(n_components=dimensions, whiten=True, svd_solver="full", random_state=seed)
    pca.fit(values)
    projection = pca.transform(values)
    inverse_error = np.square(pca.inverse_transform(projection) - values).sum(axis=1).mean()
    content = projection.astype(np.float32)
    content /= np.linalg.norm(content, axis=1, keepdims=True) + 1e-12
    if not np.isfinite(content).all():
        raise ValueError("Non-finite PCA result")
    transform = {"mean": pca.mean_, "components": pca.components_,
                 "explained_variance": pca.explained_variance_,
                 "explained_variance_ratio": pca.explained_variance_ratio_,
                 "singular_values": pca.singular_values_,
                 "total_input_variance": np.asarray(np.var(values, axis=0, ddof=1).sum()),
                 "input_reconstruction_squared_error_mean": np.asarray(inverse_error),
                 "n_samples": np.asarray(len(values), dtype=np.int64)}
    return np.ascontiguousarray(content), transform


def fit_original_pse(content: np.ndarray, index_path: Path, *, n_digit: int = 4,
                     codebook_size: int = 256, omp_threads: int = 4):
    """Original OPQ/IVF1/PQ factory, including residual and packed-code audit.

    FAISS is optional until this function is explicitly invoked. No dependency
    installation, model download, or GPU initialization is performed here.
    """
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Asset fitting requires an existing CPU faiss installation; none was installed") from exc
    values = np.ascontiguousarray(content, dtype=np.float32)
    bits = int(np.log2(codebook_size))
    if 2 ** bits != codebook_size or values.ndim != 2 or values.shape[1] % n_digit:
        raise ValueError("Invalid PQ dimensions/codebook size")
    if len(values) < max(codebook_size, 256):
        raise ValueError("OPQ training requires at least 256 real catalog vectors; PAD is excluded")
    faiss.omp_set_num_threads(int(omp_threads))
    index = faiss.index_factory(values.shape[1], f"OPQ{n_digit},IVF1,PQ{n_digit}x{bits}",
                                faiss.METRIC_INNER_PRODUCT)
    index.train(values)
    index.add(values)
    faiss.write_index(index, str(index_path))
    if not isinstance(index, faiss.IndexPreTransform) or index.chain.size() != 1:
        raise TypeError("Expected the original OPQ IndexPreTransform")
    transform = faiss.downcast_VectorTransform(index.chain.at(0))
    if transform.d_in != values.shape[1] or transform.d_out != values.shape[1]:
        raise ValueError("Expected a square OPQ transform preserving the PCA dimension")
    transformed = transform.apply_py(values).astype(np.float32, copy=False)
    ivf = faiss.downcast_index(index.index)
    extracted = faiss.extract_index_ivf(ivf)
    if (extracted.nlist != 1 or not bool(ivf.by_residual)
            or ivf.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise ValueError("Expected IVF1 residual PQ; refusing a different quantization protocol")
    if (ivf.d != transform.d_out or ivf.pq.M != n_digit
            or ivf.pq.ksub != codebook_size or ivf.pq.dsub != values.shape[1] // n_digit):
        raise ValueError("FAISS OPQ/PQ dimensions differ from the declared PSE layout")
    center = np.asarray(extracted.quantizer.reconstruct(0), dtype=np.float32)
    if center.shape != (transform.d_out,) or transformed.shape != values.shape:
        raise ValueError("Unexpected transformed-vector/coarse-center shape")
    residual = transformed - center[None, :]
    books = faiss.vector_to_array(ivf.pq.centroids).reshape(n_digit, codebook_size, -1).astype(np.float32)
    lists = extracted.invlists
    count = lists.list_size(0)
    if count != len(values):
        raise ValueError("FAISS catalog count differs")
    packed = faiss.rev_swig_ptr(lists.get_codes(0), count * lists.code_size).reshape(count, lists.code_size)
    stored_ids = faiss.rev_swig_ptr(lists.get_ids(0), count).copy()
    if not np.array_equal(np.sort(stored_ids), np.arange(count)):
        raise ValueError("FAISS stored item IDs are not a catalog permutation")
    codes = np.empty((count, n_digit), dtype=np.int64)
    for row, stored_id in zip(packed, stored_ids):
        reader = faiss.BitstringReader(faiss.swig_ptr(row), lists.code_size)
        codes[stored_id] = [reader.read(bits) for _ in range(n_digit)]
    nearest, _ = encode_residuals(residual, books)
    if not np.array_equal(nearest, codes):
        raise ValueError("Recovered IVF1 residuals do not reproduce stored FAISS SID codes")
    return residual, books, codes, center


def encode_residuals(residual: np.ndarray, centroids: np.ndarray):
    books = np.asarray(centroids, dtype=np.float32)
    slots = np.asarray(residual, dtype=np.float32).reshape(-1, books.shape[0], books.shape[2])
    codes = np.empty(slots.shape[:2], dtype=np.int64)
    errors = np.zeros(len(slots), dtype=np.float32)
    # Bound working memory even for large catalog/codebook combinations.
    for start in range(0, len(slots), 1024):
        stop = min(start + 1024, len(slots))
        for digit in range(books.shape[0]):
            squared = np.square(slots[start:stop, digit, None] - books[digit][None]).sum(-1)
            selected = squared.argmin(-1)
            codes[start:stop, digit] = selected
            errors[start:stop] += squared[np.arange(stop - start), selected]
    return codes, errors


@dataclass
class PSEAssets:
    root: Path
    manifest: dict
    centroids: torch.Tensor
    residual_features: torch.Tensor
    content_features: torch.Tensor
    catalog_sids: torch.Tensor
    reconstruction_errors: torch.Tensor
    original_item_ids: torch.Tensor

    @property
    def n_items(self) -> int:
        return len(self.original_item_ids) - 1

    @property
    def n_digit(self) -> int:
        return self.centroids.shape[0]

    @property
    def codebook_size(self) -> int:
        return self.centroids.shape[1]

    @property
    def code_dim(self) -> int:
        return self.centroids.shape[2]

    def make_tokenizer(self):
        """Only the catalog/token IDs needed by the existing backbone.

        List decoding owns collision buckets; it must not use the old lossy
        single-item ``tokens2item`` reverse dictionary.
        """
        items = [str(value) for value in self.original_item_ids[1:].tolist()]
        return SimpleNamespace(
            sid_offset=3, vocab_size=3 + self.n_digit * self.codebook_size,
            n_digit=self.n_digit, codebook_size=self.codebook_size,
            id2item=["[PAD]", *items],
            item2id={value: i + 1 for i, value in enumerate(items)},
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )

    def verify_dataset(self, dataset_dir: Path, *, verify: bool = True) -> None:
        if dataset_binding(dataset_dir, verify=verify) != self.manifest["dataset_binding"]:
            raise ValueError("PSE assets are bound to a different dataset/split/item mapping")
        expected = torch.from_numpy(catalog_original_ids(dataset_dir))
        if not torch.equal(expected, self.original_item_ids[1:]):
            raise ValueError("PSE original item ID order differs from dataset mapping")


def write_assets(output_dir: Path, *, binding: dict, original_item_ids: np.ndarray,
                 content_features: np.ndarray, residual_features: np.ndarray,
                 centroids: np.ndarray, coarse_centroid: np.ndarray,
                 encoder_metadata: dict, pca_transform: dict | None = None,
                 source_index: Path | None = None, seed: int = 2024,
                 toy: bool = False) -> PSEAssets:
    """Persist checked E0 and a legacy JointCrossPSEAlignment-compatible bundle.

    ``toy=True`` is only for explicit CPU fixtures with already-built tiny
    centroids; production requires the original FAISS index and 4x256x64.
    """
    from .runtime import engine_module

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite PSE assets: {root}")
    original = np.asarray(original_item_ids, dtype=np.int64)
    books = np.asarray(centroids, dtype=np.float32)
    content = np.asarray(content_features, dtype=np.float32)
    residual = np.asarray(residual_features, dtype=np.float32)
    center = np.asarray(coarse_centroid, dtype=np.float32)
    if books.ndim != 3 or books.shape[0] != 4 or books.shape[-1] % 4:
        raise ValueError("Four codebooks and a code dimension divisible by four heads are required")
    n, dim = len(original), books.shape[0] * books.shape[-1]
    if original.shape != (n,) or n == 0 or len(np.unique(original)) != n or np.any(original < 0):
        raise ValueError("Invalid original item IDs")
    if content.shape != (n, dim) or residual.shape != (n, dim) or center.shape != (dim,):
        raise ValueError("Content/residual/coarse-center dimensions differ from PSE")
    if not all(np.isfinite(value).all() for value in (books, content, residual, center)):
        raise ValueError("PSE assets contain non-finite values")
    if not toy:
        if books.shape != (4, 256, 64) or source_index is None or not Path(source_index).is_file():
            raise ValueError("Production requires 4x256x64 centroids and the fitted original FAISS index")
        validate_embedding_metadata(encoder_metadata, binding, 1024)
        if pca_transform is None:
            raise ValueError("Production requires frozen PCA transformation arrays")
    codes, errors = encode_residuals(residual, books)
    # Fork RNG so asset construction does not perturb recommender initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mixer = engine_module("cross_pse").CrossPSECodebookMixer(
            torch.from_numpy(books.copy()), n_heads=4, dropout=0.0,
            attention_scope="full", residual_scale_init=1.0,
            learn_residual_scale=False, zero_init_attention_output=True,
            train_base_codebooks=True, mixer_enabled=True)
    with torch.no_grad():
        if not torch.equal(mixer.crossed_codebooks(False)[0], torch.from_numpy(books)):
            raise ValueError("Full Mixer does not start exactly at E0")
    root.mkdir(parents=True)
    arrays = {
        "base_pse_codebooks.npy": books, "opq_residual_embeddings.npy": residual,
        "content_features.npy": content, "coarse_centroid.npy": center,
        "semantic_ids.npy": codes, "reconstruction_errors.npy": errors,
        "original_item_ids.npy": original,
    }
    for name, value in arrays.items():
        np.save(root / name, value, allow_pickle=False)
    # Tokens are original catalog IDs as strings; alignment maps them to 1..N.
    _dump(root / "items.json", [str(value) for value in original])
    torch.save({"model": mixer.state_dict(), "schema_version": SCHEMA_VERSION,
                "anchor": "base_pse_codebooks.npy", "initial_effective_equals_E0": True},
               root / "initial_mixer.pt")
    torch.save({"centroids": torch.from_numpy(books.copy()),
                "residual_features": torch.from_numpy(residual.copy()),
                "content_features": torch.from_numpy(content.copy()),
                "catalog_sids": torch.from_numpy(codes.copy()),
                "reconstruction_errors": torch.from_numpy(errors.copy()),
                "original_item_ids": torch.from_numpy(original.copy()),
                "coarse_centroid": torch.from_numpy(center.copy()),
                "schema_version": SCHEMA_VERSION}, root / "assets.pt")
    if pca_transform is not None:
        np.savez(root / "pca_transform.npz", **pca_transform)
    if source_index is not None:
        import shutil
        shutil.copyfile(source_index, root / "base_pse.faiss")
    _, counts = np.unique(codes, axis=0, return_counts=True)
    manifest = {
        "schema_version": SCHEMA_VERSION, "toy": bool(toy), "dataset_binding": binding,
        "n_items": n, "n_digit": 4, "codebook_size": books.shape[1], "code_dim": books.shape[2],
        "fit_scope": "full_static_catalog_no_behavior", "pad_in_fit": False,
        "seed": int(seed), "encoder": encoder_metadata, "text_template": TEXT_TEMPLATE,
        "pca": {"n_components": dim, "whiten": True, "svd_solver": "full", "normalize_after": True},
        "pse": {"factory": f"OPQ4,IVF1,PQ4x{int(np.log2(books.shape[1]))}",
                "metric": "inner_product", "by_residual": True,
                "clustering_rng": "unchanged_original_faiss_factory_defaults",
                "source_index_present": source_index is not None},
        "mixer": {"heads": 4, "dropout": 0.0, "scope": "full", "residual_scale": 1.0,
                  "learn_scale": False, "zero_output_projection": True, "anchor": "E0"},
        "collision": {"unique_sids": len(counts), "collided_items": int(counts[counts > 1].sum()),
                      "largest_bucket": int(counts.max())},
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                      for path in sorted(root.iterdir()) if path.is_file()},
    }
    _dump(root / "manifest.json", manifest)
    return load_assets(root)


def load_assets(path: Path, dataset_dir: Path | None = None, *, verify: bool = True) -> PSEAssets:
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = _json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported PSE asset schema")
    for name, record in manifest["artifacts"].items():
        target = root / name
        if Path(name).name != name or not target.is_file() or target.stat().st_size != record["bytes"]:
            raise ValueError(f"Invalid/missing PSE asset: {name}")
        if verify and sha256(target) != record["sha256"]:
            raise ValueError(f"PSE asset hash mismatch: {name}")
    bundle = torch.load(root / "assets.pt", map_location="cpu", weights_only=True)
    books = bundle["centroids"]
    n, digits, width = manifest["n_items"], manifest["n_digit"], manifest["code_dim"]
    if tuple(books.shape) != (digits, manifest["codebook_size"], width):
        raise ValueError("Asset codebook shape differs from manifest")
    if (tuple(bundle["catalog_sids"].shape) != (n, digits)
            or tuple(bundle["residual_features"].shape) != (n, digits * width)
            or tuple(bundle["content_features"].shape) != (n, digits * width)
            or tuple(bundle["original_item_ids"].shape) != (n,)
            or tuple(bundle["reconstruction_errors"].shape) != (n,)):
        raise ValueError("PSE bundle dimensions differ from manifest")
    for key in ("centroids", "residual_features", "content_features", "reconstruction_errors"):
        if not torch.isfinite(bundle[key]).all():
            raise ValueError(f"PSE bundle contains non-finite {key}")
    codes = bundle["catalog_sids"]
    if codes.dtype != torch.long or (codes < 0).any() or (codes >= books.shape[1]).any():
        raise ValueError("PSE SID codes are out of range")
    result = PSEAssets(root.resolve(), manifest, books,
        torch.cat([torch.zeros(1, digits, width), bundle["residual_features"].reshape(n, digits, width)]),
        torch.cat([torch.zeros(1, digits * width), bundle["content_features"]]),
        torch.cat([torch.full((1, digits), -1, dtype=torch.long), codes]),
        torch.cat([torch.full((1,), float("inf")), bundle["reconstruction_errors"]]),
        torch.cat([torch.full((1,), -1, dtype=torch.long), bundle["original_item_ids"]]))
    if dataset_dir is not None:
        result.verify_dataset(dataset_dir, verify=verify)
    return result
