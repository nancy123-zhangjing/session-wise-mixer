"""Protocol, numerical policy and portable state helpers."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from scripts.listwise_diffgrm.training import atomic_save, cpu_copy, rng_state, restore_rng
from scripts.listwise_diffgrm.assets import sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/09_kuairec_listwise/rqcd_h_v1.json"


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["protocol_id"] != "kuairec_rqcd_h_v1":
        raise ValueError("Unsupported protocol")
    if cfg["model"]["width"] % cfg["model"]["heads"]:
        raise ValueError("Attention width must be divisible by heads")
    if cfg["training"]["effective_batch"] < cfg["training"]["microbatch"]:
        raise ValueError("microbatch exceeds effective batch")
    return cfg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def derive_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**63 - 1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(cfg, device):
    # Must be set before the first CUDA allocation. Never enable TF32/FP8 as a
    # hidden speed optimization of this protocol.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(cfg["runtime"]["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg["runtime"]["deterministic"]
    torch.set_float32_matmul_precision("highest")
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        if cfg["training"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requires a supported CUDA device")
    return {
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tf32": False, "deterministic": cfg["runtime"]["deterministic"],
        "sdpa_backend": cfg["runtime"]["sdpa_backend"],
    }


@contextmanager
def attention_backend(name):
    if name == "auto":
        yield
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        choices = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}
        if name not in choices:
            raise ValueError(f"Unknown SDPA backend: {name}")
        with sdpa_kernel(choices[name]):
            yield


def learning_rate(update, total, warmup, peak):
    if not 1 <= update <= total:
        raise ValueError("LR update outside fixed budget")
    if update <= warmup:
        return peak * update / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (update - warmup) / max(total - warmup, 1)))


def optimizer(parameters, cfg, device, *, rq=False):
    t = cfg["training"]
    o = cfg["rq"] if rq else t
    # Fused AdamW removes many small optimizer kernels on H100. On CPU use the
    # reference path; both have exactly the same hyperparameters.
    return torch.optim.AdamW(
        parameters, lr=o["lr"], weight_decay=o["weight_decay"],
        betas=tuple(t["betas"]), eps=t["eps"],
        fused=bool(cfg["runtime"]["fused_adamw"] and torch.device(device).type == "cuda"),
    )


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, t in sorted(tensors.items()):
        t = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((str(t.dtype), tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def source_binding():
    paths = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "scripts/listwise_diffgrm/assets.py",
        ROOT / "scripts/listwise_diffgrm/metrics.py",
        ROOT / "scripts/listwise_diffgrm/decoding.py",
        ROOT / "scripts/listwise_diffgrm/training.py",
    ]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(paths)}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def preserve_rng():
    saved = rng_state()
    try:
        yield
    finally:
        restore_rng(saved)


@contextmanager
def output_lock(root, resume=False):
    """Never overwrite an old run; a stale lock requires explicit inspection."""
    root = Path(root)
    if root.exists() and not resume:
        raise FileExistsError(f"Output already exists: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    lock = root / ".running"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield root
    finally:
        lock.unlink()
