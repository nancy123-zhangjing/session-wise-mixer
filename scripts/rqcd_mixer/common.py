"""Small self-contained runtime/state helpers; no dependency on legacy trainers."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch

from .config import ROOT, DESIGN, DEFAULT_CONFIG, RUNTIME_TEMPLATE, TARGET_CONFIGS, TARGET_DESIGN


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def seed_for(seed, *labels):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, *labels], ensure_ascii=False, separators=(",", ":")).encode()).digest()[:8], "little") % (2**63-1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot exactly resume a CUDA dropout stream on CPU")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(cpu_copy(value), tmp)
    os.replace(tmp, path)


def read_state(path):
    # Checkpoints are locally produced, trusted experiment artifacts.
    return torch.load(path, map_location="cpu", weights_only=False)


@contextmanager
def output_lock(root, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=resume)
    lock = root / ".running"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    try:
        yield root
    finally:
        lock.unlink(missing_ok=True)


def append_log(root, record):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with (Path(root) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def source_binding():
    files = [*Path(__file__).parent.glob("*.py"), DEFAULT_CONFIG, RUNTIME_TEMPLATE, DESIGN,
             *TARGET_CONFIGS.values(), TARGET_DESIGN,
             *DEFAULT_CONFIG.parent.glob("requirements*.txt"),
             ROOT / "scripts/rqcd/data.py", ROOT / "scripts/listwise_diffgrm/assets.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def tensor_digest(values):
    h = hashlib.sha256()
    for k, v in sorted(values.items()):
        v = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str((v.shape, v.dtype)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def configure_runtime(runtime, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(runtime["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = runtime["deterministic"]
    torch.set_float32_matmul_precision("highest")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if runtime["precision"] == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Formal BF16 execution requires a supported CUDA GPU")
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"), "platform": platform.platform(),
            "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "runtime": runtime, "tf32": False}


def autocast(runtime, device):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=runtime["precision"] == "bf16")


@contextmanager
def attention_backend(name):
    if name == "auto":
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    kinds = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}
    with sdpa_kernel(kinds[name]):
        yield


def adamw(params, cfg, runtime, device, *, lr, weight_decay):
    t = cfg["training"]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                             betas=tuple(t["betas"]), eps=t["eps"],
                             fused=bool(runtime["fused_adamw"] and torch.device(device).type == "cuda"))


def r_learning_rate(update, total, cfg):
    peak, warm = cfg["training"]["r_lr"], cfg["training"]["warmup_updates"]
    if not 1 <= update <= total:
        raise ValueError("R update outside approved budget")
    if update <= warm:
        return peak * update / warm
    return peak * (1 + math.cos(math.pi * (update-warm)/(total-warm))) / 2


def pre_learning_rate(completed, steps_per_epoch, cfg):
    c = cfg["rq"]
    warm, total = steps_per_epoch*c["warmup_epochs"], steps_per_epoch*c["epochs"]
    if completed < warm:
        return c["lr"] * completed / max(warm, 1)
    return c["lr"] * max(0, total-completed) / max(total-warm, 1)


def gumbel_strength(update, cfg):
    from .config import total_t_steps
    q, total = cfg["quantization"], total_t_steps(cfg)
    if not 1 <= update <= total:
        raise ValueError("T update outside approved budget")
    phase = (update-1)/max(total-1, 1)
    return q["gumbel_end"]+(q["gumbel_start"]-q["gumbel_end"])*(1+math.cos(math.pi*phase))/2
