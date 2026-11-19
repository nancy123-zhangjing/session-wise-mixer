"""Load unmodified, hash-locked DIGER modules without polluting global imports."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "configs/session_wise_ar/upstream.lock.json"
_modules = None


def verify(fetch=False):
    lock = json.loads(LOCK.read_text())
    base = ROOT / lock["local_dir"]
    for entry in lock["files"]:
        path = base / entry["path"]
        if not path.exists() and fetch:
            url = f'https://raw.githubusercontent.com/junchen-fu/DIGER/{lock["commit"]}/{entry["path"]}'
            payload = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Upstream download hash mismatch: {entry['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Missing/modified upstream {path}; run upstream --fetch explicitly")
    return base


@contextmanager
def alias(name, value):
    previous = sys.modules.get(name)
    sys.modules[name] = value
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    global _modules
    if _modules is not None:
        return _modules
    base = verify()
    prefix = "_experiment1_diger_ec0a9b51"
    package = types.ModuleType(prefix)
    package.__path__ = [str(base)]
    sys.modules[prefix] = package
    layers = _load(prefix + ".layers", base / "layers.py")
    with alias("layers", layers):
        vq = _load(prefix + ".vq", base / "vq.py")
        model = _load(prefix + ".model", base / "model.py")
    pre_package = types.ModuleType(prefix + ".pre")
    pre_package.__path__ = [str(base / "scripts/rqvae/models")]
    sys.modules[pre_package.__name__] = pre_package
    pre = _load(prefix + ".pre.rqvae", base / "scripts/rqvae/models/rqvae.py")
    _modules = types.SimpleNamespace(vq=vq, model=model, pre=pre, layers=layers)
    return _modules


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    print(verify(parser.parse_args().fetch))
