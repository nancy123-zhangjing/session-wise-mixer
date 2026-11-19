"""Locate large assets without changing their locked identities (Python 3.10+).

Only restore-inputs and the explicit check command call this module's resolver.
Importing it does not read machine-specific configuration or access any assets.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = "reproducibility/asset_locations.local.json"


def resolve_archive(explicit=None, root=ROOT):
    root = Path(root).resolve()
    if explicit is not None:
        value = str(explicit)
    else:
        config = root / LOCAL_CONFIG
        if not config.is_file():
            raise ValueError(f"Pass --archive or configure {config}; no default archive is guessed")
        mapping = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(mapping, dict) or mapping.get("schema_version") != 1:
            raise ValueError("Asset locations must be a schema_version=1 object")
        if set(mapping) != {"schema_version", "migration_archive"}:
            raise ValueError("Asset locations accept only a path, never checksums or identity overrides")
        value = mapping["migration_archive"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("migration_archive must be a non-empty path")
    path = Path(value).expanduser()
    # Preserve the old CLI's cwd-relative --archive semantics. Only paths from
    # the machine map are relative to the project containing that map.
    path = path.resolve() if explicit is not None or path.is_absolute() else (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Migration archive not found: {path}")
    return path


def verify_archive(path, expected):
    path = Path(path)
    if path.stat().st_size != expected["size_bytes"]:
        raise ValueError("Migration archive size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError("Migration archive SHA256 mismatch")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check",))
    parser.add_argument("--archive", help="Overrides the ignored local location map")
    args = parser.parse_args()
    path = resolve_archive(args.archive)
    shared = json.loads((ROOT / "reproducibility/migration_assets.lock.json").read_text(encoding="utf-8"))
    stage05 = json.loads((ROOT / "experiments/05_crosspse_joint/reproducibility/assets.lock.json").read_text(encoding="utf-8"))
    if any(shared[k] != stage05[k] for k in ("sha256", "size_bytes")):
        raise ValueError("Shared and Stage 05 archive identities disagree")
    print(f"Checking {path} ({shared['size_bytes']} bytes); read-only", flush=True)
    digest = verify_archive(path, shared)
    print(json.dumps({"archive": str(path), "sha256": digest,
                      "size_bytes": shared["size_bytes"], "verified": True,
                      "independent_backup_verified": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
