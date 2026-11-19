"""Check an immutable suite and optionally package source/config only."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true", help="Seal a new local version once")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    inherited = json.loads((ROOT / "INHERITED_SOURCE.json").read_text(encoding="utf-8"))
    for name, expected in inherited["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Inherited source changed: {name}")
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    if args.seal:
        if manifest_path.exists() or (ROOT / "runs").exists():
            raise FileExistsError("Already sealed or used; make a new version instead")
        files = []
        for directory in ("code", "configs", "tests"):
            files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                         and not any(part in {"__pycache__", "tmp", ".pytest_cache"} for part in p.relative_to(ROOT).parts)
                         and p.suffix != ".pyc")
        files += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".py", ".md", ".json"}
                  and p.name not in {"PARENTS.lock.json", "SOURCE_MANIFEST.json", "BUNDLE.json"}]
        manifest = dict(version="h1_stability_followup_v1", sha256={
            p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Versioned artifact changed: {name}")
    result = dict(verified_files=len(manifest["sha256"]), inherited_files=len(inherited["sha256"]))
    if args.archive:
        path = ROOT / "stability_followup_v1.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name in sorted(manifest["sha256"]):
                tar.add(ROOT / name, arcname=name)
            tar.add(manifest_path, arcname=manifest_path.name)
            if (ROOT / "PARENTS.lock.json").exists():
                tar.add(ROOT / "PARENTS.lock.json", arcname="PARENTS.lock.json")
        result.update(archive=path.name, bytes=path.stat().st_size, sha256=digest(path))
        (ROOT / "BUNDLE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
