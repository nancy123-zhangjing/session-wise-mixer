"""Verify and freshly extract the SHA-bound validation evidence archive.

Only regular files listed by the manifest and the manifest itself are accepted.
The local acceptance receipt is written beside, never inside, the frozen payload.
An interrupted/failed extraction is preserved for inspection, never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

MANIFEST_NAME = "VALIDATION_EVIDENCE_MANIFEST.json"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_valid(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def safe_name(value):
    need(isinstance(value, str) and value and "\\" not in value and ":" not in value
         and "\x00" not in value, "Unsafe archive path spelling")
    path = PurePosixPath(value)
    need(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/"))
         and path.as_posix() == value, "Archive path must be canonical and relative")
    for part in path.parts:
        need(not part.endswith((".", " ")) and not any(ord(char) < 32 for char in part),
             "Unsafe Windows path component")
        stem = part.split(".", 1)[0].upper()
        need(stem not in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                          *[f"LPT{i}" for i in range(1, 10)]}, "Windows reserved device path")
    return path


def reject_link_ancestors(path):
    for candidate in (path, *path.parents):
        need(not candidate.is_symlink()
             and not (hasattr(candidate, "is_junction") and candidate.is_junction()),
             "Extraction destination must not traverse links or junctions")


def verify_delivery(archive, receipt_path, destination):
    archive, receipt_path = Path(archive).resolve(), Path(receipt_path).resolve()
    destination = Path(destination).absolute()
    reject_link_ancestors(destination)
    destination = destination.resolve()
    acceptance = destination.with_name(destination.name + "_LOCAL_DELIVERY_VERIFICATION.json")
    need(archive.is_file() and receipt_path.is_file(), "Archive and remote receipt are required")
    need(not archive.is_relative_to(destination) and not receipt_path.is_relative_to(destination),
         "Inputs must be outside the fresh extraction destination")
    need(not acceptance.exists(), "Refuse to overwrite an earlier local acceptance receipt")
    need(not destination.exists() or (destination.is_dir() and not any(destination.iterdir())),
         "Extraction requires a new or empty directory; no automatic resume/overwrite")
    receipt_sha = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    need(type(receipt.get("bytes")) is int and receipt["bytes"] > 0
         and type(receipt.get("files")) is int and receipt["files"] > 0
         and digest_valid(receipt.get("sha256")) and digest_valid(receipt.get("manifest_sha256")),
         "Remote receipt has invalid byte/count/hash fields")
    need(archive.stat().st_size == receipt["bytes"] and sha(archive) == receipt["sha256"],
         "Downloaded archive differs from the remote receipt")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names, canonical = {}, set()
        for member in members:
            name = safe_name(member.name).as_posix()
            need(member.isfile() and not member.issym() and not member.islnk()
                 and member.sparse is None, "Only ordinary regular files are allowed")
            need(name.casefold() not in canonical, "Duplicate or case-colliding archive members")
            canonical.add(name.casefold())
            names[name] = member
        need(MANIFEST_NAME in names and 0 < names[MANIFEST_NAME].size <= MAX_MANIFEST_BYTES,
             "Missing or oversized evidence manifest")
        with source.extractfile(names[MANIFEST_NAME]) as handle:
            manifest_bytes = handle.read(MAX_MANIFEST_BYTES + 1)
        need(len(manifest_bytes) == names[MANIFEST_NAME].size
             and hashlib.sha256(manifest_bytes).hexdigest() == receipt["manifest_sha256"],
             "Manifest bytes differ from the remote receipt")
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        files = manifest.get("files")
        need(isinstance(files, list) and len(files) == receipt["files"], "Manifest/receipt file count differs")
        expected, folded = {}, set()
        for row in files:
            need(isinstance(row, dict), "Malformed manifest entry")
            name = safe_name(row.get("path")).as_posix()
            need(name != MANIFEST_NAME and name.casefold() not in folded, "Duplicate/recursive manifest entry")
            need(type(row.get("bytes")) is int and row["bytes"] >= 0 and digest_valid(row.get("sha256")),
                 "Malformed manifest file size/hash")
            folded.add(name.casefold())
            expected[name] = row
        need(set(names) == set(expected) | {MANIFEST_NAME}, "Archive has missing or unlisted files")
        need("results/COMPLETE.json" in expected, "Archive lacks full evaluation completion evidence")
        for name, row in expected.items():
            need(names[name].size == row["bytes"], f"Tar/manifest file size differs: {name}")
        # Detect file-versus-directory conflicts before writing any payload bytes.
        for name in names:
            for parent in PurePosixPath(name).parents:
                need(parent.as_posix().casefold() not in canonical, "Archive file/directory path conflict")
        reject_link_ancestors(destination)
        destination.mkdir(parents=True, exist_ok=True)
        need(not any(destination.iterdir()), "Destination changed after preflight")
        # Python 3.12+ and the local 3.14 runtime support this safety filter.
        source.extractall(destination, members=members, filter="data")
    actual = {}
    for path in destination.rglob("*"):
        need(not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction()),
             "Extracted payload contains a link or junction")
        if path.is_file():
            actual[path.relative_to(destination).as_posix()] = path
        else:
            need(path.is_dir(), "Extracted payload contains a special file")
    need(set(actual) == set(names), "Extracted files differ from the archive inventory")
    verified_bytes = 0
    for name, row in expected.items():
        path = actual[name]
        need(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
             f"Extracted file differs from manifest: {name}")
        verified_bytes += row["bytes"]
    need(sha(actual[MANIFEST_NAME]) == receipt["manifest_sha256"], "Extracted manifest hash differs")
    need(sha(archive) == receipt["sha256"] and sha(receipt_path) == receipt_sha,
         "Archive or remote receipt changed during verification")
    result = {"status": "LOCAL_ARCHIVE_BYTES_VERIFIED", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "archive": str(archive), "archive_bytes": receipt["bytes"], "archive_sha256": receipt["sha256"],
              "remote_receipt": str(receipt_path), "remote_receipt_sha256": receipt_sha,
              "remote_archive_source": receipt.get("archive"), "destination": str(destination),
              "manifest_sha256": receipt["manifest_sha256"], "verified_manifest_files": len(expected),
              "verified_payload_bytes_excluding_manifest": verified_bytes,
              "extracted_regular_files_including_manifest": len(actual),
              "unlisted_files": 0, "links_or_special_files": 0, "verifier_sha256": sha(Path(__file__).resolve()),
              "scope": "Local archive/extracted byte integrity; scientific result acceptance is a separate review"}
    with acceptance.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_delivery(args.archive, args.receipt, args.destination), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
