"""Stage 05 evidence packaging and hash-bound replay. Python stdlib only.

No existing experiment files are moved/deleted. Training is opt-in, uses a new
working directory, and never selects an initialization by modification time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from asset_locations import resolve_archive

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/05_crosspse_joint"
PREFIX = "diff_formal_"
SPECS = [
    ("cross_frozen", "Frozen Cross-PSE", "main", "crosspse_frozen_decoder_continuation_4x256_seed2024", ["crosspse_joint_diger_v2", "crosspse_frozen_decoder_v1"], "diff_frozen_crosspse_smoke.yaml"),
    ("cross_joint_v2", "Joint Cross-PSE / DIGER v2", "main", "crosspse_joint_diger_st_v2_slowanneal_4x256_seed2024", ["crosspse_joint_diger_v2"], "diff_diger_smoke.yaml"),
    ("refined_frozen", "Frozen Refined-PSE", "main", "refinedpse_frozen_decoder_continuation_4x256_seed2024", ["refinedpse_joint_diger_v1", "refinedpse_frozen_decoder_v1"], "diff_frozen_refinedpse_smoke.yaml"),
    ("refined_joint", "Joint Refined-PSE", "main", "refinedpse_joint_diger_st_v1_slowanneal_4x256_seed2024", ["refinedpse_joint_diger_v1"], "diff_diger_smoke.yaml"),
    ("cross_sia", "Joint Cross-PSE / SIA", "history", "crosspse_joint_sia_4x256_seed2024", ["crosspse_joint_sia"], "diff_joint_smoke.yaml"),
    ("cross_diger_v1", "Joint Cross-PSE / DIGER v1", "history", "crosspse_joint_diger_st_4x256_seed2024", ["crosspse_joint_diger"], "diff_diger_smoke.yaml"),
]


def sha(path):
    with Path(path).open("rb") as handle:
        return stream_hash(handle)


def stream_hash(handle):
    digest = hashlib.sha256()
    while block := handle.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    if not records:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def safe_relative(value):
    """Only portable project-relative POSIX file paths, never links/drives."""
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or any(p in ("..", ".") or ":" in p or "\\" in p for p in path.parts):
        raise ValueError(f"Unsafe relative path: {value}")
    return path.as_posix()


def under(root, relative):
    root = Path(root).resolve()
    target = (root / safe_relative(relative)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path escapes root: {relative}")
    return target


def project_path(value):
    value = str(value).replace("\\", "/")
    marker = "/experiment1/"
    if marker in value:
        return safe_relative(value.split(marker, 1)[1])
    if value.startswith("../../"):
        return safe_relative(value[6:])
    if value.startswith("saved/"):
        return safe_relative("external/DiffGRM/" + value)
    return safe_relative(value)


def copy_evidence(source, destination, manifest, source_root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source)
    if destination.exists() and sha(destination) != digest:
        raise ValueError(f"Refusing to overwrite different evidence: {destination}")
    if not destination.exists():
        shutil.copy2(source, destination)
    manifest.append({"path": destination.relative_to(PACKAGE).as_posix(), "sha256": digest,
                     "source": Path(source).relative_to(source_root).as_posix(),
                     "source_root": Path(source_root).resolve().as_posix()})


def extract_exact(archive, members, destination):
    """Extract a prevalidated allow-list; never overwrite different files."""
    remaining = set(members)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            if name not in remaining:
                continue
            if not member.isfile():
                raise ValueError(f"Not a regular file: {name}")
            record = members[name]
            target = under(destination, record["path"])
            if target.exists():
                if not target.is_file() or sha(target) != record["sha256"]:
                    raise ValueError(f"Existing artifact differs; preserved: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".stage05-part")
                if temp.exists():
                    raise ValueError(f"Incomplete previous extraction exists: {temp}")
                with tar.extractfile(member) as src, temp.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                if sha(temp) != record["sha256"]:
                    raise ValueError(f"Hash mismatch; incomplete file retained: {temp}")
                # A concurrent writer must not be overwritten.
                if target.exists():
                    raise ValueError(f"Destination appeared during extraction: {target}")
                temp.rename(target)
            remaining.remove(name)
    if remaining:
        raise ValueError(f"Archive is missing requested members: {sorted(remaining)}")


def archive_index(archive, needed, source_paths, cache_path):
    archive = Path(archive).resolve()
    archive_sha = sha(archive)
    print(f"Migration archive SHA256: {archive_sha}", flush=True)
    checksum_file = archive.parent / "SHA256SUMS"
    if checksum_file.is_file():
        expected = {line.split()[1].lstrip("*"): line.split()[0] for line in checksum_file.read_text().splitlines() if line.strip()}
        if expected.get(archive.name) != archive_sha:
            raise ValueError("Migration archive does not match SHA256SUMS")
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached["sha256"] == archive_sha and set(cached["requested_paths"]) == set(needed | source_paths):
            # Refresh the comparison; an unchanged archive does not imply an
            # unchanged local checkout on a later build.
            for record in cached["source_comparison"]:
                record["current_sha256"] = sha(ROOT / record["path"])
                record["byte_equal"] = record["current_sha256"] == record["archive_sha256"]
            cached["missing_source_files"] = sorted(source_paths - {r["path"] for r in cached["source_comparison"]})
            write_json(cache_path, cached)
            return cached
    selected = {}
    code_comparison = []
    data_prefix = "data/cache/AmazonReviews2014/Beauty/"
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            if not name.startswith("experiment1/") or not member.isfile():
                continue
            rel = name[len("experiment1/"):]
            shared_data = rel.startswith(data_prefix + "raw/") or rel in {
                data_prefix + "processed/all_item_seqs.json", data_prefix + "processed/id_mapping.json",
                data_prefix + "processed/metadata.sentence.json",
            }
            if rel not in needed and rel not in source_paths and not shared_data:
                continue
            digest = stream_hash(tar.extractfile(member))
            if rel in source_paths:
                current = ROOT / rel
                code_comparison.append({"path": rel, "archive_sha256": digest, "current_sha256": sha(current), "byte_equal": digest == sha(current)})
            else:
                selected[rel] = {"path": rel, "archive_member": name, "sha256": digest, "size_bytes": member.size,
                                 "binding_basis": "2026-09-23 migration snapshot; not a checksum recorded at original run time"}
                print(f"Bound artifact: {rel} ({member.size} bytes)", flush=True)
    missing = sorted(needed - selected.keys())
    result = {"filename": archive.name, "sha256": archive_sha, "size_bytes": archive.stat().st_size,
              "requested_paths": sorted(needed | source_paths), "artifacts": selected,
              "missing_artifacts": missing, "source_comparison": code_comparison,
              "missing_source_files": sorted(source_paths - {r["path"] for r in code_comparison}),
              "historical_runtime_commit": None}
    write_json(cache_path, result)
    return result


def code_lock(package):
    repositories = []
    source_paths = set()
    for name, rel in (("project", "."), ("DiffGRM", "external/DiffGRM"), ("genrec", "external/genrec")):
        repo = ROOT / rel
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        record = {"name": name, "path": rel, "commit": commit, "role": "current replay baseline, not proof of original runtime version"}
        if name == "project":
            record["note"] = "Base commit only; new packaging helpers are pinned separately by package_integrity.json. Commit the package after review."
        if name != "project":
            if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
                raise ValueError(f"Commit/save tracked core changes before packaging: {repo}")
            target = package / "reproducibility/code" / f"{name}_{commit[:12]}.tar"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", f"--output={target}", commit], check=True)
            record.update(snapshot=target.relative_to(package).as_posix(), sha256=sha(target))
            tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True).splitlines()
            source_paths.update(f"{rel}/{p}" for p in tracked if (repo / p).is_file())
        repositories.append(record)
    value = {"historical_runtime_commits": "not recorded in run metadata; do not backfill current commits",
             "repositories": repositories}
    write_json(package / "reproducibility/code_versions.json", value)
    return source_paths


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def analyze_run(package, key):
    folder = package / "evidence" / key
    meta = read_json(folder / "run_metadata.json")
    summary = read_json(folder / "summary.json")
    seed_rows = rows(folder / "per_seed.csv")
    if len(seed_rows) != 1 or int(seed_rows[0]["seed"]) != 2024:
        raise ValueError(f"Expected exactly seed 2024: {key}")
    if summary["run_id"] != meta["run_id"]:
        raise ValueError(f"Run identity mismatch: {key}")
    for metric in ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "duplicate@10"):
        if not close(meta["test_results"][metric], summary[metric]["mean"]) or not close(meta["test_results"][metric], seed_rows[0][metric]):
            raise ValueError(f"Result mismatch: {key}/{metric}")
    evaluation = rows(folder / "evaluation_history.csv")
    matches = [r for r in evaluation if r["split"] == "val" and int(r["epoch"]) == meta["best_epoch"] and r["metric"] == "weighted_score"]
    if len(matches) != 1 or not close(matches[0]["value"], meta["best_val_score"]):
        raise ValueError(f"Best validation checkpoint mismatch: {key}")
    phases = rows(folder / "joint_training_history.csv")
    selected = [r for r in phases if int(r["epoch"]) <= meta["best_epoch"]]
    return meta, evaluation, phases, {
        "best_wall_epoch": meta["best_epoch"],
        "decoder_epochs_at_best": sum(r["phase"] == "recommender" for r in selected),
        "tokenizer_phases_at_best": sum(r["phase"] == "tokenizer" for r in selected),
        "decoder_updates_at_best": sum(int(r["optimizer_updates"]) for r in selected if r["phase"] == "recommender"),
        "tokenizer_updates_at_best": sum(int(r["optimizer_updates"]) for r in selected if r["phase"] == "tokenizer"),
    }


def build(args):
    raise ValueError(
        "build is retired from the daily reproduction entry point. "
        "Use scripts/legacy/import_experiments.py for a read-only historical import plan; "
        "it only executes in an isolated first-import checkout and cannot repair existing seals."
    )


def _legacy_build(args):
    source = Path(args.source).resolve()
    if source.name != "experiment1":
        source /= "experiment1"
    archive = Path(args.archive).resolve()
    PACKAGE.mkdir(parents=True, exist_ok=True)
    old = PACKAGE / "reproducibility/previous_README.md"
    if not old.exists() and (PACKAGE / "README.md").exists():
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE / "README.md", old)
    manifest = []
    source_paths = code_lock(PACKAGE)
    metadata = [p for p in (source / "outputs/run_metadata/AmazonReviews2014/DIFF_GRM").glob("*_run_metadata.json")]
    needed = {"data/diff_formal_embeddings/beauty_item_keys.json"}
    recipes = []
    for key, label, role, suffix, layers, smoke in SPECS:
        run_id = PREFIX + suffix
        matched = [(p, read_json(p)) for p in metadata if read_json(p).get("run_id") == run_id and read_json(p).get("seed") == 2024]
        if len(matched) != 1:
            raise ValueError(f"Expected one exact metadata match: {run_id}; got {len(matched)}")
        mp, meta = matched[0]
        report = source / "outputs/reports" / run_id
        dst = PACKAGE / "evidence" / key
        copy_evidence(mp, dst / "run_metadata.json", manifest, source)
        for p in sorted(report.iterdir()):
            if p.suffix in (".csv", ".json"):
                copy_evidence(p, dst / p.name, manifest, source)
        for field in ("position_metrics_csv", "order_metrics_csv", "training_metrics_csv", "evaluation_metrics_csv", "joint_metrics_csv"):
            src = source / project_path(meta[field])
            copy_evidence(src, dst / "raw" / src.name, manifest, source)
        filenames = ["diff_formal_common.yaml", "diff_formal_repo_v6.yaml"] + ["diff_formal_" + n + ".yaml" for n in layers]
        for filename in filenames + [smoke]:
            copy_evidence(source / "configs" / filename, PACKAGE / "evidence/configs" / filename, manifest, source)
        initial_decoder = project_path(meta["joint_recommender_checkpoint"])
        artifact_dir = project_path(meta["joint_crosspse_artifact_dir"])
        tokenizer_files = [artifact_dir + "/" + name for name in (
            "best.pt", "base_pse_codebooks.npy", "opq_residual_embeddings.npy", "semantic_ids.npy", "semantic_ids.json", "summary.json")]
        required = [initial_decoder, "data/diff_formal_embeddings/beauty_item_keys.json", *tokenizer_files]
        final_files = [project_path(meta[f]) for f in ("checkpoint", "joint_final_semantic_ids_path", "joint_final_crossed_codebooks_path") if meta.get(f)]
        needed.update(required + final_files)
        recipes.append({"id": key, "label": label, "role": role, "historical_run_id": run_id, "seed": 2024,
                        "historical_runtime_commit": None,
                        "configuration_status": "reconstructed from migration YAML snapshots and frozen current code defaults; not an original fully resolved run config",
                        "config_layers": ["evidence/configs/" + f for f in filenames],
                        "smoke_config": "evidence/configs/" + smoke,
                        "initial_decoder": initial_decoder, "initial_tokenizer_dir": artifact_dir,
                        "required_artifacts": required, "evaluation_artifacts": final_files,
                        "metadata": f"evidence/{key}/run_metadata.json",
                        "evaluation_level": meta["evaluation_level"],
                        "verification": {"analysis": "verified", "gpu_smoke": "not_run", "full_training": "not_run", "checkpoint_evaluation": "not_run"}})
    # Source environment is a migration-time snapshot, not proof of each run's environment.
    for name in ("grm-exp.environment.yml", "grm-exp.pip-freeze.txt", "SHA256SUMS"):
        copy_evidence(archive.parent / name, PACKAGE / "reproducibility/environment" / name, manifest, archive.parent)
    lock = archive_index(archive, needed, source_paths, PACKAGE / "reproducibility/assets.lock.json")
    shared_data = [p for p in lock["artifacts"] if p.startswith("data/cache/")]
    for recipe in recipes:
        recipe["required_artifacts"] += sorted(shared_data)
        recipe["missing_from_archive"] = [p for p in recipe["required_artifacts"] if p not in lock["artifacts"]]
        meta = read_json(PACKAGE / recipe["metadata"])
        training = meta.get("tokenizer_training_summary") or {}
        prefix = recipe["initial_tokenizer_dir"]
        recorded = {
            prefix + "/semantic_ids.json": meta.get("semantic_ids_sha256"),
            prefix + "/best.pt": training.get("best_checkpoint_sha256"),
            prefix + "/base_pse_codebooks.npy": training.get("base_pse_codebooks_sha256"),
            "data/diff_formal_embeddings/beauty_item_keys.json": training.get("items_sha256"),
        }
        recipe["historically_recorded_input_hashes"] = {p: h for p, h in recorded.items() if h}
        for path, digest in recipe["historically_recorded_input_hashes"].items():
            if path not in lock["artifacts"] or lock["artifacts"][path]["sha256"] != digest:
                raise ValueError(f"Historical tokenizer input hash mismatch: {recipe['id']}/{path}")
        write_json(PACKAGE / "reproducibility/recipes" / (recipe["id"] + ".json"), recipe)
    # De-duplicate shared configuration evidence entries.
    manifest = list({r["path"]: r for r in manifest}.values())
    write_json(PACKAGE / "reproducibility/evidence_manifest.json", {"source": source.as_posix(), "files": manifest})
    seal_controls(PACKAGE)
    analyze(PACKAGE)
    verify_package(PACKAGE)
    print(f"Stage 05 package prepared: {PACKAGE}")


def analyze(package):
    results, index, analyses = [], [], {}
    for key, label, role, *_ in SPECS:
        meta, evaluation, phases, counts = analyze_run(package, key)
        analyses[key] = (meta, evaluation, phases, counts)
        record = {"id": key, "method": label, "role": role, "seed": 2024, **counts,
                  "actual_wall_epochs": max(int(r["epoch"]) for r in phases),
                  "actual_decoder_epochs": sum(r["phase"] == "recommender" for r in phases),
                  "actual_tokenizer_phases": sum(r["phase"] == "tokenizer" for r in phases),
                  "best_val_weighted_score": meta["best_val_score"],
                  **{k: meta["test_results"][k] for k in ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "weighted_score", "legal@10", "duplicate@10")},
                  "sid_collision_rate": meta["sid_collision_stats"]["collision_rate"],
                  "fit_hours": meta["fit_elapsed_seconds"] / 3600}
        results.append(record)
        index.append({"id": key, "run_id": meta["run_id"], "seed": 2024, "role": role,
                      "metadata": f"evidence/{key}/run_metadata.json", "recipe": f"reproducibility/recipes/{key}.json",
                      "initial_decoder_recorded": meta["joint_recommender_checkpoint"],
                      "initial_tokenizer_recorded": meta["joint_crosspse_artifact_dir"],
                      "evaluation_level": meta["evaluation_level"], "selection_metric": "validation weighted_score",
                      "historical_runtime_commit": "not_recorded", "gpu_replay_status": "not_run"})
    write_csv(package / "results.csv", results)
    write_csv(package / "runs.csv", index)
    plot_validation(package, analyses)
    return results


def plot_validation(package, analyses):
    from html import escape
    output = package / "figures"
    output.mkdir(exist_ok=True)
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420" viewBox="0 0 1000 420">',
                '<rect width="1000" height="420" fill="white"/>',
                '<style>text{font-family:Arial,sans-serif;fill:#253044;font-size:13px}</style>',
                '<text x="500" y="25" text-anchor="middle" font-size="18">Validation weighted score by Decoder epochs (seed 2024)</text>']
    main_keys = ("cross_frozen", "cross_joint_v2", "refined_frozen", "refined_joint")
    scores = [float(r["value"]) for key in main_keys for r in analyses[key][1]
              if r["split"] == "val" and r["metric"] == "weighted_score"]
    ymin = math.floor(min(scores) * 1000) / 1000
    ymax = math.ceil(max(scores) * 1000) / 1000
    if ymax == ymin:
        ymax += .001
    for panel, keys in enumerate((("cross_frozen", "cross_joint_v2"), ("refined_frozen", "refined_joint"))):
        left, top, width, height = 70 + panel * 490, 90, 390, 250
        elements.append(f'<text x="{left + width/2}" y="52" text-anchor="middle">{"Cross-PSE" if panel == 0 else "Refined-PSE"}</text>')
        for tick in range(round((ymax - ymin) * 1000) + 1):
            score = ymin + tick * .001
            y = top + height * (1 - (score - ymin)/(ymax-ymin))
            elements.extend([f'<path d="M {left} {y} h {width}" stroke="#e1e5eb"/>', f'<text x="{left-8}" y="{y+4}" text-anchor="end">{score:.3f}</text>'])
        for tick in range(0, 36, 5):
            x = left + width * tick / 35
            elements.append(f'<text x="{x}" y="{top+height+21}" text-anchor="middle">{tick}</text>')
        for number, key in enumerate(keys):
            meta, evaluation, phases, counts = analyses[key]
            points = []
            color = ("#3976a8", "#dc7c26")[number]
            for row in evaluation:
                if row["split"] != "val" or row["metric"] != "weighted_score":
                    continue
                epoch = int(row["epoch"])
                dec = sum(r["phase"] == "recommender" and int(r["epoch"]) <= epoch for r in phases)
                x = left + width * dec / 35
                y = top + height * (1 - (float(row["value"])-ymin)/(ymax-ymin))
                points.append(f"{x:.2f},{y:.2f}")
                if epoch == meta["best_epoch"]:
                    elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
            elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            elements.append(f'<text x="{left + number*190}" y="75" style="fill:{color}">{escape(key)}</text>')
        elements.append(f'<text x="{left+width/2}" y="392" text-anchor="middle">Decoder epochs (initial state = 0)</text>')
    elements.append('</svg>')
    (output / "validation_decoder_epochs.svg").write_text("\n".join(elements), encoding="utf-8")


def seal_controls(package):
    """Lock newly supplied replay machinery separately from historical evidence."""
    paths = [package / "reproducibility" / name for name in
             ("assets.lock.json", "code_versions.json", "evidence_manifest.json")]
    paths += sorted((package / "reproducibility/recipes").glob("*.json"))
    previous = package / "reproducibility/previous_README.md"
    if previous.exists():
        paths.append(previous)
    value = {"basis": "current packaging revision; not historical runtime source",
             "files": [{"path": p.relative_to(package).as_posix(), "sha256": sha(p)} for p in paths],
             "helpers": [{"path": name, "sha256": sha(ROOT / name)}
                         for name in ("scripts/repro/stage05_repro.py", "tests/test_stage05_repro.py", "scripts/repro/asset_locations.py")]}
    write_json(package / "reproducibility/package_integrity.json", value)


def verify_package(package):
    integrity = read_json(package / "reproducibility/package_integrity.json")
    for record in integrity["files"]:
        if sha(under(package, record["path"])) != record["sha256"]:
            raise ValueError(f"Replay lock/recipe changed: {record['path']}")
    for record in integrity["helpers"]:
        if sha(under(ROOT, record["path"])) != record["sha256"]:
            raise ValueError(f"Replay helper changed: {record['path']}; review and rebuild before running")
    count = 0
    for record in read_json(package / "reproducibility/evidence_manifest.json")["files"]:
        path = under(package, record["path"])
        if not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError(f"Evidence missing/changed: {path}")
        count += 1
    for record in read_json(package / "reproducibility/code_versions.json")["repositories"]:
        if "snapshot" in record and sha(under(package, record["snapshot"])) != record["sha256"]:
            raise ValueError(f"Code snapshot changed: {record['name']}")
    for key, *_ in SPECS:
        analyze_run(package, key)
    print(f"Verified {count} evidence files, code snapshots and six result/metadata pairs. GPU runs not performed.")


def load_recipe(package, key):
    if key not in {s[0] for s in SPECS}:
        raise ValueError(f"Unknown experiment: {key}")
    return read_json(package / "reproducibility/recipes" / (key + ".json"))


def check_inputs(package, recipe, project_root):
    records = read_json(package / "reproducibility/assets.lock.json")["artifacts"]
    errors = []
    for rel in recipe["required_artifacts"]:
        if rel not in records:
            errors.append(f"No archived checksum: {rel}")
            continue
        path = under(project_root, rel)
        if not path.is_file():
            errors.append(f"Missing input: {path}")
        elif sha(path) != records[rel]["sha256"]:
            errors.append(f"Input checksum mismatch: {path}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Verified {len(recipe['required_artifacts'])} fixed inputs for {recipe['id']}")


def restore_inputs(args):
    verify_package(PACKAGE)
    recipe = load_recipe(PACKAGE, args.experiment)
    lock = read_json(PACKAGE / "reproducibility/assets.lock.json")
    archive = resolve_archive(args.archive, ROOT)
    if sha(archive) != lock["sha256"]:
        raise ValueError("Archive SHA256 mismatch")
    paths = recipe["required_artifacts"] + (recipe["evaluation_artifacts"] if args.include_final else [])
    if any(p not in lock["artifacts"] for p in paths):
        raise ValueError("Some required files were absent from migration archive")
    members = {lock["artifacts"][p]["archive_member"]: lock["artifacts"][p] for p in paths}
    extract_exact(archive, members, Path(args.project_root))
    check_inputs(PACKAGE, recipe, args.project_root)


def unpack_code(snapshot, target):
    if target.exists():
        raise ValueError(f"Code destination must be new: {target}")
    target.mkdir(parents=True)
    with tarfile.open(snapshot, "r:") as tar:
        for member in tar:
            rel = safe_relative(member.name.rstrip("/"))
            dst = under(target, rel)
            if member.isdir():
                dst.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dst.open("xb") as out:
                    shutil.copyfileobj(src, out)
            else:
                raise ValueError(f"Code archive contains unsupported link/type: {member.name}")


def validate_smoke_phases(records, frozen, direct):
    decoder = [r for r in records if r["phase"] == "recommender"]
    tokenizer = [r for r in records if r["phase"] == "tokenizer"]
    if not decoder or (frozen and tokenizer) or (not frozen and not tokenizer):
        raise ValueError("Smoke did not exercise the expected frozen/alternating phases")
    for row in records:
        if not math.isfinite(float(row["total_loss"])):
            raise ValueError("Non-finite smoke loss")
        own = "recommender" if row["phase"] == "recommender" else "tokenizer"
        other = "tokenizer" if own == "recommender" else "recommender"
        gradient = float(row[own + "_gradient_norm"])
        if not math.isfinite(gradient) or gradient <= 0 or float(row[other + "_gradient_norm"]) != 0:
            raise ValueError("Smoke gradient isolation failed")
    if direct and not frozen:
        gradients = [float(r["diger_direct_gradient_norm"]) for r in tokenizer]
        if not all(math.isfinite(g) for g in gradients) or max(gradients) <= 0:
            raise ValueError("No finite positive direct recommendation gradient in smoke")


def audit_formal_config(config, metadata):
    """Fail closed on recorded settings, without inventing missing history."""
    keys = ("sid_quantizer", "n_digit", "codebook_size", "masking_strategy", "guided_select",
            "guided_refresh_each_step", "diffusion_loss_reduction", "train_batch_size",
            "gradient_accumulation_steps", "warmup_steps", "lr", "mixed_precision", "max_grad_norm",
            "numerical_guard", "eval_start_epoch", "eval_interval", "patience", "early_stop_after_warmup",
            "evaluation_level", "joint_crosspse_enabled", "joint_frozen_tokenizer_control",
            "crosspse_mixer_enabled", "crosspse_train_base_codebooks")
    expected = {k: metadata[k] for k in keys if metadata.get(k) is not None}
    for key, value in (metadata.get("joint_alignment") or {}).items():
        if key in ("gradient_mode", "recommender_warmup_epochs", "cycle_epochs", "tokenizer_epochs_per_cycle", "max_tokenizer_phases") or key.startswith("diger_") and key != "diger_final_schedule_state":
            expected["joint_" + key] = value
    aliases = {"temperature": "joint_sia_temperature", "recommender_weight": "joint_recommender_sia_weight", "tokenizer_weight": "joint_tokenizer_sia_weight"}
    for source, target in aliases.items():
        if source in (metadata.get("joint_alignment") or {}):
            expected[target] = metadata["joint_alignment"][source]
    differences = {k: {"recorded": v, "resolved": config.get(k)} for k, v in expected.items() if config.get(k) != v}
    if differences:
        raise ValueError("Resolved formal config differs from recorded settings: " + json.dumps(differences))
    return {"status": "passed", "checked_settings": expected,
            "limitation": "Only metadata-recorded settings checked; original full resolved config unavailable"}


def run_replay(args):
    # No import of torch/PyYAML and no creation of output files for a plan.
    recipe = load_recipe(PACKAGE, args.experiment)
    verify_package(PACKAGE)
    check_inputs(PACKAGE, recipe, args.project_root)
    print(json.dumps({"experiment": recipe["id"], "mode": args.mode, "initial_decoder": recipe["initial_decoder"],
                      "config_layers": recipe["config_layers"], "uses_latest_checkpoint_lookup": False}, indent=2))
    if not args.execute:
        print("Plan only. Add --execute in the GPU training environment to run. No training started.")
        return
    import importlib.metadata
    try:
        import yaml
        import torch
    except ImportError as exc:
        raise ValueError("Activate the grm-exp GPU environment (PyYAML and PyTorch required)") from exc
    if not torch.cuda.is_available():
        raise ValueError("No CUDA GPU; training was not started")
    project_root = Path(args.project_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"stage05_replay_{recipe['id']}_{args.mode}_{timestamp}"
    work = project_root / "outputs/stage05_replays" / run_id
    work.mkdir(parents=True, exist_ok=False)
    lock = read_json(PACKAGE / "reproducibility/code_versions.json")
    code = next(r for r in lock["repositories"] if r["name"] == "DiffGRM")
    engine = work / "engine"
    unpack_code(under(PACKAGE, code["snapshot"]), engine)
    config = {"model": "DIFF_GRM", "dataset": "AmazonReviews2014"}
    layers = [engine / "genrec/default.yaml", engine / "genrec/datasets/AmazonReviews2014/config.yaml", engine / "genrec/models/DIFF_GRM/config.yaml"]
    layers += [under(PACKAGE, p) for p in recipe["config_layers"]]
    if args.mode == "smoke":
        layers.append(under(PACKAGE, recipe["smoke_config"]))
    for layer in layers:
        config.update(yaml.safe_load(layer.read_text(encoding="utf-8")) or {})
    # Match get_config's top-level scalar conversion without importing genrec.
    import ast
    for key, value in list(config.items()):
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                config[key] = value.lower() == "true"
            else:
                try:
                    config[key] = json.loads(value)
                except (ValueError, TypeError):
                    try:
                        config[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
    # Materialize the pinned implementation's implicit Boolean defaults.
    for key, value in (("joint_frozen_tokenizer_control", False), ("crosspse_mixer_enabled", True),
                       ("crosspse_train_base_codebooks", True)):
        config.setdefault(key, value)
    artifact = under(project_root, recipe["initial_tokenizer_dir"])
    config.update(run_id=run_id, rand_seed=2024, joint_recommender_checkpoint=str(under(project_root, recipe["initial_decoder"])),
                  joint_crosspse_artifact_dir=str(artifact), precomputed_semantic_ids_path=str(artifact / "semantic_ids.json"),
                  crosspse_summary_path=str(artifact / "summary.json"),
                  joint_crosspse_items_path=str(under(project_root, "data/diff_formal_embeddings/beauty_item_keys.json")))
    config["joint_crosspse_checkpoint"] = str(artifact / "best.pt")
    config["joint_crosspse_slots_path"] = str(artifact / "opq_residual_embeddings.npy")
    for key, subdir in (("log_dir", "logs"), ("tensorboard_log_dir", "tensorboard"), ("ckpt_dir", "checkpoints"),
                        ("run_metadata_dir", "run_metadata"), ("diffusion_diagnostics_log_dir", "diagnostics")):
        config[key] = str(work / subdir)
    # Dataset preprocessing and tokenizer caches are writable only in this new run.
    config["cache_dir"] = str(work / "cache")
    for rel in recipe["required_artifacts"]:
        if rel.startswith("data/cache/"):
            dst = work / "cache" / rel[len("data/cache/"):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(under(project_root, rel), dst)
    config_path = work / "resolved_config.json"
    write_json(config_path, config)  # JSON is valid YAML for the existing loader.
    if args.mode == "formal":
        write_json(work / "configuration_audit.json", audit_formal_config(config, read_json(PACKAGE / recipe["metadata"])))
    environment = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())}
    write_json(work / "environment.json", environment)
    command = [sys.executable, "main.py", "--model", "DIFF_GRM", "--dataset", "AmazonReviews2014", "--config", str(config_path)]
    write_json(work / "launch.json", {"argv": command, "cwd": str(engine), "recipe": recipe,
                                     "code": code, "config_sha256": sha(config_path), "status": "started"})
    env = os.environ.copy()
    env.update(HF_ENDPOINT=env.get("HF_ENDPOINT", "https://hf-mirror.com"), HF_HOME=str(work / "huggingface"),
               TOKENIZERS_PARALLELISM="false", TORCHDYNAMO_DISABLE="1", WANDB_MODE="disabled")
    with (work / "console.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=engine, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = child.wait()
    write_json(work / "completion.json", {"exit_code": code, "completed_utc": datetime.now(timezone.utc).isoformat()})
    if code:
        raise ValueError(f"Training failed with exit {code}; preserved log: {work / 'console.log'}")
    if args.mode == "smoke":
        metrics = list((work / "diagnostics").rglob("*_joint_metrics.csv"))
        if len(metrics) != 1:
            raise ValueError("Expected exactly one joint-metrics file in this new smoke run")
        validate_smoke_phases(rows(metrics[0]), bool(config.get("joint_frozen_tokenizer_control", False)),
                              config.get("joint_gradient_mode") == "diger_st")
        write_json(work / "smoke_check.json", {"status": "passed", "metrics": str(metrics[0]),
                                             "scope": "finite loss and expected gradient routes; not metric reproduction"})
    print(f"Run completed; outputs: {work}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build", help="Retired; use the guarded historical import entry point")
    build_parser.add_argument("--source", help="Legacy option; build is retired")
    build_parser.add_argument("--archive", help="Legacy option; build is retired")
    sub.add_parser("analyze")
    sub.add_parser("verify")
    for name in ("check-inputs", "restore-inputs", "run"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", choices=[r[0] for r in SPECS], required=True)
        p.add_argument("--project-root", required=True)
        if name == "restore-inputs":
            p.add_argument("--archive", help="Explicit archive; otherwise use the ignored local location map")
            p.add_argument("--include-final", action="store_true", help="Also restore final model and paired SID exports; does not evaluate them")
        if name == "run":
            p.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
            p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        build(args)
    elif args.action == "analyze":
        verify_package(PACKAGE)
        analyze(PACKAGE)
    elif args.action == "verify":
        verify_package(PACKAGE)
    elif args.action == "restore-inputs":
        restore_inputs(args)
    elif args.action == "check-inputs":
        verify_package(PACKAGE)
        check_inputs(PACKAGE, load_recipe(PACKAGE, args.experiment), args.project_root)
    else:
        run_replay(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
