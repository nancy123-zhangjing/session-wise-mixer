"""Non-destructive research packaging for stages 01-04 (05 remains sealed).

Original archives/files are read-only. Outputs are derived tables, byte-exact
evidence copies, source snapshots, manifests and opt-in isolated replay jobs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile

import stage05_repro as safe
from asset_locations import resolve_archive

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "reproducibility"
SOURCE = ROOT / "archive/evidence_bundles/repro_sources_01_04"
DATA = "data/cache/AmazonReviews2014/Beauty/"
STAGES = {
    "01": "01_ar_baseline_3x256", "02": "02_ar_capacity_ablation",
    "03": "03_diffgrm_tokenizer_comparison", "04": "04_crosspse_offline",
}
SMALL_ARCHIVES = [
    "01_ar_baseline_full_evidence.tar.gz", "01_ar_baseline_summary.tar.gz",
    "01_ar_baseline_attention.tar.gz", "02_ar_capacity_evidence_seed2024.tar.gz",
    "02_ar_capacity_attention_seed2024.tar.gz", "03_diffgrm_tokenizers_evidence_seed2024.tar.gz",
]
TEXT_EXT = {".json", ".csv", ".yaml", ".yml", ".md", ".txt", ".py", ".sh", ".ps1"}
METRICS = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "legal@10", "duplicate@10")
AR_LOGS = {
    "Aug-08-2026_03-18-07": "ar_formal_rqkmeans_seed2024_20260808_031804.log",
    "Aug-08-2026_08-48-38": "ar_formal_rqkmeans_seed2025_20260808_084836.log",
    "Aug-08-2026_14-30-32": "ar_formal_rqkmeans_seed2026_20260808_143029.log",
    "Aug-07-2026_08-29-16": "ar_formal_rqvae_seed2024_20260807_082913.log",
    "Aug-08-2026_06-04-42": "ar_formal_rqvae_seed2025_20260808_060440.log",
    "Aug-08-2026_11-37-10": "ar_formal_rqvae_seed2026_20260808_113708.log",
    "Aug-14-2026_05-17-06": "ar_formal_formal_pse_3x256_seed2024_20260814_051703.log",
    "Aug-14-2026_07-34-19": "ar_formal_formal_rqkmeans_128x256x512_seed2024_20260814_073417.log",
    "Aug-14-2026_10-19-23": "ar_formal_formal_rqkmeans_512x256x128_seed2024_20260814_101921.log",
}
sha = safe.sha
read_json = safe.read_json
write_json = safe.write_json
rows = safe.rows
under = safe.under
relative = safe.safe_relative


def specs():
    result = []
    dates = {
        "rqk": ["Aug-08-2026_03-18-07", "Aug-08-2026_08-48-38", "Aug-08-2026_14-30-32"],
        "rqvae": ["Aug-07-2026_08-29-16", "Aug-08-2026_06-04-42", "Aug-08-2026_11-37-10"],
    }
    def add(stage, key, method, run, date, config, seed=2024, artifact=None, sizes=None, reference=None):
        model = "AR_GRM" if stage in ("01", "02") else "DIFF_GRM"
        result.append(dict(stage=stage, id=key, method=method, run_id=run, seed=seed, model=model,
                           stamp="AmazonReviews2014_" + date, config=config, artifact_dir=artifact,
                           semantic_codebook_sizes=sizes or ([256] * (3 if model == "AR_GRM" else 4)),
                           reference=reference))
    for method in ("rqk", "rqvae"):
        for seed, date in zip((2024, 2025, 2026), dates[method]):
            kind = "rqkmeans" if method == "rqk" else "rqvae"
            add("01", f"{method}_seed{seed}", "RQ-KMeans" if method == "rqk" else "RQ-VAE",
                f"ar_formal_{kind}_3x256", date, ["ar_formal_common.yaml", f"ar_formal_{kind}.yaml"], seed,
                f"outputs/tokenizers/{kind}/Beauty/seed{seed}")
    add("02", "pse_seed2024", "PSE", "ar_formal_pse_3x256", "Aug-14-2026_05-17-06",
        ["ar_formal_common.yaml", "ar_formal_pse.yaml"])
    for sizes, date in (([128,256,512], "Aug-14-2026_07-34-19"), ([512,256,128], "Aug-14-2026_10-19-23")):
        layout = "x".join(map(str, sizes))
        key = "rqk_" + "_".join(map(str, sizes)) + "_seed2024"
        add("02", key, "RQ-KMeans " + layout, "ar_formal_rqkmeans_" + layout, date,
            ["ar_formal_common.yaml", "ar_formal_rqkmeans_" + "_".join(map(str, sizes)) + ".yaml"],
            artifact=f"outputs/tokenizers/rqkmeans_{layout}/Beauty/seed2024", sizes=sizes)
    for item in list(result):
        if item["stage"] == "01" and item["seed"] == 2024:
            result.append({**item, "stage": "02", "reference": {"stage": "01", "id": item["id"]}})
    common = ["diff_formal_common.yaml", "diff_formal_repo_v6.yaml"]
    add("03", "pse_seed2024", "PSE", "diff_formal_pse_4x256_repo_repro_v6_bf16", "Aug-11-2026_14-09-45",
        [common[0], "diff_formal_pse.yaml", common[1]])
    add("03", "rqk_seed2024", "RQ-KMeans", "diff_formal_rqkmeans_4x256_v6_bf16", "Aug-12-2026_06-37-17",
        common + ["diff_formal_rqkmeans_v6.yaml"])
    add("03", "rqvae_seed2024", "RQ-VAE", "diff_formal_rqvae_4x256_v6_bf16", "Aug-12-2026_12-04-25",
        common + ["diff_formal_rqvae_v6.yaml"], artifact="outputs/tokenizers/diff_rqvae/Beauty/seed2024")
    add("04", "refined_seed2024", "Refined-PSE", "diff_formal_refinedpse_4x256_v2_catalog_bf16", "Aug-28-2026_14-15-28",
        common + ["diff_formal_refinedpse_v2_catalog.yaml"], artifact="outputs/tokenizers/diff_refinedpse/Beauty/v2_catalog_seed2024")
    add("04", "cross_seed2024", "Cross-PSE", "diff_formal_crosspse_4x256_v3_catalog_bf16", "Aug-28-2026_19-01-02",
        common + ["diff_formal_crosspse_v3_catalog.yaml"], artifact="outputs/tokenizers/diff_crosspse/Beauty/v3_catalog_seed2024")
    pse = next(x for x in result if x["stage"] == "03" and x["method"] == "PSE")
    result.append({**pse, "stage": "04", "reference": {"stage": "03", "id": pse["id"]}})
    return result


def package(stage):
    return ROOT / "experiments" / STAGES[stage]


def immutable_copy(source, destination):
    """Copies are additive. An existing, different destination is an error."""
    digest = sha(source)
    if destination.exists():
        if not destination.is_file() or sha(destination) != digest:
            raise ValueError(f"Preserved differing existing file: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return digest


def immutable_bytes(data, destination):
    digest = hashlib.sha256(data).hexdigest()
    if destination.exists():
        if sha(destination) != digest:
            raise ValueError(f"Preserved differing existing file: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as out:
            out.write(data)
    return digest


def write_csv(path, records, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def prepare_sources(archive):
    """One streaming pass indexes assets and recovers only lightweight evidence."""
    SHARED.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name in SMALL_ARCHIVES:
        path = ROOT / "archive/evidence_bundles" / name
        digest = sha(path)
        dest = SOURCE / name.removesuffix(".tar.gz")
        count = 0
        with tarfile.open(path, "r:gz") as tar:
            for entry in tar:
                if not entry.isfile() or Path(entry.name).suffix not in TEXT_EXT:
                    continue
                target = under(dest, entry.name.removeprefix("./"))
                immutable_bytes(tar.extractfile(entry).read(), target)
                count += 1
        receipts.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=digest, recovered_text_files=count))
    archive = Path(archive).resolve()
    digest = sha(archive)
    if digest != "a74836899900bccb9d44a18f6403b4220b50615f53b26503c82825dcb2f9bc09":
        raise ValueError("Unexpected migration archive; review its identity before rebuilding locks")
    lock_path = SHARED / "migration_assets.lock.json"
    if not lock_path.exists():
        assets, extracted = {}, []
        target_root = SOURCE / "migration"
        with tarfile.open(archive, "r|gz") as tar:
            for entry in tar:
                name = entry.name.removeprefix("./")
                if not entry.isfile() or not name.startswith("experiment1/"):
                    continue
                rel = relative(name[len("experiment1/"):])
                is_asset = rel.startswith((DATA, "data/formal_embeddings/", "data/diff_formal_embeddings/",
                                          "outputs/tokenizers/", "external/DiffGRM/saved/", "outputs/console/"))
                is_text = (rel.startswith(("outputs/run_metadata/", "outputs/position_loss_logs/", "outputs/diffusion_diagnostics/",
                                           "outputs/reports/", "outputs/attention/", "outputs/tokenizers/", "configs/", "scripts/", "docs/"))
                           and Path(rel).suffix in TEXT_EXT)
                is_text = is_text or rel in {"outputs/console/" + x for x in AR_LOGS.values()}
                # Original console stays in the immutable archive; selected formal
                # logs are copied later if explicitly needed for derived history.
                if not is_asset and not is_text:
                    continue
                handle = tar.extractfile(entry)
                if is_text:
                    payload = handle.read()
                    h = immutable_bytes(payload, under(target_root, rel))
                    extracted.append(dict(path=rel, sha256=h, size_bytes=entry.size, archive_member=name))
                else:
                    h = safe.stream_hash(handle)
                if is_asset:
                    assets[rel] = dict(path=rel, archive_member=name, sha256=h, size_bytes=entry.size,
                                       provenance="2026-09-23 migration snapshot; not automatically a historical runtime hash")
        write_json(lock_path, dict(filename=archive.name, sha256=digest, size_bytes=archive.stat().st_size,
                                   artifacts=assets, extracted_evidence=extracted))
    migration = read_json(lock_path)
    if migration["sha256"] != digest:
        raise ValueError("Migration lock/archive mismatch")
    for record in migration["extracted_evidence"]:
        if sha(under(SOURCE / "migration", record["path"])) != record["sha256"]:
            raise ValueError(f"Recovered migration evidence changed: {record['path']}")
    for name in ("grm-exp.environment.yml", "grm-exp.pip-freeze.txt", "SHA256SUMS"):
        immutable_copy(archive.parent / name, SHARED / "environment" / name)
    receipts.append(dict(path=archive.as_posix(), sha256=digest, size_bytes=archive.stat().st_size,
                         note="Keep an independent backup; large assets are not stored in Git"))
    write_json(SHARED / "sources.lock.json", {"sources": receipts})
    print(f"Indexed {len(migration['artifacts'])} archive assets; recovered {len(migration['extracted_evidence'])} lightweight source files", flush=True)
    return migration


def code_snapshots():
    entries = []
    for name, rel in (("project", "."), ("DiffGRM", "external/DiffGRM"), ("genrec", "external/genrec")):
        repo = ROOT / rel
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        paths = ["scripts", "configs", "patches", "requirements-linux.txt", "requirements-windows.txt", "environment.yml", "upstreams.lock"] if name == "project" else []
        dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--", *paths], text=True).splitlines()
        if any(not p.endswith(".md") for p in dirty):
            raise ValueError(f"Uncommitted runtime source in {name}; preserve/commit before binding")
        out = SHARED / "code" / f"{name}_{commit[:12]}.tar"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "--output=" + str(out), commit, *paths], check=True)
        entries.append(dict(name=name, commit=commit, snapshot=out.relative_to(ROOT).as_posix(), sha256=sha(out),
                            role="current replay source snapshot, not the historical runtime commit"))
    write_json(SHARED / "code_versions.json", dict(repositories=entries, historical_runtime_commit=None))
    comparison = []
    project = next(r for r in entries if r["name"] == "project")
    with tarfile.open(ROOT / project["snapshot"], "r:") as archive:
        for entry in archive:
            if not entry.isfile() or not entry.name.startswith(("scripts/", "configs/")) or entry.name.endswith(".md"):
                continue
            original = under(SOURCE / "migration", entry.name)
            if original.is_file():
                current, migrated = archive.extractfile(entry).read(), original.read_bytes()
                comparison.append(dict(path=entry.name, byte_equal=current == migrated,
                                       equal_after_crlf_to_lf=current.replace(b"\r\n", b"\n") == migrated.replace(b"\r\n", b"\n")))
    write_json(SHARED / "source_comparison.json", {
        "scope": "Current project commit scripts/configs versus later migration snapshot, not original run-time proof",
        "files": comparison})
    return entries


def evidence_copy(source, pack, target, manifest):
    path = under(pack, target)
    digest = immutable_copy(source, path)
    manifest.append(dict(path=target, sha256=digest, source=source.relative_to(ROOT).as_posix(),
                         copy_kind="byte_exact"))


def find_report(item):
    migration = SOURCE / "migration"
    if item["stage"] == "01":
        return SOURCE / "01_ar_baseline_full_evidence/formal_3x256_evidence_v2/reports"
    if item["stage"] == "02":
        return SOURCE / "02_ar_capacity_evidence_seed2024/outputs/reports/ar_internal_study"
    if item["stage"] == "04" and not item["reference"]:
        base = ROOT / "archive/deprecated_and_temp/tmp_browser_cache/crosspse_report_20260829/crosspse_report_20260829_094937"
        return base / "reports" / item["run_id"]
    return migration / "outputs/reports" / item["run_id"]


def recipe_inputs(item, metadata, assets):
    required = [DATA + "raw/" + f for f in ("meta_Beauty.json.gz", "reviews_Beauty_5.json.gz")]
    required += [DATA + "processed/" + f for f in ("all_item_seqs.json", "id_mapping.json", "metadata.sentence.json")]
    seed, method = item["seed"], item["method"]
    artifact = item["artifact_dir"]
    tokenizer_assets = [p for p in assets if artifact and p.startswith(artifact + "/")]
    if item["model"] == "AR_GRM":
        required += ["data/formal_embeddings/" + f for f in ("beauty_sentence_t5_xl.npy", "beauty_item_keys.json", "beauty_sentence_t5_xl_manifest.json")]
        if method.startswith("RQ-KMeans"):
            layout = "3x8" if item["semantic_codebook_sizes"] == [256]*3 else "_" + "x".join(map(str, item["semantic_codebook_sizes"]))
            sid = DATA + f"processed/sentence-t5-xl_pca32_RQKMEANS{layout}_seed{seed}_it20_inputpca32.sem_ids"
            required += [DATA + f"processed/sentence-t5-xl_pca32_fitseed{seed}.sent_emb" + suffix for suffix in ("", ".pkl")]
            required += [artifact + "/" + f for f in ("reconstructed_embeddings.npy", "quantizer_artifacts.npz", "semantic_ids.npy", "semantic_ids.json")]
        elif method == "PSE":
            sid = DATA + "processed/sentence-t5-xl_pca0_OPQ3,IVF1,PQ3x8.sem_ids"
        else:
            sid = artifact + "/semantic_ids.json"
    elif method in ("PSE", "RQ-KMeans"):
        suffix = "OPQ4,IVF1,PQ4x8" if method == "PSE" else "RQKMEANS4x8_seed2024_it20_PCA"
        sid = DATA + f"processed/sentence-t5-base_pca256_{suffix}.sem_ids"
        required.append(DATA + "processed/sentence-t5-base_pca256.sent_emb")
    else:
        sid = artifact + "/semantic_ids.json"
        required.append(artifact + "/summary.json")
    required.append(sid)
    extra_inputs = []
    if item["model"] == "DIFF_GRM":
        extra_inputs += [p for p in assets if p.startswith("data/diff_formal_embeddings/")]
    if item["stage"] == "04" and not item["reference"]:
        extra_inputs += [p for p in assets if p.startswith("outputs/tokenizers/diff_pse_base/Beauty/seed2024/")]
    expected_sid = metadata.get("semantic_ids_sha256")
    if expected_sid and sid in assets and assets[sid]["sha256"] != expected_sid:
        raise ValueError(f"Historical SID hash mismatch: {item['id']}")
    checkpoint = "external/DiffGRM/saved/" + item["stamp"] + "/pytorch_model.bin"
    return dict(required_artifacts=sorted(set(required)), tokenizer_artifacts=sorted(set(tokenizer_assets + extra_inputs)),
                evaluation_artifacts=[checkpoint], initial_sid=sid, historical_initial_sid_sha256=expected_sid,
                decoder_initialization={"type": "random_initialization_with_seed", "seed": seed, "checkpoint": None},
                historical_final_checkpoint=checkpoint)


def copy_run_evidence(item, pack, manifest):
    migration = SOURCE / "migration"
    meta_path = migration / f"outputs/run_metadata/AmazonReviews2014/{item['model']}/{item['stamp']}_run_metadata.json"
    metadata = read_json(meta_path)
    if metadata["run_id"] != item["run_id"] or int(metadata["seed"]) != item["seed"]:
        raise ValueError(f"Run identity mismatch: {meta_path}")
    folder = f"evidence/{item['id']}"
    evidence_copy(meta_path, pack, folder + "/run_metadata.json", manifest)
    for field, value in metadata.items():
        if field.endswith("_csv") and isinstance(value, str):
            original = migration / safe.project_path(value)
            if original.is_file():
                evidence_copy(original, pack, folder + "/raw/" + original.name, manifest)
    report = find_report(item)
    for p in sorted(report.glob("*")):
        if p.suffix in (".json", ".csv"):
            evidence_copy(p, pack, f"evidence/reports/{report.name}/" + p.name, manifest)
    for filename in item["config"]:
        evidence_copy(migration / "configs" / filename, pack, "evidence/configs/" + filename, manifest)
    if item["artifact_dir"]:
        for p in (migration / item["artifact_dir"]).glob("*"):
            if p.suffix in (".csv", ".json"):
                evidence_copy(p, pack, folder + "/tokenizer/" + p.name, manifest)
    if item["model"] == "AR_GRM":
        if item["run_id"].endswith("_3x256") and item["method"] != "PSE":
            quantizer = "rq_kmeans" if item["method"] == "RQ-KMeans" else "rqvae"
            attention = migration / f"outputs/attention/formal_3x256/{quantizer}/seed{item['seed']}"
        else:
            variant = "pse_256x256x256" if item["method"] == "PSE" else "rqk_" + "x".join(map(str, item["semantic_codebook_sizes"]))
            attention = migration / f"outputs/attention/ar_internal_study/{variant}/seed2024"
    else:
        attention = migration / f"outputs/attention/{item['run_id']}/seed2024"
    for p in attention.glob("*"):
        if p.suffix in (".csv", ".json"):
            evidence_copy(p, pack, folder + "/attention/" + p.name, manifest)
    return metadata, folder, report


def copy_history(stage, pack, manifest):
    migration = SOURCE / "migration"
    selected = {x["stamp"] for x in specs()}
    if stage == "03":
        # Original PSE metadata predates per-position utilization logging. The
        # later tokenizer summary explicitly contains the unchanged base PSE.
        evidence_copy(migration / "outputs/tokenizers/diff_refinedpse/Beauty/v2_catalog_seed2024/summary.json",
                      pack, "evidence/supplementary/pse_base_statistics_from_stage04.json", manifest)
        for p in (ROOT / "outputs/run_metadata/AmazonReviews2014/DIFF_GRM").glob("*_run_metadata.json"):
            if p.name.removesuffix("_run_metadata.json") not in selected:
                evidence_copy(p, pack, "evidence/history/" + p.name, manifest)
    if stage == "04":
        base = ROOT / "archive/deprecated_and_temp/tmp_browser_cache/crosspse_report_20260829/crosspse_report_20260829_094937/tokenizers/pilots"
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".csv", ".json"):
                evidence_copy(p, pack, "evidence/history/" + p.relative_to(base).as_posix(), manifest)


def parse_ar_console(text):
    """Recover scalar histories without evaluating Python from a log."""
    training, evaluation, seen_train, seen_val = [], [], set(), set()
    number = r"[+\-\d.eE]+"
    for match in re.finditer(r"\[Epoch (\d+)\] Train Loss:\s*(" + number + r")", text):
        epoch, value = int(match[1]), float(match[2])
        if epoch in seen_train or not math.isfinite(value):
            raise ValueError("Duplicate/non-finite AR training epoch")
        seen_train.add(epoch)
        training.append(dict(epoch=epoch, train_loss=value))
    for match in re.finditer(r"\[Epoch (\d+)\] Val Results: ([^\r\n]*)", text):
        epoch = int(match[1])
        if epoch in seen_val:
            raise ValueError("Duplicate AR validation epoch")
        seen_val.add(epoch)
        for metric in re.finditer(r"\('([^']+)',\s*(?:np\.float\d+\()?(" + number + r")\)?\)", match[2]):
            value = float(metric[2])
            if not math.isfinite(value):
                raise ValueError("Non-finite AR validation metric")
            evaluation.append(dict(epoch=epoch, split="val", metric=metric[1], value=value))
    return training, evaluation


def derive_ar_history(item, pack, migration):
    console = "outputs/console/" + AR_LOGS[item["stamp"].removeprefix("AmazonReviews2014_")]
    record = migration["artifacts"][console]
    original = SOURCE / "migration" / console
    if sha(original) != record["sha256"]:
        raise ValueError("AR console source changed: " + console)
    training, evaluation = parse_ar_console(original.read_text(encoding="utf-8", errors="replace"))
    if not training or not evaluation:
        raise ValueError("AR scalar history could not be recovered")
    dest = pack / "derived" / item["id"]
    write_csv(dest / "training_history.csv", training)
    write_csv(dest / "evaluation_history.csv", evaluation)
    write_json(dest / "provenance.json", {
        "kind": "derived_from_console_not_original_csv", "source": record,
        "archive_sha256": migration["sha256"], "parser": "scripts/repro/project_repro.py:parse_ar_console",
        "note": "Numbers transcribed by regex; original console remains in the archive and recovered source cache."})


def build(args):
    raise ValueError(
        "build is retired from the daily reproduction entry point. "
        "Use scripts/legacy/import_experiments.py for a read-only historical import plan; "
        "it only executes in an isolated first-import checkout and cannot repair existing seals."
    )


def _legacy_build(args):
    migration = prepare_sources(args.archive)
    versions = code_snapshots()
    assets = migration["artifacts"]
    for stage in STAGES:
        pack = package(stage)
        previous = pack / "reproducibility/previous_README.md"
        if not previous.exists():
            immutable_copy(pack / "README.md", previous)
        manifest, stage_assets = [], set()
        for item in [x for x in specs() if x["stage"] == stage]:
            meta, folder, report = copy_run_evidence(item, pack, manifest)
            inputs = recipe_inputs(item, meta, assets)
            for field in ("required_artifacts", "tokenizer_artifacts", "evaluation_artifacts"):
                stage_assets.update(inputs[field])
            console = None
            if item["model"] == "AR_GRM":
                console = "outputs/console/" + AR_LOGS[item["stamp"].removeprefix("AmazonReviews2014_")]
                stage_assets.add(console)
                derive_ar_history(item, pack, migration)
            recipe = {**item, **inputs, "metadata": folder + "/run_metadata.json",
                      "replay_mode": "frozen_historical_sid_random_decoder",
                      "historical_runtime_commit": None,
                      "config_layers": ["evidence/configs/" + x for x in item["config"]],
                      "config_basis": "migration YAML plus pinned replay source defaults; not original fully resolved config",
                      "selection_metric": "ndcg@10" if item["model"] == "AR_GRM" else "weighted_score",
                      "evaluation_level": "item" if item["model"] == "AR_GRM" else "sid",
                      "report_per_seed": f"evidence/reports/{report.name}/per_seed.csv",
                      "historical_console": console,
                      "missing_inputs": [p for p in inputs["required_artifacts"] if p not in assets],
                      "status": {"analysis": "verified_at_build", "gpu_smoke": "not_run", "full_retraining": "not_run", "tokenizer_retraining": "not_run", "checkpoint_evaluation": "not_run"}}
            write_json(pack / "reproducibility/recipes" / (item["id"] + ".json"), recipe)
        copy_history(stage, pack, manifest)
        write_json(pack / "reproducibility/assets.lock.json", {"archive_filename": migration["filename"], "archive_sha256": migration["sha256"],
                    "artifacts": {p: assets[p] for p in sorted(stage_assets) if p in assets}, "missing": sorted(stage_assets - assets.keys())})
        write_json(pack / "reproducibility/code_versions.json", {"repositories": versions, "historical_runtime_commit": None,
                    "shared_paths_relative_to": "project_root", "new_helpers": ["scripts/repro/project_repro.py", "scripts/repro/project_replay.py", "scripts/repro/stage05_repro.py"]})
        unique = {r["path"]: r for r in manifest}
        write_json(pack / "reproducibility/evidence_manifest.json", {"files": list(unique.values()), "original_files_deleted": False})
        analyze(stage)
        validate_scientific_evidence(stage)
        seal(stage)
        verify(stage)
    registry()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def analyze(stage, write=True):
    pack = package(stage)
    results, index = [], []
    for item in [x for x in specs() if x["stage"] == stage]:
        recipe = read_json(pack / "reproducibility/recipes" / (item["id"] + ".json"))
        meta = read_json(pack / recipe["metadata"])
        if meta["run_id"] != item["run_id"] or int(meta["seed"]) != item["seed"]:
            raise ValueError(f"Metadata run identity mismatch: {stage}/{item['id']}")
        candidates = [r for r in rows(pack / recipe["report_per_seed"]) if int(r["seed"]) == item["seed"]
                      and item["stamp"] in r.get("metadata", "")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one exact reported result for {stage}/{item['id']}: {len(candidates)}")
        reported = candidates[0]
        if int(reported["best_epoch"]) != int(meta["best_epoch"]):
            raise ValueError(f"Best epoch mismatch: {stage}/{item['id']}")
        if "fit_hours" in reported and not close(reported["fit_hours"], meta["fit_elapsed_seconds"] / 3600):
            raise ValueError(f"Fit time mismatch: {stage}/{item['id']}")
        for metric in METRICS:
            if not close(reported[metric], meta["test_results"][metric]):
                raise ValueError(f"Reported result/metadata mismatch: {item['id']} {metric}")
        result = dict(id=item["id"], method=item["method"], run_id=item["run_id"], seed=item["seed"],
                      role="reused_baseline" if item["reference"] else "primary", reference_stage=item["reference"]["stage"] if item["reference"] else "",
                      model=item["model"], evaluation_level=recipe["evaluation_level"], semantic_positions=len(item["semantic_codebook_sizes"]),
                      suffix_positions=int(item["model"] == "AR_GRM"), semantic_codebook_sizes="x".join(map(str, item["semantic_codebook_sizes"])),
                      best_epoch=meta["best_epoch"], best_val_score=meta.get("best_val_score", ""), selection_metric=recipe["selection_metric"],
                      **{k: meta["test_results"][k] for k in METRICS}, fit_hours=meta["fit_elapsed_seconds"]/3600)
        results.append(result)
        index.append(dict(id=item["id"], run_id=item["run_id"], seed=item["seed"], metadata=recipe["metadata"],
                          recipe="reproducibility/recipes/" + item["id"] + ".json", model=item["model"],
                          evaluation_level=recipe["evaluation_level"], role=result["role"], reference_stage=result["reference_stage"],
                          historical_runtime_commit="not_recorded", replay_mode=recipe["replay_mode"], gpu_smoke="not_run",
                          checkpoint=meta["checkpoint"], initial_sid=recipe.get("initial_sid", "")))
    if write:
        write_csv(pack / "results.csv", results)
        write_csv(pack / "runs.csv", index)
    grouped = {}
    for row in results:
        grouped.setdefault(row["method"], []).append(row)
    summaries = []
    for method, group in grouped.items():
        row = dict(method=method, n_seeds=len(group), seeds=",".join(str(r["seed"]) for r in group), evaluation_level=group[0]["evaluation_level"], std_ddof=1)
        for metric in METRICS:
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values)>1 else ""
        summaries.append(row)
    if write:
        write_csv(pack / "summary.csv", summaries)
    return results


def validate_scientific_evidence(stage):
    """Check actual selection history and all saved positional metrics."""
    pack, checks = package(stage), []
    for item in [x for x in specs() if x["stage"] == stage]:
        recipe = load_recipe(stage, item["id"])
        meta = read_json(pack / recipe["metadata"])
        folder = pack / "evidence" / item["id"] / "raw"
        if item["model"] == "AR_GRM":
            evaluation = pack / "derived" / item["id"] / "evaluation_history.csv"
        else:
            evaluation = folder / Path(safe.project_path(meta["evaluation_metrics_csv"])).name
        candidates = [r for r in rows(evaluation) if r["split"] == "val" and r["metric"] == recipe["selection_metric"]]
        if not candidates:
            raise ValueError("Missing recorded validation selection history")
        best = max(candidates, key=lambda r: (float(r["value"]), -int(r["epoch"])))
        if int(best["epoch"]) != int(meta["best_epoch"]) or not close(best["value"], meta["best_val_score"]):
            raise ValueError(f"Validation history does not support chosen checkpoint: {stage}/{item['id']}")
        positions = folder / Path(safe.project_path(meta["position_metrics_csv"])).name
        position_rows = rows(positions)
        for row in position_rows:
            for field in ("nll", "accuracy"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Non-finite {field} in {positions}")
        for split, epoch in (("train", meta["best_epoch"]), ("test", 0)):
            selected = [r for r in position_rows if r["split"] == split and int(r["epoch"]) == int(epoch)]
            if len(selected) != 4 or len({r["position"] for r in selected}) != 4:
                raise ValueError(f"Incomplete {split} positional metrics for {item['id']}")
        checks.append(dict(id=item["id"], best_epoch=int(best["epoch"]), selection_metric=recipe["selection_metric"],
                           selection_value=float(best["value"]), position_rows=len(position_rows),
                           train_at_best_positions=4, test_positions=4))
    write_json(pack / "reproducibility/analysis_checks.json", {"runs": checks, "gpu_replay_performed": False})


def seal(stage):
    pack = package(stage)
    files = [pack / "reproducibility" / x for x in ("assets.lock.json", "code_versions.json", "evidence_manifest.json", "previous_README.md")]
    files += sorted((pack / "reproducibility/recipes").glob("*.json"))
    files += [pack / "reproducibility/analysis_checks.json"]
    files += [pack / name for name in ("results.csv", "summary.csv", "runs.csv")]
    files += sorted((pack / "derived").rglob("*.*"))
    helpers = [ROOT / x for x in ("scripts/repro/project_repro.py", "scripts/repro/project_replay.py", "tests/test_project_repro.py", "scripts/repro/stage05_repro.py", "scripts/repro/asset_locations.py")]
    helpers += [SHARED / x for x in ("sources.lock.json", "code_versions.json", "migration_assets.lock.json", "source_comparison.json")]
    helpers += sorted((SHARED / "environment").glob("*"))
    write_json(pack / "reproducibility/package_integrity.json", {
        "files": [{"path": p.relative_to(pack).as_posix(), "sha256": sha(p)} for p in files],
        "shared": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in helpers]})


def verify(stage):
    pack = package(stage)
    integrity = read_json(pack / "reproducibility/package_integrity.json")
    for r in integrity["files"]:
        if sha(under(pack, r["path"])) != r["sha256"]:
            raise ValueError(f"Stage {stage} lock/recipe changed: {r['path']}")
    for r in integrity["shared"]:
        if sha(under(ROOT, r["path"])) != r["sha256"]:
            raise ValueError(f"Replay helper changed: {r['path']}; review before rebuilding")
    evidence = read_json(pack / "reproducibility/evidence_manifest.json")["files"]
    for r in evidence:
        if sha(under(pack, r["path"])) != r["sha256"]:
            raise ValueError(f"Stage {stage} evidence changed: {r['path']}")
    for r in read_json(pack / "reproducibility/code_versions.json")["repositories"]:
        if sha(under(ROOT, r["snapshot"])) != r["sha256"]:
            raise ValueError(f"Source snapshot changed: {r['snapshot']}")
    analyze(stage, write=False)
    print(f"Stage {stage}: verified {len(evidence)} evidence copies, recipes and source snapshots; GPU replay not run")


def registry():
    records = []
    for stage in STAGES:
        for row in rows(package(stage) / "runs.csv"):
            records.append({"stage": stage, "package": "experiments/" + STAGES[stage], **row})
    for row in rows(ROOT / "experiments/05_crosspse_joint/runs.csv"):
        records.append({"stage": "05", "package": "experiments/05_crosspse_joint", **row})
    write_json(ROOT / "experiments/experiment_registry.json", {"runs": records,
                    "warning": "References reuse historical runs. AR item-level and DiffGRM SID-level are not one leaderboard."})


def load_recipe(stage, experiment):
    if experiment not in {x["id"] for x in specs() if x["stage"] == stage}:
        raise ValueError(f"Unknown experiment {stage}/{experiment}")
    return read_json(package(stage) / "reproducibility/recipes" / (experiment + ".json"))


def check_inputs(stage, recipe, project_root):
    assets = read_json(package(stage) / "reproducibility/assets.lock.json")["artifacts"]
    errors = []
    for name in recipe["required_artifacts"]:
        path = under(project_root, name)
        if name not in assets:
            errors.append("Not bound in archive: " + name)
        elif not path.is_file():
            errors.append("Missing: " + name)
        elif sha(path) != assets[name]["sha256"]:
            errors.append("Hash mismatch: " + name)
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Verified {len(recipe['required_artifacts'])} exact inputs for {stage}/{recipe['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="Retired; use the guarded historical import entry point")
    b.add_argument("--archive", help="Legacy option; this command always refuses to rebuild sealed packages")
    for action in ("verify", "analyze", "list"):
        p = sub.add_parser(action)
        p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    for action in ("check-inputs", "restore-inputs", "plan", "run"):
        p = sub.add_parser(action)
        p.add_argument("--stage", choices=STAGES, required=True)
        p.add_argument("--experiment", required=True)
        p.add_argument("--project-root", default=str(ROOT))
        if action == "restore-inputs":
            p.add_argument("--archive", help="Explicit archive; otherwise use the ignored local location map")
            p.add_argument("--include-final", action="store_true")
            p.add_argument("--include-tokenizer", action="store_true")
        if action == "run":
            p.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
            p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        build(args)
    elif args.action == "run":
        from project_replay import run_replay
        run_replay(args)
    elif args.action in ("verify", "analyze", "list"):
        for stage in STAGES if args.stage == "all" else [args.stage]:
            if args.action != "list":
                verify(stage)
            if args.action == "analyze":
                analyze(stage)
            elif args.action == "list":
                for r in rows(package(stage) / "runs.csv"):
                    print(stage, r["id"], r["run_id"], r["seed"], r["role"])
    else:
        verify(args.stage)
        recipe = load_recipe(args.stage, args.experiment)
        if args.action == "check-inputs":
            check_inputs(args.stage, recipe, args.project_root)
        elif args.action == "plan":
            print(json.dumps(recipe, indent=2, ensure_ascii=False))
            print("Plan only. No training or filesystem changes performed.")
        else:
            lock = read_json(package(args.stage) / "reproducibility/assets.lock.json")
            archive = resolve_archive(args.archive, ROOT)
            if sha(archive) != lock["archive_sha256"]:
                raise ValueError("Archive hash mismatch")
            paths = recipe["required_artifacts"] + (recipe["evaluation_artifacts"] if args.include_final else []) + (recipe["tokenizer_artifacts"] if args.include_tokenizer else [])
            selected = [lock["artifacts"][p] for p in paths]
            safe.extract_exact(archive, {r["archive_member"]: r for r in selected}, args.project_root)
            check_inputs(args.stage, recipe, args.project_root)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
