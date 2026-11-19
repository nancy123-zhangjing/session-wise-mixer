"""Formal G01 single-card continuation from the authenticated H1 R17 boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import KuaiRecListDataset
from .recovery_r17_release1 import GAMMA, R_BLOCK, T_BLOCK, G01Controller, NAME, build_g01, gate_metrics
from .recovery_run import catalog_event, evaluation_event, save
from .recovery_evaluation import load_subset, write_json
from .run import microbatches, phase_indices, run_lock, evaluate
from .training import atomic_save, seed_all, sha256


BEST_EPOCHS = (20,)
FULL_VALIDATION_EPOCHS = (20,)


def _append(path, value):
    path = Path(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _reference_r17(output, reference):
    summary = json.loads(Path(reference).read_text(encoding="utf-8"))
    record = dict(kind="reference", epoch=17, summary=str(reference),
                  summary_sha256=sha256(reference), ndcg5=summary["metrics"]["list/ndcg@5"],
                   r_updates=summary["r_updates"], t_updates=summary["t_updates"],
                   full_split=True, checkpoint=str(Path(output) / "initial_R17.pt"),
                   checkpoint_sha256=sha256(Path(output) / "initial_R17.pt"))
    write_json(Path(output) / "validation_records" / "R17.json", record)
    write_json(Path(output) / "reference_R17.json", record)
    return record


def _diagnostic_labels(output):
    return sorted(p.name for p in (Path(output) / "evaluations").glob("validation_after_T*")
                  if (p / "complete.json").exists())


def _selection(output):
    records = {str(epoch): json.loads((Path(output) / "validation_records" / f"R{epoch:02d}.json").read_text(encoding="utf-8"))
               for epoch in BEST_EPOCHS}
    selected_epoch = min(BEST_EPOCHS, key=lambda epoch: (-records[str(epoch)]["ndcg5"], epoch))
    selected = dict(records[str(selected_epoch)], epoch=selected_epoch)
    result = dict(criterion="fixed_R20_no_model_selection",
                  candidates=records, selected=selected,
                  full_validation_epochs=list(FULL_VALIDATION_EPOCHS),
                  reference_R17_reused=True)
    write_json(Path(output) / "selection.json", result)
    return result


def train(args):
    if not args.device.startswith("cuda"):
        raise ValueError("G01 formal continuation requires the authorized H100")
    controller, assets, data = build_g01(args.parent, args.device, args.dataset, args.assets)
    trainer = controller.trainer
    output = Path(args.output)
    manifest = load_subset(args.subset, trainer.config["dataset"])
    if manifest["count"] != 8192 or manifest["population"] != 71657:
        raise ValueError("Locked diagnostic subset changed")
    validation = KuaiRecListDataset(trainer.config["dataset"], "validation", assets=assets)
    request = dict(name=NAME, parent=args.parent, parent_sha256=sha256(args.parent),
                   protocol=controller.protocol, gamma=GAMMA, t_block=T_BLOCK,
                   r_block=R_BLOCK, validation_epochs=list(FULL_VALIDATION_EPOCHS),
                   subset_sha256=sha256(args.subset))
    with run_lock(output, resume=bool(args.resume)):
        if args.resume:
            controller.restore(torch.load(args.resume, map_location="cpu", weights_only=False))
            if json.loads((output / "request.json").read_text(encoding="utf-8")) != request:
                raise ValueError("Resume request changed")
        else:
            (output / "validation_records").mkdir(parents=True, exist_ok=True)
            write_json(output / "request.json", request)
            save(controller, output, "initial_R17.pt")
            _reference_r17(output, args.r17_reference)
            catalog_event(controller, output, "initial_R17")
        started = time.monotonic()
        while not controller.done:
            phase = controller.progress["phase"]
            row = dict(controller.row)
            indices = phase_indices(trainer.config, len(data), phase, row)
            if controller.progress["update"] == 0 and not controller.progress.get("boundary_prepared"):
                seed_all(trainer.config["seed"] + 100003 * (phase + 1))
            while controller.progress["update"] < row["updates"]:
                begin = controller.progress["update"] * trainer.config["training"]["effective_batch_size"]
                selected = indices[begin:begin + trainer.config["training"]["effective_batch_size"]]
                before = time.perf_counter()
                values = controller.step(
                    microbatches(data, selected, trainer.config["training"]["microbatch_size"]), row["kind"])
                values.update(epoch=row["epoch"], kind=row["kind"], update=controller.progress["update"],
                              seconds=time.perf_counter() - before,
                              cumulative_seconds=time.monotonic() - started,
                              sample_indices_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest())
                values["mixer_gate"] = gate_metrics(trainer)
                _append(output / "training.jsonl", values)
                if controller.progress["update"] % (10 if row["kind"] == "T" else 100) == 0:
                    save(controller, output)
                    print(json.dumps(dict(event="g01_progress", **values), ensure_ascii=False), flush=True)
            save(controller, output)
            controller.prepare_boundary()
            if row["kind"] == "T":
                catalog_event(controller, output, f"T{trainer.t_updates:03d}")
                evaluation_event(controller, assets, validation, manifest, output,
                                 f"after_T{trainer.t_updates:03d}", full=False)
            else:
                label = f"R{row['epoch']:02d}"
                evaluation_event(controller, assets, validation, manifest, output, label,
                                 full=row["epoch"] in FULL_VALIDATION_EPOCHS)
            boundary = output / f"phase{phase:02d}_{row['kind']}_R{row['epoch']:02d}_T{trainer.t_updates:03d}.pt"
            save(controller, output, boundary.name)
            if row["kind"] == "R" and row["epoch"] in FULL_VALIDATION_EPOCHS:
                receipt = json.loads((output / "evaluations" / label / "complete.json").read_text(encoding="utf-8"))
                summary_path = output / "evaluations" / label / receipt["attempt"] / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                write_json(output / "validation_records" / f"{label}.json", dict(
                    kind="formal", epoch=row["epoch"], summary=str(summary_path),
                    summary_sha256=sha256(summary_path), ndcg5=summary["metrics"]["list/ndcg@5"],
                    r_updates=summary["r_updates"], t_updates=summary["t_updates"], full_split=True,
                    checkpoint=str(boundary), checkpoint_sha256=sha256(boundary)))
            controller.complete_phase()
            save(controller, output)
            print(json.dumps(dict(event="g01_boundary", epoch=row["epoch"], kind=row["kind"],
                                  r_updates=trainer.r_updates, t_updates=trainer.t_updates), ensure_ascii=False), flush=True)
        selected = _selection(output)
        write_json(output / "complete.json", dict(status="training_and_validation_complete", name=NAME,
            r_updates=trainer.r_updates, t_updates=trainer.t_updates, phase=controller.progress["phase"],
            selected_epoch=selected["selected"]["epoch"], last_sha256=sha256(output / "last.pt"),
            elapsed_seconds=time.monotonic() - started))


def test(args):
    output = Path(args.output)
    json.loads((output / "complete.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    targets = [json.loads((output / "validation_records/R20.json").read_text(encoding="utf-8"))]
    unique = []
    for target in targets:
        if target["epoch"] not in [row["epoch"] for row in unique]:
            unique.append(target)
    controller, assets, _ = build_g01(args.parent, args.device, args.dataset, args.assets)
    results = []
    for target in unique:
        checkpoint = Path(target["checkpoint"])
        if sha256(checkpoint) != target["checkpoint_sha256"]:
            raise ValueError("Test checkpoint changed")
        controller.restore(torch.load(checkpoint, map_location="cpu", weights_only=False))
        destination = output / "tests" / f"R{target['epoch']:02d}"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            evaluate(controller.trainer, assets, split="test", destination=destination)
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        results.append(dict(epoch=target["epoch"], summary=str(destination / "summary.json"),
                            predictions=str(destination / "per_sample.jsonl.gz"),
                            summary_sha256=sha256(destination / "summary.json"),
                            predictions_sha256=sha256(destination / "per_sample.jsonl.gz"),
                            ndcg5=summary["metrics"]["list/ndcg@5"]))
    write_json(output / "test_predictions_complete.json", dict(results=results,
        selection_sha256=sha256(output / "selection.json"), test_protocol="full_test_beam32_fp32"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "test"))
    for key in ("parent", "dataset", "assets", "subset", "r17-reference", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    return train(args) if args.command == "train" else test(args)


if __name__ == "__main__":
    main()
