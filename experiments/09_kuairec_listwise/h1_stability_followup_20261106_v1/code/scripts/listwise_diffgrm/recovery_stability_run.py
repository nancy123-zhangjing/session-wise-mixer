"""Shared single-card runner for four continuations; original training primitives."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import KuaiRecListDataset
from .recovery import fingerprint
from .recovery_evaluation import load_subset, write_json as original_write_json
from .recovery_run import catalog_event, evaluation_event as original_evaluation, save
from .recovery_stability import build_followup, gate_metrics
from .recovery_stability_protocol import load_spec
from .run import evaluate, microbatches, phase_indices, run_lock
from .training import seed_all, sha256


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    original_write_json(path, value)


def evaluation_event(controller, assets, data, manifest, output, label, full):
    binding = dict(gate=fingerprint(controller.gate.state()), protocol=controller.protocol)
    path = output / "evaluation_gate_bindings" / f"{label}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Evaluation gate/arm binding changed")
    else:
        if (output / "evaluations" / label / "complete.json").exists():
            raise ValueError("Evaluation receipt is missing its gate binding")
        write_json(path, binding)
    return original_evaluation(controller, assets, data, manifest, output, label, full=full)


def select_best(output, spec):
    records = {str(epoch): json.loads((output / "validation_records" / f"R{epoch}.json").read_text(encoding="utf-8"))
               for epoch in spec["validation_epochs"]}
    epoch = min(spec["validation_epochs"], key=lambda e: (-records[str(e)]["ndcg5"], e))
    selection = dict(spec=spec, criterion="full_validation_ndcg5_tie_earlier",
                     candidates=records, selected=records[str(epoch)])
    write_json(output / "selection.json", selection)
    return selection


def build(args):
    spec = load_spec(args.config)
    controller, assets, data, locked = build_followup(
        spec, args.parents, args.device, args.dataset, args.assets)
    return spec, controller, assets, data, locked


def initial_reference(output, spec, locked):
    source = Path(locked["summary"])
    if sha256(source) != locked["summary_sha256"]:
        raise ValueError("Parent validation summary changed")
    summary = json.loads(source.read_text(encoding="utf-8"))
    checkpoint = output / f"initial_R{spec['start_r']}.pt"
    record = dict(kind="reference", epoch=spec["start_r"], ndcg5=summary["metrics"]["list/ndcg@5"],
                  r_updates=summary["r_updates"], t_updates=summary["t_updates"],
                  full_split=True, summary=str(source), summary_sha256=sha256(source),
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    write_json(output / "validation_records" / f"R{spec['start_r']}.json", record)


def train(args):
    if not args.device.startswith("cuda"):
        raise ValueError("Formal training requires single-card CUDA")
    spec, controller, assets, data, locked = build(args)
    trainer, output = controller.trainer, Path(args.output)
    manifest = load_subset(args.subset, trainer.config["dataset"])
    if (manifest["count"], manifest["population"]) != (8192, 71657):
        raise ValueError("Diagnostic subset population changed")
    validation = KuaiRecListDataset(trainer.config["dataset"], "validation", assets=assets)
    request = dict(spec=spec, protocol=controller.protocol, parent=locked,
                   subset_sha256=sha256(args.subset), execution_binding=controller.execution_binding)
    with run_lock(output, resume=bool(args.resume)):
        if args.resume:
            if json.loads((output / "request.json").read_text(encoding="utf-8")) != request:
                raise ValueError("Resume request changed")
            controller.restore(torch.load(args.resume, map_location="cpu", weights_only=False))
        else:
            write_json(output / "request.json", request)
            save(controller, output, f"initial_R{spec['start_r']}.pt")
            initial_reference(output, spec, locked)
            catalog_event(controller, output, f"initial_R{spec['start_r']}")
        started = time.monotonic()
        while not controller.done:
            phase, row = controller.progress["phase"], dict(controller.row)
            skip = row["kind"] == "T" and spec["mode"] == "freeze"
            indices = phase_indices(trainer.config, len(data), phase, row)
            if controller.progress["update"] == 0 and not controller.progress["boundary_prepared"]:
                seed_all(trainer.config["seed"] + 100003 * (phase + 1))
            while not skip and controller.progress["update"] < row["updates"]:
                begin = controller.progress["update"] * trainer.config["training"]["effective_batch_size"]
                selected = indices[begin:begin + trainer.config["training"]["effective_batch_size"]]
                before = time.perf_counter()
                values = controller.step(microbatches(data, selected, trainer.config["training"]["microbatch_size"]), row["kind"])
                values.update(epoch=row["epoch"], kind=row["kind"], update=controller.progress["update"],
                              seconds=time.perf_counter() - before,
                              sample_indices_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                              mixer_gate=gate_metrics(trainer))
                with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(values, allow_nan=False) + "\n")
                if controller.progress["update"] % (10 if row["kind"] == "T" else 100) == 0:
                    save(controller, output)
                    print(json.dumps(dict(event="progress", arm=spec["arm"], **values)), flush=True)
            save(controller, output)
            controller.prepare_boundary()
            if row["kind"] == "T" and not skip:
                catalog_event(controller, output, f"T{trainer.t_updates}")
                evaluation_event(controller, assets, validation, manifest, output,
                                 f"after_T{trainer.t_updates}", False)
            elif row["kind"] == "R":
                label = f"R{row['epoch']}"
                full = row["epoch"] in spec["validation_epochs"]
                evaluation_event(controller, assets, validation, manifest, output, label, full)
                if full:
                    checkpoint = output / "checkpoints" / f"R{row['epoch']}_T{trainer.t_updates}.pt"
                    save(controller, output, str(checkpoint.relative_to(output)))
                    receipt = json.loads((output / "evaluations" / label / "complete.json").read_text(encoding="utf-8"))
                    path = output / "evaluations" / label / receipt["attempt"] / "summary.json"
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    write_json(output / "validation_records" / f"{label}.json", dict(
                        kind="formal", epoch=row["epoch"], full_split=True,
                        ndcg5=summary["metrics"]["list/ndcg@5"], r_updates=trainer.r_updates,
                        t_updates=trainer.t_updates, summary=str(path), summary_sha256=sha256(path),
                        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint)))
            controller.complete_phase()
            save(controller, output)
            print(json.dumps(dict(event="boundary", arm=spec["arm"], epoch=row["epoch"],
                                  kind=row["kind"], skipped=skip, r_updates=trainer.r_updates,
                                  t_updates=trainer.t_updates)), flush=True)
        selection = select_best(output, spec)
        if (trainer.r_updates, trainer.t_updates) != (35 * 693, spec["stop_t"]):
            raise ValueError("Final budget mismatch")
        write_json(output / "complete.json", dict(status="training_validation_complete", spec=spec,
                   best_epoch=selection["selected"]["epoch"], last_sha256=sha256(output / "last.pt"),
                   r_updates=trainer.r_updates, t_updates=trainer.t_updates,
                   elapsed_seconds=time.monotonic() - started))


def test(args):
    spec, controller, assets, _, _ = build(args)
    output = Path(args.output)
    completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if completion["spec"] != spec or selection["spec"] != spec:
        raise ValueError("Wrong arm selection")
    targets = {row["epoch"]: row for row in (selection["selected"], selection["candidates"]["35"])}
    results = []
    with run_lock(output / "test_lock", resume=True):
        for epoch, target in targets.items():
            checkpoint = Path(target["checkpoint"])
            if sha256(checkpoint) != target["checkpoint_sha256"]:
                raise ValueError("Test checkpoint changed")
            controller.restore(torch.load(checkpoint, map_location="cpu", weights_only=False))
            binding = dict(spec=spec, checkpoint_sha256=sha256(checkpoint),
                           gate=fingerprint(controller.gate.state()), source=controller.execution_binding)
            root = output / "tests" / f"R{epoch}"
            receipt_path = root / "complete.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["binding"] != binding:
                    raise ValueError("Test cache belongs to another state")
                destination = root / receipt["attempt"]
                for name in ("summary.json", "per_sample.jsonl.gz"):
                    if sha256(destination / name) != receipt["files"][name]:
                        raise ValueError("Test artifact hash changed")
            else:
                root.mkdir(parents=True, exist_ok=True)
                attempt = 1
                while (root / f"attempt{attempt}").exists():
                    attempt += 1
                destination = root / f"attempt{attempt}"
                evaluate(controller.trainer, assets, split="test", destination=destination)
                receipt = dict(binding=binding, attempt=destination.name, files={
                    name: sha256(destination / name) for name in ("summary.json", "per_sample.jsonl.gz")})
                write_json(receipt_path, receipt)
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            if summary["examples"] != 53194 or summary["split"] != "test":
                raise ValueError("Test split/count changed")
            results.append(dict(epoch=epoch, ndcg5=summary["metrics"]["list/ndcg@5"],
                                summary=str(destination / "summary.json"), files=receipt["files"]))
    write_json(output / "test_predictions_complete.json", dict(spec=spec, results=results,
               selection_sha256=sha256(output / "selection.json")))


def verify(args):
    """Round-trip saved full states; no optimizer update or model inference."""
    spec, controller, _, _, _ = build(args)
    output = Path(args.output)
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    records = []
    for target in selection["candidates"].values():
        path = Path(target["checkpoint"])
        if sha256(path) != target["checkpoint_sha256"]:
            raise ValueError("Saved checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=False)
        controller.restore(state)
        if fingerprint(controller.checkpoint()) != fingerprint(state):
            raise ValueError("Checkpoint round-trip differs")
        records.append(dict(epoch=target["epoch"], path=str(path), sha256=sha256(path)))
    write_json(output / "restore_verified.json", dict(status="passed", spec=spec,
               updates_performed=0, checkpoints=records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "test", "verify"))
    for name in ("config", "parents", "dataset", "assets", "subset", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    args = parser.parse_args()
    {"train": train, "test": test, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
