"""Batched, target-free generation with asynchronous batched prediction writing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from .catalog import generate
from .common import preserve_rng, attention_backend, write_json, sha256
from .screen import generation_dim


def packed_l5o_metrics(data, predictions):
    """Score the existing generated lists with the shared L5O evaluator."""
    from scripts.kuairec_data.l5o_metrics import load_predictions, slate_metrics

    pred = load_predictions(predictions, data.sample_ids)
    target = data.target.cpu().numpy()
    return {key: slate_metrics(pred, target, data.watch_ratios, threshold)
            for key, threshold in (('metrics', 1.0), ('sensitivity_ev2', 2.0))}


def item_metrics(prediction, target, catalog_size):
    """Same item-level semantics as listwise_diffgrm.metrics, without hot-loop assertions."""
    width = prediction.shape[1]
    pos = torch.arange(width, device=prediction.device)
    duplicate = ((prediction[:, :, None] == prediction[:, None, :]) &
                 (pos[None, :] < pos[:, None])[None]).any(-1)
    valid = (prediction > 0) & (prediction <= catalog_size) & ~duplicate
    relevant = (prediction[:, :, None] == target[:, None, :]).any(-1) & valid
    discount = 1 / torch.log2(pos.float()+2)
    hit = relevant.sum(-1)
    return {"ndcg@5": (relevant*discount).sum(-1)/discount[:min(width, target.shape[1])].sum(),
            "recall@5": hit.float()/target.shape[1], "hr@5": (hit > 0).float(),
            "completion_rate": valid.all(-1).float()}


def _write_batch(stream, data, start, prediction, truth, metrics, diagnostics):
    rows = []
    for i in range(len(prediction)):
        rows.append(json.dumps({"sample_id": data.sample_ids[start+i], "user_id": data.users[start+i],
                               "first_target_timestamp_us": int(data.first_target_time[start+i]),
                               "prediction": prediction[i], "target": truth[i],
                               "metrics": {k: v[i] for k, v in metrics.items()},
                               "diagnostics": {k: v[i] for k, v in diagnostics.items()}},
                              ensure_ascii=False, allow_nan=False))
    stream.write("\n".join(rows)+"\n")


@torch.inference_mode()
def evaluate(model, diffusion, catalog, data, cfg, runtime, noise_path, output, identity):
    output, noise_path = Path(output), Path(noise_path)
    noise_meta = json.loads(noise_path.with_suffix(".json").read_text(encoding="utf-8"))
    sample_hash = hashlib.sha256(json.dumps(data.sample_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if (noise_meta["split"] != data.split or noise_meta["sha256"] != sha256(noise_path) or
            noise_meta["sample_ids_sha256"] != sample_hash or
            noise_meta["seed"] != cfg["evaluation"]["seed"] or
            noise_meta["noise_version"] != cfg["evaluation"]["noise_version"]):
        raise ValueError("Evaluation noise identity/order/hash mismatch")
    if (output/"summary.json").exists():
        result = json.loads((output/"summary.json").read_text(encoding="utf-8"))
        if (result["identity"] != identity or result["prediction_sha256"] != sha256(output/"predictions.jsonl.gz") or
                result["noise_sha256"] != noise_meta["sha256"] or result["catalog_hash"] != catalog.version_hash):
            raise ValueError("Existing evaluation has different checkpoint/protocol or corrupt output")
        return result
    output.mkdir(parents=True, exist_ok=True)
    array = np.load(noise_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(data), cfg["list_length"], generation_dim(cfg)):
        raise ValueError("Noise table geometry mismatch")
    start_time, original_mode = time.perf_counter(), model.training
    model.eval()
    total, diag_total = {}, {}
    tmp, final = output/"predictions.jsonl.gz.partial", output/"predictions.jsonl.gz"
    pending = []
    try:
        with preserve_rng(), attention_backend(runtime["attention_backend"]), \
                gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as stream, \
                ThreadPoolExecutor(max_workers=1) as writer:
            for start in range(0, len(data), runtime["evaluation_batch"]):
                stop = min(start+runtime["evaluation_batch"], len(data))
                history, truth = data.batch(torch.arange(start, stop), catalog.device)
                noise = torch.from_numpy(np.array(array[start:stop], copy=True)).to(catalog.device, non_blocking=True)
                prediction, diagnostics = generate(model, diffusion, catalog, history, noise, runtime,
                    filter_history=cfg['evaluation'].get('filter_history', False))
                metrics = item_metrics(prediction, truth, cfg["catalog_size"])
                # One correctness decision per generated batch, not per DDIM
                # step / beam / item. Failed outputs are never accepted.
                if not diagnostics["finite_vector"].all() or not metrics["completion_rate"].bool().all():
                    raise RuntimeError("Nonfinite generated vector or incomplete/invalid five-item output")
                for key, value in metrics.items():
                    total[key] = total.get(key, 0)+value.double().sum()
                for key, value in diagnostics.items():
                    diag_total[key] = diag_total.get(key, 0)+value.double().sum(0)
                m_cpu = {k: v.cpu().tolist() for k, v in metrics.items()}
                d_cpu = {k: v.cpu().tolist() for k, v in diagnostics.items()}
                pending.append(writer.submit(_write_batch, stream, data, start, prediction.cpu().tolist(),
                                             truth.cpu().tolist(), m_cpu, d_cpu))
                if len(pending) >= runtime["evaluation_write_batches"]:
                    pending.pop(0).result()
                if (start//runtime["evaluation_batch"]+1) % 20 == 0 or stop == len(data):
                    write_json(output/"progress.json", {"identity": identity, "completed": stop,
                                                        "total": len(data), "seconds": time.perf_counter()-start_time})
            for future in pending:
                future.result()
        if catalog.device.type == "cuda":
            torch.cuda.synchronize(catalog.device)
        tmp.replace(final)
        result = {"identity": identity, "count": len(data),
                  "metrics": {k: float(v/len(data)) for k, v in total.items()},
                  "diagnostics_by_position": {k: (v/len(data)).cpu().tolist() for k, v in diag_total.items()},
                  "seconds": time.perf_counter()-start_time, "candidate_lists": 1,
                  "ddim_steps": cfg["diffusion"]["sample_steps"], "eta": 0,
                  "prediction_sha256": sha256(final), "noise_sha256": noise_meta["sha256"],
                  "catalog_version": catalog.version, "catalog_hash": catalog.version_hash}
        if hasattr(data, 'watch_ratios'):
            result['legacy_metrics'] = result['metrics']
            result.update(packed_l5o_metrics(data, final))
        write_json(output/"summary.json", result)
        return result
    finally:
        model.train(original_mode)
