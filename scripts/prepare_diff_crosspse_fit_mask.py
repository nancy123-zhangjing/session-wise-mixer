"""Export the exact item mask used to fit the DiffGRM PSE tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from accelerate import Accelerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
sys.path.insert(0, str(DIFF_ROOT))

from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer  # noqa: E402
from genrec.utils import get_config, get_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    os.chdir(DIFF_ROOT)
    config = get_config(
        model_name="DIFF_GRM",
        dataset_name="AmazonReviews2014",
        config_file=args.config,
        config_dict=None,
    )
    accelerator = Accelerator()
    config["accelerator"] = accelerator
    config["device"] = str(accelerator.device)
    config["use_ddp"] = False
    dataset = get_dataset("AmazonReviews2014")(config)
    expected_items = [dataset.id_mapping["id2item"][i] for i in range(1, dataset.n_items)]
    stored_items = json.loads(args.items.resolve().read_text(encoding="utf-8"))
    if stored_items != expected_items:
        raise ValueError("Shared item list does not match the current DiffGRM catalog")

    helper = DIFF_GRMTokenizer.__new__(DIFF_GRMTokenizer)
    helper.config = config
    helper.log = lambda message, level="info": print(message, flush=True)
    mask = helper._get_items_for_training(dataset).astype(np.bool_, copy=False)
    if mask.shape != (len(expected_items),) or not mask.any() or mask.all():
        raise ValueError(f"Unexpected PSE fit mask: shape={mask.shape}, fit={mask.sum()}")

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.resolve(), mask)
    summary = {
        "items": len(mask),
        "fit_items": int(mask.sum()),
        "validation_items": int((~mask).sum()),
        "split": config.get("split"),
        "train_sliding": bool(config.get("train_sliding", False)),
        "min_hist_len": int(config.get("min_hist_len", 2)),
        "max_hist_len": int(config.get("max_hist_len", 50)),
    }
    args.summary_output.resolve().write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
