"""Verify downloaded small BGE files and exercise the real pinned tokenizer.

Weights are explicitly not verified/loaded by this small configuration check.
Use model_manifest's full verification after the weight download completes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from .encode_text import untruncated_token_lengths
from .model_manifest import MODEL_ID, verify_snapshot


def audit_tokenizer(snapshot: Path, manifest_path: Path) -> dict:
    from transformers import AutoTokenizer
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID:
        raise ValueError("Wrong model in tokenizer manifest")
    records = [row for row in manifest["files"] if row["path"] != "pytorch_model.bin"]
    checked = verify_snapshot(snapshot, records)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    bert = json.loads((Path(snapshot) / "sentence_bert_config.json").read_text(encoding="utf-8"))
    adapter = SimpleNamespace(tokenizer=tokenizer,
                              _first_module=lambda: SimpleNamespace(do_lower_case=bert.get("do_lower_case", False)))
    texts = ["", "标题：猫咪跳舞\n类别：萌宠 / 猫", "商品推荐 " * 2000]
    lengths = untruncated_token_lengths(adapter, texts)
    truncated = tokenizer(texts, truncation=True, max_length=512, padding=False)["input_ids"]
    actual = list(map(len, truncated))
    if actual != [min(value, 512) for value in lengths] or lengths[2] <= 512:
        raise ValueError("Real tokenizer truncation behavior differs from audit counts")
    return {"schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID, "revision": manifest["revision"], "weights_loaded": False,
            "weights_verified": False, "gpu_used": False, "checked_files": checked,
            "tokenizer_class": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
            "untruncated_lengths": lengths, "truncated_lengths": actual,
            "empty_text_preserved": True, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new tokenizer smoke output path")
    report = audit_tokenizer(args.model_path, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "untruncated": report["untruncated_lengths"],
                      "truncated": report["truncated_lengths"], "weights_loaded": False}))


if __name__ == "__main__":
    main()
