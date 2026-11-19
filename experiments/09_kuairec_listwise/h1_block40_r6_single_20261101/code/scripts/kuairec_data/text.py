"""Lossless, audited parsing of the released KuaiRec static item text CSV.

The release contains four unquoted caption newlines. We repair only the
observed 3-column + 8-column pattern for those four IDs and fail closed on any
other malformed record. Raw files are never rewritten. No behavior is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "video_id", "manual_cover_text", "caption", "topic_tag",
    "first_level_category_id", "first_level_category_name",
    "second_level_category_id", "second_level_category_name",
    "third_level_category_id", "third_level_category_name",
]
KNOWN_BROKEN_IDS = frozenset({1378, 1613, 5910, 6331})
MISSING = frozenset({"", "unknown", "nan", "[]", "null", "none", "-124"})
REPAIR_RULE = "join_known_unquoted_caption_3_plus_8_fields_preserving_newline_v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _missing(value: str) -> bool:
    return value.strip().casefold() in MISSING


def _validate(fields: list[str], line: int) -> None:
    if len(fields) != len(FIELDS) or not fields[0].isdigit():
        raise ValueError(f"Invalid item record at physical line {line}: expected 10 fields and numeric ID")
    for position in (4, 6, 8):
        try:
            int(fields[position])
        except ValueError as exc:
            raise ValueError(f"Invalid category ID at physical line {line}") from exc
    if not (fields[3].startswith("[") and fields[3].endswith("]")):
        raise ValueError(f"Invalid topic list at physical line {line}")


def _text(record: dict[str, Any]) -> str:
    parts = []
    for field, label in (("caption", "标题"), ("manual_cover_text", "封面文字"),
                         ("topic_tag", "主题")):
        if not _missing(record[field]):
            parts.append(f"{label}：{record[field].strip()}")
    categories = [record[f"{level}_level_category_name"].strip()
                  for level in ("first", "second", "third")
                  if not _missing(record[f"{level}_level_category_name"])]
    if categories:
        parts.append("类别：" + " / ".join(categories))
    return "\n".join(parts)


def parse_text(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return item rows and lossless repair ledger, without creating files.

    Line numbers are 1-based physical CSV lines, including the header. Legal
    CSV quoted multiline fields are handled by csv.reader, not repaired.
    """
    with Path(source).open("r", encoding="utf-8-sig", newline="") as stream:
        lines = stream.readlines()
    reader = csv.reader(lines, strict=True)
    if next(reader, None) != FIELDS:
        raise ValueError("Unexpected KuaiRec text header")
    rows, repairs, seen = [], [], set()
    while True:
        start = reader.line_num + 1
        fields = next(reader, None)
        if fields is None:
            break
        repaired = False
        if len(fields) != len(FIELDS):
            if (len(fields) != 3 or not fields[0].isdigit()
                    or int(fields[0]) not in KNOWN_BROKEN_IDS):
                raise ValueError(f"Unknown malformed text record at physical line {start}")
            prefix_end = reader.line_num
            if prefix_end != start:
                raise ValueError(f"Unexpected multiline malformed prefix at physical line {start}")
            # A genuine row boundary must still be present in the raw source.
            last_line = lines[prefix_end - 1]
            separator = "\r\n" if last_line.endswith("\r\n") else (
                "\n" if last_line.endswith("\n") else "\r" if last_line.endswith("\r") else "")
            continuation = next(reader, None)
            if (not separator or continuation is None or len(continuation) != 8
                    or reader.line_num != prefix_end + 1 or continuation[0].isdigit()):
                raise ValueError(f"Unexpected caption continuation at physical line {start}")
            fields = fields[:2] + [fields[2] + separator + continuation[0]] + continuation[1:]
            repaired = True
        end = reader.line_num
        _validate(fields, start)
        item_id = int(fields[0])
        if item_id in seen:
            raise ValueError(f"Duplicate video_id {item_id} in static item table")
        seen.add(item_id)
        original = dict(zip(FIELDS, fields))
        record = {**original, "video_id": item_id, "source_line_start": start,
                  "source_line_end": end, "repaired": repaired}
        record["text"] = _text(record)
        record["text_missing"] = not bool(record["text"])
        record["caption_missing"] = _missing(record["caption"])
        rows.append(record)
        if repaired:
            raw_record = "".join(lines[start - 1:end])
            repairs.append({"video_id": item_id, "source_line_start": start,
                            "source_line_end": end, "raw_record": raw_record,
                            "raw_record_sha256": hashlib.sha256(raw_record.encode("utf-8")).hexdigest(),
                            "rule": REPAIR_RULE, "parsed_fields": original})
    return sorted(rows, key=lambda row: row["video_id"]), sorted(repairs, key=lambda row: row["video_id"]), len(lines)


def repair_text(source: Path, output_dir: Path) -> dict[str, Any]:
    """Write deterministic item_text.jsonl and text_repairs.jsonl in a new path.

    Existing output files cause an error. The caller owns item-ID remapping.
    """
    source, output_dir = Path(source), Path(output_dir)
    targets = [output_dir / name for name in ("item_text.jsonl", "text_repairs.jsonl")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite existing text outputs")
    rows, repairs, physical_lines = parse_text(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, values in zip(targets, (rows, repairs)):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(_json(value) + "\n")
    return {
        "schema_version": 1, "items": len(rows), "physical_source_lines": physical_lines,
        "repaired_items": len(repairs), "repaired_video_ids": [row["video_id"] for row in repairs],
        "caption_missing_items": sum(row["caption_missing"] for row in rows),
        "text_missing_items": sum(row["text_missing"] for row in rows),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "outputs": {path.name: {"bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in targets},
        "text_policy": "Static caption, cover text, topic and category names only; raw fields retained. No behavior statistics.",
        "catalog_assumption": "Released full static catalog; publication-time metadata is not proven to be an event-time snapshot.",
        "repair_rule": REPAIR_RULE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(repair_text(arguments.source, arguments.output), ensure_ascii=True, sort_keys=True))
