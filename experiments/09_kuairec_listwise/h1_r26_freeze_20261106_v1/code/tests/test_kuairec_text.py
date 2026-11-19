"""Boundary checks for deterministic KuaiRec static-text repair (CPU only)."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kuairec_data.text import FIELDS, parse_text, repair_text


class TextRepairTests(unittest.TestCase):
    def tempdir(self):
        root = Path(__file__).resolve().parents[1] / "outputs" / "kuairec_text_tests"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="text-", dir=root)

    def source(self, root, body, newline="\n"):
        path = Path(root) / "input.csv"
        path.write_bytes((",".join(FIELDS) + newline + body).encode("utf-8"))
        return path

    def test_known_split_preserves_all_characters_and_provenance(self):
        with self.tempdir() as root:
            source = self.source(root, '1378,UNKNOWN,first\r\n second,[],6,name,-124,UNKNOWN,-124,UNKNOWN\r\n', "\r\n")
            original = source.read_bytes()
            stats = repair_text(source, Path(root) / "out")
            row = json.loads((Path(root) / "out/item_text.jsonl").read_text("utf-8"))
            ledger = json.loads((Path(root) / "out/text_repairs.jsonl").read_text("utf-8"))
            self.assertEqual(row["caption"], "first\r\n second")
            self.assertEqual((row["source_line_start"], row["source_line_end"]), (2, 3))
            self.assertTrue(row["repaired"])
            self.assertEqual(ledger["raw_record"].encode("utf-8"), original.split(b"\r\n", 1)[1])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(stats["repaired_video_ids"], [1378])

    def test_valid_quoted_multiline_comma_and_quote_are_not_repaired(self):
        with self.tempdir() as root:
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerow([7, "UNKNOWN", 'a, "quote"\nnext', "[]", 6, "name", -124, "UNKNOWN", -124, "UNKNOWN"])
            rows, repairs, _ = parse_text(self.source(root, buffer.getvalue()))
            self.assertEqual(rows[0]["caption"], 'a, "quote"\nnext')
            self.assertFalse(repairs)
            self.assertFalse(rows[0]["repaired"])

    def test_unknown_bad_record_is_not_silently_joined_or_skipped(self):
        with self.tempdir() as root:
            source = self.source(root, '42,UNKNOWN,bad\n tail,[],6,name,-124,UNKNOWN,-124,UNKNOWN\n')
            with self.assertRaisesRegex(ValueError, "Unknown malformed"):
                repair_text(source, Path(root) / "out")
            self.assertFalse((Path(root) / "out").exists())

    def test_wrong_continuation_and_duplicate_id_fail(self):
        for body in ('1378,UNKNOWN,bad\n1,UNKNOWN,next,[],6,name,-124,UNKNOWN,-124,UNKNOWN\n',
                     '1,UNKNOWN,x,[],6,name,-124,UNKNOWN,-124,UNKNOWN\n' * 2):
            with self.subTest(body=body), self.tempdir() as root:
                with self.assertRaises(ValueError):
                    parse_text(self.source(root, body))

    def test_missing_text_is_flagged_without_inventing_content(self):
        with self.tempdir() as root:
            rows, _, _ = parse_text(self.source(root, '0,UNKNOWN,,[],-124,UNKNOWN,-124,UNKNOWN,-124,UNKNOWN\n'))
            self.assertEqual(rows[0]["text"], "")
            self.assertTrue(rows[0]["text_missing"])
            self.assertTrue(rows[0]["caption_missing"])
            self.assertEqual(rows[0]["manual_cover_text"], "UNKNOWN")

    def test_repeat_outputs_are_identical_sorted_and_overwrite_is_refused(self):
        with self.tempdir() as root:
            source = self.source(root, '2,UNKNOWN,z,[],6,name,-124,UNKNOWN,-124,UNKNOWN\n1,UNKNOWN,y,[],6,name,-124,UNKNOWN,-124,UNKNOWN\n')
            first = repair_text(source, Path(root) / "one")
            second = repair_text(source, Path(root) / "two")
            self.assertEqual(first, second)
            ids = [json.loads(line)["video_id"] for line in (Path(root) / "one/item_text.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(ids, [1, 2])
            with self.assertRaises(FileExistsError):
                repair_text(source, Path(root) / "one")


if __name__ == "__main__":
    unittest.main()
