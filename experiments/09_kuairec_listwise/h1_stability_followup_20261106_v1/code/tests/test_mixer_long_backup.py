"""Exercise complete byte-backup transport; fixtures are NOT experiment evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('long_backup', ROOT / 'scripts/analysis/pack_mixer_long_delivery.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def encoded(value):
    return json.dumps(value).encode()


class BackupTransportTests(unittest.TestCase):
    def fixture(self, directory, *, omitted=None, hostile=None):
        """Small arbitrary byte streams; no fabricated training receipts or models."""
        files = {name: b'BYTE TRANSPORT FIXTURE ONLY' for name in backup.mandatory_names()}
        inputs = []
        for i in range(14):
            name = 'input_%02d.bin' % i
            value = (name + ': fixture').encode()
            files['inputs/' + name] = value
            inputs.append(dict(path=name, size_bytes=len(value), sha256=hashlib.sha256(value).hexdigest()))
        files['delivery/INPUTS.json'] = encoded(dict(campaign_id=backup.CAMPAIGN, records=inputs))
        if omitted:
            files.pop(omitted)
        if hostile:
            files[hostile] = b'hostile path fixture'
        records = [dict(member=name, size_bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                   for name, value in sorted(files.items())]
        manifest = encoded(dict(format=backup.FORMAT, campaign_id=backup.CAMPAIGN,
                                files=records, all_14_locked_inputs_included=True))
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name, value in [*files.items(), ('MANIFEST.json', manifest)]:
                info = tarfile.TarInfo(name); info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
        raw = stream.getvalue()
        directory.mkdir()
        parts = []
        for i, start in enumerate(range(0, len(raw), 67)):
            value = raw[start:start + 67]
            name = 'fixture.part%04d' % i
            (directory / name).write_bytes(value)
            parts.append(dict(member=name, size_bytes=len(value), sha256=hashlib.sha256(value).hexdigest()))
        (directory / 'PACKAGE.json').write_bytes(encoded(dict(format=backup.FORMAT, campaign_id=backup.CAMPAIGN,
            parts=parts, archive_sha256=hashlib.sha256(raw).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest).hexdigest(), files=len(records))))
        return files

    def test_multipart_roundtrip_hashes_every_member_without_claiming_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); files = self.fixture(root / 'package')
            result = backup.verify(root / 'package', root / 'extracted')
            self.assertTrue(result['all_member_bytes_verified'])
            self.assertFalse(result['model_restore_verified'])
            self.assertFalse(result['goal_complete'])
            for name, value in files.items():
                self.assertEqual((root / 'extracted' / name).read_bytes(), value)

    def test_corrupted_or_reordered_part_is_rejected_before_extracting_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root / 'package')
            path = root / 'package/fixture.part0001'
            path.write_bytes(b'x' * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'part damaged'):
                backup.verify(root / 'package', root / 'extracted')
            self.assertFalse((root / 'extracted').exists())

    def test_even_rehashed_archive_cannot_omit_required_model_or_input(self):
        for missing in ('formal/mixer_joint/last.pt', 'inputs/input_07.bin'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); self.fixture(root / 'package', omitted=missing)
                with self.assertRaisesRegex(ValueError, 'omitted'):
                    backup.verify(root / 'package', root / 'extracted')

    def test_traversal_and_windows_device_names_are_rejected(self):
        for name in ('../outside', '/outside', 'C:/outside', 'a/CON.txt', 'a/part.','a\\outside'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); self.fixture(root / 'package', hostile=name)
                with self.assertRaises(ValueError):
                    backup.verify(root / 'package', root / 'extracted')
                self.assertFalse((root / 'outside').exists())

    def test_partial_audit_cannot_authorize_a_complete_backup_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            goal = Path(temp) / backup.CAMPAIGN; goal.mkdir()
            (goal / 'audit.json').write_bytes(encoded(dict(status='monitoring_not_completion')))
            (goal / 'release.json').write_bytes(encoded({}))
            with self.assertRaisesRegex(ValueError, 'Complete independent artifact audit'):
                backup.collect(goal, goal / 'audit.json', goal / 'release.json')


if __name__ == '__main__':
    unittest.main()
