"""Stdlib-only checks for complete, stable, safely named backup payloads."""
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.analysis import package_h1_recovery as backup


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else backup.json_bytes(value)
    path.write_bytes(data)
    return data


class RecoveryBackupTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='h1_backup_test_', dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'campaign'
        self.parent = Path(self.temporary.name) / 'R20.pt'
        parent_data = write(self.parent, b'authenticated parent state')
        sources = {'model.py': b'original model bytes\n', 'external/DiffGRM/engine.py': b'engine bytes\n'}
        for name, data in sources.items():
            write(self.root / 'code' / name, data)
        self.binding = dict(config=dict(dataset='/external/dataset', assets='/external/content_pse'),
            data=dict(files={'train.parquet': dict(bytes=99, sha256='d' * 64)}, manifest_sha256='e' * 64),
            asset_manifest='f' * 64, code={'model.py': sha(sources['model.py'])},
            engine={'engine.py': sha(sources['external/DiffGRM/engine.py'])})
        binding_data = write(self.root / 'code/RECOVERY_PARENT_BINDING.json', self.binding)
        execution = dict(workspace=dict(package_sources_uncommitted=False),
            parent=dict(bytes=len(parent_data), sha256=sha(parent_data)),
            parent_binding_json_sha256=sha(binding_data),
            execution_file_count=len(sources),
            execution_files={name: dict(bytes=len(data), sha256=sha(data)) for name, data in sources.items()})
        execution_data = write(self.root / 'code/RECOVERY_EXECUTION_MANIFEST.json', execution)
        self.addCleanup(patch.stopall)
        patch.object(backup, 'PARENT_SHA256', sha(parent_data)).start()
        patch.object(backup, 'EXECUTION_MANIFEST_SHA256', sha(execution_data)).start()
        self.complete = {}
        for policy, count in [('decay', 280), ('freeze', 200)]:
            directory = self.root / f'runs/H1-{policy}'
            payload = write(directory / 'last.pt', b'final checkpoint ' + policy.encode())
            write(directory / 'phase26.pt', b'intermediate checkpoint')
            write(directory / 'predictions.jsonl.gz', b'prediction bytes')
            self.complete[policy] = dict(policy=policy, r_updates=18018, t_updates=count,
                phase=33, endpoint_epoch=26, endpoint_checkpoint_sha256=sha(payload))
            write(directory / 'complete.json', self.complete[policy])
        write(self.root / 'formal_jobs/H1-decay/training.log', b'training log\n')
        write(self.root / 'formal_jobs/H1-freeze/training.log', b'training log\n')
        write(self.root / 'delivery/validation_subset_8192.json', {'fixture': True})
        write(self.root / 'delivery/formal_job.py', b'launcher bytes')
        write(self.root / 'delivery/h1_recovery_code.tar.gz', b'omit duplicate code archive')
        write(self.root / 'preflight_r1/decay/report.json', {'passed': True})
        write(self.root / 'preflight_r1/decay/preflight_only.pt', b'omit non-formal checkpoint')
        write(self.root / 'backup/previous/ignore.bin', b'never recurse into old backups')

    def package(self, name='new'):
        return backup.package(self.root, self.parent, name=name, chunk_bytes=137)

    def test_archive_manifest_all_bytes_chunks_and_dependencies(self):
        before = backup.scan_sources(self.root, self.parent)[2]
        result = self.package()
        output = Path(result['output'])
        payload = (output / result['archive']).read_bytes()
        self.assertEqual(result['sha256'], sha(payload))
        self.assertEqual(result['bytes'], len(payload))
        position = 0
        for index, block in enumerate(result['chunks']):
            self.assertEqual((block['index'], block['offset']), (index, position))
            self.assertEqual(block['sha256'], sha(payload[position:position + block['bytes']]))
            position += block['bytes']
        self.assertEqual(position, len(payload))
        with tarfile.open(output / result['archive']) as archive:
            names = archive.getnames()
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(member.isfile() for member in archive.getmembers()))
            manifest_bytes = archive.extractfile('MANIFEST.json').read()
            self.assertEqual(manifest_bytes, (output / 'MANIFEST.json').read_bytes())
            self.assertEqual(result['manifest_sha256'], sha(manifest_bytes))
            manifest = json.loads(manifest_bytes)
            self.assertEqual(set(names), set(manifest['files']) | {'MANIFEST.json'})
            for name, entry in manifest['files'].items():
                data = archive.extractfile(name).read()
                self.assertEqual((entry['bytes'], entry['sha256']), (len(data), sha(data)))
            self.assertEqual(archive.extractfile('references/R20_best.pt').read(), self.parent.read_bytes())
            dependencies = json.load(archive.extractfile('references/DEPENDENCIES.json'))
            self.assertEqual(dependencies['dataset']['binding'], self.binding['data'])
            self.assertEqual(dependencies['content_pse']['manifest_sha256'], self.binding['asset_manifest'])
            self.assertFalse(manifest['self_contained_dataset'])
            self.assertFalse(any('preflight_only.pt' in name or 'h1_recovery_code.tar' in name or 'backup/' in name for name in names))
            self.assertIn('preflight_r1/decay/report.json', names)
        self.assertEqual(before, backup.scan_sources(self.root, self.parent)[2])
        self.assertFalse(result['local_delivery_verified'])
        self.assertFalse(result['checkpoint_restore_verified'])
        with self.assertRaises(FileExistsError):
            self.package()

    def test_completion_counters_checkpoint_and_parent_must_match(self):
        for key, bad in [('r_updates', 18017), ('t_updates', 279), ('phase', 32),
                         ('endpoint_epoch', 27), ('policy', 'freeze')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.complete['decay'])
                changed[key] = bad
                write(self.root / 'runs/H1-decay/complete.json', changed)
                with self.assertRaisesRegex(ValueError, 'expected R26'):
                    self.package()
        write(self.root / 'runs/H1-decay/complete.json', self.complete['decay'])
        path = self.root / 'runs/H1-freeze/last.pt'
        original = path.read_bytes()
        path.write_bytes(b'wrong final checkpoint')
        with self.assertRaisesRegex(ValueError, 'completion/checkpoint SHA'):
            self.package()
        path.write_bytes(original)
        self.parent.write_bytes(b'wrong parent')
        with self.assertRaisesRegex(ValueError, 'R20 parent'):
            self.package()

    def test_code_mismatch_does_not_get_success_receipt(self):
        (self.root / 'code/model.py').write_bytes(b'changed source bytes')
        with self.assertRaisesRegex(ValueError, 'Execution source differs'):
            self.package()
        self.assertTrue((self.root / 'backup/new/archive.tar.gz.partial').is_file())
        self.assertFalse((self.root / 'backup/new/ARCHIVE_RECEIPT.json').exists())

    def test_unbound_code_file_is_rejected(self):
        write(self.root / 'code/extra.py', b'unbound source')
        with self.assertRaisesRegex(ValueError, 'code inventory differs'):
            self.package()
        self.assertFalse((self.root / 'backup/new/ARCHIVE_RECEIPT.json').exists())

    def test_change_during_read_is_rejected(self):
        target = self.root / 'runs/H1-decay/predictions.jsonl.gz'
        original_read = backup.StableReader.read
        changed = False

        def mutate(reader, size=-1):
            nonlocal changed
            result = original_read(reader, size)
            if reader.path == target and not changed:
                changed = True
                with target.open('ab') as stream:
                    stream.write(b'late writer')
            return result

        with patch.object(backup.StableReader, 'read', mutate):
            with self.assertRaisesRegex(ValueError, 'changed during read'):
                self.package()
        self.assertFalse((self.root / 'backup/new/ARCHIVE_RECEIPT.json').exists())

    def test_new_input_during_packaging_is_rejected(self):
        original_add = backup.add_bytes

        def add(archive, name, data):
            result = original_add(archive, name, data)
            if name == 'references/DEPENDENCIES.json':
                write(self.root / 'runs/H1-decay/late.log', b'late log')
            return result

        with patch.object(backup, 'add_bytes', add):
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                self.package()
        self.assertFalse((self.root / 'backup/new/ARCHIVE_RECEIPT.json').exists())

    def test_unsafe_paths_and_linked_input_rejected(self):
        for name in ('', '.', '..', '../escape', '/absolute', 'a/../b', 'a\\b', 'C:/a', 'a//b', 'a\x00b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                backup.safe_member(name)
        with self.assertRaisesRegex(ValueError, 'safe directory component'):
            self.package('../escape')
        target = self.root / 'delivery/linked.json'
        try:
            target.symlink_to(self.parent)
        except OSError:
            self.skipTest('This Windows account cannot create symlinks')
        with self.assertRaisesRegex(ValueError, 'Linked path'):
            self.package()


if __name__ == '__main__':
    unittest.main()
