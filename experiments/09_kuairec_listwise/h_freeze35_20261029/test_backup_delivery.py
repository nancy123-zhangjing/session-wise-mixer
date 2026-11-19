"""Synthetic safety/coverage/relocation checks, never real-model acceptance."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('backup_delivery.py')
spec = importlib.util.spec_from_file_location('freeze35_backup_under_test', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ROOT = Path(__file__).resolve().parents[3]


def fp(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class BackupTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'tmp/freeze35_backup'; base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def archive(self, *, files=None, actual=None, extra=None, manifest_override=None):
        files = files or {'goal/a.txt': b'synthetic data'}
        records = {n: dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in files.items()}
        manifest = manifest_override or {'files': records}
        raw = m.encoded(manifest); archive = self.root / 'synthetic.tar'
        with m.tarfile.open(archive, 'w', format=m.tarfile.PAX_FORMAT) as tar:
            info = m.tarfile.TarInfo(m.MANIFEST); info.size = len(raw); tar.addfile(info, io.BytesIO(raw))
            for name, data in (files if actual is None else actual).items():
                info = m.tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
            if extra: extra(tar)
        return archive, m.sha(archive), hashlib.sha256(raw).hexdigest(), manifest

    def test_all_63_immutable_slots_and_three_branch_clocks_are_explicit(self):
        self.assertEqual({b: len(m.node_specs(b)) for b in 'ABC'}, {'A': 14, 'B': 6, 'C': 43})
        self.assertEqual(sum(len(m.node_specs(b)) for b in 'ABC'), 63)
        for branch in 'ABC':
            specs = m.node_specs(branch)
            self.assertIn('R35_training_boundary.pt', specs)
            self.assertEqual(specs['R35_training_boundary.pt']['R'], 24255)
            self.assertEqual(specs['R35_training_boundary.pt']['T'], 360 if branch == 'B' else 200)
        self.assertEqual(m.node_specs('C')['R20_pre_freeze.pt']['phase'], 25)
        self.assertEqual(m.node_specs('C')['phase25_T_R20_T200.pt']['T'], 200)

    def test_unsafe_paths_reserved_names_and_case_collisions_are_rejected(self):
        for name in ('../x', '/x', 'a\\x', 'C:x', './a', 'a/../b', 'a.', 'a ', 'CON.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError): m.member_name(name)
        for names in [('A.txt', 'a.txt'), ('a', 'a/b'), ('a/b', 'a')]:
            known = {}; m.register(names[0], known)
            with self.assertRaises(ValueError): m.register(names[1], known)

    def test_synthetic_byte_verification_does_not_claim_formal_backup_pass(self):
        archive, a, b, manifest = self.archive(); destination = self.root / 'fresh'
        self.assertEqual(m.extract_bytes(archive, destination, a, b), manifest)
        self.assertEqual((destination / 'goal/a.txt').read_bytes(), b'synthetic data')
        self.assertFalse((destination / 'BACKUP_VERIFICATION.json').exists())
        with self.assertRaises(ValueError): m.extract_bytes(archive, destination, a, b)

    def test_archive_or_manifest_sha_failure_creates_no_destination(self):
        archive, a, b, _ = self.archive(); destination = self.root / 'fresh'
        with self.assertRaises(ValueError): m.extract_bytes(archive, destination, '0' * 64, b)
        self.assertFalse(destination.exists())
        with self.assertRaises(ValueError): m.extract_bytes(archive, destination, a, '0' * 64)
        self.assertFalse(destination.exists())

    def test_changed_payload_is_rejected_even_if_archive_outer_sha_is_known(self):
        archive, a, b, _ = self.archive(actual={'goal/a.txt': b'changed bytes!'})
        destination = self.root / 'fresh'
        with self.assertRaises(ValueError): m.extract_bytes(archive, destination, a, b)
        self.assertFalse((destination / 'BACKUP_VERIFICATION.json').exists())

    def test_missing_or_extra_members_are_rejected(self):
        archive, a, b, _ = self.archive(actual={})
        with self.assertRaisesRegex(ValueError, 'members differ'): m.extract_bytes(archive, self.root / 'fresh', a, b)

    def test_links_and_traversal_are_rejected_before_extraction(self):
        for kind, name in [(m.tarfile.SYMTYPE, 'link'), (m.tarfile.LNKTYPE, 'hard'), (m.tarfile.REGTYPE, '../escape')]:
            def add(tar, kind=kind, name=name):
                info = m.tarfile.TarInfo(name); info.type = kind; info.linkname = 'goal/a.txt'; tar.addfile(info)
            archive, a, b, _ = self.archive(extra=add)
            destination = self.root / 'fresh'
            with self.assertRaises(ValueError): m.extract_bytes(archive, destination, a, b)
            self.assertFalse(destination.exists())

    def test_incomplete_real_format_never_gets_an_official_verification_receipt(self):
        archive, a, b, _ = self.archive(manifest_override={'format': m.FORMAT, 'files': {}, 'self_contained': False}, actual={})
        destination = self.root / 'fresh'
        with self.assertRaises(ValueError): m.verify(archive, destination, archive_sha=a, manifest_sha=b)
        self.assertFalse((destination / 'BACKUP_VERIFICATION.json').exists())

    def test_external_dependencies_require_exact_bytes_and_disclosed_mapping(self):
        deps = {'historical_results': []}
        for group in ('dataset', 'assets'):
            directory = self.root / group; directory.mkdir()
            (directory / 'manifest.json').write_text('{}')
            (directory / 'data.bin').write_bytes(group.encode())
            deps[group] = dict(local='old/' + group, remote='/remote/' + group,
                manifest_sha256=m.sha(directory / 'manifest.json'), files={'data.bin': m.file_record(directory / 'data.bin')})
        row = self.root / 'old_prediction.gz'; row.write_bytes(b'synthetic')
        deps['historical_results'] = [dict(local='old/prediction.gz', remote='/remote/prediction.gz', **m.file_record(row))]
        mapping = {'old/dataset': str(self.root / 'dataset'), 'old/assets': str(self.root / 'assets'), 'old/prediction.gz': str(row)}
        result = m.verify_dependencies(deps, 'local', mappings=mapping)
        self.assertEqual(result['historical_results'], 1)
        row.write_bytes(b'tampered')
        with self.assertRaises(ValueError): m.verify_dependencies(deps, 'local', mappings=mapping)

    def relocation_fixture(self, gate=False):
        parent = {'source': 'authenticated'}
        old = dict(config=dict(dataset='/remote/data', assets='/remote/pse', hidden=256), code={'f': 'a' * 64})
        new = copy.deepcopy(old); new['config'].update(dataset='/local/data', assets='/local/pse')
        protocol = dict(branch='C', stop_phase=45, source_proof={'relocations': {}, 'extensions': {}})
        current_protocol = copy.deepcopy(protocol)
        current_protocol['source_proof']['relocations'] = {'dataset': {'parent': '/remote/data', 'execution': '/local/data'}}
        controller = SimpleNamespace(execution_binding=new, parent_binding=parent, parent_hashes={'model': 'm'},
            protocol=current_protocol, protocol_sha256='c' * 64)
        inner = dict(binding=old, model=[1], alignment=[2], r_optimizer=[3], t_optimizer=[4], scheduler=[5],
                     rng={'python': [6], 'cuda': [7]}, progress={'phase': 25}, r_updates=13860, t_updates=200)
        c20 = {'inner': copy.deepcopy(inner)}
        saved = dict(inner=inner, execution_binding=old, parent_binding=parent, parent_hashes={'model': 'm'},
                     protocol=protocol, replay_gate=None, replay_gate_state_sha256=None)
        if gate:
            saved['replay_gate_state_sha256'] = fp(c20['inner'])
            saved['replay_gate'] = dict(checkpoint_state_sha256=fp(c20['inner']), protocol_sha256='b' * 64)
        def verifier(a, b):
            left, right = copy.deepcopy(a), copy.deepcopy(b)
            for key in ('dataset', 'assets'): left['config'].pop(key); right['config'].pop(key)
            if left != right: raise ValueError('Scientific binding changed')
            return {'extensions': {}, 'relocations': current_protocol['source_proof']['relocations']}
        return saved, controller, verifier, c20

    def test_memory_path_relocation_preserves_original_scientific_state(self):
        saved, controller, verifier, c20 = self.relocation_fixture(gate=True); before = fp(saved)
        adapted, proof = m.memory_relocation(saved, controller, fp, verifier, c20=c20)
        self.assertEqual(fp(saved), before)
        for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'progress'):
            self.assertEqual(adapted['inner'][key], saved['inner'][key])
        self.assertNotEqual(adapted['replay_gate_state_sha256'], saved['replay_gate_state_sha256'])
        self.assertEqual(adapted['replay_gate']['protocol_sha256'], controller.protocol_sha256)
        self.assertTrue(proof['original_archive_bytes_untouched'])

    def test_relocation_does_not_silently_accept_new_science_sources_or_missing_c20(self):
        saved, controller, verifier, c20 = self.relocation_fixture(gate=True)
        with self.assertRaisesRegex(ValueError, 'archived R20'): m.memory_relocation(saved, controller, fp, verifier)
        bad = copy.deepcopy(saved); bad['protocol']['stop_phase'] = 46
        with self.assertRaisesRegex(ValueError, 'Scientific recovery'): m.memory_relocation(bad, controller, fp, verifier, c20=c20)
        with self.assertRaisesRegex(ValueError, 'not add runtime'):
            m.memory_relocation(saved, controller, fp, lambda a, b: {'extensions': {'new.py': 'x'}, 'relocations': {}}, c20=c20)

    def test_pack_requires_real_release_evidence_before_writing_archive(self):
        root = self.root / 'goal'; root.mkdir(); release = self.root / 'release.json'
        release.write_text(json.dumps({'status': 'pending', 'all_goal_jobs_released': False, 'evidence': []}))
        archive = self.root / 'full.tar'
        with patch.object(m, 'execution_inventory'):
            with self.assertRaisesRegex(ValueError, 'released-job'):
                m.pack(root, archive, execution_sha='a' * 64, release_receipt=release)
        self.assertFalse(archive.exists())


if __name__ == '__main__': unittest.main()
