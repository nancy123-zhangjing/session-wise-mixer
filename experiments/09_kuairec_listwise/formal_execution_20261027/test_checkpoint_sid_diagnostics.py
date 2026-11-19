"""Small CPU fixtures only; never run final acceptance or inspect real models."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).absolute().parent
WORKSPACE = HERE.parents[2]
SCRIPT = WORKSPACE / 'experiments/09_kuairec_listwise/formal_execution_20261027/checkpoint_sid_diagnostics.py'
spec = importlib.util.spec_from_file_location('checkpoint_sid_diagnostics_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch = m.cpu_libraries()
manifest = m.read_bound_json(SCRIPT.with_name('FORMAL_BRANCH_BINDING.json'), m.BINDING_SHA)
statistics, sources = m.load_statistics(WORKSPACE / 'outputs/stage09_restore/goal_20261027_r1/code', manifest['expected_bindings'])


def saved(sid, epoch):
    return dict(epoch=epoch, **m.clocks(epoch), checkpoint_precision='bf16',
        stored_model_floating_dtype='torch.float32', sid_snapshot=dict(field='alignment.hard_codes',
        shape=[10729, 4], pad_row=-1, real_code_range=[0, 255], fingerprint=m.sid_fingerprint(sid)))


def state(arm, epoch, selection):
    sid = torch.zeros(10729, 4, dtype=torch.int64)
    sid[0] = -1
    clock = m.clocks(epoch)
    return dict(format='kuairec_listwise_v1', arm=arm, binding=manifest['expected_bindings'][arm[0]],
        precision='bf16', r_updates=clock['r_updates'], t_updates=clock['t_updates'],
        progress=dict(phase=clock['phase'], update=0, best_epoch=selection['best_epoch'], best_score=selection['score']),
        alignment={'hard_codes': sid})


def receipt_fixture():
    sid = torch.zeros(10729, 4, dtype=torch.int64)
    sid[0] = -1
    result = dict(schema='stage09_four_arm_final_selection_v1', status='passed', root=m.REMOTE_ROOT,
        execution_source=m.SOURCE, branch_binding_sha256=m.BINDING_SHA, branch_helper_sha256=m.BRANCH_HELPER_SHA,
        script_sha256=m.FINAL_HELPER_SHA, distinct_full_validations=42, test_used_for_selection=False, gpu_used=False,
        arms={}, four_best_sha256={}, four_best_bytes={})
    for arm in m.ARMS:
        row = dict(selection=dict(metric='list/ndcg@5', split='validation', beam=32, tie_break='earlier',
            score=.1, best_epoch=8, best_t_updates=40), shared_initial_exact=True,
            best_state=saved(sid, 8), final_state=saved(sid, 35))
        for role in ('best', 'last'):
            row[role] = dict(path=m.REMOTE_ROOT + '/runs/' + arm + '/' + role + '.pt', bytes=1,
                            sha256=('1' if role == 'best' else '2') * 64)
        result['arms'][arm] = row
        result['four_best_sha256'][arm] = row['best']['sha256']
        result['four_best_bytes'][arm] = row['best']['bytes']
    return result


class Checks(unittest.TestCase):
    def test_hand_count_keeps_true_zero_sid(self):
        sid = torch.tensor([[-1]*4, [0]*4, [0]*4, [1]*4, [1]*4, [1]*4, [2]*4], dtype=torch.int64)
        result = m.summarize_sid(sid, statistics, expected_items=6)
        self.assertEqual(result['real_all_zero_sid_items'], 2)
        expected = dict(items=6, unique_sids=3, collided_items=5, collision_rate=5/6,
                        colliding_buckets=2, largest_bucket=3)
        self.assertEqual({key: result['collision'][key] for key in expected}, expected)
        self.assertEqual(result['collision']['per_digit'][0]['usage'][:3], [2, 3, 1])
        # Verify exact fingerprint compatibility with the existing pinned helper.
        path = SCRIPT.with_name('formal_branch_acceptance.py')
        self.assertEqual(m.file_identity(path)['sha256'], m.BRANCH_HELPER_SHA)
        spec = importlib.util.spec_from_file_location('sid_test_branch_fingerprint', path)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        self.assertEqual(result['fingerprint'], helper.fingerprint(sid))

    def test_wrong_sid_rejected(self):
        good = torch.tensor([[-1]*4, [0]*4, [255]*4], dtype=torch.int64)
        bad = [good.float(), good[:, :3], good[1:]]
        for row, value in ((0, 0), (1, -1), (2, 256)):
            changed = good.clone(); changed[row, 0] = value; bad.append(changed)
        for value in bad:
            with self.subTest(dtype=value.dtype, shape=value.shape), self.assertRaises(RuntimeError):
                m.summarize_sid(value, statistics, expected_items=2)

    def test_receipt_four_best_and_last_identity(self):
        receipt = receipt_fixture()
        m.validate_selection(receipt)
        mutations = [lambda x: x.update(status='pending'), lambda x: x.update(execution_source='wrong'),
            lambda x: x.update(branch_binding_sha256='0'*64), lambda x: x['arms'].pop('H1'),
            lambda x: x['four_best_sha256'].update(J0='0'*64),
            lambda x: x['arms']['J0']['last'].update(path=m.REMOTE_ROOT+'/runs/J0/best.pt'),
            lambda x: x['arms']['J0']['final_state'].update(t_updates=399)]
        for mutation in mutations:
            changed = copy.deepcopy(receipt); mutation(changed)
            with self.assertRaises(RuntimeError): m.validate_selection(changed)

    def test_checkpoint_sha_sid_and_role_binding(self):
        selection = dict(best_epoch=8, score=.1)
        with tempfile.TemporaryDirectory(prefix='sid_diagnostics_fixture_', dir=HERE) as folder:
            paths, refs, snapshots = {}, {}, {}
            for role, epoch in (('best', 8), ('last', 35)):
                checkpoint = state('J0', epoch, selection)
                path = Path(folder) / (role + '.pt')
                torch.save(checkpoint, path)
                ref = dict(path=m.REMOTE_ROOT+'/runs/J0/'+role+'.pt', **m.file_identity(path))
                snapshot = saved(checkpoint['alignment']['hard_codes'], epoch)
                result = m.inspect_checkpoint(path, ref, snapshot, arm='J0', selection=selection,
                    binding=manifest['expected_bindings']['J'], statistics=statistics)
                self.assertEqual(result['epoch'], epoch)
                self.assertEqual(result['sid']['collision']['items'], 10728)
                paths[role], refs[role], snapshots[role] = path, ref, snapshot
            bad_cases = [(refs['last'], snapshots['best']),
                         (dict(refs['best'], sha256='0'*64), snapshots['best']),
                         (refs['best'], dict(snapshots['best'], sid_snapshot=dict(snapshots['best']['sid_snapshot'], fingerprint='0'*64))),
                         (refs['best'], snapshots['last'])]
            for ref, snapshot in bad_cases:
                with self.assertRaises(RuntimeError):
                    m.inspect_checkpoint(paths['best'], ref, snapshot, arm='J0', selection=selection,
                        binding=manifest['expected_bindings']['J'], statistics=statistics)
            with self.assertRaises(RuntimeError):
                m.inspect_checkpoint(paths['best'], refs['best'], snapshots['best'], arm='J1', selection=selection,
                    binding=manifest['expected_bindings']['J'], statistics=statistics)

    def test_json_sha_rejects_and_cpu_remains_cpu(self):
        with self.assertRaises(RuntimeError):
            m.read_bound_json(SCRIPT.with_name('FORMAL_BRANCH_BINDING.json'), '0'*64)
        self.assertIsNone(torch.version.cuda)
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(len(sources), 3)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), helper_sha256=m.file_identity(SCRIPT)['sha256'],
        real_final_checkpoint_loaded=False, final_selection_executed=False, gpu_initialized=torch.cuda.is_initialized(),
        scope='Synthetic SID/checkpoint fixtures and SHA-bound frozen statistics function only.')))
    raise SystemExit(0 if result.wasSuccessful() else 1)
