"""Only test the new freeze transition and exact continuation; no real-data runs."""
import unittest
from scripts.session_wise_ar.cf_freeze import FrozenTrainer
from tests.test_rqcd_mixer import tiny_config, fixture, SyntheticData, assert_tree_equal


class FreezeTests(unittest.TestCase):
    def test_frozen_state_and_resume(self):
        for mode in ('none', 'intra'):
            cfg = tiny_config()
            cfg['training'].update(epochs=35, preheat_epochs=5, t_steps_per_cycle=10)
            _, _, original = fixture(mode=mode, cfg=cfg)
            data = SyntheticData(cfg)
            while original.r_epoch < 9:
                original.advance(data)
            parent = original.state()
            original.__class__ = FrozenTrainer
            original.restore(parent)
            for _ in range(4):
                original.advance(data)
            state = original.state()
            self.assertEqual(state['r_epoch'], 11)
            self.assertEqual(state['t_updates'], 40)
            for key in ('tokenizer', 'catalog', 'optimizer_t', 't_stream'):
                assert_tree_equal(self, parent[key], state[key])
            for key in ('T_times', 'T_epsilon', 'T_gumbel'):
                assert_tree_equal(self, parent['generators'][key], state['generators'][key])
            original.advance(data)
            expected = original.state()
            original.restore(state)
            original.advance(data)
            assert_tree_equal(self, expected, original.state())
            with self.assertRaises(ValueError):
                original.refresh_catalog()
            with self.assertRaises(ValueError):
                original.update(None, None, 'T')


if __name__ == '__main__':
    unittest.main()
