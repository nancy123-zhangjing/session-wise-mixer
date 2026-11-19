"""Verify actual frozen schedule function and phase IDs without a GPU or training."""
import ast
import json
import math
import runpy
import unittest
from pathlib import Path
from protocol import counts, skipped

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
tree = ast.parse((REPO / 'outputs/stage09_restore/goal_20261027_r1/code/scripts/listwise_diffgrm/training.py').read_text(encoding='utf-8'))
definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'phase_plan')
namespace = {'math': math}
exec(compile(ast.Module(body=[definition], type_ignores=[]), '<original phase_plan>', 'exec'), namespace)
old = runpy.run_path(str(HERE.parent / 'multiseed_freeze_20261109_v1/protocol.py'))


class ProtocolChecks(unittest.TestCase):
    def test_schedule_matches_existing_h1_freeze(self):
        for seed in (2025, 2026):
            cfg = json.loads((HERE / f'config_seed{seed}.json').read_text())
            plan = namespace['phase_plan'](cfg, 709443)
            self.assertEqual(len(plan), 45)
            self.assertEqual(counts(plan, {'phase': 25, 'update': 0}, 'H0'), {'R': 13860, 'T': 200})
            self.assertEqual(counts(plan, {'phase': 45, 'update': 0}, 'H0'), {'R': 24255, 'T': 200})
            actual = [(i, row) for i, row in enumerate(plan) if not skipped(row, 'H0')]
            expected = [(i, row) for i, row in enumerate(plan) if not old['skipped'](row, 'H1')]
            self.assertEqual(actual, expected)
            self.assertEqual(sum(row['updates'] for i, row in actual if i >= 25), 10395)
            self.assertEqual([row['epoch'] for i, row in actual if i >= 25 and row['validate']], [23, 26, 29, 32, 35])
            self.assertEqual((HERE / f'config_seed{seed}.json').read_bytes(),
                             (HERE.parent / 'multiseed_freeze_20261109_v1' / f'config_seed{seed}.json').read_bytes())


if __name__ == '__main__': unittest.main()
