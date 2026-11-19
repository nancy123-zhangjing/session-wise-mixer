"""CPU structural gate for the candidate runtime; no model execution or inference."""
import ast
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    code = Path('outputs/stage09_spread200_20261031/code').resolve()
    spec = importlib.util.spec_from_file_location('candidate_async', root / 'async_resume.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    old = ast.parse((code / 'scripts/listwise_diffgrm/spread200_run.py').read_text(encoding='utf-8'))
    new = ast.parse((root / 'async_resume.py').read_text(encoding='utf-8'))
    def calls(tree, method):
        return [ast.dump(x, include_attributes=False) for x in ast.walk(tree)
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                and isinstance(x.func.value, ast.Name) and x.func.value.id == 'c' and x.func.attr == method]
    for method in ('step', 'prepare_boundary', 'complete_phase'):
        assert calls(old, method) == calls(new, method), method + ' changed'
    sys.path.insert(0, str(code))
    import torch
    from scripts.listwise_diffgrm.training import phase_plan
    state = torch.load(root / 'cancellation_r1/last.pt', map_location='cpu', weights_only=False)
    config = state['inner']['binding']['config']
    plan = phase_plan(config, 709443)
    jobs = [(phase, runtime.task_kind(row), row['epoch']) for phase, row in enumerate(plan) if phase >= 5]
    assert len(plan) == 45 and sum(row['updates'] for row in plan[5:]) == 20990
    validations = [epoch for _, kind, epoch in jobs if kind == 'validation']
    assert validations == [8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    assert sum(kind == 'diagnostic' for _, kind, _ in jobs) == 20
    remaining = [x for x in jobs if x[0] >= state['inner']['progress']['phase'] and x[1]]
    assert len(remaining) == 11
    assert [e for _, k, e in remaining if k == 'validation'] == [26, 29, 32, 35]
    # The frozen boundary transition changes only progress; it calls no RNG,
    # model, optimizer, catalog or evaluation operation.
    spread = ast.parse((code / 'scripts/listwise_diffgrm/spread200.py').read_text(encoding='utf-8'))
    transition = next(x for x in ast.walk(spread) if isinstance(x, ast.FunctionDef) and x.name == 'complete_phase')
    assert all(not isinstance(x.func, ast.Attribute) for x in ast.walk(transition) if isinstance(x, ast.Call))
    result = dict(status='passed_CPU_async_structure_only', model_execution=False, candidate_deployed=False,
                  runtime_sha256=runtime.sha(root / 'async_resume.py'),
                  scientific_manifest_sha256=runtime.sha(code / 'SPREAD200_EXECUTION_MANIFEST.json'),
                  preserved_update_calls=['step', 'prepare_boundary', 'complete_phase'],
                  total_phases=45, new_updates=20990, full_validation_epochs=validations,
                  new_subset_diagnostics=20, initial_reused_diagnostic=1, remaining_tasks=remaining,
                  limitation='Structural checks do not replace numerical concurrency, task execution, final audit or resource-fit gates.')
    (root / 'CPU_ASYNC_STRUCTURE.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
