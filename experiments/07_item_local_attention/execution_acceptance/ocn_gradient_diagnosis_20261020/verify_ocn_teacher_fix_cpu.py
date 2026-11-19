"""Verify a proposed scoring-only autocast fix without editing runtime source."""
from contextlib import redirect_stdout
import difflib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import types
from unittest import mock
import torch

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ocn_fix_fixture', root/'tests/test_item_local_attention.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
filename = 'genrec/models/DIFF_GRM/model.py'
old_source = subprocess.check_output(['git', '-C', str(fixture.ENGINE), 'show', '39e94e6:'+filename]).decode()
new_source = (fixture.ENGINE/filename).read_text(encoding='utf-8')
needle = '                self.eval()\n                with torch.no_grad():\n'
replacement = ('                self.eval()\n'
               '                # Teacher scoring must not cache detached casts for training.\n'
               '                with torch.no_grad(), torch.autocast(\n'
               '                    device_type=device.type, dtype=torch.get_autocast_dtype(device.type),\n'
               '                    enabled=torch.is_autocast_enabled(device.type), cache_enabled=False\n'
               '                ):\n')
results = []
for architecture, source in [('pinned_baseline', old_source), ('item_local', new_source)]:
    assert source.count(needle) == 1, 'Teacher scoring patch must identify exactly one source location'
    candidate_source = source.replace(needle, replacement)
    for behavior, selected in [('original', source), ('teacher_cache_fix', candidate_source)]:
        module = types.ModuleType(fixture.PACKAGE+'.'+architecture+'_'+behavior)
        module.__package__ = fixture.PACKAGE
        with mock.patch.dict('sys.modules', {'genrec.model': fixture.abstract_module,
                                           'genrec.dataset': fixture.dataset_module,
                                           'genrec.tokenizer': fixture.tokenizer_module}):
            exec(compile(selected, architecture+'_'+behavior+'.py', 'exec'), module.__dict__)
        config = fixture.tiny_config(True if architecture == 'item_local' else None)
        config['decoder_n_layer'] = 4
        torch.manual_seed(2024)
        with redirect_stdout(io.StringIO()):
            model = module.DIFF_GRM(config, object(), fixture.TinyTokenizer()).train()
            with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=True):
                loss = model(fixture.tiny_batch()).loss
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        results.append({'architecture': architecture, 'behavior': behavior,
                        'decoder_layers': 4, 'loss': loss.item(), 'parameters': len(list(model.parameters())),
                        'missing_gradient_names': missing,
                        'all_present_gradients_finite': all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)})
    proposed = ''.join(difflib.unified_diff(source.splitlines(keepends=True), candidate_source.splitlines(keepends=True),
                                         fromfile='a/'+filename, tofile='b/'+filename))
    (root/'.runtime/stage07'/('proposed_'+architecture+'_ocn_cache_fix.patch')).write_text(proposed, encoding='utf-8')
for architecture in ('pinned_baseline', 'item_local'):
    original, fixed = [row for row in results if row['architecture'] == architecture]
    assert len(original['missing_gradient_names']) == 40
    assert not fixed['missing_gradient_names'] and fixed['all_present_gradients_finite']
    assert original['loss'] == fixed['loss']
output = root/'.runtime/stage07/OCN_TEACHER_FIX_CPU_VERIFICATION_20261020.json'
with output.open('x', encoding='utf-8') as handle:
    json.dump({'scope': 'Four-layer synthetic CPU reproduction and in-memory candidate; frozen source unchanged, no GPU or real training',
               'status': 'PASS', 'results': results}, handle, indent=2)
print(json.dumps({'status': 'PASS', 'results': [{key: value for key, value in row.items() if key != 'missing_gradient_names'} |
                                             {'missing_gradient_tensors': len(row['missing_gradient_names'])} for row in results]}, indent=2))
