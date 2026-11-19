"""Small CPU-only comparison of gradient flow in pinned old/new OCN models."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('item_local_test_fixture', root/'tests/test_item_local_attention.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
previous_class = fixture.load_previous_model()
results = []
for architecture, cls, enabled in [('pinned_baseline', previous_class, None), ('item_local', fixture.DIFF_GRM, True)]:
    for precision, cache in [('float32', True), ('bf16', True), ('bf16', False)]:
        torch.manual_seed(2024)
        model = fixture.make_model(enabled, cls=cls).train()
        with redirect_stdout(io.StringIO()):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bf16', cache_enabled=cache):
                loss = model(fixture.tiny_batch()).loss
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        results.append({'architecture': architecture, 'precision': precision, 'autocast_cache_enabled': cache,
                        'loss': loss.item(), 'finite_loss': bool(torch.isfinite(loss)),
                        'parameters': len(list(model.parameters())), 'missing_gradient_names': missing})
output = root/'.runtime/stage07/OCN_AUTOCAST_CPU_DIAGNOSIS_20261020.json'
with output.open('x', encoding='utf-8') as handle:
    json.dump({'scope': 'Synthetic CPU reproduction; no real dataset, Trainer, CUDA, or platform work', 'results': results}, handle, indent=2)
print(json.dumps(results, indent=2))
