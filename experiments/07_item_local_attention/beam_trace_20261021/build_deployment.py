from pathlib import Path
import hashlib
import json
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
destination = here / 'deployment'
destination.mkdir(exist_ok=False)
sources = {
    'scripts/analysis/evaluate_item_local_beam_trace.py': root / 'scripts/analysis/evaluate_item_local_beam_trace.py',
    'scripts/analysis/item_local_beam_trace.py': root / 'scripts/analysis/item_local_beam_trace.py',
    'tests/test_item_local_beam_trace.py': root / 'tests/test_item_local_beam_trace.py',
    'external/DiffGRM/genrec/models/DIFF_GRM/beam.py': root / 'external/DiffGRM/genrec/models/DIFF_GRM/beam.py',
    'run_gpu_trace.py': here / 'run_gpu_trace.py',
    'prepare_cpu.py': here / 'prepare_cpu.py',
    'PROTOCOL.md': here / 'PROTOCOL.md',
    'package_evaluation.py': here.parent / 'beam_validation_20261021/package_evaluation.py',
}
rows = []
for rel, source in sources.items():
    target = destination / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    rows.append({'path': rel, 'size_bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(destination / 'DEPLOYMENT_MANIFEST.json').write_text(json.dumps({'files': rows}, indent=2)+'\n')
print(json.dumps({'destination': str(destination), 'files': rows}))
