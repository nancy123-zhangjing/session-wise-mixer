import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'h1_r14_g01_20261105_r1'
assert not (ROOT/'code').exists()
shutil.copytree(OLD/'code',ROOT/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('recovery_r17_release1.py','recovery_r17_release1_run.py'):
    shutil.copyfile(ROOT/name,ROOT/'code/scripts/listwise_diffgrm'/name)
shutil.copyfile(OLD/'validation_subset_8192.json',ROOT/'validation_subset_8192.json')
inputs=json.loads((OLD/'INPUTS.json').read_text())
inputs['parent']=str(OLD/'runs/R14-G01/phase20_R_R17_T160.pt')
inputs['r17_reference']=str(OLD/'runs/R14-G01/evaluations/R17/attempt1/summary.json')
inputs.pop('r14_reference')
(ROOT/'INPUTS.json').write_text(json.dumps(inputs,indent=2))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').rglob('*') if p.is_file()}
(ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'prepared':True,'files':len(manifest)}))
