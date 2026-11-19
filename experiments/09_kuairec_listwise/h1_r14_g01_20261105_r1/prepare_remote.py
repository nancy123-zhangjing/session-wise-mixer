"""Prepare isolated G01 extension; never alter existing experiment sources."""
import json, shutil, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OLD=BASE/'h1_mixer_g01_20261104_r1'
REPLAY=BASE/'h1_replay_r5_r20_20261105_r1'
ORIGINAL=BASE/'goal_20261027_r1'
if (ROOT/'code').exists():
    raise FileExistsError('Inspect existing preparation before rerun')
shutil.copytree(OLD/'code',ROOT/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('recovery_r14_g01.py','recovery_r14_g01_run.py'):
    shutil.copyfile(ROOT/name,ROOT/'code/scripts/listwise_diffgrm'/name)
shutil.copyfile(OLD/'validation_subset_8192.json',ROOT/'validation_subset_8192.json')
inputs=dict(parent=str(REPLAY/'runs/H1-Replay-R20/checkpoints/R14_T120.pt'),
dataset=str(ORIGINAL/'inputs/dataset'),assets=str(ORIGINAL/'assets/pse_v1'),
r14_reference=str(ORIGINAL/'runs/H1/validation_R14/summary.json'))
(ROOT/'INPUTS.json').write_text(json.dumps(inputs,indent=2))
manifest={}
for p in (ROOT/'code').rglob('*'):
    if p.is_file():
        manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(prepared=True,files=len(manifest),root=str(ROOT))))
