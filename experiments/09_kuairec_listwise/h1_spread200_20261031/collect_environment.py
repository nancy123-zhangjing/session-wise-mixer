"""Record package metadata only; no CUDA/model execution or environment changes."""
import argparse
import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--scope',required=True)
a=p.parse_args()
packages={}
for package in ('torch','numpy','pyarrow','pandas','scipy','matplotlib','PyYAML','tqdm','einops','transformers'):
    try: packages[package]=metadata.version(package)
    except metadata.PackageNotFoundError: packages[package]=None
value=dict(format='stage09_environment_metadata_v1',scope=a.scope,
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=sys.version,executable=sys.executable,platform=platform.platform(),
    packages=packages,collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cuda_device_inspection=False,model_execution=False,
    limitation='Package metadata from this interpreter; not proof of GPU availability or numerical restoration.')
with a.output.open('x',encoding='utf-8') as f:
    json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(value))
