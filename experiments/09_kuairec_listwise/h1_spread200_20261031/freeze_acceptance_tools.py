"""Version a new offline acceptance source bundle without overwriting receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--revision',required=True)
p.add_argument('--include',action='append',default=[])
a=p.parse_args()
if not re.fullmatch(r'r[0-9]+',a.revision):raise ValueError('Expected revision rN')
here=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=here/f'TOOLS_MANIFEST_{a.revision.upper()}.json'
archive=here/f'acceptance_tools_{a.revision}.tar.gz'
receipt=here/f'TOOLS_BUNDLE_{a.revision.upper()}.json'
if any(path.exists() for path in (manifest,archive,receipt)):raise FileExistsError('Preserve all previous revisions')
names=set(read(here/'TOOLS_MANIFEST_R1.json')['files'])
names.update(('audit_async_delivery.py','audit_completed_validation.py','source_check_validation.py',
    'check_readonly_validation_binding.py','check_locked_selection.py','audit_all_T200.py',
    'observe_async_once_v2.py','freeze_acceptance_tools.py','RECOVERY.md',
    'ENVIRONMENT_LOCAL.json','ENVIRONMENT_SHARED_INTERPRETER.json',
    'ALL_T200_ACCEPTED.json','ALL_T200_SNAPSHOT.json','ASYNC_FIRST30_ACCEPTED.json'))
for epoch in (8,11,14,17,20,23,26,29,32,35):
    folder='first_validation_R08' if epoch==8 else f'saved_validation_R{epoch:02d}'
    names.update(f'{folder}/{name}' for name in ('ACCEPTED.json','audit.json','record.json','summary.json'))
names.add('saved_validation_R35/SELECTION_ACCEPTED.json')
for name in a.include:
    path=(here/name).resolve();path.relative_to(here)
    if not path.is_file():raise ValueError('Additional source/evidence file missing')
    names.add(path.relative_to(here).as_posix())
files={name:dict(sha256=sha(here/name),bytes=(here/name).stat().st_size) for name in sorted(names)}
with manifest.open('x',encoding='utf-8') as f:
    json.dump(dict(status='prepared_source_bundle_not_final_acceptance',revision=a.revision,files=files),f,indent=2);f.write('\n')
with archive.open('xb') as raw:
    with tarfile.open(fileobj=raw,mode='w:gz',compresslevel=6) as tar:
        for name in sorted(files):tar.add(here/name,arcname=name,recursive=False)
        tar.add(manifest,arcname=manifest.name,recursive=False)
for name,metadata in files.items():
    if sha(here/name)!=metadata['sha256']:raise ValueError('Source changed while bundling')
result=dict(status='prepared_source_bundle_not_final_acceptance',revision=a.revision,
    archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    manifest_sha256=sha(manifest),files=len(files),
    limitation='Source and completed validation receipts only. Final fresh-extraction audits, report and resource closure remain pending.')
with receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
