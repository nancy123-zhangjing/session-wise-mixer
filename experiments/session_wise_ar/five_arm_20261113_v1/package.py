"""Build a small immutable source package; no data/checkpoints or remote writes."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    reference = ROOT/'experiments/rqcd_mixer/targets_formal_20261113_v1'
    cf = json.loads((reference/'SOURCE_MANIFEST.json').read_text())
    paths = {ROOT/name for name in cf['files'] if name.startswith(('scripts/','configs/','docs/'))}
    for folder in ('scripts/session_wise_ar','configs/session_wise_ar'):
        paths.update(p for p in (ROOT/folder).glob('*') if p.suffix in ('.py','.json','.txt'))
    paths.update((ROOT/'docs/research/session_wise_ar_mixer').glob('*.md'))
    paths.update(reference/name for name in ('SOURCE_MANIFEST.json','FINAL_AUDIT.json','runtime.locked.json'))
    for name in ('scripts/__init__.py','scripts/rqcd/__init__.py','scripts/listwise_diffgrm/__init__.py'):
        if (ROOT/name).exists(): paths.add(ROOT/name)
    for p in paths:
        relative = p.relative_to(ROOT).as_posix()
        subprocess.run(['git','ls-files','--error-unmatch',relative],cwd=ROOT,capture_output=True,check=True)
        if subprocess.check_output(['git','status','--porcelain','--',relative],cwd=ROOT).strip():
            raise RuntimeError('Dirty scientific input: '+relative)
    mappings = {p.relative_to(ROOT).as_posix():p for p in paths}
    lock = json.loads((ROOT/'configs/session_wise_ar/upstream.lock.json').read_text())
    for entry in lock['files']:
        p = ROOT/lock['local_dir']/entry['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError('Upstream changed: '+str(p))
        mappings[p.relative_to(ROOT).as_posix()] = p
    mappings['delivery/worker.py'] = HERE/'worker.py'
    manifest = {'scientific_commit':'b66a50f', 'deployment_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'files':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(mappings.items())}}
    manifest_path = HERE/'SOURCE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive = ROOT/'.runtime/session_wise_ar_five_code.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,p in sorted(mappings.items()): tar.add(p,arcname=name)
        tar.add(manifest_path,arcname='SOURCE_MANIFEST.json')
    receipt = {'archive':str(archive),'bytes':archive.stat().st_size,
               'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(mappings)}
    (HERE/'PACKAGE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__': main()
