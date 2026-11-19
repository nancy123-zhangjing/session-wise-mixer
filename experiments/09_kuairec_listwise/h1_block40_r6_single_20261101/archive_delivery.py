"""Seal a terminal run or verify/extract its local archive. Never deletes files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n')

def seal(root):
    expected=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1')
    if root!=expected: raise ValueError('Only this owned new run may be sealed')
    pointer=root/'FORMAL_COMPLETION.json'
    if pointer.is_file():
        completion=read(pointer)
        if completion['attempt'] not in ('formal_r1',):
            raise ValueError('Unrecognized continuation attempt')
        attempt=root/'jobs'/completion['attempt']
        if (sha(attempt/'result.json')!=completion['result_sha256']
                or sha(attempt/'resume_gate.json')!=completion['resume_gate_sha256']):
            raise ValueError('Continuation success pointer does not bind its evidence')
        result=read(attempt/'result.json')
    else:
        result=read(root/'jobs/formal_r1/result.json')
    if result['status']!='training_validation_test_predictions_complete': raise ValueError('Formal job is not successfully terminal')
    run=root/'runs/H1-Block40-R6-R35'
    if read(run/'complete.json')['t_updates']!=200 or not (run/'test_predictions_complete.json').is_file():
        raise ValueError('Scientific run incomplete')
    destination=root/'delivery'
    destination.mkdir(exist_ok=True)
    manifest=destination/'DELIVERY_MANIFEST.json'
    archive=destination/'stage09_h1_block40_single_r1.tar.gz'
    receipt=destination/'ARCHIVE_RECEIPT.json'
    if any(p.exists() for p in (manifest,archive,receipt)): raise FileExistsError('Never overwrite delivery artifacts')
    files={}
    for path in sorted(root.rglob('*')):
        if destination==path or destination in path.parents: continue
        if path.is_symlink(): raise ValueError('Symlink not permitted in sealed run')
        if not path.is_file(): continue
        name=path.relative_to(root).as_posix()
        files[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    write(manifest,dict(format='stage09_block40_single_delivery_v1',files=files,
        total_bytes=sum(v['bytes'] for v in files.values()),source_root=str(root),
        external_dependencies=read(root/'INPUTS.json'),
        external_dependency_policy='Original parent, dataset and PSE assets are existing preserved dependencies; paths may be explicitly relocated with source hashes verified.'))
    with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for name in sorted(files):
                path=root/name
                if path.stat().st_size!=files[name]['bytes'] or sha(path)!=files[name]['sha256']:
                    raise ValueError('Source changed while sealing')
                info=tar.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
                with path.open('rb') as f: tar.addfile(info,f)
            info=tar.gettarinfo(str(manifest),arcname='DELIVERY_MANIFEST.json')
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with manifest.open('rb') as f: tar.addfile(info,f)
    for name,value in files.items():
        if sha(root/name)!=value['sha256']: raise ValueError('Source changed after sealing')
    result=dict(status='sealed',archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=sha(manifest),files=len(files),source_bytes=sum(v['bytes'] for v in files.values()),
        source_hashes_rechecked=True,scientific_analysis_pending=True,resource_release_requires_separate_verification=True)
    write(receipt,result); print(json.dumps(result),flush=True)

def extract(archive,receipt_path,destination):
    receipt=read(receipt_path)
    if sha(archive)!=receipt['archive_sha256']: raise ValueError('Downloaded archive SHA mismatch')
    if destination.exists(): raise FileExistsError('Independent extraction must be into a fresh directory')
    if shutil.disk_usage(destination.parent).free<receipt['source_bytes']+2*1024**3:
        raise ValueError('Insufficient free space with 2 GiB reserve; do not delete existing work')
    with tarfile.open(archive,'r:gz') as tar:
        manifest_members=[m for m in tar.getmembers() if m.name=='DELIVERY_MANIFEST.json']
        if len(manifest_members)!=1: raise ValueError('Archive lacks unique manifest')
        content=tar.extractfile(manifest_members[0]).read()
        if hashlib.sha256(content).hexdigest()!=receipt['manifest_sha256']: raise ValueError('Embedded manifest SHA mismatch')
        manifest=json.loads(content)
        members=tar.getmembers(); names=set()
        for member in members:
            path=PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name
                    or not member.isfile() or member.name in names):
                raise ValueError('Unsafe archive member')
            names.add(member.name)
            if member.name!='DELIVERY_MANIFEST.json':
                if member.name not in manifest['files'] or member.size!=manifest['files'][member.name]['bytes']:
                    raise ValueError('Archive member differs from manifest')
        if names!=set(manifest['files'])|{'DELIVERY_MANIFEST.json'}: raise ValueError('Archive file set differs')
        destination.mkdir()
        for member in members:
            target=destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as output: shutil.copyfileobj(source,output,4*1024*1024)
    for name,value in manifest['files'].items():
        path=destination/name
        if path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']: raise ValueError('Extracted payload SHA mismatch')
    write(destination/'LOCAL_EXTRACTION_VERIFIED.json',dict(status='passed',archive_sha256=receipt['archive_sha256'],
        manifest_sha256=receipt['manifest_sha256'],files=len(manifest['files']),all_file_hashes_verified=True,
        independent_fresh_directory=str(destination),checkpoint_cold_restore_pending=True,at_unix=time.time()))
    print(json.dumps(dict(status='passed',files=len(manifest['files']),destination=str(destination))),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('seal','extract'))
    p.add_argument('--root',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--receipt',type=Path);p.add_argument('--destination',type=Path)
    a=p.parse_args()
    if a.command=='seal': seal(a.root.resolve())
    else: extract(a.archive.resolve(),a.receipt.resolve(),a.destination.resolve())

if __name__=='__main__': main()
