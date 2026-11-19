"""Complete stage05 byte backup, including inputs and recoverable states.

Packing requires a completed artifact audit and platform release evidence.
Verification hashes every archive part and extracted member. Neither mode
claims the experimental Goal delivered or substitutes for model restoration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
CAMPAIGN = 'goal_20261020_long_mixer_v1'
FORMAT = 'stage05_long_full_byte_backup_v1'
ARMS = ('no_mixer_frozen','no_mixer_joint','mixer_frozen','mixer_joint')
CODE_COMMIT = '69682698c15ba796b98e6329c950061a2f68a05b'
ENGINE_COMMIT = '1c879252f6395d6e64462511823c3929b9c42c00'


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def safe_name(value):
    need(isinstance(value,str) and value and '\\' not in value and ':' not in value, 'Unsafe archive name')
    path = PurePosixPath(value)
    need(not path.is_absolute() and not any(part in ('','..','.') for part in value.split('/')), 'Archive traversal')
    reserved={'CON','PRN','AUX','NUL'}|{prefix+str(i) for prefix in ('COM','LPT') for i in range(1,10)}
    need(all(not part.endswith((' ','.')) and part.split('.')[0].upper() not in reserved for part in path.parts),
         'Nonportable Windows archive name')
    return value


def mandatory_names():
    names={'delivery/INPUTS.json','formal/common/initial.pt','formal/identity.json','formal/CONTROL_PASS.json',
           'formal/SELECTION_SEALED.json','formal/COMPUTE_COMPLETE.json',
           'prep/code.bundle','prep/code_r2.bundle','prep/engine.bundle',
           'code/configs/05_crosspse_joint/long/mixer_long_seed2024.json','engine/genrec/models/DIFF_GRM/model.py'}
    names.update('formal/'+a+'/'+n for a in ARMS for n in
                 ('last.pt','best.pt','TRAIN_COMPLETE.json','progress.json','TEST_STARTED.json','TEST_RESULT.json'))
    return names


def regular(path, root):
    root, path = Path(root).resolve(), Path(path)
    need(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(), 'Missing/nonregular/escaped source: '+str(path))
    parent = path.parent
    while parent != root:
        need(parent.is_relative_to(root) and not parent.is_symlink(), 'Symlink source parent')
        parent = parent.parent
    return path


def collect(goal, audit_path, release_path):
    goal = Path(goal).resolve()
    need(goal.name == CAMPAIGN, 'Wrong campaign')
    audit, release = read(audit_path), read(release_path)
    need(audit['status'] == 'FORMAL_ARTIFACT_AUDIT_PASS_NOT_DELIVERY' and audit['tests_checked'] is True
         and audit['checkpoints_checked'] is True and audit['goal_complete'] is False, 'Complete independent artifact audit required')
    need(release['campaign_id'] == CAMPAIGN and release['all_goal_jobs_released'] is True
         and release['actual_allocated_gpu_seconds'] <= 43200 and release['acceptance_allocated_gpu_seconds'] <= 1800,
         'Platform-verified cumulative release/budget required')
    sources = {}
    def add(path):
        path = regular(path,goal)
        name = safe_name(path.relative_to(goal).as_posix())
        need(name not in sources or sources[name] == path, 'Duplicate archive member')
        sources[name] = path
    source_locks = {}
    for directory, expected in (('code',CODE_COMMIT), ('engine',ENGINE_COMMIT)):
        cwd = goal / directory
        head = subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
        dirty = subprocess.check_output(['git','-C',str(cwd),'status','--porcelain'],text=True).strip()
        need(head == expected and not dirty, 'Frozen source changed')
        files = subprocess.check_output(['git','-C',str(cwd),'ls-files','-z']).decode().split('\0')
        for name in files:
            if name:
                add(cwd / name)
        boundary = cwd / '.git/shallow'
        source_locks[directory] = dict(commit=head,tracked_files=len([v for v in files if v]),
                 shallow_boundary=boundary.read_text().strip().splitlines() if boundary.exists() else [])
    inputs = read(goal / 'delivery/INPUTS.json')
    for row in inputs['records']:
        path = goal / 'inputs' / safe_name(row['path'])
        need(sha(path) == row['sha256'] and path.stat().st_size == row['size_bytes'], 'Immutable input changed')
        add(path)
    for arm in ARMS:
        for name in ('last.pt','best.pt','TRAIN_COMPLETE.json','progress.json','TEST_STARTED.json','TEST_RESULT.json'):
            add(goal / 'formal' / arm / name)
    for path in (goal / 'formal').glob('*.json'):
        add(path)
    add(goal / 'formal/common/initial.pt')
    # Keep GPU-gate states as well as receipts, so the real recovery evidence is portable.
    for path in (goal / 'smoke').rglob('*'):
        if path.is_file() and 'runtime' not in path.relative_to(goal / 'smoke').parts and path.suffix in ('.json','.pt','.log','.csv','.npy'):
            add(path)
    for directory in ('delivery','prep'):
        for path in (goal / directory).rglob('*'):
            relative = path.relative_to(goal / directory)
            if any(part.startswith(('final_package','package_extract','restore_check','full_backup')) for part in relative.parts):
                continue
            if path.is_file() and path.suffix in ('.json','.csv','.log','.yaml','.yml','.txt','.md','.py','.bundle','.gz','.png','.svg','.pdf','.npy'):
                add(path)
    add(audit_path); add(release_path)
    return sources, dict(source_locks=source_locks,artifact_audit_sha256=sha(audit_path),resource_release_sha256=sha(release_path),
                         all_14_locked_inputs_included=len(inputs['records']) == 14,
                         audit_script_sha256=audit.get('auditor_sha256'),
                         packer_sha256=sha(Path(__file__)),goal_complete=False)


def pack(goal, output, audit, release, part_mib):
    need(not output.exists(), 'Use a new package directory')
    sources, metadata = collect(goal,audit,release)
    need(metadata['all_14_locked_inputs_included'], 'Complete required input streams not collected')
    output.mkdir(parents=True)
    records = [dict(member=name,size_bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(sources.items())]
    manifest = dict(format=FORMAT,campaign_id=CAMPAIGN,utc=datetime.now(timezone.utc).isoformat(),files=records,**metadata)
    manifest_data = (json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    archive = output / 'mixer_long_backup.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as tar:
            for row in records:
                source = sources[row['member']]
                tar.add(source,arcname=row['member'],recursive=False)
                need(source.stat().st_size == row['size_bytes'] and sha(source) == row['sha256'], 'Source changed during packaging')
            info = tarfile.TarInfo('MANIFEST.json'); info.size=len(manifest_data)
            tar.addfile(info,io.BytesIO(manifest_data))
    archive_sha, archive_size = sha(archive), archive.stat().st_size
    parts = []
    if archive_size > part_mib*1024*1024:
        with archive.open('rb') as handle:
            while block := handle.read(part_mib*1024*1024):
                path = output / ('mixer_long_backup.tar.gz.part%04d' % (len(parts)+1))
                with path.open('xb') as target:
                    target.write(block)
                parts.append(dict(member=path.name,size_bytes=path.stat().st_size,sha256=sha(path)))
        # Retain the original as a remote redundant copy; download only listed parts.
    else:
        parts.append(dict(member=archive.name,size_bytes=archive_size,sha256=archive_sha))
    result = dict(format=FORMAT,status='PACKAGED_BYTES_NOT_COMPLETE_DELIVERY',campaign_id=CAMPAIGN,
                  archive_sha256=archive_sha,archive_size_bytes=archive_size,manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
                  files=len(records),uncompressed_bytes=sum(r['size_bytes'] for r in records),parts=parts,goal_complete=False)
    write(output / 'PACKAGE.json',result)
    return result


class ChainedReader(io.RawIOBase):
    def __init__(self,paths):
        self.paths=iter(paths); self.handle=None; self.digest=hashlib.sha256()
    def readable(self): return True
    def read(self,n=-1):
        need(n>=0,'Bounded stream reads required')
        blocks=[]; remaining=n
        while remaining:
            if self.handle is None:
                path=next(self.paths,None)
                if path is None: break
                self.handle=Path(path).open('rb')
            block=self.handle.read(remaining)
            if not block:
                self.handle.close(); self.handle=None; continue
            self.digest.update(block); blocks.append(block); remaining-=len(block)
        return b''.join(blocks)
    def readinto(self,buffer):
        value=self.read(len(buffer)); buffer[:len(value)]=value; return len(value)
    def close(self):
        if self.handle is not None: self.handle.close()
        super().close()


def verify(package,output):
    need(not output.exists(),'Use a new extraction directory')
    receipt=read(package/'PACKAGE.json')
    need(receipt['format']==FORMAT and receipt['campaign_id']==CAMPAIGN and receipt['parts'], 'Wrong package receipt')
    paths=[]
    for row in receipt['parts']:
        name=safe_name(row['member']); path=regular(package/name,package)
        need('/' not in name and path.stat().st_size==row['size_bytes'] and sha(path)==row['sha256'],'Archive part damaged')
        paths.append(path)
    output.mkdir(parents=True)
    records={}; manifest_data=None
    stream=ChainedReader(paths)
    try:
        with tarfile.open(fileobj=stream,mode='r|gz') as tar:
            for member in tar:
                name=safe_name(member.name)
                need(member.isfile() and name not in records,'Nonregular/duplicate archive member')
                target=(output/name).resolve()
                need(target.is_relative_to(output.resolve()),'Archive member escaped extraction')
                target.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256(); size=0
                with tar.extractfile(member) as source, target.open('xb') as destination:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        digest.update(block); size+=len(block); destination.write(block)
                need(size==member.size,'Truncated member')
                records[name]=dict(member=name,size_bytes=size,sha256=digest.hexdigest())
                if name=='MANIFEST.json': manifest_data=target.read_bytes()
        while stream.read(1024*1024): pass
        need(stream.digest.hexdigest()==receipt['archive_sha256'],'Concatenated archive hash differs')
    finally:
        stream.close()
    need(manifest_data is not None and hashlib.sha256(manifest_data).hexdigest()==receipt['manifest_sha256'],'Manifest is missing/damaged')
    manifest=json.loads(manifest_data)
    need(manifest['format']==FORMAT and manifest['campaign_id']==CAMPAIGN, 'Wrong internal manifest')
    expected={safe_name(r['member']):r for r in manifest['files']}
    need(len(expected)==len(manifest['files']) and set(records)==set(expected)|{'MANIFEST.json'},'Archive file inventory differs')
    need(all(records[name]==row for name,row in expected.items()) and len(expected)==receipt['files'],'Member byte hashes differ')
    need(mandatory_names().issubset(expected), 'Backup omitted required model/selection/input/source members')
    inputs=read(output/'delivery/INPUTS.json')
    need(inputs['campaign_id']==CAMPAIGN and len(inputs['records'])==14,'Backup lacks exactly fourteen locked input streams')
    for row in inputs['records']:
        name='inputs/'+safe_name(row['path'])
        need(name in expected and expected[name]['sha256']==row['sha256'] and expected[name]['size_bytes']==row['size_bytes'],
             'Backup omitted or changed a locked input byte stream')
    result=dict(status='FULL_BACKUP_BYTE_VERIFICATION_PASS_NOT_RESTORE',campaign_id=CAMPAIGN,
                package_sha256=sha(package/'PACKAGE.json'),manifest_sha256=receipt['manifest_sha256'],files=len(expected),
                all_member_bytes_verified=True,all_14_locked_inputs_included=manifest['all_14_locked_inputs_included'],
                model_restore_verified=False,goal_complete=False)
    write(output/'BYTE_VERIFICATION.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pack','verify'),required=True)
    parser.add_argument('--goal-root',type=Path)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--package-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--part-mib',type=int,default=512)
    args=parser.parse_args()
    need(args.part_mib>=1,'Invalid part size')
    if args.mode=='pack':
        need(all(v is not None for v in (args.goal_root,args.audit,args.release)),'Packing requires explicit goal/audit/release')
        result=pack(args.goal_root.resolve(),args.output.resolve(),args.audit.resolve(),args.release.resolve(),args.part_mib)
    else:
        need(args.package_dir is not None,'Verification requires package directory')
        result=verify(args.package_dir.resolve(),args.output.resolve())
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
