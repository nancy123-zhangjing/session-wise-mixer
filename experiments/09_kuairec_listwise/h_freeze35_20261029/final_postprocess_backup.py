"""Archive committed delivery files and verify a fresh extraction and Git clone."""
from __future__ import annotations
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
TASK_REL = 'experiments/09_kuairec_listwise/h_freeze35_20261029'
PATHS = [TASK_REL, 'INSPIRE.md', 'docs/guides/PROJECT_HANDOFF.md']

def git(*args, cwd=ROOT):
    p = subprocess.run(['git','-c','core.longpaths=true',*map(str,args)],cwd=cwd,capture_output=True)
    if p.returncode:
        # Preserve local Git diagnostics without changing any global or repository configuration.
        print(p.stderr.decode('utf-8',errors='replace')[-8000:],file=sys.stderr)
        p.check_returncode()
    return p.stdout

def digest_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()

def main():
    commit=git('rev-parse',sys.argv[1]+'^{commit}').decode().strip()
    out=(ROOT/'outputs/stage09_restore'/f'h_freeze35_20261029_postprocess_{commit[:12]}').resolve()
    out.mkdir(exist_ok=False)
    bundle=out/'delivery.bundle'
    assert git('rev-parse','HEAD').decode().strip()==commit
    git('bundle','create',bundle,'HEAD')
    verification=git('bundle','verify',bundle).decode()
    # Preserve raw receipt bytes even when Git normalizes CRLF on commit.
    # Every source must still match its committed blob except documented CRLF normalization.
    archive=out/'delivery.tar'
    names=git('ls-tree','-r','--name-only','-z',commit,'--',*PATHS).decode().split('\0')
    names=[n for n in names if n]
    members={}
    with tarfile.open(archive,'x:') as tf:
        for name in names:
            source=ROOT/name
            assert source.is_file() and not source.is_symlink()
            raw=source.read_bytes()
            blob=git('show',commit+':'+name)
            assert raw==blob or raw.replace(b'\r\n',b'\n')==blob, name
            members[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'git_blob_sha256':hashlib.sha256(blob).hexdigest(),'git_normalized_crlf':raw!=blob}
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644
            tf.addfile(info,io.BytesIO(raw))
        manifest={'format':'freeze35_postprocess_archive_v1','content_commit':commit,
            'paths':PATHS,'members':members,'file_count':len(members),
            'scope':'Raw task records, reports, statistics, code helpers and handoff bound to the committed tree, allowing only Git CRLF normalization. Runtime models and external dependencies are separate. Closure receipt is excluded.'}
        manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        info=tarfile.TarInfo('POSTPROCESS_MANIFEST.json');info.size=len(manifest_bytes);info.mode=0o644
        tf.addfile(info,io.BytesIO(manifest_bytes))
    extraction=out/'extracted'
    extraction.mkdir()
    extracted=set()
    with tarfile.open(archive,'r:') as tf:
        for m in tf:
            pos=PurePosixPath(m.name)
            assert not pos.is_absolute() and '..' not in pos.parts and '\\' not in m.name
            if m.isdir(): continue
            assert m.isfile() and not m.issym() and not m.islnk()
            assert m.name=='POSTPROCESS_MANIFEST.json' or m.name in members
            assert m.name not in extracted
            data=tf.extractfile(m).read()
            if m.name=='POSTPROCESS_MANIFEST.json':
                assert data==manifest_bytes
            else:
                assert hashlib.sha256(data).hexdigest()==members[m.name]['sha256']
                assert digest_file(ROOT/m.name)==members[m.name]['sha256']
            dest=extraction/pos
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f: f.write(data)
            h=hashlib.sha256(data).hexdigest()
            assert digest_file(dest)==h
            extracted.add(m.name)
    assert extracted==set(members)|{'POSTPROCESS_MANIFEST.json'}
    clone=out/'git_restored'
    git('clone','--no-checkout',bundle,clone)
    # Restore the bound commit using Git's own checkout; no branch or existing worktree is touched.
    git('checkout','--detach',commit,cwd=clone)
    assert git('rev-parse','HEAD',cwd=clone).decode().strip()==commit
    git('fsck','--no-dangling',cwd=clone)
    for name,meta in members.items():
        # Windows checkout line-ending policy may differ; compare Git blobs and separately record disk bytes.
        blob=git('show','HEAD:'+name,cwd=clone)
        assert hashlib.sha256(blob).hexdigest()==meta['git_blob_sha256']
        assert (clone/name).is_file()
        assert digest_file(ROOT/name)==meta['sha256']
    manifest_path=out/'MANIFEST.json'
    manifest_path.write_bytes(manifest_bytes)
    receipt={'status':'postprocess_archive_and_git_restore_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'content_commit':commit,'script_sha256':digest_file(__file__),
        'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest_file(archive)},
        'git_bundle':{'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':digest_file(bundle)},
        'manifest':{'path':str(manifest_path),'sha256':digest_file(manifest_path)},
        'extraction':str(extraction),'git_restore':str(clone),'verified_files':len(members),
        'fresh_tar_members_and_disk_verified':True,'git_bundle_verified':True,'fresh_git_clone_fsck_passed':True,
        'git_blob_bytes_verified':True,'raw_source_hashes_unchanged_before_after':True,
        'windows_long_paths':'Per-command git -c core.longpaths=true; global/repository config unchanged',
        'git_crlf_normalized_files':[n for n,d in members.items() if d['git_normalized_crlf']],
        'note':'Tar and manifest preserve original raw receipt bytes. Git bundle preserves committed canonical blobs; explicit per-file hashes distinguish CRLF normalization. Runtime models/data are separate and never repackaged by this helper.'}
    (out/'VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'receipt':str(out/'VERIFICATION.json'),'receipt_sha256':digest_file(out/'VERIFICATION.json'),'files':len(members)}))

if __name__=='__main__': main()
