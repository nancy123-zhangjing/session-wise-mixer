"""Resume an E archive using authenticated 64MiB chunks and one scratch file.

Only run after the prior SCP process has exited. Existing complete prefix bytes
are checked, not retransferred. No scientific artifacts or baselines are deleted.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
D=HERE/'delivery'
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1/delivery/transfer_chunks_r1'
CLI=[str(Path('C:/Users/nancy/AppData/Roaming/uv/tools/inspire-skill/Scripts/python.exe')),'-X','utf8',str(ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/inspire_compat.py'),'--json','--account','sii-shared','notebook','scp','nancy-s09-e-prep-1101-r1','--workspace','CPU资源空间','--timeout','600','--download']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    from transfer_archive_chunks import manifest
    m=manifest(D/'transfer_chunks_r1/MANIFEST.json')
    receipt=json.loads((D/'ARCHIVE_RECEIPT.json').read_text(encoding='utf-8'))
    if (m['archive_sha256'],m['archive_bytes'])!=(receipt['archive_sha256'],receipt['archive_bytes']):raise ValueError('Unbound manifest')
    archive=D/receipt['archive']
    if archive.stat().st_size>m['archive_bytes']:raise ValueError('Oversized partial archive')
    scratch=D/'transfer_chunks_r1';scratch.mkdir(exist_ok=True)
    reused=0
    with archive.open('r+b') as target:
        for row in m['chunks']:
            length=os.fstat(target.fileno()).st_size
            present=max(0,min(row['bytes'],length-row['offset']))
            target.seek(row['offset']);prefix=target.read(present)
            if present==row['bytes']:
                if hashlib.sha256(prefix).hexdigest()!=row['sha256']:raise ValueError('Existing prefix chunk differs')
                reused+=present
                continue
            part=scratch/row['name']
            if not (part.exists() and part.stat().st_size==row['bytes'] and sha(part)==row['sha256']):
                child=subprocess.Popen(CLI+[REMOTE+'/'+row['name'],str(part)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
                try:
                    output,_=child.communicate(timeout=630)
                except subprocess.TimeoutExpired:
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    else:child.kill()
                    child.communicate()
                    raise RuntimeError('Chunk transport timed out and was stopped; preserve verified prefix')
                if child.returncode:
                    print(output,flush=True)
                    raise RuntimeError('Chunk transfer failed; keep existing verified archive prefix')
            if part.stat().st_size!=row['bytes'] or sha(part)!=row['sha256']:raise ValueError('Downloaded chunk differs')
            with part.open('rb') as source:
                if source.read(present)!=prefix:raise ValueError('Partial chunk prefix differs; preserve and investigate')
                target.seek(row['offset']+present)
                for block in iter(lambda:source.read(4*1024*1024),b''):target.write(block)
            target.flush();os.fsync(target.fileno())
            target.seek(row['offset'])
            if hashlib.sha256(target.read(row['bytes'])).hexdigest()!=row['sha256']:raise ValueError('Appended chunk differs')
            # Only this verified, disposable transfer chunk is removed.
            part.unlink()
            print(json.dumps(dict(event='verified_chunk_appended',index=row['index'],bytes=os.fstat(target.fileno()).st_size,total=m['archive_bytes'])),flush=True)
    if archive.stat().st_size!=m['archive_bytes'] or sha(archive)!=m['archive_sha256']:raise ValueError('Final full-archive SHA differs')
    out=dict(status='passed_authenticated_prefix_resume_and_full_archive',reused_complete_prefix_bytes=reused,archive_bytes=archive.stat().st_size,archive_sha256=m['archive_sha256'],manifest_sha256=sha(D/'transfer_chunks_r1/MANIFEST.json'))
    with (D/'RESUMED_DOWNLOAD_ACCEPTED.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
