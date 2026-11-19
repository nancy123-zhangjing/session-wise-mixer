"""Resume the sealed archive in bounded small transfers; never rerun science."""
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from resume_archive_download import HERE,ROOT,D,sha
from transfer_archive_chunks import manifest

FOLDER='transfer_chunks_8m_r2'
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1/delivery/'+FOLDER
CLI=['C:/Users/nancy/AppData/Roaming/uv/tools/inspire-skill/Scripts/python.exe','-X','utf8',str(ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/inspire_compat.py'),'--json','--account','sii-shared','notebook','scp','nancy-s09-e-prep-1101-r1','--workspace','CPU资源空间','--timeout','150','--download']

def emit(**row):
    row['time']=time.time()
    with (D/'TRANSFER_8M_PROGRESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def fetch(row):
    path=D/FOLDER/row['name']
    if path.exists() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']:return path
    child=subprocess.Popen(CLI+[REMOTE+'/'+row['name'],str(path.relative_to(ROOT))],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    started=time.monotonic();verified_before_close=False;full_since=None
    while True:
        try:
            output,err=child.communicate(timeout=10)
            break
        except subprocess.TimeoutExpired:
            # A transport can hang during session shutdown after every byte is
            # present. The sealed source SHA, not process exit alone, proves
            # delivery. Stop only this owned transfer tree and recheck bytes.
            full=path.exists() and path.stat().st_size==row['bytes']
            full_since=(full_since or time.monotonic()) if full else None
            try:ready=full and sha(path)==row['sha256']
            except OSError:ready=False
            # Windows SCP can hold an exclusive file handle until shutdown.
            # Stable complete size permits closing our transport, never acceptance.
            # Acceptance still requires the full source SHA after the handle closes.
            close_locked_full=full_since is not None and time.monotonic()-full_since>=20
            if ready or close_locked_full or time.monotonic()-started>=175:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                output,err=child.communicate()
                if not (ready or close_locked_full):raise RuntimeError('Small chunk transfer timeout; preserve partial archive')
                if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Chunk changed while closing transport')
                verified_before_close=True
                emit(event='complete_chunk_sha_verified_transport_closed',index=row['index'],transport_exit=child.returncode)
                break
    if child.returncode and not verified_before_close:
        detail=re.sub(r'(?:wss?|https?)://\S+','<redacted-url>',output+'\n'+err)
        emit(event='transfer_failed',index=row['index'],returncode=child.returncode,detail=detail[-3000:])
        raise RuntimeError('Small chunk transport failed')
    if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Small chunk SHA mismatch')
    return path

def main():
    mpath=D/FOLDER/'MANIFEST.json'
    if sha(mpath)!='de5cc1f59674b275c4d03c2e953362a6f0ac39ed6223438b6cef07b3f7908da2':raise ValueError('Unbound small-chunk manifest')
    m=manifest(mpath);receipt=json.loads((D/'ARCHIVE_RECEIPT.json').read_text())
    if (m['archive_sha256'],m['archive_bytes'])!=(receipt['archive_sha256'],receipt['archive_bytes']):raise ValueError('Archive binding differs')
    archive=D/receipt['archive'];initial=archive.stat().st_size
    if initial>m['archive_bytes']:raise ValueError('Oversized archive')
    # Salvage authenticated pieces of the interrupted 64 MiB scratch transfer.
    old=D/'transfer_chunks_r1/chunk_0015.bin';old_offset=15*64*1024**2
    if old.exists():
        with old.open('rb') as f:
            for row in m['chunks']:
                pos=row['offset']-old_offset
                if pos<0 or pos+row['bytes']>old.stat().st_size or row['offset']+row['bytes']<=initial:continue
                f.seek(pos);data=f.read(row['bytes'])
                if hashlib.sha256(data).hexdigest()!=row['sha256']:continue
                path=D/FOLDER/row['name']
                if not path.exists():path.write_bytes(data)
    remaining=[]
    with archive.open('r+b') as target:
        for row in m['chunks']:
            present=max(0,min(row['bytes'],initial-row['offset']))
            if present==row['bytes']:
                target.seek(row['offset'])
                if hashlib.sha256(target.read(present)).hexdigest()!=row['sha256']:raise ValueError('Existing complete prefix differs')
            else:remaining.append(row)
        emit(event='authenticated_resume_started',pid=os.getpid(),bytes=initial,total=m['archive_bytes'],parallel_transfers=1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for start in range(0,len(remaining)):
                batch=remaining[start:start+1]
                futures=[pool.submit(fetch,row) for row in batch]
                for row,future in zip(batch,futures):
                    part=future.result();length=os.fstat(target.fileno()).st_size
                    if not row['offset']<=length<=row['offset']+row['bytes']:raise ValueError('Noncontiguous append')
                    present=length-row['offset'];target.seek(row['offset']);prefix=target.read(present)
                    with part.open('rb') as source:
                        if source.read(present)!=prefix:raise ValueError('Existing tail differs')
                        target.seek(length);target.write(source.read())
                    target.flush();os.fsync(target.fileno());target.seek(row['offset'])
                    if hashlib.sha256(target.read(row['bytes'])).hexdigest()!=row['sha256']:raise ValueError('Appended chunk differs')
                    part.unlink()
                    emit(event='verified_chunk_appended',index=row['index'],bytes=os.fstat(target.fileno()).st_size,total=m['archive_bytes'])
    if archive.stat().st_size!=m['archive_bytes'] or sha(archive)!=m['archive_sha256']:raise ValueError('Full archive SHA mismatch')
    out=dict(status='passed_authenticated_prefix_resume_and_full_archive',reused_initial_prefix_bytes=initial,archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],manifest_sha256=sha(mpath))
    with (D/'RESUMED_DOWNLOAD_ACCEPTED.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    emit(event='full_archive_accepted',**out)

if __name__=='__main__':main()
