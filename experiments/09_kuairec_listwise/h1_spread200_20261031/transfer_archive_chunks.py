"""Bounded-memory archive split, authenticated prefix reuse and assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def verified(path,row):
    if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
        raise ValueError('Chunk bytes mismatch: '+str(path))
def manifest(path):
    m=read(path);offset=0
    for i,row in enumerate(m['chunks']):
        if row['index']!=i or row['name']!=f'chunk_{i:04d}.bin' or row['offset']!=offset:
            raise ValueError('Noncontiguous or unsafe chunk manifest')
        if not 0<row['bytes']<=m['chunk_bytes']:raise ValueError('Invalid chunk width')
        offset+=row['bytes']
    if offset!=m['archive_bytes']:raise ValueError('Chunk coverage differs')
    return m
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('split','seed','verify','assemble'))
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--archive',type=Path)
    p.add_argument('--partial',type=Path)
    p.add_argument('--index',type=int)
    a=p.parse_args();d=a.directory
    if a.command=='split':
        expected=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1/delivery')
        if a.archive.resolve()!=expected/'stage09_h1_spread200_r1.tar.gz' or d.resolve()!=expected/'transfer_chunks_r1':
            raise ValueError('Only the owned sealed archive may be split')
        receipt=read(a.receipt)
        if receipt['status']!='sealed' or a.archive.stat().st_size!=receipt['archive_bytes']:raise ValueError('Sealed archive required')
        d.mkdir(exist_ok=True);rows=[];total=hashlib.sha256();offset=0;width=64*1024**2
        with a.archive.open('rb') as f:
            for i,block in enumerate(iter(lambda:f.read(width),b'')):
                total.update(block)
                row=dict(index=i,name=f'chunk_{i:04d}.bin',offset=offset,bytes=len(block),sha256=hashlib.sha256(block).hexdigest())
                path=d/row['name']
                if path.exists():verified(path,row)
                else:
                    with path.open('xb') as output:output.write(block)
                rows.append(row);offset+=len(block)
        if total.hexdigest()!=receipt['archive_sha256']:raise ValueError('Sealed archive SHA changed')
        m=dict(status='split_full_archive_sha_verified',archive_bytes=offset,archive_sha256=total.hexdigest(),chunk_bytes=width,chunks=rows,splitter_sha256=sha(Path(__file__)))
        target=d/'MANIFEST.json'
        if target.exists():
            if read(target)!=m:raise ValueError('Existing chunk manifest differs')
        else:write(target,m)
        print(json.dumps(dict(status=m['status'],chunks=len(rows),archive_sha256=m['archive_sha256'],manifest_sha256=sha(target))),flush=True)
        return
    m=manifest(a.manifest)
    if a.command=='verify':
        row=m['chunks'][a.index];verified(d/row['name'],row)
        print(json.dumps(dict(status='chunk_sha_verified',index=a.index,bytes=row['bytes'])))
    elif a.command=='seed':
        reused=[];length=a.partial.stat().st_size
        with a.partial.open('rb') as source:
            for row in m['chunks']:
                if row['offset']+row['bytes']>length:break
                block=source.read(row['bytes'])
                if hashlib.sha256(block).hexdigest()!=row['sha256']:raise ValueError('Existing prefix differs from sealed archive')
                path=d/row['name']
                if path.exists():verified(path,row)
                else:
                    with path.open('xb') as output:output.write(block)
                reused.append(row['index'])
        result=dict(status='partial_prefix_complete_chunks_authenticated',reused_indices=reused,
            reused_bytes=sum(m['chunks'][i]['bytes'] for i in reused),partial_bytes=length,
            partial_sha256=sha(a.partial),manifest_sha256=sha(a.manifest),unverified_partial_tail_not_reused=True)
        write(d/'PREFIX_REUSE_ACCEPTED.json',result);print(json.dumps(result))
    else:
        receipt=read(a.receipt)
        if m['archive_sha256']!=receipt['archive_sha256'] or m['archive_bytes']!=receipt['archive_bytes']:
            raise ValueError('Chunk manifest not bound to the sealed archive')
        if a.archive.exists():raise FileExistsError('Never overwrite a previous archive')
        if shutil.disk_usage(d).free<m['archive_bytes']+receipt['source_bytes']+2*1024**3:
            raise ValueError('Insufficient room for assembly and independent extraction')
        for row in m['chunks']:verified(d/row['name'],row)
        total=hashlib.sha256()
        with a.archive.open('xb') as output:
            for row in m['chunks']:
                with (d/row['name']).open('rb') as source:
                    for block in iter(lambda:source.read(4*1024**2),b''):total.update(block);output.write(block)
        if total.hexdigest()!=receipt['archive_sha256'] or a.archive.stat().st_size!=receipt['archive_bytes']:
            raise ValueError('Final assembled archive differs from sealed source')
        result=dict(status='all_chunks_and_assembled_archive_sha_verified',chunks=len(m['chunks']),
            archive_bytes=a.archive.stat().st_size,archive_sha256=total.hexdigest(),manifest_sha256=sha(a.manifest),assembler_sha256=sha(Path(__file__)))
        write(d/'ASSEMBLED_ACCEPTED.json',result);print(json.dumps(result))

if __name__=='__main__':main()
