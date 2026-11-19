"""Seal/verify a small science-result companion to the full model archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','verify'))
    a=p.parse_args();d=HERE/'delivery'
    archive=d/'science_delivery_r1.tar.gz';manifest=d/'SCIENCE_DELIVERY_MANIFEST_R1.json';receipt=d/'SCIENCE_DELIVERY_RECEIPT_R1.json'
    if a.command=='build':
        if any(path.exists() for path in (archive,manifest,receipt)):raise FileExistsError('Preserve previous result bundles')
        for folder,key in (('final_analysis','ANALYSIS.json'),('final_temporal','TEMPORAL.json'),('final_process','PROCESS.json')):
            if read(HERE/folder/key)['status']!='passed':raise ValueError('Passed science audits required')
        if read(HERE/'final_report/VISUAL_REVIEW_ACCEPTED.json')['status']!='passed_manual_PNG_visual_review_and_report_table_binding':
            raise ValueError('Actual visual/table review required')
        files={}
        for folder in ('final_analysis','final_temporal','final_process','final_report','closure_platform_r1'):
            for path in sorted((HERE/folder).rglob('*')):
                if path.is_symlink():raise ValueError('No symlinks')
                if path.is_file():files[path.relative_to(HERE).as_posix()]=path
        names=('COLD_RESTORE.json','OPERATIONAL_AUDIT.json','GPU_RELEASE_AND_BUDGET.json','GPU_RELEASE_RECHECK.json',
            'RECOVERY.md','PROTOCOL.md','AUTHORIZATION.json','INPUTS.json','CPU_SCHEDULE_CHECK.json','RUN_GATE.json',
            'FIRST_R_BOUNDARY_OBSERVED.json','FIRST_T_READONLY_OBSERVED.json',
            'EIGHT_GPU_CODE_DELETION.json','REMOTE_EIGHT_GPU_RETIREMENT.json',
            'delivery/ARCHIVE_RECEIPT.json','delivery/DELIVERY_MANIFEST.json','delivery/RESUMED_DOWNLOAD_ACCEPTED.json',
            'delivery/transfer_chunks_8m_r2/MANIFEST.json',
            'FEISHU_DELIVERY_VERIFIED.json','COMPARISON_DOCUMENT_PLAN.md',
            'analyze_e.py','process_e.py','temporal_e.py','cold_restore.py','audit_operational.py',
            'archive_delivery.py','report_e.py','build_comparison_doc.py','pack_science_delivery.py','finish_local_acceptance.py')
        files.update({name:HERE/name for name in names})
        for path in sorted(HERE.glob('ACCEPTANCE_*.log')):files[path.name]=path
        files['MODEL_ARCHIVE_EXTRACTION_VERIFIED.json']=HERE.parents[2]/'outputs/stage09_block40_single_20261101_fresh_r1/LOCAL_EXTRACTION_VERIFIED.json'
        doc=read(HERE/'FEISHU_DELIVERY_VERIFIED.json')
        if doc['status']!='passed_content_tables_images_readback':raise ValueError('Verified Feishu document required')
        work=HERE.parents[2]/'draft_3cf6198c_folder'
        for path in sorted(work.iterdir()):
            if path.is_file():files['feishu_source/'+path.name]=path
        files['prior_E_gpu_ledger.json']=HERE.parent/'h1_block40_r6_8gpu_20261101/GPU_LEDGER.json'
        metadata={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in sorted(files.items())}
        write(manifest,dict(status='science_result_files_sealed',files=metadata,
            model_archive=read(d/'ARCHIVE_RECEIPT.json'),
            external_dependency_guide='RECOVERY.md; original parent, dataset, PSE and accepted A/D/C/H0 comparisons stay at the explicitly authenticated existing paths.',
            boundary='Science, restoration and GPU-release evidence. CPU-stop and final Goal receipts are produced after this bundle is uploaded.'))
        with archive.open('xb') as raw:
            with tarfile.open(fileobj=raw,mode='w:gz',compresslevel=6) as tar:
                for name,path in sorted(files.items()):tar.add(path,arcname=name,recursive=False)
                tar.add(manifest,arcname=manifest.name,recursive=False)
        for name,path in files.items():
            if sha(path)!=metadata[name]['sha256']:raise ValueError('Science output changed while sealing')
        write(receipt,dict(status='science_result_bundle_sealed',archive=archive.name,archive_sha256=sha(archive),
            archive_bytes=archive.stat().st_size,manifest_sha256=sha(manifest),files=len(files),packer_sha256=sha(Path(__file__))))
    r=read(receipt);m=read(manifest)
    if sha(archive)!=r['archive_sha256'] or sha(manifest)!=r['manifest_sha256']:raise ValueError('Result bundle SHA differs')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(m['files'])+1:raise ValueError('Result bundle coverage differs')
        for member in members:
            if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:raise ValueError('Unsafe result member')
            content=tar.extractfile(member).read()
            expected=r['manifest_sha256'] if member.name==manifest.name else m['files'][member.name]['sha256']
            if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Result bundle member SHA differs')
    proof=dict(status='passed_science_result_bundle_all_member_hashes',archive_sha256=r['archive_sha256'],
        manifest_sha256=r['manifest_sha256'],files=r['files'],verifier_sha256=sha(Path(__file__)))
    destination=d/'SCIENCE_DELIVERY_LOCAL_VERIFIED.json'
    if destination.exists():
        if read(destination)!=proof:raise ValueError('Existing bundle verification differs')
    else:write(destination,proof)
    print(json.dumps(dict(**r,local_all_members_verified=True)))

if __name__=='__main__':main()
