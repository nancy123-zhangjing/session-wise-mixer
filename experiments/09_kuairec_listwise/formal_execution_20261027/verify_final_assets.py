"""Recheck the delivered files, source repositories, predictions and release ledger."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
F=Path(__file__).resolve().parent
D=ROOT/'outputs/stage09_delivery/goal_20261027_r1'
R=ROOT/'outputs/stage09_restore/goal_20261027_r1'
A=F.parent/'final_analysis_20261027'
checks=[]
def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(p,entry):
    need(p.is_file() and not p.is_symlink(),f'Missing or linked file: {p}')
    need('bytes' not in entry or p.stat().st_size==entry['bytes'],f'Wrong size: {p}')
    actual=sha(p);need(actual==entry['sha256'],f'Wrong SHA: {p}')
    checks.append(dict(path=str(p),bytes=p.stat().st_size,sha256=actual))
def passed(p):
    v=read(p);need(v.get('status','').lower() in ('pass','passed') or v.get('passed') is True,f'Failed evidence: {p}')
    return v

packages=[]
for run in ('warmup_J','warmup_H','J0','J1','H0','H1'):
    mp=D/f'STAGE09_RUN_{run}_R1.json';m=passed(mp)
    rp=(F/f'{run[-1]}_WARMUP_LOCAL_RESTORE.json') if run.startswith('warmup') else F/f'RESTORE_{run}.json'
    rec=passed(rp);need(sha(mp)==rec['manifest_sha256'],'Run manifest receipt differs')
    verify(D/Path(m['archive']['path']).name,m['archive'])
    need(len(m['files'])==m['file_count'],'Run file count')
    for e in m['files']:verify(R/e['path'],e)
    packages.append(dict(kind='model_run',run=run,manifest=str(mp),manifest_sha256=sha(mp),archive=m['archive'],restore_receipt=str(rp)))

asset_receipt=F.parent/'assets_preparation_20261027/GENERATED_ASSETS_LOCAL_RESTORE.json'
assets=passed(asset_receipt)
verify(Path(assets['archive']['path']),assets['archive'])
verify(Path(assets['source_manifest']['path']),assets['source_manifest'])
for e in assets['files']:verify(R/e['path'],e)
packages.append(dict(kind='generated_assets',archive=assets['archive'],restore_receipt=str(asset_receipt)))
for scope in ('operations','evaluations'):
    mp=D/f'STAGE09_{scope.upper()}_R1.json';m=passed(mp)
    rp=R/f'sidecars/RESTORE_{scope}.json';rec=passed(rp)
    need(rec['manifest_sha256']==sha(mp),'Sidecar manifest differs')
    verify(D/Path(m['archive']['path']).name,m['archive'])
    for e in m['files']:verify(R/f'sidecars/{scope}'/e['path'],e)
    packages.append(dict(kind='sidecar',scope=scope,manifest=str(mp),manifest_sha256=sha(mp),archive=m['archive'],restore_receipt=str(rp)))

inputs=read(D/'INPUT_TRANSFER_MANIFEST.json')
for e in inputs['files']:
    rel=Path(e['path'])
    if e['path'].startswith('inputs/dataset/'):
        p=ROOT/'outputs/kuairec_listwise_v1/run1'/rel.relative_to('inputs/dataset')
    elif e['path'].startswith('delivery/'):
        p=D/rel.relative_to('delivery')
    else:raise RuntimeError('Unexpected input manifest mapping')
    verify(p,e)
source=passed(F/'LOCAL_SOURCE_RESTORE_ACCEPTANCE.json')
for repo in source['repositories']:
    p=Path(repo['path'])
    head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    clean=subprocess.check_output(['git','-C',str(p),'status','--porcelain','--untracked-files=all'],text=True).strip()
    need(head==repo['head'] and not clean,'Restored frozen source changed')
for e in source['bound_source_files']:verify(Path(e['path']),e)
freeze=passed(F/'FOUR_BEST_FREEZE_20261028.json')
need(freeze['distinct_full_validations']==42 and not freeze['test_used_for_selection'],'Selection protocol')
for arm,v in freeze['arms'].items():
    complete=read(R/f'runs/{arm}/complete.json')
    verify(R/f'runs/{arm}/complete.json',v['complete'])
    need(complete['r_updates']==24255 and complete['t_updates']==400,'Training counts')
    for role in ('initial','best','last'):verify(R/f'runs/{arm}/{role}.pt',v[role])
    need(v['shared_initial_exact'],'Shared start not accepted')
for name in ('J1_CPU_RESTORE_20261028.json','H1_CPU_RESTORE_20261028.json','J_WARMUP_CPU_ACCEPTANCE.json','H_WARMUP_CPU_ACCEPTANCE.json','CHECKPOINT_SID_DIAGNOSTICS_20261028.json','CHECKPOINT_GEOMETRY_CHECK_20261028.json'):
    passed(F/name)

metric_audits=[]
for arm in ('J0','J1','H0','H1'):
    for variant in ('test_beam32','validation_subset_beam32','validation_subset_beam256'):
        p=A/f'{arm}_{variant}_audit.json';v=passed(p)
        n=53194 if variant=='test_beam32' else 1024
        need(v['examples']==n and v['mismatches']==0 and v['source_labels_verified'] and v['source_cold_groups_verified'],'Metric audit')
        folder=R/'sidecars/evaluations/evaluations'/(arm+'_'+variant+('_r2' if arm=='H1' else ''))
        verify(folder/'summary.json',dict(sha256=v['summary_sha256']))
        verify(folder/'per_sample.jsonl.gz',dict(sha256=v['predictions_sha256']))
        metric_audits.append(dict(path=str(p),sha256=sha(p),examples=n))
for x in read(A/'full_validation_selected_audits.json')['rows']:
    p=Path(x['audit_path']);verify(p,dict(sha256=x['audit_sha256']));v=passed(p)
    need(v['examples']==71657 and v['mismatches']==0,'Full validation replay')
for arch in ('J','H'):
    for variant in ('test_beam32','validation_subset_beam32','validation_subset_beam256'):
        v=read(A/f'{arch}_{variant}_paired_users_10000.json')
        need(v['bootstrap']['repeats']==10000 and v['bootstrap']['confidence']==.95,'Bootstrap configuration')
        for label,arm in [('baseline',arch+'0'),('treatment',arch+'1')]:
            audit=read(A/f'{arm}_{variant}_audit.json')
            need(v['inputs'][label+'_sha256']==audit['predictions_sha256'],'Bootstrap input differs')

release=passed(F/'ALL_GPU_RELEASE_ACCOUNTING_20261028.json')
need(release['all_gpu_released'] and release['all_final_evaluations_succeeded'] and release['job_count']==13,'Resource closure')
verify(Path(release['ledger_path']),dict(sha256=release['ledger_sha256']))
seconds=sum(j['allocated_gpu_seconds'] for j in release['jobs'])
need(seconds==148316 and seconds==release['allocated_gpu_seconds'] and seconds/3600<75,'Budget calculation')
for j in release['jobs']:need(j['resource_released'],'Unreleased job')
for p in ('training_validation_curves.png','validation_curve.csv','best_and_R35_validation.csv','training_phase_loss.csv','curve_sources.json'):
    need((A/'curves'/p).is_file(),'Missing curve artifact')

out=dict(schema='stage09_final_local_asset_verification_v1',status='PASS',created_at_utc=datetime.now(timezone.utc).isoformat(),
    remote_root='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1',
    archive_root=str(D),restore_root=str(R),packages=packages,files_verified=len(checks),verified_files=checks,
    repositories=source['repositories'],final_metric_audits=metric_audits,full_validation_roles=8,paired_user_bootstraps=6,
    all_gpu_released=True,gpu_seconds=seconds,gpu_hours=seconds/3600,settlement_verified=False,
    check_script_sha256=sha(Path(__file__)),limitations=['Single training seed; no new tuning.',
    'Cost is allocated GPU time times verified price, not billing settlement.',
    'Historical source receipt manifests retain their original paths; DEPLOYMENT binds global_user relocation.'])
with (F/'FINAL_LOCAL_ASSET_VERIFICATION_20261028.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',files_verified=len(checks),packages=len(packages),gpu_hours=seconds/3600)),flush=True)
