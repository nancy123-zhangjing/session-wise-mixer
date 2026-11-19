"""One targeted CPU check: original orders/seeds, E counts, unchanged delivered science; no model updates."""
import hashlib,json,math,py_compile,sys,tarfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=HERE/'code'
D=ROOT/'outputs/stage09_spread200_20261031_fresh_r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(CODE))
from scripts.listwise_diffgrm.block40_single import e_plan,original_phase,paired_indices,index_sha
from scripts.listwise_diffgrm.spread200 import paired_indices as d_indices
from scripts.listwise_diffgrm.training import phase_plan,expected_update_counts
from scripts.listwise_diffgrm.run import phase_indices
parent=json.loads((ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/PARENT_BINDING.json').read_text(encoding='utf-8'))
cfg=parent['binding']['config'];count=709443
original=phase_plan(cfg,count);plan=e_plan(cfg,count)
assert expected_update_counts(plan,dict(phase=len(plan),update=0))=={'R':24255,'T':200}
assert len(plan[5:])==35
assert [r['epoch'] for r in plan[5:] if r['kind']=='T']==[5,11,17,23,29]
rows=[]
dcfg=json.loads(json.dumps(cfg));dcfg['training']['t_updates_per_cycle']=20
for phase,row in enumerate(plan[5:],5):
    indices=paired_indices(cfg,count,phase,row)
    refphase=next(i for i,r in enumerate(original) if
        (r['kind']==row['kind'] and (r['epoch']==row['epoch'] if row['kind']=='R' else r['cycle']==row['cycle'])))
    assert original_phase(row)==refphase
    assert np.array_equal(indices,phase_indices(cfg,count,refphase,original[refphase]))
    if row['kind']=='T':
        pair=row['cycle']-1
        halves=[d_indices(dcfg,count,5+4*(2*pair+h),dict(kind='T',updates=20)) for h in (0,1)]
        assert np.array_equal(indices,np.concatenate(halves))
    rows.append(dict(phase=phase,kind=row['kind'],epoch=row['epoch'],original_phase=refphase,
        indices_sha256=index_sha(indices),original_phase_seed=cfg['seed']+100003*(refphase+1)))
unchanged=[]
for p in (D/'code').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
        rel=p.relative_to(D/'code');assert sha(p)==sha(CODE/rel),str(rel);unchanged.append(str(rel))
for p in [CODE/'scripts/listwise_diffgrm/block40_single.py',CODE/'scripts/listwise_diffgrm/block40_single_run.py',HERE/'runtime.py',HERE/'job.py']:
    py_compile.compile(str(p),doraise=True)
record=dict(status='passed_schedule_and_unchanged_science',r_updates=24255,t_updates=200,
    noise_clock_total=400,original_config_unchanged=True,original_t_and_r_samples_and_phase_seeds_exact=True,
    phase_orders=rows,unchanged_delivered_files=len(unchanged),new_gpu_test=False,
    reused_single_gpu_acceptance='D final delivered training/update/resume/readonly implementation',
    limitation='New E schedule checked here; runtime first T/R boundaries still require observation')
(HERE/'CPU_SCHEDULE_CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
files={p.relative_to(CODE).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size)
       for p in sorted(CODE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='E_EXECUTION_MANIFEST.json'}
(CODE/'E_EXECUTION_MANIFEST.json').write_text(json.dumps(dict(files=files,parent_sha256=parent['sha256']),indent=2)+'\n',encoding='utf-8')
prior=3287
historical=json.loads((ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/FINAL_GPU_RESOURCE_AUDIT.json').read_text())
forecast=historical['cumulative_budget_upper_seconds']*1.35+1800
assert forecast<72000 and prior+72000+180<86400
gate=dict(status='accepted_single_e_runtime',runtime_sha256=sha(HERE/'runtime.py'),parent_sha256=parent['sha256'],
    scientific_manifest_sha256=sha(CODE/'E_EXECUTION_MANIFEST.json'),schedule_check_sha256=sha(HERE/'CPU_SCHEDULE_CHECK.json'),
    consumed_gpu_seconds_upper_bound=prior,platform_max_seconds=72000,inner_max_seconds=71820,
    release_allowance_seconds=180,forecast_seconds=forecast,forecast_basis='D measured all-attempt upper bound x1.35 +1800s; same R/T work and fewer diagnostics; update using live E throughput',
    preflight_seconds_upper_bound=prior,low_utilization_stop_seconds=9000,utilization_sample_seconds=30,
    numerical_acceptance_basis='Unchanged D single-GPU scientific and read-only implementation; user replaced the eight-GPU branch and requested minimal checks')
(HERE/'RUN_GATE.json').write_text(json.dumps(gate,indent=2)+'\n',encoding='utf-8')
archive=HERE/'execution.tar.gz'
with tarfile.open(archive,'x:gz') as a:
    for name in files:a.add(CODE/name,arcname='code/'+name)
    a.add(CODE/'E_EXECUTION_MANIFEST.json',arcname='code/E_EXECUTION_MANIFEST.json')
    for name in ('runtime.py','job.py','RUN_GATE.json','CPU_SCHEDULE_CHECK.json','INPUTS.json','validation_subset_8192.json'):
        a.add(HERE/name,arcname=name)
receipt=dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,gate_sha256=sha(HERE/'RUN_GATE.json'),
    launcher_sha256=sha(HERE/'job.py'),runtime_sha256=sha(HERE/'runtime.py'),forecast_hours=forecast/3600,
    prior_gpu_hours=prior/3600,authorized_total_gpu_hours=24,backend='single H100')
(HERE/'DEPLOYMENT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
