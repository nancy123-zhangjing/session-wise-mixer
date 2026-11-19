"""Final requirement-by-requirement closure, only after actual CPU shutdown."""
import hashlib
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FRESH=ROOT/'outputs/stage09_spread200_20261031_fresh_r1'
RUN=FRESH/'runs/H1-Spread200-R35'
EXPECTED_PARENT='991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1'
EXPECTED_MANIFEST='c875e203a0c07ecc109b0ee6e89502cdcc31ab4887fab0d4506f4173e30b3f07'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(value,message):
    if not value:raise ValueError(message)

def main():
    target=HERE/'FINAL_DELIVERY_CLOSURE.json'
    require(not target.exists(),'Never overwrite a final closure receipt')
    analysis=read(HERE/'final_analysis/ANALYSIS.json');temporal=read(HERE/'final_temporal/TEMPORAL.json')
    process=read(HERE/'final_process/PROCESS.json');cold=read(HERE/'COLD_RESTORE.json')
    operational=read(HERE/'ASYNC_DELIVERY_AUDIT.json');gpu=read(HERE/'FINAL_GPU_RESOURCE_AUDIT.json')
    review=read(HERE/'final_report_r2/VISUAL_REVIEW_ACCEPTED.json')
    require(all(x['status']=='passed' for x in (analysis,temporal,process,cold)),'Incomplete science/cold restoration')
    require(read(HERE/'CPU_INPUT_ACCEPTANCE.json')['status']=='passed','CPU pairing preflight missing')
    require(read(HERE/'CPU_TESTS.json')['exit_code']==0,'Targeted CPU tests failed')
    pre=read(HERE/'PREFLIGHT_ACCEPTED.json')
    require(pre['status']=='accepted' and pre['parent_sha256']==EXPECTED_PARENT and pre['manifest_sha256']==EXPECTED_MANIFEST,'GPU preflight binding differs')
    require(sha(HERE/'PREFLIGHT_REPORT.json')==pre['report_sha256'],'GPU preflight report changed')
    require(sha(FRESH/'code/SPREAD200_EXECUTION_MANIFEST.json')==EXPECTED_MANIFEST,'Scientific source manifest changed')
    require(cold['parent_sha256']==EXPECTED_PARENT and sha(ROOT/'outputs/stage09_restore/goal_20261027_r1/runs/H1/initial.pt')==EXPECTED_PARENT,'Original shared parent differs')
    require(analysis['updates']==dict(R=24255,T=200,new_logged_updates=20990),'Training endpoint/counters differ')
    require({p.name for p in (FRESH/'runs').iterdir() if p.is_dir()}=={'H1-Spread200-R35'},'Unexpected new scientific branch')
    pair=read(HERE/'ALL_T200_ACCEPTED.json')
    require(pair['T_updates']==200 and pair['T_blocks']==10 and pair['sample_occurrences']==204800,'T pairing coverage incomplete')
    require(pair['all_batch_hashes_exact'] and pair['ordered_stream_matches_original_first200T'] and pair['original_noise_total_steps']==400,'T samples or noise clock differ')
    require(pair['codebook_lr']==1e-5 and pair['mixer_lr']==1e-4 and pair['original_35R_cosine_lr_preserved_at_every_T'],'Learning-rate protocol differs')
    require(len(cold['checkpoints'])==7 and len(cold['all_boundary_checkpoint_fields_and_clocks_verified'])==41,'Cold restoration scope incomplete')
    require(all(all(r['restored_equal'].values()) and r['paired_t_rng_cpu_equal'] for r in cold['checkpoints']),'Actual restored state equality failed')
    require(operational['status']=='passed_async_operational_delivery_audit' and len(operational['readonly_tasks'])==11 and all(r['readonly_state_exact'] for r in operational['readonly_tasks']),'Readonly execution audit incomplete')
    points=list(analysis['points'].values())
    require(sorted(p['epoch'] for p in points if p['split']=='validation')==[8,11,14,17,20,23,26,29,32,35],'Full validation grid differs')
    require(sorted(p['epoch'] for p in points if p['split']=='test')==[23,35] and analysis['selected_epoch']==23,'Test/selection limits differ')
    for label in ('fixed_test','best_test','fixed_validation'):
        comparison=read(HERE/f'final_analysis/comparisons/{label}.json')
        require(comparison['bootstrap']['repeats']==2000 and comparison['bootstrap']['seed']==2024,'Bootstrap budget differs')
        require(comparison['primary_estimand'].startswith('sample mean;') and comparison['bootstrap']['unit'].startswith('user;'),'Bootstrap estimand/unit differs')
        require(comparison['primary']==analysis['comparisons'][label],'Reported comparison not bound to full accepted comparison')
        require(all(math.isfinite(v['delta']) and len(v['confidence_interval'])==2 for v in comparison['primary'].values()),'Invalid paired interval')
    require(process['unique_diagnostic_states']==31 and process['newly_inferred_subset_states']==20 and process['new_full_validation_extractions']==10 and process['historical_initial_reuse']==1,'Process-state coverage differs')
    require(len(process['catalogs'])==11 and process['before_after_catalog_observations']==20,'Catalog coverage differs')
    for catalog in process['catalogs'].values():
        require(len(catalog['per_slot'])==catalog['n_digit']==4,'Incomplete per-slot audit')
        require(catalog['used_codes_per_slot']==[r['used_code_count'] for r in catalog['per_slot']],'Slot usage inconsistent')
        require(all(catalog['independent_snapshot_check'][k] for k in ('state_binding','collision_and_usage','effective_displacement','mixer_correction')),'Catalog independent checks incomplete')
    require(temporal['cutoff_us']==1598745441188500 and temporal['population']==dict(early=38644,late=33013,full=71657),'Time population changed')
    require(temporal['bootstrap']['repeats']==2000 and temporal['bootstrap']['unit']=='user' and temporal['bootstrap']['estimand']=='sample mean','Temporal bootstrap differs')
    require(set(temporal['descriptive_decomposition'])=={'5','20','50'},'Temporal decomposition coverage differs')
    require(all(d['closure_error']<=2e-12 and not d['causal'] and len(d['classes'])==5 for d in temporal['descriptive_decomposition'].values()),'Temporal decomposition invalid')
    require(review['status']=='passed_manual_PNG_visual_review_and_report_table_binding' and len(review['figures'])==4,'Four-figure/table review incomplete')
    require(sha(HERE/'final_report_r2/REPORT.md')==review['report_sha256'],'Final Chinese report changed')
    require(gpu['status']=='passed_all_GPU_attempts_terminal_released_and_within_budget' and len(gpu['jobs'])==5 and gpu['max_concurrent_gpus']==1,'GPU closure incomplete')
    require(gpu['cumulative_budget_upper_seconds']<=86400 and gpu['preflight_upper_seconds']<=1800,'Authorized GPU budget exceeded')
    for name in ('nancy-s09-d-prep-1031-r1','nancy-s09-d-prep-1031-r2'):
        state=read(HERE/f'closure_cpu_r1/{name}_status.json')
        require(state['success'] and state['data']['name']==name and state['data']['status']=='STOPPED','Temporary CPU entrance not confirmed stopped')
    require(read(HERE/'LOCAL_WORK_PROCESSES_CLOSED.json')['matching_owned_process_count']==0,'Owned local work process remains')
    extraction=read(FRESH/'LOCAL_EXTRACTION_VERIFIED.json');archive=read(HERE/'delivery/ARCHIVE_RECEIPT.json')
    require(extraction['status']=='passed' and extraction['files']==3464 and extraction['all_file_hashes_verified'],'Independent extraction incomplete')
    require(sha(HERE/'delivery/stage09_h1_spread200_r1.tar.gz')==archive['archive_sha256']==extraction['archive_sha256'],'Local full model archive changed')
    companion=read(HERE/'delivery/SCIENCE_DELIVERY_RECEIPT_R1.json')
    require(read(HERE/'delivery/SCIENCE_DELIVERY_LOCAL_VERIFIED.json')['archive_sha256']==companion['archive_sha256'],'Science bundle unverified')
    remote=read(HERE/'delivery/SCIENCE_DELIVERY_REMOTE_VERIFIED.json')
    require(remote['status']=='passed_remote_sha_matches_local' and remote['archive_sha256']==companion['archive_sha256'],'Remote science backup incomplete')
    require(sha(HERE/'delivery/science_delivery_r1.tar.gz')==companion['archive_sha256'],'Local science backup changed')
    tool_manifest=read(HERE/'TOOLS_MANIFEST_R3.json')
    for name,record in tool_manifest['files'].items():require(sha(HERE/name)==record['sha256'],'Actual final tool/evidence differs from r3 bundle')
    inputs={}
    for source in (analysis,temporal,process,read(HERE/'final_report_r2/REPORT_BINDING.json')):
        for name,digest in source['input_sha256'].items():
            require(name not in inputs or inputs[name]==digest,'Conflicting scientific source digest')
            inputs[name]=digest
    for name,digest in inputs.items():require(sha(name)==digest,'Scientific input changed after its accepted audit: '+name)
    evidence=['PROTOCOL.md','PARENT_BINDING.json','CPU_INPUT_ACCEPTANCE.json','CPU_TESTS.json','PREFLIGHT_ACCEPTED.json',
        'ALL_T200_ACCEPTED.json','COLD_RESTORE.json','ASYNC_DELIVERY_AUDIT.json','FINAL_GPU_RESOURCE_AUDIT.json',
        'final_analysis/ANALYSIS.json','final_temporal/TEMPORAL.json','final_process/PROCESS.json',
        'final_report_r2/REPORT.md','final_report_r2/VISUAL_REVIEW_ACCEPTED.json','TOOLS_MANIFEST_R3.json',
        'delivery/ARCHIVE_RECEIPT.json','delivery/SCIENCE_DELIVERY_RECEIPT_R1.json','delivery/SCIENCE_DELIVERY_REMOTE_VERIFIED.json',
        'LOCAL_WORK_PROCESSES_CLOSED.json']
    evidence += [str(p.relative_to(HERE)).replace('\\','/') for p in sorted((HERE/'closure_cpu_r1').glob('*.json'))]
    criteria=[
        dict(requirement=1,scope='Frozen protocol/code/config/data/full R5 parent; old evidence preserved',evidence=['PROTOCOL.md','PARENT_BINDING.json','PREFLIGHT_ACCEPTED.json']),
        dict(requirement=2,scope='Only D; reused R5, 10*(20T+3R), strict R35/T200 and original scientific configuration',evidence=['final_analysis/ANALYSIS.json','COLD_RESTORE.json']),
        dict(requirement=3,scope='All ordered paired batches, saved T RNG across R, original 400T noise and 35R LR clocks',evidence=['ALL_T200_ACCEPTED.json','COLD_RESTORE.json','final_analysis/ANALYSIS.json']),
        dict(requirement=4,scope='Targeted CPU and real single-H100 CUDA next-T/R, isolation and safe continuation',evidence=['CPU_TESTS.json','PREFLIGHT_ACCEPTED.json','ASYNC_DELIVERY_AUDIT.json']),
        dict(requirement=5,scope='All five GPU attempts including failure/start/release; <=24h, short tests <=0.5h, max one H100; listed cost vs settlement separated',evidence=['FINAL_GPU_RESOURCE_AUDIT.json']),
        dict(requirement=6,scope='Ten full validations, locked common-grid best, only two new tests, all new metrics and 2000 paired-user bootstrap; historical A/C reused',evidence=['final_analysis/ANALYSIS.json']),
        dict(requirement=7,scope='31 diagnostic states/11 catalog snapshots/4-slot usage and unchanged fixed time/frequency groups; H0 separately historical; no causal claim',evidence=['final_process/PROCESS.json','final_temporal/TEMPORAL.json']),
        dict(requirement=8,scope='Chinese report/4 reviewed curves; full SHA archive/fresh extraction/7 real cold restores/all41 states/science backup; every GPU released and both CPUs stopped',evidence=['COLD_RESTORE.json','final_report_r2/VISUAL_REVIEW_ACCEPTED.json','FINAL_GPU_RESOURCE_AUDIT.json','delivery/SCIENCE_DELIVERY_REMOTE_VERIFIED.json','closure_cpu_r1/'])]
    for row in criteria:row['status']='passed'
    result=dict(status='complete_all_requested_science_delivery_restore_and_resource_checks',goal_complete=True,
        at_unix=time.time(),criteria=criteria,report='final_report_r2/REPORT.md',selected_epoch=23,
        fixed_test=analysis['comparisons']['fixed_test']['list/ndcg@5'],best_test=analysis['comparisons']['best_test']['list/ndcg@5'],
        gpu_hours_upper=gpu['cumulative_gpu_hours_upper'],preflight_seconds_upper=gpu['preflight_upper_seconds'],
        actual_settlement_verified=False,evidence_sha256={name:sha(HERE/name) for name in evidence},
        verified_scientific_input_files=len(inputs),auditor_sha256=sha(Path(__file__)),
        limits=['One seed and post-hoc schedule; user bootstrap is conditional on fixed models.',
            'No MEDA reset, new H0/seed/T-budget branch, old-model inference, Feishu write or external message added.',
            'CPU cold restoration omits applying CUDA RNG; authenticated CUDA states remain intact and CUDA next-step restoration was tested separately.'])
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    lines=['# 本次Goal逐项最终验收','',f'完成时间（Unix）：{result["at_unix"]}。全部8项通过。','',
        '| 项目 | 核验范围 | 结果 |','|---|---|---|']
    lines += [f'| {r["requirement"]} | {r["scope"]} | passed |' for r in criteria]
    lines += ['', '权威中文报告：[REPORT.md](final_report_r2/REPORT.md)。完整证据及SHA：[FINAL_DELIVERY_CLOSURE.json](FINAL_DELIVERY_CLOSURE.json)。',
        '', '本次结果为负，按原协议完成；未为超过基线追加实验。实际点券结算未核实，GPU时间及报价估算与账单已明确区分。','']
    with (HERE/'FINAL_GOAL_ACCEPTANCE.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],goal_complete=True,criteria=len(criteria),gpu_hours_upper=result['gpu_hours_upper'],verified_input_files=len(inputs))))

if __name__=='__main__':main()
