"""Read-only acceptance of completed R14-G01; no training or inference."""
import hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
os.environ['DIFFGRM_ENGINE_ROOT']=str(ROOT/'code/external/DiffGRM')
from audit_recovery_support import audit_recovery_predictions
from audit_analysis_core import audit_predictions
from scripts.listwise_diffgrm.run import phase_indices
from scripts.listwise_diffgrm.training import phase_plan

def read(path): return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False))

def main():
    out=ROOT/'runs/R14-G01'
    assert read(ROOT/'jobs/result.json')['status']=='training_validation_test_complete'
    assert read(ROOT/'CHECKPOINT_RESTORE.json')['status']=='passed'
    assert read(ROOT/'GPU_GATE.json')['status']=='passed'
    finish=read(out/'complete.json')
    assert (finish['r_updates'],finish['t_updates'],finish['phase'])==(13860,200,25)
    inputs=read(ROOT/'INPUTS.json')
    config=read(ROOT.parent/'goal_20261027_r1/runs/H1/resolved_config.json')
    plan=phase_plan(config,709443)
    assert len(plan)==45 and sum(p['updates'] for p in plan if p['kind']=='T')==400
    log=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    cursor=0; r=9702; t=120
    for phase in range(17,25):
        row=plan[phase]
        order=phase_indices(config,709443,phase,row)
        for update in range(1,row['updates']+1):
            event=log[cursor]; cursor+=1
            r+=row['kind']=='R'; t+=row['kind']=='T'
            assert (event['phase_index'],event['update'],event['kind'],event['r_updates'],event['t_updates'])==(phase,update,row['kind'],r,t)
            batch=config['training']['effective_batch_size']
            selected=order[(update-1)*batch:update*batch]
            assert hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()==event['sample_indices_sha256']
            assert event['mixer_gate']['gamma']==0.1
            assert event['learning_rates_used']['codebook']==config['training']['codebook_lr']
            assert event['learning_rates_used']['mixer']==config['training']['mixer_lr']
    assert cursor==len(log)==4238 and (r,t)==(13860,200)
    dest=ROOT/'acceptance'; dest.mkdir(exist_ok=True)
    manifest=read(ROOT/'validation_subset_8192.json')
    audits={}; comparisons=[]
    for label in ('after_T160','R15','R16','R17','after_T200','R18','R19','R20'):
        receipt=read(out/f'evaluations/{label}/complete.json')
        path=out/f'evaluations/{label}'/receipt['attempt']/'per_sample.jsonl.gz'
        audit=audit_recovery_predictions(path,dataset_dir=inputs['dataset'],subset_manifest=manifest)
        assert audit['passed'] and audit['source_labels_verified'],label
        write(dest/f'{label}_audit.json',audit)
        audits[label]=dict(examples=audit['examples'],ndcg5=audit['metrics']['list/ndcg@5'],passed=True)
    for epoch in (17,20):
        record=read(out/f'validation_records/R{epoch}.json')
        assert digest(record['checkpoint'])==record['checkpoint_sha256']
        original=read(ROOT.parent/f'goal_20261027_r1/runs/H1/validation_R{epoch}/summary.json')
        baseline=original['metrics']['list/ndcg@5']; value=audits[f'R{epoch}']['ndcg5']
        comparisons.append(dict(epoch=epoch,original_ndcg5=baseline,treatment_ndcg5=value,relative_change=value/baseline-1))
    test=out/'tests/R20/per_sample.jsonl.gz'
    audit=audit_predictions(test,dataset_dir=inputs['dataset'])
    assert audit['passed'] and audit['examples']==53194
    write(dest/'R20_test_audit.json',audit)
    test_receipt=read(out/'test_predictions_complete.json')
    assert len(test_receipt['results'])==1 and test_receipt['results'][0]['epoch']==20
    assert test_receipt['results'][0]['predictions_sha256']==digest(test)
    baseline_path=ROOT/'BASELINE_R20_TEST.json'
    assert digest(baseline_path)=='2e588690946d14d6c406acc0f00612c9c631c65aaa94adb201eec351afeae1e0'
    baseline=read(baseline_path)
    current_test=read(test.parent/'summary.json')
    for key in ('dataset_binding','metric_protocol','beam','inference_precision','examples'):
        assert baseline[key]==current_test[key],key
    original_test=baseline['metrics']['list/ndcg@5']
    source=read(ROOT/'SOURCE_MANIFEST.json')
    for name,sha in source.items(): assert digest(ROOT/name)==sha,name
    artifacts={}
    for directory in (out,ROOT/'jobs'):
        for path in directory.rglob('*'):
            if path.is_file(): artifacts[str(path.relative_to(ROOT))]=dict(bytes=path.stat().st_size,sha256=digest(path))
    write(dest/'ARTIFACT_MANIFEST.json',artifacts)
    result=dict(status='passed',updates=dict(R_added=4158,T_added=80,R_total=13860,T_total=200),
                sample_order_verified=True,source_unchanged=True,validation=audits,
                validation_comparisons=comparisons,test_ndcg5=audit['metrics']['list/ndcg@5'],
                test_comparison=dict(original_ndcg5=original_test,treatment_ndcg5=audit['metrics']['list/ndcg@5'],relative_change=audit['metrics']['list/ndcg@5']/original_test-1),
                test_metrics=audit['metrics'],checkpoint_restore=read(ROOT/'CHECKPOINT_RESTORE.json'),
                catalog={p.stem:read(p) for p in (out/'catalog').glob('*.json')},
                limitation='Single seed. R14 to R20 only; does not establish stability after R20. Resource release checked separately.')
    write(dest/'RESULT.json',result)
    print(json.dumps(dict(status='passed',validation=comparisons,test_ndcg5=result['test_ndcg5'])),flush=True)
if __name__=='__main__': main()
