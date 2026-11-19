"""CPU-only final audit and three-seed paired descriptive comparisons."""
import json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGE=ROOT.parent
sys.path[:0]=[str(STAGE/'h_freeze35_20261029_r1/code'),str(STAGE/'h1_no_code_tag_20261108_v1')]
METRICS=('list/ndcg@5','list/hr@5','list/recall@5')

def main():
    import torch
    from audit_analysis_core import audit_predictions
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.training import sha256
    rows=[];audits=[];pairs={}
    for seed in (2025,2026):
        pair=[]
        for arm in ('H0','H1'):
            r=ROOT/f'seed{seed}_{arm}'
            complete=json.loads((r/'GPU_COMPLETE.json').read_text())
            train=json.loads((r/'TRAINING_COMPLETE.json').read_text())
            if complete['status']!='passed' or train['R']!=35 or train['T']!=(400 if arm=='H0' else 200): raise ValueError('Incomplete run')
            pair.append(json.loads((r/'R5_PAIR.json').read_text())['hashes'])
            state=torch.load(r/'last.pt',map_location='cpu',weights_only=False)
            if (state['r_updates'],state['t_updates'])!=(24255,train['T']): raise ValueError('Wrong endpoint')
            if arm=='H1':
                actual={k:fingerprint(state[k]) for k in ('alignment','t_optimizer')}
                if actual!=json.loads((r/'FREEZE_R20.json').read_text())['hashes']: raise ValueError('Frozen state drift')
            expected_epochs=(5,8,11,14,17,20,23,26,29,32,35)
            scores={}
            for epoch in expected_epochs:
                receipt=json.loads((r/'checkpoints'/f'R{epoch:02d}.json').read_text())
                if sha256(receipt['checkpoint'])!=receipt['sha256']: raise ValueError('Checkpoint changed')
                summary=json.loads((Path(receipt['validation'])/'summary.json').read_text())
                if summary['examples']!=71657 or not summary['full_split'] or summary['beam']!=32: raise ValueError('Invalid validation')
                scores[epoch]=summary['metrics']['list/ndcg@5']
            if max(scores,key=scores.get)!=train['best_epoch']: raise ValueError('Wrong best selection')
            seen_updates=set();n={'R':0,'T':0}
            for line in (r/'training.jsonl').read_text().splitlines():
                v=json.loads(line);key=(v['phase_index'],v['update'])
                if key in seen_updates: raise ValueError('Duplicate update log')
                seen_updates.add(key);n[v['phase']]+=1
            if n!={'R':24255,'T':train['T']}: raise ValueError('Training log count mismatch')
            for record in complete['tests']:
                if record.get('reused'):
                    src=next(x for x in rows if x['seed']==seed and x['arm']==arm and x['selection']==record['reused'])
                    rows.append(dict(src,selection=record['label']));continue
                audit=audit_predictions(record['predictions'],dataset_dir=STAGE/'goal_20261027_r1/inputs/dataset')
                if not audit['passed'] or not audit['source_labels_verified'] or audit['examples']!=53194: raise ValueError('Test audit failed')
                for key in ('predictions_sha256','summary_sha256'):
                    if audit[key]!=record[key]: raise ValueError('Changed test artifact')
                audits.append(dict(seed=seed,arm=arm,selection=record['label'],passed=True,max_absolute_error=audit['max_absolute_error']))
                rows.append(dict(seed=seed,arm=arm,selection=record['label'],epoch=train['best_epoch'] if record['label']=='best' else 35,metrics={k:record['metrics'][k] for k in METRICS}))
            del state
        if pair[0]!=pair[1]: raise ValueError('R5 pair mismatch')
        pairs[str(seed)]=dict(passed=True,hashes=pair[0])
    if pairs['2025']['hashes']['model']==pairs['2026']['hashes']['model']: raise ValueError('New seeds not independent initializations')
    old=[('H0','best',32,STAGE/'goal_20261027_r1/evaluations/H0_test_beam32/summary.json'),('H0','R35',35,STAGE/'h0_all_nodes_test_20261109_v1/evaluations/test_R35/summary.json'),('H1','best',32,STAGE/'h_freeze35_20261029_r1/runs/H1-Freeze20-R35/evaluations/test_R32/attempt1/summary.json'),('H1','R35',35,STAGE/'h_freeze35_20261029_r1/runs/H1-Freeze20-R35/evaluations/test_R35/attempt1/summary.json')]
    for arm,selection,epoch,path in old:
        v=json.loads(path.read_text())
        if v['split']!='test' or v['examples']!=53194 or v['beam']!=32 or v['inference_precision']!='fp32': raise ValueError('Historical protocol mismatch')
        rows.append(dict(seed=2024,arm=arm,selection=selection,epoch=epoch,source=str(path),source_sha256=sha256(path),metrics={k:v['metrics'][k] for k in METRICS}))
    comparisons=[]
    for selection in ('R35','best'):
        for metric in METRICS:
            by_arm={arm:[next(x['metrics'][metric] for x in rows if x['seed']==seed and x['arm']==arm and x['selection']==selection) for seed in (2024,2025,2026)] for arm in ('H0','H1')}
            deltas=[a-b for a,b in zip(by_arm['H1'],by_arm['H0'])]
            comparisons.append(dict(selection=selection,metric=metric,arms={a:dict(values=v,mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for a,v in by_arm.items()},paired_deltas=deltas,paired_delta_mean=statistics.mean(deltas),paired_delta_sd=statistics.stdev(deltas),positive_seeds=sum(d>0 for d in deltas)))
    result=dict(status='passed',scope='H1 R20 freeze versus H0 continued alternation; whole-protocol comparison',rows=rows,pairs=pairs,audits=audits,comparisons=comparisons)
    (ROOT/'FINAL_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
