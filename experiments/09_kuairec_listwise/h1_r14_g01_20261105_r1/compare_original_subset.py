"""CPU-only comparison from archived predictions; no model inference."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
manifest=REPO/'experiments/09_kuairec_listwise/h1_mixer_g01_20261104_r1/validation_subset_8192.json'
wanted={r['sample_id']:r['index'] for r in json.loads(manifest.read_text())['rows']}
assert len(wanted)==8192
weights=[1/math.log2(i+2) for i in range(5)]
result={'subset_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'examples':8192,'original':{}}
for epoch in (14,17,20):
    folder=REPO/f'outputs/stage09_restore/goal_20261027_r1/runs/H1/validation_R{epoch}'
    pred=folder/'per_sample.jsonl.gz'
    summary=json.loads((folder/'summary.json').read_text())
    found={}; all_values=[]; error=0.; seen_ids=set()
    with gzip.open(pred,'rt') as stream:
        for line in stream:
            row=json.loads(line); sid=row['sample_id']
            assert sid not in seen_ids
            seen_ids.add(sid)
            target=set(row['targets']); seen=set(); dcg=0.
            for item,w in zip(row['predictions'][0],weights):
                if 1<=item<=10728 and item not in seen and item in target: dcg+=w
                seen.add(item)
            value=dcg/sum(weights)
            error=max(error,abs(value-row['metrics']['list/ndcg@5']))
            all_values.append(value)
            if sid in wanted:
                assert row['dataset_index']==wanted[sid]
                found[sid]=value
    assert set(found)==set(wanted) and len(all_values)==71657
    full=sum(all_values)/len(all_values)
    assert error<1e-6 and abs(full-summary['metrics']['list/ndcg@5'])<1e-6
    result['original'][f'R{epoch}']=dict(subset_ndcg5=sum(found.values())/8192,full_ndcg5=full,
        subset_hit_samples=sum(v>0 for v in found.values()),r_updates=summary['r_updates'],t_updates=summary['t_updates'],
        predictions_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),max_saved_metric_error=error)
result['gated']=json.loads((ROOT/'DIAGNOSTICS.json').read_text())['ndcg5']
result['missing_original_nodes']=['after_T160','R15','R16','after_T200','R18','R19']
(ROOT/'ORIGINAL_SUBSET_COMPARISON.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
