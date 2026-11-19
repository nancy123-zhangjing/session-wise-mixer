"""Read saved predictions only; reproduce the locked temporal split."""
import gzip,hashlib,json,math
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
CUT=1598745441188500
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
result={'cutoff_us':CUT,'rule':'first target timestamp < cutoff is early','no_training_or_inference':True,'models':{},'inputs_sha256':{}}
weights=[1/math.log2(i+2) for i in range(5)]
dataset=BASE/'goal_20261027_r1/inputs/dataset'
truth={};hashes={}
for split in ('validation','test'):
    p=dataset/(split+'.parquet');hashes[split]=sha(p)
    truth[split]=pq.read_table(p,columns=['sample_id','user_id','target_item_ids','target_timestamp_us']).to_pylist()
for name,folder,run in [('R14_fixed01','h1_r14_g01_20261105_r1','R14-G01'),('R17_release1','h1_r17_release1_20261105_r1','R17-Release1')]:
    metrics={}
    for split in ('validation','test'):
        root=BASE/folder/'runs'/run
        path=root/('evaluations/R20/attempt1' if split=='validation' else 'tests/R20')
        summary=json.loads((path/'summary.json').read_text())
        assert summary['dataset_binding']['files'][split+'.parquet']['sha256']==hashes[split]
        sums={'early':0.,'late':0.,'full':0.};counts={k:0 for k in sums};maxerr=0.
        pred=path/'per_sample.jsonl.gz';before=sha(pred)
        with gzip.open(pred,'rt') as f:
            for i,line in enumerate(f):
                row=json.loads(line);target=truth[split][i]
                assert row['dataset_index']==i and row['sample_id']==target['sample_id'] and row['user_id']==target['user_id'] and row['targets']==target['target_item_ids']
                seen=set();dcg=0
                for item,w in zip(row['predictions'][0],weights):
                    if 1<=item<=10728 and item not in seen and item in target['target_item_ids']:dcg+=w
                    seen.add(item)
                value=dcg/sum(weights);maxerr=max(maxerr,abs(value-row['metrics']['list/ndcg@5']))
                seg='early' if target['target_timestamp_us'][0]<CUT else 'late'
                for k in (seg,'full'):sums[k]+=value;counts[k]+=1
        assert counts['full']==len(truth[split])==summary['examples']
        assert maxerr<1e-6 and abs(sums['full']/counts['full']-summary['metrics']['list/ndcg@5'])<1e-6
        assert before==sha(pred)
        result['inputs_sha256'][str(pred)]=before
        if split=='validation':
            assert counts['early']==38644 and counts['late']==33013
            metrics.update({k:sums[k]/counts[k] for k in sums})
        else:metrics['test']=sums['full']/counts['full']
    result['models'][name]=metrics
result['status']='passed'
(ROOT/'TEMPORAL_R20.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
