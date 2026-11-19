"""New D/E best temporal aggregation from saved predictions; no model execution."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
P=ROOT/'experiments/09_kuairec_listwise'
OLD=P/'h_freeze35_20261029/cold_popularity_audit_20261030/initial_r1'
INPUTS={}
def bind(path, expected=None):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    digest=h.hexdigest()
    if expected is not None:assert digest==expected,str(path)
    INPUTS[str(path)]=digest
    return path
def read(path,expected=None):return json.loads(bind(path,expected).read_text(encoding='utf-8-sig'))
def write(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def main():
    assert not (HERE/'BEST_TEMPORAL.json').exists(),'Do not repeat completed statistics'
    bind(__file__)
    complete=read(OLD/'COMPLETE.json')
    with np.load(bind(OLD/'shared_arrays.npz',complete['files']['shared_arrays.npz']),allow_pickle=False) as archive:
        shared={key:archive[key] for key in archive.files}
    assert shared['metric_keys'][0]=='list/ndcg@5'
    n=len(shared['users']);assert n==71657
    masks={'early':shared['starts']<1598745441188500,'late':shared['starts']>=1598745441188500}
    assert [int(x.sum()) for x in masks.values()]==[38644,33013]
    values={};models={}
    for name,stem in [('H0_best','H0_best_R32'),('H1_original_best','H1_R20')]:
        ar=np.load(bind(OLD/f'arrays_{stem}.npz',complete['files'][f'arrays_{stem}.npz']),allow_pickle=False)
        values[name]=ar['metric_values'][:,0]
        old=read(OLD/f'MODEL_{stem}.json',complete['files'][f'MODEL_{stem}.json'])
        models[name]={seg:old['segments'][seg]['metrics']['list/ndcg@5'] for seg in ('early','late','full')}
        for seg in masks:assert abs(values[name][masks[seg]].mean()-models[name][seg])<2e-12
    discount=1/np.log2(np.arange(5)+2);idcg=discount.sum()
    for name,folder,label in [('D_best','h1_spread200_20261031','D_R23_validation'),('E_best','h1_block40_r6_single_20261101','E_R29_validation')]:
        accepted=read(P/folder/'final_analysis/ANALYSIS.json');assert accepted['status']=='passed'
        point=accepted['points'][label];assert point['epoch']==accepted['selected_epoch']
        pred=bind(point['paths']['predictions'],point['predictions_sha256'])
        if name=='D_best' and (HERE/'D_best_ndcg5.npy').exists():
            # Earlier interrupted invocation completed and checked all D rows before saving.
            bind(HERE/'analyze_best_initial.py')
            arr=np.load(bind(HERE/'D_best_ndcg5.npy'),allow_pickle=False)
            assert arr.shape==(n,) and abs(arr.mean()-point['metrics']['list/ndcg@5'])<2e-12
            values[name]=arr
            models[name]={seg:float(arr[mask].mean()) for seg,mask in masks.items()}
            models[name]['full']=float(arr.mean())
            print('Reused completed D aggregation',flush=True)
            continue
        arr=np.empty(n,dtype=np.float64);count=0
        with gzip.open(pred,'rt',encoding='utf-8') as f:
            for i,line in enumerate(f):
                row=json.loads(line)
                assert i<n and row['dataset_index']==i and row['sample_id']==shared['sample_id'][i]
                assert row['user_id']==int(shared['users'][i]) and row['targets']==shared['targets'][i].tolist()
                top=row['predictions'][0];assert len(top)==5 and len(set(top))==5 and min(top)>0
                targets=set(row['targets']);arr[i]=sum(discount[j] for j,item in enumerate(top) if item in targets)/idcg
                count+=1
        assert count==n and abs(arr.mean()-point['metrics']['list/ndcg@5'])<2e-12
        values[name]=arr
        models[name]={seg:float(arr[mask].mean()) for seg,mask in masks.items()}
        models[name]['full']=float(arr.mean())
        np.save(HERE/(name+'_ndcg5.npy'),arr,allow_pickle=False)
        print(name,models[name],flush=True)
    # New D/E-to-H0 paired comparisons; same global-user draw across both periods.
    inverse=shared['inverse_users'];counts=shared['user_segment_count'];u=counts.shape[1]
    delta_sums=np.zeros((2,u,2))
    for j,name in enumerate(('D_best','E_best')):
        diff=values[name]-values['H0_best']
        for si,mask in enumerate(masks.values()):np.add.at(delta_sums[si,:,j],inverse[mask],diff[mask])
    draws=np.zeros((2000,2,2));rng=np.random.default_rng(2024)
    for start in range(0,2000,32):
        size=min(32,2000-start);indices=rng.integers(0,u,size=(size,u))
        weights=np.stack([np.bincount(row,minlength=u) for row in indices]).astype(float)
        for si in range(2):draws[start:start+size,si]=(weights@delta_sums[si])/(weights@counts[si])[:,None]
    comparisons={}
    for j,name in enumerate(('D_best','E_best')):
        comparison={}
        for si,seg in enumerate(masks):
            comparison[seg]={'delta':models[name][seg]-models['H0_best'][seg],'relative_delta':models[name][seg]/models['H0_best'][seg]-1,'ci95':np.quantile(draws[:,si,j],[.025,.975]).tolist()}
        comparison['late_minus_early']={'delta':comparison['late']['delta']-comparison['early']['delta'],'ci95':np.quantile(draws[:,1,j]-draws[:,0,j],[.025,.975]).tolist()}
        comparisons[name+'_minus_H0_best']=comparison
    for path,digest in list(INPUTS.items()):bind(path,digest)
    write(HERE/'BEST_TEMPORAL.json',{'status':'passed','scope':'new D_R23/E_R29 best validation slices versus cached H0_R32 best; historical H1_R20 reused','cutoff_us':1598745441188500,'population':{k:int(v.sum()) for k,v in masks.items()},'models':models,'comparisons':comparisons,'bootstrap':{'repeats':2000,'seed':2024,'unit':'user','estimand':'sample mean','draws':'same global validation users jointly across periods'},'inputs_sha256':INPUTS,'limitations':['Validation used for checkpoint selection; not independent confirmatory evidence.','One training seed; unadjusted exploratory intervals.','Different R/T and selection grids; not isolated causal schedule effect.','Static validation slices do not demonstrate adaptation to online new data.'],'no_training_or_inference':True})
    print(json.dumps(comparisons,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
