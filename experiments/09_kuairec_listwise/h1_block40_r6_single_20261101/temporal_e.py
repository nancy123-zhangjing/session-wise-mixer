"""CPU-only E35 vs accepted A35 and D35 temporal diagnosis; reuse historical arrays.

The old H0 result is a separately labelled historical reference. No old model
statistics are rerun. A35 and D35 temporal statistics and arrays are reused without decoding or replaying their predictions.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
from analyze_e import bind, read, write, sha, INPUTS
OLD=ROOT/'experiments/09_kuairec_listwise/h_freeze35_20261029/cold_popularity_audit_20261030'
LOCKED=OLD/'initial_r1'
GROUPS=['never_seen','history_only','target_head','target_mid','target_tail']
KS=(5,20,50)
DISCOUNT=1/np.log2(np.arange(50)+2)
IDCG=float(DISCOUNT[:5].sum())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    bind(__file__)
    accepted=read(a.analysis/'ANALYSIS.json')
    if accepted['status']!='passed': raise ValueError('Independent E audit is required first')
    completion=read(LOCKED/'COMPLETE.json')
    cache=bind(LOCKED/'shared_arrays.npz','cc14b074a0a0e9bbd7c1049cb1294a55c243958ecd1fce8f9eb36d6656c0cc9b')
    shared=np.load(cache,allow_pickle=False)
    ids,users,targets,starts=[shared[k] for k in ('sample_id','users','targets','starts')]
    fine,target_group,train_count=[shared[k] for k in ('fine_group','target_group','train_count')]
    inverse=shared['inverse_users']; u=len(shared['user_ids']); n=len(ids)
    masks={'early':starts<1598745441188500,'late':starts>=1598745441188500,'full':np.ones(n,dtype=bool)}
    if [int(m.sum()) for m in masks.values()]!=[38644,33013,71657]: raise ValueError('Locked time population differs')
    counts=shared['user_segment_count']
    if not np.array_equal(np.bincount(fine[1:],minlength=5),[1914,799,802,3206,4007]):
        raise ValueError('Locked training groups differ')
    # Reuse only pure functions; the historical main would repeat old statistics.
    spec=importlib.util.spec_from_file_location('_accepted_temporal_functions',bind(OLD/'analyze.py','3270debadddbd528a5b9b423a5b6ccae89246ab1351ac02d27279eca34ea7267'))
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    if list(shared['feature_keys'])!=helper.FEATURE_KEYS: raise ValueError('Feature semantics differ')
    metrics_keys=list(shared['metric_keys'])
    epoint=accepted['points']['E_R35_validation']
    points={'E_Block40_R35':dict(predictions=epoint['paths']['predictions'],sha=epoint['predictions_sha256'],metrics=epoint['metrics'])}
    prior_root=ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031/final_temporal'
    prior=read(prior_root/'TEMPORAL.json','21815f2303417cc3f3f7105ef186152e78113f27e64f4bb2c6c890a933395316')
    if prior['status']!='passed' or prior['cutoff_us']!=1598745441188500: raise ValueError('Historical temporal protocol differs')
    features=[]; user_sums=[]; class_arrays=[]; models={}
    for name,digest in [('A_Freeze_R35','ca0d6deb874ff750f433e528ee12ba5f5832a34ffd07425a8898ecdc50354d62'),('D_Spread_R35','ac0c52ea64e09f902e475adbc62cbe9614c17f7e568c23865e34180961918430')]:
        cached=np.load(bind(prior_root/(name+'_arrays.npz'),digest),allow_pickle=False)
        class_arrays.append(cached['class_ndcg']); features.append(helper.feature_matrix(cached['class_ndcg'])); user_sums.append(cached['user_segment_sums'])
        models[name]=prior['models'][name]
    for name,point in points.items():
        pred5=np.zeros((n,5),dtype=np.int32); pred50=np.zeros((n,50),dtype=np.int32)
        count=0
        with gzip.open(bind(point['predictions'],point['sha']),'rt',encoding='utf-8') as f:
            for i,line in enumerate(f):
                row=json.loads(line)
                if (i>=n or row['dataset_index']!=i or row['sample_id']!=ids[i] or row['user_id']!=int(users[i])
                    or row['targets']!=targets[i].tolist()): raise ValueError('Temporal source sample mismatch')
                pred5[i]=row['predictions'][0]
                ordered=sorted(range(len(row['scores'])),key=lambda j: (-(row['scores'][j] if row['scores'][j] is not None else -math.inf),j))
                collected=[]; seen=set(); used=0
                for j in ordered:
                    if not (row['beam_mask'][j] and row['complete_mask'][j]): continue
                    if used==32: break
                    used+=1
                    for item in row['predictions'][j]:
                        if item not in seen: seen.add(item); collected.append(item)
                        if len(collected)==50: break
                    if len(collected)==50: break
                pred50[i,:len(collected)]=collected; count+=1
        if count!=n: raise ValueError('Temporal prediction count differs')
        rank5=helper.rank_targets(pred5,targets); rank50=helper.rank_targets(pred50,targets)
        gains=np.stack([helper.gains(rank5 if k==5 else rank50,k) for k in KS],axis=1)
        contrib=np.stack([(gains*(target_group==g)[:,None,:]).sum(axis=2)/IDCG for g in range(5)],axis=2)
        feature=helper.feature_matrix(contrib)
        metric_columns=[]
        for ranks,k in [(rank5,5)]+[(rank50,k) for k in (10,20,50)]:
            hit=(ranks>0)&(ranks<=k)
            metric_columns.extend([helper.gains(ranks,k).sum(axis=1)/IDCG,hit.any(axis=1).astype(float),hit.mean(axis=1)])
        values=np.stack(metric_columns,axis=1)
        if max(abs(float(values[:,j].mean())-point['metrics'][key]) for j,key in enumerate(metrics_keys))>2e-12:
            raise ValueError('Temporal metrics do not close to accepted full validation')
        sums=np.zeros((2,u,len(helper.FEATURE_KEYS)))
        for si,seg in enumerate(('early','late')): np.add.at(sums[si],inverse[masks[seg]],feature[masks[seg]])
        features.append(feature); user_sums.append(sums); class_arrays.append(contrib)
        segments={}
        for seg,mask in masks.items():
            classes={}
            for gi,group in enumerate(GROUPS):
                pick=target_group[mask]==gi; occurrences=int(pick.sum()); byk={}
                for ki,k in enumerate(KS):
                    rr=(rank5 if k==5 else rank50)[mask]
                    hits=int(((rr>0)&(rr<=k)&pick).sum()); gain=float((gains[mask,ki]*pick).sum())
                    byk[str(k)]=dict(target_hits=hits,target_hit_rate=hits/occurrences if occurrences else None,
                        ndcg_contribution=gain/(mask.sum()*IDCG))
                classes[group]=dict(target_occurrences=occurrences,metrics=byk)
            concentration={}
            for label,pred in (('list5',pred5),('aggregate50',pred50)):
                flat=pred[mask].reshape(-1); valid=flat[flat>0]; frequency=np.bincount(valid,minlength=len(fine))
                top=sorted(np.flatnonzero(frequency),key=lambda i:(-frequency[i],i))[:10]
                concentration[label]=dict(valid_slots=len(valid),unique_items=int(np.count_nonzero(frequency)),
                    top10_share=float(frequency[top].sum()/len(valid)) if len(valid) else None,
                    class_shares={g:float((fine[valid]==gi).mean()) if len(valid) else None for gi,g in enumerate(GROUPS)},
                    top10_items=[dict(item=int(i),count=int(frequency[i]),group=GROUPS[fine[i]],train_count=int(train_count[i])) for i in top])
            segments[seg]=dict(metrics={key:float(values[mask,j].mean()) for j,key in enumerate(metrics_keys)},
                classes=classes,recommendation=concentration)
        models[name]=segments
        np.savez_compressed(out/(name+'_arrays.npz'),class_ndcg=contrib,metric_values=values,user_segment_sums=sums)
        print(json.dumps(dict(event='new_temporal_model',name=name)),flush=True)
    all_comparisons={}; all_decompositions={}
    for base_index,base_name in enumerate(('A','D')):
        paired_sums=[user_sums[base_index],user_sums[2]]
        paired_features=[features[base_index],features[2]]
        comparisons={}
        for seg,mask in masks.items():
            count=counts.sum(axis=0) if seg=='full' else counts[0 if seg=='early' else 1]
            selected=count>0
            sums=[s.sum(axis=0) if seg=='full' else s[0 if seg=='early' else 1] for s in paired_sums]
            draws=helper.bootstrap([s[selected] for s in sums],count[selected],repeats=2000,seed=2024)
            observed=paired_features[1][mask].mean(axis=0)-paired_features[0][mask].mean(axis=0)
            ci=np.quantile(draws[:,1]-draws[:,0],[.025,.975],axis=0)
            comparisons[seg]={key:dict(delta=float(observed[j]),ci=ci[:,j].tolist()) for j,key in enumerate(helper.FEATURE_KEYS)}
        draws=helper.shift_bootstrap(paired_sums,counts,repeats=2000,seed=2024)
        observed=(paired_features[1][masks['late']]-paired_features[0][masks['late']]).mean(axis=0)-(paired_features[1][masks['early']]-paired_features[0][masks['early']]).mean(axis=0)
        ci=np.quantile(draws[:,1]-draws[:,0],[.025,.975],axis=0)
        comparisons['late_minus_early']={key:dict(delta=float(observed[j]),ci=ci[:,j].tolist()) for j,key in enumerate(helper.FEATURE_KEYS)}
        decomposition={}
        for ki,k in enumerate(KS):
            rows=[]
            for gi,g in enumerate(GROUPS):
                period=[]
                for seg in ('early','late'):
                    mask=masks[seg]; pick=target_group[mask]==gi; num=int(pick.sum())
                    delta=float((class_arrays[2][mask,ki,gi]-class_arrays[base_index][mask,ki,gi]).sum()*IDCG)
                    period.append((num/(mask.sum()*5),delta/num if num else None))
                (we,de),(wl,dl)=period
                de0=de if de is not None else 0.; dl0=dl if dl is not None else 0.
                rows.append(dict(group=g,weight_early=we,weight_late=wl,gain_delta_early=de,gain_delta_late=dl,
                    both_periods_observed=de is not None and dl is not None,
                    composition=.5*(wl-we)*(dl0+de0)*5/IDCG,within=.5*(wl+we)*(dl0-de0)*5/IDCG))
            predicted=sum(r['composition']+r['within'] for r in rows)
            actual=comparisons['late_minus_early'][f'{"list" if k==5 else "aggregate"}/ndcg@{k}/overall']['delta']
            if abs(predicted-actual)>2e-12: raise ValueError('Temporal decomposition does not close')
            decomposition[str(k)]=dict(classes=rows,observed_change=actual,closure_error=abs(predicted-actual),causal=False)
        all_comparisons['E_minus_'+base_name]=comparisons
        all_decompositions['E_minus_'+base_name]=decomposition
    reference=read(LOCKED/'MODEL_H0_best_R32.json','81fa77b849834472e2b62078e35e41e60627feca935df751fc6a83bd6af9c477')
    composition=read(LOCKED/'DATA_COMPOSITION.json',completion['files']['DATA_COMPOSITION.json'])
    for file,digest in INPUTS.items():
        if sha(file)!=digest: raise ValueError('Input changed')
    write(out/'TEMPORAL.json',dict(status='passed',split='validation',cutoff_us=1598745441188500,
        population={s:int(m.sum()) for s,m in masks.items()},models=models,comparisons=all_comparisons,
        descriptive_decomposition=all_decompositions,historical_arrays_reused=['A_Freeze_R35','D_Spread_R35'],data_composition_reused=composition,
        old_H0_best_R32_historical_reference=reference,bootstrap=dict(repeats=2000,seed=2024,unit='user',estimand='sample mean',temporal_contrast='joint global-user draw'),
        input_sha256=INPUTS,limitations=['Validation used in model selection; post-hoc design and single seed.',
        'Old H0 R32/T360 is a historical reference, not a controlled tokenizer schedule comparison.',
        'Decomposition is descriptive; it does not establish the cause of time degradation.']))

if __name__=='__main__': main()
