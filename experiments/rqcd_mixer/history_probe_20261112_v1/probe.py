"""Read-only paired history ablation on the previously locked validation subset."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--subset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source))
    import numpy as np
    import torch
    from scripts.rqcd_mixer.common import read_state,sha256,write_json,configure_runtime,attention_backend,autocast,tensor_digest
    from scripts.rqcd_mixer.catalog import Catalog
    from scripts.rqcd_mixer.data import PackedSplit
    from scripts.rqcd_mixer.model import Recommender,Diffusion
    from scripts.rqcd_mixer.tokenizer import distances
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'COMPLETE.json').exists():raise RuntimeError('Already complete')
    meta=json.loads((a.source/'prepared/PREPARED.json').read_text());cfg=meta['config']
    for n,h in meta['source'].items():assert sha256(a.source/n)==h,n
    runtime=json.loads((a.source/'runtime.locked.json').read_text());env=configure_runtime(runtime,'cuda')
    subset=json.loads((a.subset_root/'SUBSET.json').read_text());idx=np.asarray(subset['indices']);n=len(idx);assert n==8192
    data=PackedSplit(meta['dataset_path'],'validation',cfg,'cpu')
    assert [data.sample_ids[i] for i in idx]==subset['sample_ids']
    users=np.asarray([str(data.users[i]) for i in idx]);lengths=(data.history[idx]!=0).sum(-1).numpy()
    # A true permutation, no same-user donor. Exact-length groups where feasible.
    rng=np.random.default_rng(20261112);groups=[];rare=[]
    for length in np.unique(lengths):
        g=np.flatnonzero(lengths==length).tolist();mx=max(Counter(users[g]).values())
        (groups if 2*mx<=len(g) else rare).append(g)
    if rare:
        pooled=[i for g in rare for i in g]
        # Merge rare lengths with the largest valid group so a derangement is possible.
        largest=max(range(len(groups)),key=lambda j:len(groups[j]));pooled+=groups.pop(largest);groups.append(pooled)
    donor=np.empty(n,dtype=np.int64)
    for g in groups:
        byuser=defaultdict(list)
        for i in g:byuser[users[i]].append(i)
        keys=list(byuser);rng.shuffle(keys);order=[]
        for u in keys:rng.shuffle(byuser[u]);order+=byuser[u]
        shift=max(map(len,byuser.values()));assert 2*shift<=len(order)
        donor[np.asarray(order)]=np.roll(np.asarray(order),shift)
    assert len(np.unique(donor))==n and np.all(users!=users[donor]) and np.all(donor!=np.arange(n))
    shuffle={'seed':20261112,'source_subset_sha256':sha256(a.subset_root/'SUBSET.json'),'indices':idx.tolist(),
        'donor_subset_positions':donor.tolist(),'different_user_fraction':float(np.mean(users!=users[donor])),
        'same_history_length_fraction':float(np.mean(lengths==lengths[donor])),
        'mean_absolute_length_difference':float(np.mean(np.abs(lengths-lengths[donor]))),
        'method':'Within exact length, shuffled user blocks cyclic derangement; rare groups pooled with largest feasible group'}
    write_json(a.output/'SHUFFLE.json',shuffle)
    noise_path=a.source/'prepared/validation_noise.npy';nm=json.loads(noise_path.with_suffix('.json').read_text());assert sha256(noise_path)==nm['sha256']
    noise=np.load(noise_path,mmap_mode='r')
    # Independent fixed epsilon for teacher-noised targets, identical across arms/history/t.
    eps=np.random.default_rng(2026111201).standard_normal((n,cfg['rq']['latent_dim'])).astype('float32')
    np.save(a.output/'probe_epsilon.npy',eps)
    protocol={'training':False,'arms':['B0','B1'],'count':n,'split':'validation','positions':[1],
        'checkpoint':'original validation-selected best','runtime':runtime,'environment':env,
        'time_levels':{'low':100,'medium':500,'high':900},'free_generation':'original 20 DDIM steps from original first-item noise',
        'teacher_probe':'single denoising prediction at fixed t of true first target plus fixed epsilon; no prefix',
        'rank':'squared L2 to all catalog prequantization z; ties broken by ascending catalog ID',
        'metric_note':'first-target accuracy/rank and any-of-five hit are separate; these are not list NDCG@5',
        'shuffle_sha256':sha256(a.output/'SHUFFLE.json'),'epsilon_sha256':sha256(a.output/'probe_epsilon.npy'),
        'noise_sha256':nm['sha256'],'script_sha256':sha256(Path(__file__))}
    write_json(a.output/'PROTOCOL.json',protocol)

    def measure(cat,x,truth,clean):
        z=x.float()*cat.stats['scale']+cat.stats['mean'];score=distances(z,cat.z)
        target=truth[:,0];ts=score.gather(1,(target-1)[:,None]);ids=torch.arange(1,cat.count+1,device='cuda')[None]
        rank=1+((score<ts)|((score==ts)&(ids<target[:,None]))).sum(-1)
        nn=score.argmin(-1)+1;empty=torch.empty((len(x),0),dtype=torch.long,device='cuda');rq,_=cat.land(z,empty)
        return {'mse':(x.float()-clean).square().mean(-1),'rank':rank.float(),'mrr':1/rank.float(),
            'nn_first_hit':(nn==target).float(),'nn_any5_hit':(nn[:,None]==truth).any(-1).float(),
            'nn_first_recall10':(rank<=10).float(),'nn_first_recall100':(rank<=100).float(),
            'rq_first_hit':(rq==target).float(),'rq_any5_hit':(rq[:,None]==truth).any(-1).float(),
            'nn_item':nn,'rq_item':rq}

    all_results={};started=time.perf_counter()
    with torch.inference_mode(),attention_backend(runtime['attention_backend']):
        for arm in ('B0','B1'):
            best=json.loads((a.source/arm/'BEST.json').read_text());cp=a.source/arm/best['checkpoint'];assert sha256(cp)==best['sha256']
            st=read_state(cp);assert st['config']==cfg and st['runtime']==runtime
            cat=Catalog(st['catalog'],'cuda');model=Recommender(cfg).cuda().eval();model.load_state_dict(st['model'])
            diffusion=Diffusion(cfg).cuda().eval();diffusion.load_state_dict(st['diffusion']);before=tensor_digest(model.state_dict())
            records=defaultdict(lambda:defaultdict(list));outputs=defaultdict(list);free_real=[]
            for b in range(0,n,256):
                e=min(b+256,n);hist,truth=data.batch(torch.from_numpy(idx[b:e]),'cuda');shuf,_=data.batch(torch.from_numpy(idx[donor[b:e]]),'cuda')
                clean=cat.x_pad[truth[:,0]];epsilon=torch.from_numpy(eps[b:e]).cuda();initial=torch.from_numpy(np.array(noise[idx[b:e],0],copy=True)).cuda()
                caches={}
                for mode,h in [('real',hist),('shuffled',shuf)]:
                    with autocast(runtime,'cuda'):mem,valid=model.encode_history(model.lookup(cat.sid_pad[h]),h!=0);caches[mode]=model.start_cache(mem,valid)
                for level,t in [('free',None),('low',100),('medium',500),('high',900)]:
                    if t is not None:
                        times=torch.full((len(hist),),t,dtype=torch.long,device='cuda');noisy=diffusion.noise(clean,times,epsilon)
                        base=measure(cat,noisy,truth,clean)
                        for k,v in base.items():records[level+'/before'][k].append(v.cpu())
                    pair={}
                    for mode in ('real','shuffled'):
                        def predictor(x,times):
                            with autocast(runtime,'cuda'):return model.predict_cached(x,times,caches[mode])
                        x=diffusion.sample(initial,predictor) if t is None else predictor(noisy,times).float()
                        assert torch.isfinite(x).all();pair[mode]=x
                        measures=measure(cat,x,truth,clean)
                        for k,v in measures.items():records[level+'/'+mode][k].append(v.cpu())
                        outputs[level+'/'+mode].append(x.cpu())
                        if level=='free' and mode=='real':free_real.append(measures['rq_item'].cpu())
                    records[level+'/sensitivity']['output_mse'].append((pair['real']-pair['shuffled']).square().mean(-1).cpu())
                print(json.dumps({'arm':arm,'completed':e,'total':n}),flush=True)
            arrays={key:{k:torch.cat(v).numpy() for k,v in d.items()} for key,d in records.items()}
            prior=np.load(a.subset_root/arm/'vectors_and_predictions.npz')
            assert np.array_equal(torch.cat(free_real).numpy(),prior['predictions'][:,0,0])
            zfree=torch.cat(outputs['free/real']).numpy()*float(cat.stats['scale'])+cat.stats['mean'].cpu().numpy()
            vector_maxdiff=float(np.max(np.abs(zfree-prior['z'][:,0])))
            summary={}
            for key,d in arrays.items():
                summary[key]={k:float(v.astype('float64').mean()) for k,v in d.items() if not k.endswith('_item')}
                if 'rank' in d:summary[key]['median_rank']=float(np.median(d['rank']))
            _,inv=np.unique(users,return_inverse=True);counts=np.bincount(inv);paired={}
            names=['mse','mrr','nn_first_recall10','nn_any5_hit','rq_any5_hit']
            for level in ('free','low','medium','high'):
                diffs=np.stack([arrays[level+'/real'][k].astype('float64')-arrays[level+'/shuffled'][k] for k in names],-1)
                sums=np.stack([np.bincount(inv,weights=diffs[:,k]) for k in range(len(names))],-1)
                rng=np.random.default_rng(20261112);boots=[]
                for _ in range(2000):
                    pick=rng.integers(0,len(counts),len(counts));boots.append(sums[pick].sum(0)/counts[pick].sum())
                ci=np.quantile(boots,[.025,.975],axis=0)
                paired[level]={k:{'real_minus_shuffled':float(diffs[:,j].mean()),'user_bootstrap_95pct':ci[:,j].tolist()} for j,k in enumerate(names)}
                paired[level]['nn_item_changed_fraction']=float(np.mean(arrays[level+'/real']['nn_item']!=arrays[level+'/shuffled']['nn_item']))
                paired[level]['rq_item_changed_fraction']=float(np.mean(arrays[level+'/real']['rq_item']!=arrays[level+'/shuffled']['rq_item']))
            armout=a.output/arm;armout.mkdir(exist_ok=True)
            np.savez_compressed(armout/'per_sample.npz',**{key.replace('/','__')+'__'+k:v for key,d in arrays.items() for k,v in d.items()},
                **{key.replace('/','__')+'__x':torch.cat(v).numpy() for key,v in outputs.items()})
            result={'arm':arm,'R':st['r_epoch'],'checkpoint_sha256':best['sha256'],'count':n,
                'alpha_bar':{str(t):float(diffusion.alpha[t]) for t in (100,500,900)},'summary':summary,'paired':paired,
                'original_first_rq_prediction_exact_match':True,'original_first_generated_z_max_abs_difference':vector_maxdiff,
                'model_unchanged':tensor_digest(model.state_dict())==before,'per_sample_sha256':sha256(armout/'per_sample.npz')}
            assert result['model_unchanged'];write_json(armout/'SUMMARY.json',result);all_results[arm]=result
            print(json.dumps({'arm_complete':arm,'free':{k:v for k,v in summary.items() if k.startswith('free')}}),flush=True)
            del st,model,diffusion,cat;torch.cuda.empty_cache()
    write_json(a.output/'COMPLETE.json',{'status':'HISTORY_PROBE_COMPLETE','training_started':False,'seconds':time.perf_counter()-started,
        'results':all_results,'shuffle_summary':{k:v for k,v in shuffle.items() if k not in ['indices','donor_subset_positions']},
        'protocol_sha256':sha256(a.output/'PROTOCOL.json'),'script_sha256':sha256(Path(__file__))})

if __name__=='__main__':main()
