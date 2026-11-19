"""Full validation/test paired readout evaluation. No training or optimizer."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    from scripts.rqcd_mixer.common import read_state, sha256, write_json, configure_runtime, attention_backend, tensor_digest
    from scripts.rqcd_mixer.catalog import Catalog, generate
    from scripts.rqcd_mixer.data import PackedSplit
    from scripts.rqcd_mixer.evaluation import item_metrics
    from scripts.rqcd_mixer.model import Recommender, Diffusion
    from scripts.rqcd_mixer.tokenizer import distances

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.source/'prepared/PREPARED.json').read_text())
    for name, expected in meta['source'].items():
        assert sha256(args.source/name) == expected, name
    cfg = meta['config']
    runtime = json.loads((args.source/'runtime.locked.json').read_text())
    environment = configure_runtime(runtime, 'cuda')
    protocol = {'source': str(args.source), 'training': False, 'checkpoint': 'original validation-selected best',
        'splits': {'validation': 71657, 'test': 53194}, 'arms': ['B0','B1','B2'],
        'runtime': runtime, 'environment': environment, 'script_sha256': sha256(Path(__file__)),
        'steps': cfg['diffusion']['sample_steps'], 'noise': 'unchanged original full split noise table',
        'readouts': ['original catalog-constrained RQ', 'full-catalog squared L2 nearest prequantization z_i'],
        'prefix': 'original RQ prefixes shared; NN posthoc readout never fed back',
        'exclusions': 'each readout excludes its own previously selected items; history items allowed',
        'selection': 'No new checkpoint selection on either split'}
    write_json(out/'PROTOCOL.json', protocol)
    started = time.perf_counter()
    all_results = {}
    with torch.inference_mode(), attention_backend(runtime['attention_backend']):
        for arm in protocol['arms']:
            best = json.loads((args.source/arm/'BEST.json').read_text())
            cp = args.source/arm/best['checkpoint']
            cp_hash = sha256(cp)
            assert cp_hash == best['sha256']
            state = read_state(cp)
            assert state['config'] == cfg and state['runtime'] == runtime
            cat = Catalog(state['catalog'], 'cuda')
            model = Recommender(cfg).cuda().eval()
            model.load_state_dict(state['model'])
            diffusion = Diffusion(cfg).cuda().eval()
            diffusion.load_state_dict(state['diffusion'])
            before = tensor_digest(model.state_dict())
            all_results[arm] = {}
            for split in protocol['splits']:
                destination = out/arm/split
                destination.mkdir(parents=True, exist_ok=True)
                if (destination/'SUMMARY.json').exists():
                    existing = json.loads((destination/'SUMMARY.json').read_text())
                    assert existing['checkpoint_sha256'] == cp_hash
                    assert existing['script_sha256'] == protocol['script_sha256']
                    for name, expected in existing['artifacts'].items():
                        assert sha256(destination/name) == expected
                    all_results[arm][split] = existing
                    continue
                begin_time = time.perf_counter()
                data = PackedSplit(meta['dataset_path'], split, cfg, device='cpu')
                assert len(data) == protocol['splits'][split]
                noise_path = args.source/'prepared'/f'{split}_noise.npy'
                noise_meta = json.loads(noise_path.with_suffix('.json').read_text())
                assert sha256(noise_path) == noise_meta['sha256']
                sample_hash = hashlib.sha256(json.dumps(data.sample_ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                assert sample_hash == noise_meta['sample_ids_sha256']
                noise = np.load(noise_path, mmap_mode='r')
                preds, truths, zs, raw_valids = [], [], [], []
                for begin in range(0, len(data), runtime['evaluation_batch']):
                    stop = min(begin+runtime['evaluation_batch'],len(data))
                    history, truth = data.batch(torch.arange(begin,stop), 'cuda')
                    initial = torch.from_numpy(np.array(noise[begin:stop],copy=True)).cuda()
                    captured = []
                    original_land = cat.land
                    def capture(z,selected):
                        captured.append(z.detach().clone())
                        return original_land(z,selected)
                    cat.land = capture
                    try:
                        rq, diag = generate(model,diffusion,cat,history,initial,runtime)
                    finally:
                        cat.land = original_land
                    zlist = torch.stack(captured,1)
                    assert torch.isfinite(zlist).all()
                    nn = torch.empty((len(history),0),dtype=torch.long,device='cuda')
                    for j in range(cfg['list_length']):
                        score = distances(zlist[:,j],cat.z)
                        if nn.shape[1]:
                            score.scatter_(1,nn-1,float('inf'))
                        item = score.argmin(-1)+1
                        nn = torch.cat((nn,item[:,None]),1)
                    assert item_metrics(rq,truth,cat.count)['completion_rate'].all()
                    assert item_metrics(nn,truth,cat.count)['completion_rate'].all()
                    preds.append(torch.stack((rq,nn),1).cpu())
                    truths.append(truth.cpu())
                    zs.append(zlist.cpu())
                    raw_valids.append(diag['raw_valid'].cpu())
                    if (begin//runtime['evaluation_batch']+1)%20==0 or stop==len(data):
                        progress={'arm':arm,'split':split,'completed':stop,'total':len(data),'seconds':time.perf_counter()-begin_time}
                        write_json(out/'PROGRESS.json',progress)
                        print(json.dumps(progress),flush=True)
                pred,truth=torch.cat(preds),torch.cat(truths)
                values=[item_metrics(pred[:,k],truth,cat.count) for k in (0,1)]
                metrics=[{k:float(v.double().mean()) for k,v in x.items()} for x in values]
                # Reproduction of all saved predictions, in original order, is mandatory.
                if split=='test':
                    orig=args.source/arm/'test/best'
                else:
                    orig=args.source/arm/'validation'/f"R{state['r_epoch']:02d}"
                orig_summary=json.loads((orig/'summary.json').read_text())
                assert orig_summary['identity']['checkpoint_sha256']==cp_hash
                matched=0
                with gzip.open(orig/'predictions.jsonl.gz','rt') as f:
                    for i,line in enumerate(f):
                        row=json.loads(line)
                        assert row['sample_id']==data.sample_ids[i]
                        assert row['target']==truth[i].tolist()
                        matched+=int(row['prediction']==pred[i,0].tolist())
                assert i+1==len(data)
                assert matched==len(data), (arm,split,'reproduction mismatch',matched)
                for key in metrics[0]:
                    assert abs(metrics[0][key]-orig_summary['metrics'][key])<1e-10
                diff=(values[1]['ndcg@5']-values[0]['ndcg@5']).double().numpy()
                users,inv=np.unique(np.asarray([str(u) for u in data.users]),return_inverse=True)
                counts=np.bincount(inv); sums=np.bincount(inv,weights=diff)
                rng=np.random.default_rng(20261112); boot=[]
                for _ in range(3000):
                    sampled=rng.integers(0,len(users),len(users))
                    boot.append(sums[sampled].sum()/counts[sampled].sum())
                np.savez_compressed(destination/'vectors_and_predictions.npz',z=torch.cat(zs).numpy(),
                    prediction=pred.numpy(),target=truth.numpy())
                with gzip.open(destination/'predictions.jsonl.gz','wt',encoding='utf-8') as f:
                    for i,sid in enumerate(data.sample_ids):
                        f.write(json.dumps({'sample_id':sid,'user_id':str(data.users[i]),'target':truth[i].tolist(),
                            'rq':pred[i,0].tolist(),'nn':pred[i,1].tolist(),
                            'rq_ndcg':float(values[0]['ndcg@5'][i]),'nn_ndcg':float(values[1]['ndcg@5'][i])})+'\n')
                result={'arm':arm,'split':split,'R':state['r_epoch'],'T':state['t_updates'],'count':len(data),
                    'checkpoint':str(cp),'checkpoint_sha256':cp_hash,'script_sha256':protocol['script_sha256'],
                    'sample_ids_sha256':sample_hash,'noise_sha256':noise_meta['sha256'],
                    'rq':metrics[0],'nn':metrics[1],
                    'relative_ndcg_change':metrics[1]['ndcg@5']/metrics[0]['ndcg@5']-1,
                    'paired_user_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
                    'unique_users':len(users),'ndcg_difference':float(diff.mean()),
                    'rq_hit_sessions':int(values[0]['hr@5'].sum()),'nn_hit_sessions':int(values[1]['hr@5'].sum()),
                    'both_hit_sessions':int(((values[0]['hr@5']>0)&(values[1]['hr@5']>0)).sum()),
                    'prediction_agreement_by_position':(pred[:,0]==pred[:,1]).double().mean(0).tolist(),
                    'raw_valid_by_position':torch.cat(raw_valids).double().mean(0).tolist(),
                    'first_position_hit_any_target':{name:float((pred[:,k,0,None]==truth).any(-1).double().mean()) for k,name in enumerate(('rq','nn'))},
                    'original_exact_list_agreement':matched/len(data),'model_unchanged':tensor_digest(model.state_dict())==before,
                    'seconds':time.perf_counter()-begin_time,
                    'artifacts':{n:sha256(destination/n) for n in ('vectors_and_predictions.npz','predictions.jsonl.gz')}}
                assert result['model_unchanged']
                write_json(destination/'SUMMARY.json',result)
                all_results[arm][split]=result
                print(json.dumps({'complete':arm+'/'+split,'rq':metrics[0],'nn':metrics[1]}),flush=True)
                del data,preds,truths,zs,raw_valids,pred,truth
            del state,model,diffusion,cat
            torch.cuda.empty_cache()
    write_json(out/'COMPLETE.json',{'status':'FULL_NN_EVALUATION_COMPLETE','training_started':False,
        'results':all_results,'seconds':time.perf_counter()-started,'script_sha256':protocol['script_sha256'],
        'protocol_sha256':sha256(out/'PROTOCOL.json')})


if __name__=='__main__':
    main()
