"""Independent replay of all D process predictions and saved SID snapshots.

No model forward pass. Catalog-before-T references reuse the previous T snapshot,
only after every intervening R checkpoint has passed Tokenizer isolation audit.
"""
import argparse
import csv
import gzip
import importlib
import json
import math
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from analyze_d import bind,read,write,sha,local,INPUTS,CORE_SHA

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','cold-restore','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    bind(__file__);bind(ROOT/'scripts/listwise_diffgrm/analysis.py',CORE_SHA)
    core=importlib.import_module('scripts.listwise_diffgrm.analysis')
    restored=read(a.cold_restore)
    if restored['status']!='passed' or len(restored['all_boundary_checkpoint_fields_and_clocks_verified'])!=41:
        raise ValueError('Actual restore and all-R Tokenizer isolation audit required')
    root=run.parents[1]
    subset=read(root/'validation_subset_8192.json','7359700f6483f315287d502b200375471d6bb50203dbcc43b567af8b9327f130')
    wanted={r['sample_id']:r['index'] for r in subset['rows']}
    if len(wanted)!=8192:raise ValueError('Diagnostic population changed')
    data=ROOT/'outputs/kuairec_listwise_v1/run1'
    request=read(run/'request.json')
    bind(ROOT/'outputs/stage09_restore/goal_20261027_r1/runs/warmup_H/validation_R05/per_sample.jsonl.gz',request['warmup_predictions_sha256'])
    import pyarrow.parquet as pq
    bind(data/'validation.parquet','9303e01107e4182a5fb33577cb96d9f256cbc85f737f62f349d8b78a61eaa3d4')
    table=pq.read_table(data/'validation.parquet',columns=['sample_id','user_id','target_item_ids']).to_pylist()
    truth={row['sample_id']:(i,row['user_id'],row['target_item_ids']) for i,row in enumerate(table)}
    points={}
    labels=['initial_R05_T000']
    for b in range(10):
        phase=5+4*b; epoch=5+3*b;t=20*(b+1)
        labels += [f'phase{phase:02d}_T_R{epoch:02d}_T{t:03d}',f'phase{phase+1:02d}_R_R{epoch+1:02d}_T{t:03d}',f'phase{phase+3:02d}_R_R{epoch+3:02d}_T{t:03d}']
    for label in labels:
        receipt=read(run/'diagnostics'/label/'complete.json')
        if 'reused' in receipt and label=='initial_R05_T000':
            # It is an authenticated extraction of an old prediction, not a new
            # inference. Preserve the reuse receipt without rerunning old stats.
            result=receipt['result']
            if result['examples']!=8192 or receipt['source_sha256']!=request['warmup_predictions_sha256']:
                raise ValueError('Original R5 subset source/count differs')
            points[label]=dict(metrics=result['metrics'],operation='authenticated historical R5 extraction reuse',
                source_sha256=receipt['source_sha256'])
            continue
        key='reused' if 'reused' in receipt else 'predictions'
        source=bind(local(run,receipt[key]),receipt['source_sha256' if key=='reused' else 'predictions_sha256'])
        meta=read(source.parent/'summary.json');_,arguments=core._metadata(source)
        seen=set();sums={};saved_sums={};max_error=0.
        with gzip.open(source,'rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line); sid=row['sample_id']
                if sid not in wanted:continue
                if sid in seen or (row['dataset_index'],row['user_id'],row['targets'])!=truth[sid] or row['dataset_index']!=wanted[sid]:
                    raise ValueError('Diagnostic sample identity mismatch')
                seen.add(sid);values=core.recompute_row(row,**arguments)
                if set(values)!=set(row['metrics']):raise ValueError('Diagnostic row metric coverage differs')
                for metric,value in values.items():
                    max_error=max(max_error,abs(value-row['metrics'][metric]));sums[metric]=sums.get(metric,0.)+value
                    saved_sums[metric]=saved_sums.get(metric,0.)+row['metrics'][metric]
        if seen!=wanted.keys() or max_error>2e-6:raise ValueError('Diagnostic replay failed')
        means={k:v/8192 for k,v in sums.items()}
        # Production rows hold FP32 metrics; independent ranking uses Python
        # FP64. Authenticate the receipt against its own saved rows tightly,
        # then use the same 2e-6 FP32/FP64 tolerance as the frozen core auditor.
        reported=receipt['result']['metrics']
        if (receipt['result']['examples']!=8192 or set(reported)!=set(means)
                or max(abs(saved_sums[k]/8192-reported[k]) for k in means)>2e-12):
            raise ValueError('Diagnostic receipt does not match saved row aggregation')
        summary_error=max(abs(means[k]-reported[k]) for k in means)
        if summary_error>2e-6:raise ValueError('Diagnostic receipt does not match independent replay')
        if key=='predictions' and (meta['examples']!=8192 or meta['subset']!=subset or meta['full_split']):
            raise ValueError('Diagnostic subset metadata changed')
        points[label]=dict(metrics=means,operation='independent new diagnostic replay' if key=='predictions' else 'independent subset extraction from new full validation',
            source_sha256=sha(source),max_saved_row_error=max_error,
            max_independent_summary_error=summary_error,independent_fp32_tolerance=2e-6,
            saved_row_aggregation_tolerance=2e-12)
        print(json.dumps(dict(event='diagnostic_replay_passed',label=label)),flush=True)
    import torch
    torch.set_num_threads(4)
    bind(ROOT/'scripts/listwise_diffgrm/recovery.py',sha(root/'code/scripts/listwise_diffgrm/recovery.py'))
    from scripts.listwise_diffgrm.recovery import fingerprint
    catalogs={};previous=None
    catalog_labels=['initial_R05_T000']+[labels[1+3*b] for b in range(10)]
    for label in catalog_labels:
        metrics=read(run/'catalog'/(label+'.json'))
        snapshot=torch.load(bind(run/'catalog'/(label+'.pt'),metrics['snapshot_sha256']),map_location='cpu',weights_only=False)
        hard=snapshot['hard_codes'].numpy();books=snapshot['effective_codebooks'].numpy().astype(np.float64)
        checkpoint=run/(label+'.pt')
        checkpoint_receipt=read(run/'checkpoint_receipts'/(checkpoint.name+'.json'))
        state=torch.load(bind(checkpoint,checkpoint_receipt['sha256']),map_location='cpu',weights_only=False)
        if fingerprint(state)!=checkpoint_receipt['state_sha256']:
            raise ValueError('Catalog source checkpoint contents differ from receipt')
        catalog_state=state
        if label!='initial_R05_T000':
            # Both frozen runners diagnose a T boundary before complete_phase,
            # which changes only the four progress fields. The immutable phase
            # checkpoint is saved immediately after that cursor transition.
            phase=int(label.split('_',1)[0][5:])
            expected_progress=dict(phase=phase+1,update=0,boundary_prepared=False,rng_prepared=False)
            if state['inner']['progress']!=expected_progress:raise ValueError('Unexpected catalog checkpoint cursor')
            catalog_state=dict(state);catalog_state['inner']=dict(state['inner'])
            catalog_state['inner']['progress']=dict(phase=phase,update=20,boundary_prepared=True,rng_prepared=True)
        if (metrics['state']!=fingerprint(catalog_state)
                or metrics['r_updates']!=state['inner']['r_updates']
                or metrics['t_updates']!=state['inner']['t_updates']):
            raise ValueError('Catalog belongs to a different saved checkpoint')
        base=state['inner']['alignment']['mixer.base_codebooks'].numpy().astype(np.float64)
        base_norm=float(np.linalg.norm(base));effective_norm=float(np.linalg.norm(books))
        mixer_ratio=float(np.linalg.norm(books-base))/base_norm if base_norm else None
        def same(actual,saved):
            return actual is None and saved is None or actual is not None and saved is not None and math.isclose(actual,saved,rel_tol=1e-10,abs_tol=1e-10)
        if (not same(base_norm,metrics['base_codebook_norm'])
                or not same(effective_norm,metrics['effective_codebook_norm'])
                or not same(mixer_ratio,metrics['mixer_delta_over_base_global_norm'])):
            raise ValueError('Independent Mixer correction or codebook norm differs')
        del state,catalog_state
        if (hard.shape!=(10728,metrics['n_digit']) or metrics['num_items']!=10728
                or books.shape[:2]!=(metrics['n_digit'],metrics['codebook_size'])
                or (hard<0).any() or (hard>=metrics['codebook_size']).any() or not np.isfinite(books).all()):
            raise ValueError('Catalog shape/count/code range differs')
        _,counts=np.unique(hard,axis=0,return_counts=True)
        if len(counts)!=metrics['unique_sid_count'] or int(counts[counts>1].sum())!=metrics['collision_item_count']:
            raise ValueError('Independent collision/SID count differs')
        if (not same(float(counts[counts>1].sum()/len(hard)),metrics['collision_item_fraction'])
                or int(counts.max())!=metrics['largest_bucket_size']
                or len(hard)-len(counts)!=metrics['duplicate_row_count']):
            raise ValueError('Independent collision fraction or bucket size differs')
        for slot,row in enumerate(metrics['per_slot']):
            frequency=np.bincount(hard[:,slot],minlength=metrics['codebook_size'])
            probability=frequency/len(hard);positive=probability[frequency>0]
            entropy=float(-(positive*np.log(positive)).sum())
            normalized=entropy/math.log(metrics['codebook_size']) if metrics['codebook_size']>1 else 0.
            if (frequency.tolist()!=row['counts'] or int((frequency>0).sum())!=row['used_code_count']
                    or not np.allclose(probability,row['frequencies'],rtol=0,atol=1e-14)
                    or not same(entropy,row['entropy_nats']) or not same(normalized,row['normalized_entropy'])
                    or not same(math.exp(entropy),row['effective_code_count'])):
                raise ValueError('Independent per-slot code usage differs')
        if previous is not None:
            changed=hard!=previous[0];change=float(changed.any(axis=1).mean())
            displacement=float(np.linalg.norm(books-previous[1]))
            old=metrics['previous_change']
            previous_norm=float(np.linalg.norm(previous[1]))
            relative=displacement/previous_norm if previous_norm else None
            if (abs(change-old['sid_item_change_fraction'])>1e-14 or not same(displacement,old['effective_displacement_norm'])
                    or not same(relative,old['effective_displacement_over_previous_global_norm'])
                    or not same(float(changed.mean()),old['sid_digit_change_fraction'])
                    or not np.allclose(changed.mean(axis=0),old['sid_change_fraction_per_slot'],rtol=0,atol=1e-14)):
                raise ValueError('Independent SID/embedding displacement differs')
        metrics['independent_snapshot_check']=dict(checkpoint_sha256=checkpoint_receipt['sha256'],
            state_binding=True,progress_transition='initial unchanged' if label=='initial_R05_T000' else 'T boundary before complete_phase cursor only',
            collision_and_usage=True,effective_displacement=True,mixer_correction=True)
        catalogs[label]=metrics;previous=(hard,books)
    curves=[]
    for b in range(10):
        before=labels[0] if b==0 else labels[3*b]
        after,first,end=labels[1+3*b:4+3*b]
        catalog_before='initial_R05_T000' if b==0 else labels[1+3*(b-1)]
        for stage,label in (('before_T',before),('after_20T',after),('after_first_R',first),('cycle_end',end)):
            curves.append(dict(cycle=b+1,stage=stage,source=label,**{k:points[label]['metrics'][k] for k in ('list/ndcg@5','list/hr@5','list/recall@5')}))
        catalogs[after]['before_T_snapshot_reference']=catalog_before
        catalogs[after]['before_T_reuse_proof']='All intervening R checkpoint alignment and T optimizer hashes are identical in the passed cold_restore report.'
    with (out/'PROCESS_CURVE.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    for path,digest in INPUTS.items():
        if sha(path)!=digest:raise ValueError('Diagnostic input changed')
    write(out/'PROCESS.json',dict(status='passed',diagnostic_population=8192,unique_diagnostic_states=len(points),
        newly_inferred_subset_states=20,new_full_validation_extractions=10,historical_initial_reuse=1,
        points=points,catalogs=catalogs,before_after_catalog_observations=20,independent_catalog_snapshots=11,
        curves=curves,input_sha256=INPUTS,limitations='Process diagnostics are descriptive. Before-T catalog reuse is valid because Tokenizer is unchanged throughout intervening R, verified from every checkpoint.'))

if __name__=='__main__':main()
