"""Independent CPU acceptance of E; reuses accepted A and D audits, never re-decodes.

Run only after downloading a complete E run. Does not alter scientific code.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
GRID = (20, 23, 26, 29, 32, 35)
VAL = (8, 11, 14, 17, *GRID)
CORE_SHA = '4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd'
A_SHA = '66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937'
INPUTS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def bind(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Input bytes changed: ' + str(path))
    if str(path) in INPUTS and INPUTS[str(path)] != digest:
        raise ValueError('Input changed during analysis')
    INPUTS[str(path)] = digest
    return path

def read(path, expected=None):
    return json.loads(bind(path, expected).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def local(run, saved):
    marker = '/H1-Block40-R6-R35/'
    if str(saved).count(marker) != 1:
        raise ValueError('Unrecognized remote artifact path')
    path = (run / str(saved).split(marker)[1]).resolve()
    path.relative_to(run)
    return path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run, out = a.run.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bind(__file__)
    bind(ROOT / 'scripts/listwise_diffgrm/analysis.py', CORE_SHA)
    core = importlib.import_module('scripts.listwise_diffgrm.analysis')
    data = ROOT / 'outputs/kuairec_listwise_v1/run1'
    complete = read(run / 'complete.json')
    if (complete['status'], complete['r_updates'], complete['t_updates']) != ('training_and_validation_complete', 24255, 200):
        raise ValueError('E endpoint incomplete')
    selection = read(run / 'selection.json', complete['selection_sha256'])
    tests = read(run / 'test_predictions_complete.json')
    if tests['selection_sha256'] != sha(run / 'selection.json'):
        raise ValueError('Test selection binding differs')
    cpu = read(HERE / 'CPU_SCHEDULE_CHECK.json')
    expected_orders = {x['phase']: dict(x, sha256=x['indices_sha256'], lists=40960 if x['kind']=='T' else 709443) for x in cpu['phase_orders']}
    for phase, expected in expected_orders.items():
        order = read(run / 'sample_orders' / f'phase{phase:02d}.json')
        if order['indices_sha256_int64_le'] != expected['sha256'] or order['lists'] != expected['lists']:
            raise ValueError('Formal phase samples differ from paired acceptance')
    # Clocks advance exactly once per logged update, not just at endpoints.
    r, t, n = 3465, 0, 0
    expected_phase, phase_update, indices = 5, 0, None
    for line in bind(run / 'training.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        kind = 'T' if (expected_phase-5)%7==0 else 'R'
        width = 40 if kind=='T' else 693
        if indices is None:
            block=(expected_phase-5)//7
            epoch=5+6*block+(expected_phase-5)%7
            source_phase=5+4*block if kind=='T' else 6+4*((epoch-6)//3)+(epoch-6)%3
            if source_phase!=expected_orders[expected_phase]['original_phase']: raise ValueError('Original phase mapping differs')
            generator=np.random.default_rng(np.random.SeedSequence([2024,7919,source_phase]))
            indices=generator.permutation(709443)
            if kind=='T': indices=indices[:40960]
            order_hash=hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()
            if order_hash!=expected_orders[expected_phase]['sha256']: raise ValueError('Independent phase order differs')
        batch=indices[phase_update*1024:(phase_update+1)*1024]
        batch_hash=hashlib.sha256(batch.astype('<i8',copy=False).tobytes()).hexdigest()
        if (row['phase']!=kind or row['phase_index']!=expected_phase or row['update']!=phase_update+1
                or row['sample_indices_sha256']!=batch_hash or row['lists']!=len(batch)):
            raise ValueError('Formal logged batch order differs from original paired sample stream')
        if row['noise_used']['optimizer_step']!=t or row['noise_used']['total_optimizer_steps']!=400:
            raise ValueError('Original 400T Gumbel clock changed')
        expected_noise=.2+.4*(1+math.cos(math.pi*t/400))
        if (row['noise_used']['softmax_temperature']!=1
                or abs(row['noise_used']['gumbel_noise_scale']-expected_noise)>1e-14):
            raise ValueError('Original Gumbel noise amplitude or temperature changed')
        lrs=row['learning_rates_used']
        if lrs['codebook']!=1e-5 or lrs['mixer']!=1e-4:
            raise ValueError('Tokenizer learning rates changed')
        expected_lr=.001*.5*(1+math.cos(math.pi*((r-500)/(24255-500))))
        if abs(lrs['R']-expected_lr)>1e-14:
            raise ValueError('Original 35R learning-rate schedule changed or restarted')
        if row['phase'] == 'R': r += 1
        elif row['phase'] == 'T': t += 1
        else: raise ValueError('Unknown training phase')
        if (row['r_updates'], row['t_updates']) != (r, t):
            raise ValueError('Duplicated or missing training update')
        n += 1
        phase_update+=1
        if phase_update==width:
            expected_phase+=1; phase_update=0; indices=None
    if (r,t,n) != (24255,200,20990):
        raise ValueError('Update count differs from protocol')
    if expected_phase!=40 or phase_update!=0: raise ValueError('Wrong phase endpoint')
    points = {}
    def point(record, split):
        epoch = record['epoch']
        files = {key: bind(local(run, record[key]), record[key+'_sha256']) for key in ('summary','predictions','checkpoint')}
        meta = read(files['summary'])
        expected = dict(split=split, examples=71657 if split=='validation' else 53194,
            full_split=True, beam=32, inference_precision='fp32', architecture='H', seed=2024,
            r_updates=epoch*693, t_updates=(1+(epoch-6)//6)*40)
        if (any(meta.get(k) != v for k,v in expected.items()) or meta.get('subset') is not None
                or meta.get('diagnostic_prefix_limit') is not None):
            raise ValueError('New evaluation protocol or clocks differ')
        cached=HERE/('first_validation_R08' if epoch==8 else f'saved_validation_R{epoch:02d}')/'ACCEPTED.json'
        operation='new independent CPU replay'
        if split=='validation' and cached.is_file():
            accepted=read(cached)
            if (accepted['status']!='passed_one_completed_validation_only'
                    or accepted['epoch']!=epoch
                    or accepted['summary_sha256']!=record['summary_sha256']
                    or accepted['predictions_sha256']!=record['predictions_sha256']
                    or accepted['independent_core_sha256']!=CORE_SHA
                    or accepted['dataset_binding']!=meta['dataset_binding']):
                raise ValueError('Completed validation audit does not bind this exact source')
            for name in ('validation.parquet','train_seen.json'):
                bind(data/name,meta['dataset_binding']['files'][name]['sha256'])
            result=read(cached.parent/'audit.json',accepted['audit_sha256'])
            if (result['predictions_sha256']!=record['predictions_sha256']
                    or result['summary_sha256']!=record['summary_sha256']
                    or result['mismatches']!=0 or not result['source_cold_groups_verified']
                    or result['examples']!=meta['examples']
                    or result['examples']!=accepted['examples'] or result['users']!=accepted['users']):
                raise ValueError('Accepted completed validation audit has incomplete scope')
            operation=f'reuse exact previously accepted new-E R{epoch:02d} independent replay'
        else:
            result = core.audit_predictions(files['predictions'], dataset_dir=data)
        if not result['passed'] or not result['source_labels_verified']:
            raise ValueError('Independent E replay failed')
        name = f'E_R{epoch:02d}_{split}'
        if record['ndcg5']!=meta['metrics']['list/ndcg@5']:
            raise ValueError('Locked selection score differs from its authenticated production summary')
        write(out / 'audits' / (name+'.json'), result)
        points[name] = dict(epoch=epoch, split=split, audit_operation=operation, paths={k:str(v) for k,v in files.items()},
            metrics=result['metrics'], selection_score=record['ndcg5'], predictions_sha256=result['predictions_sha256'])
        print(json.dumps(dict(event='independent_E_audit', model=name, passed=True)), flush=True)
        return points[name]
    raw_validations={e:read(run/'validation_records'/f'R{e:02d}.json') for e in VAL}
    validations = {e: point(raw_validations[e], 'validation') for e in VAL}
    # Selection uses the score locked during training, as in the original A
    # protocol. Independent FP64 replay checks that score within tolerance; it
    # must not silently reselect a model due to tiny aggregation differences.
    best = min(GRID, key=lambda e: (-validations[e]['selection_score'], e))
    if (selection['selected']!=raw_validations[best]
            or selection['candidates']!=[raw_validations[e] for e in GRID]):
        raise ValueError('Common-grid best selection mismatch')
    if {x['epoch'] for x in tests['results']} != {35,best} or len(tests['results']) != len({35,best}):
        raise ValueError('Unauthorized or missing new test')
    for record in tests['results']:
        chosen=raw_validations[record['epoch']]
        if (record['checkpoint_sha256']!=chosen['checkpoint_sha256']
                or local(run,record['checkpoint'])!=local(run,chosen['checkpoint'])):
            raise ValueError('New test is not bound to its preselected validation checkpoint')
        point(record, 'test')
    apath = ROOT / 'experiments/09_kuairec_listwise/h_freeze35_20261029/final_analysis/A/analysis_report.json'
    old = read(apath, A_SHA)
    if old['status'] != 'passed' or old['selected_epoch'] != 32:
        raise ValueError('Unexpected accepted A comparison')
    oldpoints = {}
    for epoch,split in ((35,'test'),(32,'test'),(35,'validation')):
        name = f'A_R{epoch}_{split}'
        item = old['models'][name]
        audit = read(item['audit_path'], item['audit_sha256'])
        if not audit['passed'] or audit['predictions_sha256'] != item['predictions_sha256']:
            raise ValueError('Historical accepted audit mismatch')
        for key in ('summary','predictions','checkpoint'): bind(item[key], item[key+'_sha256'])
        oldpoints[name] = item
    comparisons = {}
    for label, ae, de, split in (('E_minus_A_fixed_test',35,35,'test'),('E_minus_A_best_test',32,best,'test'),('E_minus_A_fixed_validation',35,35,'validation')):
        ap = oldpoints[f'A_R{ae}_{split}']['predictions']
        dp = points[f'E_R{de:02d}_{split}']['paths']['predictions']
        comparison = core.paired_comparison(ap, dp, repeats=2000, seed=2024)
        write(out / 'comparisons' / (label+'.json'), comparison)
        comparisons[label] = comparison['primary']
    dpath = ROOT / 'experiments/09_kuairec_listwise/h1_spread200_20261031/final_analysis/ANALYSIS.json'
    d = read(dpath, '5cc04a637140d94d198c1a2e0132cf09b83d03cac152e82ac9e5da675add44e8')
    if d['status']!='passed' or d['selected_epoch']!=23: raise ValueError('Unexpected accepted D report')
    for name in ('D_R35_test','D_R23_test','D_R35_validation'):
        item=d['points'][name]
        for path in item['paths'].values(): bind(path,d['input_sha256'][path])
        audit_path=dpath.parent/'audits'/(name+'.json')
        audit=read(audit_path)
        if not audit['passed'] or audit['predictions_sha256']!=item['predictions_sha256'] or audit['metrics']!=item['metrics']:
            raise ValueError('Historical D audit mismatch')
        oldpoints[name]=item
    comparison=core.paired_comparison(oldpoints['D_R35_test']['paths']['predictions'],points['E_R35_test']['paths']['predictions'],repeats=2000,seed=2024)
    write(out/'comparisons'/'E_minus_D_fixed_test.json',comparison)
    comparisons['E_minus_D_fixed_test']=comparison['primary']
    with (out/'METRICS.csv').open('x',encoding='utf-8-sig',newline='') as f:
        rows = [dict(model=name,**value['metrics']) for name,value in points.items()]
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for path, expected in INPUTS.items():
        if sha(path) != expected: raise ValueError('Input changed during analysis: '+path)
    write(out/'ANALYSIS.json',dict(status='passed', scope='new E full prediction replay and E-D/E-A paired comparisons; temporal/catalog/restore/resource checks separate',
        selected_epoch=best, points=points, comparisons=comparisons, updates=dict(R=r,T=t,new_logged_updates=n),
        historical_audits_reused=list(oldpoints), historical_points=oldpoints, input_sha256=INPUTS,
        limitations='One seed, post-hoc schedule choice, sample-mean user cluster bootstrap; no cross-seed or causal claim'))

if __name__ == '__main__': main()
