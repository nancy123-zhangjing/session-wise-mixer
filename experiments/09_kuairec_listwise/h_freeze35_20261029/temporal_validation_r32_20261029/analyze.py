"""Fixed historical time-split diagnosis; CPU replay only, no training/inference."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, gzip, hashlib, csv
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = OUT.parent
sys.path.insert(0, str(ROOT))
from scripts.listwise_diffgrm import analysis as core
CORE_SHA = '4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd'
DATA_SHA = '9303e01107e4182a5fb33577cb96d9f256cbc85f737f62f349d8b78a61eaa3d4'
MID = 1598745441188500
N = 71657
REPORTS = {
 'A': (TASK/'final_analysis/A/analysis_report.json','66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937'),
 'B': (TASK/'final_analysis/B/analysis_report.json','7596546bb014f56b69d0a574d9cfd3a991e267bf667af6b80603cc6f48361a7e')}
SPECS = {'H0_best_R32': ('B','B_R32_validation'), 'H1_R20': ('A','A_R20_validation'),
         'H1_freeze_R32': ('A','A_R32_validation')}
PRIMARY = [f'list/{k}@5' for k in ('ndcg','hr','recall')]
PRIMARY += [f'aggregate/{k}@{n}' for n in (10,20,50) for k in ('ndcg','hr','recall')]
bound = {}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8388608), b''): h.update(chunk)
 return h.hexdigest()
def bind(p, expected=None):
 p=Path(p).resolve(); digest=sha(p)
 if expected is not None and digest != expected: raise ValueError('Changed source: '+str(p))
 bound[str(p)]=digest
 return digest
def save(p, obj):
 with Path(p).open('x',encoding='utf-8') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def log(**v): print(json.dumps(v,ensure_ascii=False),flush=True)

def main():
 bind(__file__);bind(ROOT/'scripts/listwise_diffgrm/analysis.py',CORE_SHA)
 bind(ROOT/'scripts/listwise_diffgrm/subset.py')
 reports={}
 for name,(path,digest) in REPORTS.items():
  bind(path,digest);reports[name]=read(path)
  assert reports[name]['status']=='passed' and reports[name]['analysis_sha256']==CORE_SHA
 points={}; metas={}
 for name,(branch,key) in SPECS.items():
  p=reports[branch]['models'][key];points[name]=p
  for field in ('summary','predictions','checkpoint','audit_path'):
   expected=p['audit_sha256'] if field=='audit_path' else p[field+'_sha256']
   bind(p[field],expected)
  audit=read(p['audit_path'])
  assert audit['passed'] and audit['source_labels_verified'] and audit['mismatches']==0
  assert audit['predictions_sha256']==p['predictions_sha256']
  assert audit['summary_sha256']==p['summary_sha256'] and audit['metrics']==p['metrics']
  meta=read(p['summary']);metas[name]=meta
  assert meta['split']=='validation' and meta['full_split'] and meta['examples']==N
  assert meta['architecture']=='H' and meta['beam']==32 and meta['seed']==2024
  assert meta['inference_precision']=='fp32'
  assert meta['r_updates']==(20 if name=='H1_R20' else 32)*693
  assert meta['t_updates']==(360 if name=='H0_best_R32' else 200)
 ref=metas['H0_best_R32']
 for meta in metas.values():
  for field in ('dataset_binding','metric_protocol','catalog_size','subset','diagnostic_prefix_limit'):
   assert meta.get(field)==ref.get(field)
 d=ROOT/'outputs/kuairec_listwise_v1/run1'
 bind(d/'validation.parquet',DATA_SHA)
 bind(d/'train_seen.json',ref['dataset_binding']['files']['train_seen.json']['sha256'])
 seen_items=set(read(d/'train_seen.json')['item_ids'])
 old_path=ROOT/'experiments/09_kuairec_listwise/generalization_audit_20261028/RESULTS.json'
 bind(old_path);old=read(old_path)
 assert old['temporal_validation']['midpoint']=='2020-08-30T07:57:21.188500+08:00'
 bind(ROOT/'experiments/09_kuairec_listwise/generalization_audit_20261028/analyze.py')
 truth=pq.read_table(d/'validation.parquet',columns=['sample_id','user_id','target_item_ids','target_timestamp_us']).to_pylist()
 assert len(truth)==N and len({x['sample_id'] for x in truth})==N
 users=np.asarray([x['user_id'] for x in truth],dtype=np.int64)
 starts=np.asarray([x['target_timestamp_us'][0] for x in truth],dtype=np.int64)
 ends=np.asarray([x['target_timestamp_us'][-1] for x in truth],dtype=np.int64)
 assert (int(starts.min())+int(ends.max()))//2==MID
 masks={'early':starts<MID,'late':starts>=MID,'full':np.ones(N,dtype=bool)}
 assert [int(masks[k].sum()) for k in masks]==[38644,33013,N]
 pop={k:dict(examples=int(v.sum()),users=len(set(users[v].tolist()))) for k,v in masks.items()}
 assert pop['early']['users']==4011 and pop['late']['users']==3827 and pop['full']['users']==4733
 request=dict(created_utc=datetime.now(timezone.utc).isoformat(),models=SPECS,points=points,
  split='validation',cutoff_us=MID,cutoff_local='2020-08-30T07:57:21.188500+08:00',
  rule='first target timestamp < midpoint: early; otherwise late; keep five-item blocks intact',
  population=pop,crossing_target_blocks=int(((starts<MID)&(ends>=MID)).sum()),
  primary_metrics=PRIMARY,bootstrap=dict(repeats=2000,seed=2024,confidence=.95,unit='paired user clusters within each time segment',estimand='sample mean'),
  inference_or_training=False,reselect_models=False,implementation_sha256=CORE_SHA,
  source_sha256=dict(bound),python=sys.version,numpy=np.__version__,pyarrow=pyarrow.__version__,argv=sys.argv)
 save(OUT/'REQUEST.json',request)
 all_keys=sorted(points['H0_best_R32']['metrics'])
 arrays={}; old_checks={}
 for name,p in points.items():
  protocol=metas[name]['metric_protocol']
  kwargs=dict(catalog_size=metas[name]['catalog_size'],list_length=protocol['list_length'],
              aggregate_ks=tuple(protocol['aggregate_ks']),max_candidate_lists=protocol['candidate_lists'])
  values=np.empty((N,len(all_keys)),dtype=np.float64)
  with gzip.open(p['predictions'],'rt',encoding='utf-8') as f:
   count=0
   for i,line in enumerate(f):
    assert i<N
    row=json.loads(line); t=truth[i]
    assert row['dataset_index']==i and row['sample_id']==t['sample_id']
    assert row['user_id']==t['user_id'] and row['targets']==t['target_item_ids']
    assert row['cold_target']==any(item not in seen_items for item in t['target_item_ids'])
    metrics=core.recompute_row(row,**kwargs)
    assert sorted(metrics)==all_keys
    assert max(abs(metrics[k]-row['metrics'][k]) for k in all_keys)<2e-6
    values[i]=[metrics[k] for k in all_keys];count+=1
   assert count==N
  means=values.mean(axis=0)
  assert max(abs(float(means[i])-p['metrics'][k]) for i,k in enumerate(all_keys))<2e-12
  arrays[name]=values
  if name!='H1_freeze_R32':
   arm,epoch=('H0',32) if name=='H0_best_R32' else ('H1',20)
   legacy=next(x for x in old['temporal_validation']['curves'] if x['arm']==arm and x['epoch']==epoch)
   idx=all_keys.index('list/ndcg@5')
   old_checks[name]={seg:abs(float(values[mask,idx].mean())-legacy[seg]) for seg,mask in masks.items()}
   assert max(old_checks[name].values())<2e-12
  log(event='prediction_replayed',model=name,examples=N,full_ndcg5=float(means[all_keys.index('list/ndcg@5')]))
 summaries={seg:{name:{k:float(values[mask,j].mean()) for j,k in enumerate(all_keys)}
                  for name,values in arrays.items()} for seg,mask in masks.items()}
 comparisons={}
 idxs=[all_keys.index(k) for k in PRIMARY]
 pairs=[('H0_best_R32','H1_freeze_R32'),('H1_R20','H1_freeze_R32'),('H0_best_R32','H1_R20')]
 for seg in ('early','late'):
  mask=masks[seg]; uid,inverse=np.unique(users[mask],return_inverse=True)
  counts=np.bincount(inverse,minlength=len(uid))
  sums={}
  for name,v in arrays.items():
   accum=np.zeros((len(uid),len(PRIMARY)))
   np.add.at(accum,inverse,v[mask][:,idxs]);sums[name]=accum
  for baseline,treatment in pairs:
   e=core.cluster_bootstrap(counts,sums[baseline],sums[treatment],repeats=2000,seed=2024,confidence=.95)
   def fmt(kind):
    out={}
    for j,k in enumerate(PRIMARY):
     a,b,delta=(float(e[kind+'_'+field][j]) for field in ('baseline','treatment','delta'))
     out[k]=dict(baseline=a,treatment=b,delta=delta,relative_delta=delta/a if a else None,confidence_interval=e[kind+'_ci'][j].tolist())
    return out
   label=seg+'__'+treatment+'_minus_'+baseline
   comparisons[label]=dict(segment=seg,baseline=baseline,treatment=treatment,population=pop[seg],
                            primary=fmt('sample'),user_average=fmt('user'))
   log(event='paired_done',comparison=label,ndcg5=comparisons[label]['primary']['list/ndcg@5'])
 for p,digest in bound.items(): assert sha(p)==digest,('Source changed',p)
 results=dict(status='passed',format='freeze35_fixed_temporal_validation_v1',
  checked_utc=datetime.now(timezone.utc).isoformat(),request_sha256=sha(OUT/'REQUEST.json'),
  population=pop,cutoff_us=MID,summaries=summaries,comparisons=comparisons,
  historical_ndcg_reproduction_absolute_errors=old_checks,source_sha256=bound,
  limitations=['Post-hoc diagnosis; both time segments contributed to full-validation checkpoint selection.',
   'A single training seed; user-cluster intervals exclude training-seed and model-selection uncertainty.',
   'H1 R20 and frozen R32 have T200; continued recommender optimization is a bundled intervention including fixed encoding and the original LR schedule.',
   'H0 best R32 has T360 and a different training trajectory; not a matched causal Mixer comparison.',
   'Aggregated @10/20/50 combine candidate five-item lists and are not one generated 50-item list.',
   'Not an independent future-time test; does not establish a unique cause of temporal generalization.'])
 save(OUT/'RESULTS.json',results)
 with (OUT/'METRICS.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['segment','examples','users','model',*all_keys])
  for seg,models in summaries.items():
   for name,m in models.items():w.writerow([seg,pop[seg]['examples'],pop[seg]['users'],name,*[m[k] for k in all_keys]])
 with (OUT/'PAIRED.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['comparison','metric','baseline','treatment','delta','relative_delta','ci_low','ci_high'])
  for label,c in comparisons.items():
   for k,m in c['primary'].items():w.writerow([label,k,m['baseline'],m['treatment'],m['delta'],m['relative_delta'],*m['confidence_interval']])
 save(OUT/'COMPLETE.json',dict(status='passed',finished_utc=datetime.now(timezone.utc).isoformat(),
  files={name:sha(OUT/name) for name in ('analyze.py','REQUEST.json','RESULTS.json','METRICS.csv','PAIRED.csv')},
  only_cpu_statistics=True,training_or_inference_started=False,full_goal_complete=False))
 log(event='complete',output=str(OUT))
if __name__=='__main__': main()

