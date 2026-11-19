"""Post-hoc CPU audit of H best validation/test reversal. No inference/training."""
import os,sys,json,gzip,hashlib,csv,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.listwise_diffgrm.analysis import cluster_bootstrap
O=Path(__file__).resolve().parent;D=ROOT/'outputs/kuairec_listwise_v1/run1'
R=ROOT/'outputs/stage09_restore/goal_20261027_r1';E=ROOT/'experiments/09_kuairec_listwise'
NITEM=10728;CUT80=1598609167593000;CUT90=1598881785624000
TZ=timezone(timedelta(hours=8));discount=1/np.log2(np.arange(5)+2);ideal=discount.sum()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def date(us):return datetime.fromtimestamp(int(us)/1e6,TZ).isoformat()
def log(**x):print(json.dumps(x),flush=True)
def counts(items):return np.bincount(np.asarray(items).reshape(-1),minlength=NITEM+1)
def tv(a,b):return float(abs(a/a.sum()-b/b.sum()).sum()/2)
refs=[]
def bind(p,expected=None):
 h=sha(p)
 if expected is not None:assert h==expected,(p,h,expected)
 refs.append(dict(path=str(p),sha256=h));return h
transfer=read(ROOT/'outputs/stage09_delivery/goal_20261027_r1/INPUT_TRANSFER_MANIFEST.json')
for e in transfer['files']:
 if e['path'].startswith('inputs/dataset/') and Path(e['path']).name in ['train.parquet','validation.parquet','test.parquet','train_seen.json','config.json']:
  bind(D/Path(e['path']).name,e['sha256'])
seen=read(D/'train_seen.json');seenmask=np.zeros(NITEM+1,dtype=bool);seenmask[seen['item_ids']]=True
traincount=np.zeros(NITEM+1,dtype=np.int64);trainusers=Counter();train_min=10**30;train_max=0
for batch in pq.ParquetFile(D/'train.parquet').iter_batches(batch_size=32768,columns=['user_id','target_item_ids','target_timestamp_us']):
 b=batch.to_pydict();traincount+=counts(b['target_item_ids']);trainusers.update(b['user_id'])
 times=np.asarray(b['target_timestamp_us']);train_min=min(train_min,int(times.min()));train_max=max(train_max,int(times.max()))
assert train_max<CUT80
seen_rank=sorted(np.flatnonzero(traincount),key=lambda i:(-traincount[i],i));head=set(seen_rank[:math.ceil(len(seen_rank)*.1)])
tail=set(seen_rank[math.ceil(len(seen_rank)*.5):]);popgroup=np.array(['no_train_target']*(NITEM+1),dtype=object)
popgroup[seen_rank]='mid';popgroup[list(head)]='head_top10pct_items';popgroup[list(tail)]='tail_bottom50pct_items'
data={};split_report={}
for split in ['validation','test']:
 cols=['sample_id','user_id','session_start_us','session_end_us','history_item_ids','history_timestamp_us','target_item_ids','target_timestamp_us','target_total_play_ms']
 vals={k:[] for k in ['sample_id','user','start','end','day','history_cold','repeat_target','session_hist','gap','cold','play']};targets=[]
 for batch in pq.ParquetFile(D/f'{split}.parquet').iter_batches(batch_size=8192,columns=cols):
  for x in batch.to_pylist():
   h=np.asarray(x['history_item_ids']);ht=np.asarray(x['history_timestamp_us']);t=x['target_item_ids'];tt=x['target_timestamp_us'];h=h[h>0];ht=ht[ht>0]
   assert len(h)==len(ht) and ht.max()<tt[0] and len(set(t))==5
   if split=='validation':assert x['session_start_us']>=CUT80 and x['session_end_us']<CUT90
   else:assert x['session_start_us']>=CUT90
   vals['sample_id'].append(x['sample_id']);vals['user'].append(x['user_id']);vals['start'].append(tt[0]);vals['end'].append(tt[-1]);vals['day'].append(date(tt[0])[:10]);vals['history_cold'].append(float((~seenmask[h]).mean()))
   vals['repeat_target'].append(sum(i in h for i in t)/5);vals['session_hist'].append(int(sum(ht>=x['session_start_us'])));vals['gap'].append((tt[0]-ht[-1])/1e6);vals['cold'].append(bool((~seenmask[t]).any()));vals['play'].append(x['target_total_play_ms']/1000);targets.append(t)
 z={k:np.array(v) for k,v in vals.items()};z['targets']=np.array(targets,dtype=np.int64);z['itemcount']=counts(targets);data[split]=z
 split_report[split]=dict(examples=len(targets),users=len(set(z['user'])),target_items=int((z['itemcount']>0).sum()),start=date(z['start'].min()),end=date(z['end'].max()),cold_sample_rate=float(z['cold'].mean()),cold_target_occurrence_rate=float((~seenmask[z['targets']]).mean()),history_cold_occurrence_rate=float(z['history_cold'].mean()),repeat_target_rate=float(z['repeat_target'].mean()),history_target_gap_median_seconds=float(np.median(z['gap'])),current_session_history_median=float(np.median(z['session_hist'])),play_seconds_mean=float(z['play'].mean()),train_unseen_users=len(set(z['user'])-set(trainusers)),train_unseen_user_samples=int(sum(u not in trainusers for u in z['user'])),target_mass_by_train_popularity={g:float(np.mean(popgroup[z['targets']]==g)) for g in sorted(set(popgroup))})
 log(stage='data_read',split=split,examples=len(targets))

manifests={a:read(ROOT/f'outputs/stage09_delivery/goal_20261027_r1/STAGE09_RUN_{a}_R1.json') for a in ['H0','H1']}
allval={};bestpred={};daily=[]
def replay(path,z,expected_sha):
 bind(path,expected_sha);out=np.zeros(len(z['user']));pred=np.zeros((len(out),5),dtype=np.int64);used=np.zeros(len(out),dtype=bool)
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   x=json.loads(line);i=x['dataset_index'];assert not used[i] and x['sample_id']==z['sample_id'][i] and x['user_id']==z['user'][i] and x['targets']==z['targets'][i].tolist()
   p=x['predictions'][0];assert len(p)==len(set(p))==5 and min(p)>0
   value=sum(discount[j] for j,item in enumerate(p) if item in x['targets'])/ideal
   assert abs(value-x['metrics']['list/ndcg@5'])<2e-6
   out[i]=value;pred[i]=p;used[i]=True
 assert used.all();return out,pred
for a in ['H0','H1']:
 allval[a]={}
 refsbyname={x['path']:x['sha256'] for x in manifests[a]['files']}
 for epoch in range(8,36,3):
  rel=f'runs/{a}/validation_R{epoch:02d}/per_sample.jsonl.gz';path=R/rel
  score,pred=replay(path,data['validation'],refsbyname[rel]);allval[a][epoch]=score
  if epoch=={'H0':32,'H1':20}[a]:bestpred[('validation',a)]=pred
  log(stage='validation_checkpoint_replayed',arm=a,epoch=epoch,ndcg=float(score.mean()))
testscore={}
for a in ['H0','H1']:
 path=R/'sidecars/evaluations/evaluations'/(a+'_test_beam32'+('_r2' if a=='H1' else ''))/'per_sample.jsonl.gz'
 audit=read(E/f'final_analysis_20261027/{a}_test_beam32_audit.json');testscore[a],bestpred[('test',a)]=replay(path,data['test'],audit['predictions_sha256']);assert abs(testscore[a].mean()-audit['metrics']['list/ndcg@5'])<1e-12
scores={'validation':np.stack([allval['H0'][32],allval['H1'][20]],axis=1),'test':np.stack([testscore['H0'],testscore['H1']],axis=1)}
def means(v):return dict(H0=float(v[:,0].mean()),H1=float(v[:,1].mean()),delta=float(np.diff(v,axis=1).mean()),relative_delta=float(v[:,1].mean()/v[:,0].mean()-1) if v[:,0].mean() else None)
groups={};paired={}
commonusers=set(data['validation']['user'])&set(data['test']['user'])
for split,z in data.items():
 v=scores[split];g={}
 groupsets={'all':np.ones(len(v),dtype=bool),'common_users':np.isin(z['user'],list(commonusers)),'cold_target':z['cold'],'seen_targets':~z['cold'],'no_cold_history':z['history_cold']==0,'has_cold_history':z['history_cold']>0,'target_repeated_in_history':z['repeat_target']>0,'no_target_in_history':z['repeat_target']==0}
 for name,mask in groupsets.items():
  if mask.any():g[name]=dict(examples=int(mask.sum()),**means(v[mask]))
 groups[split]=g
 u,idx=np.unique(z['user'],return_inverse=True);cnt=np.bincount(idx);sums=np.stack([np.bincount(idx,weights=v[:,j]) for j in range(2)],axis=1)
 b=cluster_bootstrap(cnt,sums[:,0:1],sums[:,1:2],repeats=10000)
 paired[split]=dict(**means(v),sample_delta_ci95=b['sample_ci'][0].tolist(),user_H0=float(b['user_baseline'][0]),user_H1=float(b['user_treatment'][0]),user_delta=float(b['user_delta'][0]),user_delta_ci95=b['user_ci'][0].tolist(),users=len(u))
 # Equal weights for the same users removes new-user and activity-composition differences.
 commonidx=np.isin(u,list(commonusers));paired[split]['common_users_equal_weight']=means(sums[commonidx]/cnt[commonidx,None])
 for day in sorted(set(z['day'])):
  mask=z['day']==day;daily.append(dict(split=split,day=day,examples=int(mask.sum()),**means(v[mask]),cold_sample_rate=float(z['cold'][mask].mean())))

# A symmetric descriptive decomposition of cold/seen sample mix; not causal attribution.
pv=float(data['validation']['cold'].mean());pt=float(data['test']['cold'].mean());mix=0;within=0
for g,wv,wt in [('cold_target',pv,pt),('seen_targets',1-pv,1-pt)]:
 dv=groups['validation'][g]['delta'];dt=groups['test'][g]['delta'];mix+=(wt-wv)*(dv+dt)/2;within+=(dt-dv)*(wv+wt)/2
assert abs(mix+within-(paired['test']['delta']-paired['validation']['delta']))<1e-12

# Item-specific DCG contributions sum exactly to the overall paired NDCG difference.
itemdelta={};itemdetail=[];prediction_profile={}
for split,z in data.items():
 contribution=np.zeros((NITEM+1,2));profile={}
 for j,a in enumerate(['H0','H1']):
  p=bestpred[(split,a)];hit=(p[:,:,None]==z['targets'][:,None,:]).any(axis=2)
  for rank in range(5):np.add.at(contribution[:,j],p[hit[:,rank],rank],discount[rank]/ideal)
  pc=counts(p);top=np.argsort(-pc)[1:1] # populated explicitly below, exclude PAD
  top=sorted(range(1,NITEM+1),key=lambda i:(-pc[i],i))[:20]
  profile[a]=dict(distinct_predicted_items=int((pc>0).sum()),top10_prediction_mass=float(sum(pc[top[:10]])/pc.sum()),top20=[dict(internal_item_id=int(i),predicted_slots=int(pc[i]),prediction_share=float(pc[i]/pc.sum()),target_occurrences=int(z['itemcount'][i]),target_share=float(z['itemcount'][i]/z['itemcount'].sum()),train_target_count=int(traincount[i])) for i in top])
 assert np.allclose(contribution.sum(axis=0)/len(z['user']),scores[split].mean(axis=0),atol=1e-12)
 itemdelta[split]=np.diff(contribution,axis=1)[:,0];prediction_profile[split]=profile
cv=data['validation']['itemcount'];ct=data['test']['itemcount'];nv=len(data['validation']['user']);nt=len(data['test']['user']);common=(cv>0)&(ct>0)
v=np.divide(itemdelta['validation'],cv,out=np.zeros(NITEM+1),where=cv>0);t=np.divide(itemdelta['test'],ct,out=np.zeros(NITEM+1),where=ct>0)
itemmix=float(np.sum(((ct/nt-cv/nv)*(v+t)/2)[common]));itemwithin=float(np.sum(((t-v)*(ct/nt+cv/nv)/2)[common]));unique=float((itemdelta['test'][~common]/nt-itemdelta['validation'][~common]/nv).sum())
reversal=itemdelta['test']/nt-itemdelta['validation']/nv
assert abs(itemmix+itemwithin+unique-reversal.sum())<1e-12
for i in sorted(range(1,NITEM+1),key=lambda i:reversal[i]):
 itemdetail.append(dict(internal_item_id=i,train_target_count=int(traincount[i]),val_target_count=int(cv[i]),test_target_count=int(ct[i]),val_target_share=float(cv[i]/cv.sum()),test_target_share=float(ct[i]/ct.sum()),val_ndcg_delta_contribution=float(itemdelta['validation'][i]/nv),test_ndcg_delta_contribution=float(itemdelta['test'][i]/nt),reversal_contribution=float(reversal[i])))

# Retrospective validation-time robustness: all these samples have already been used.
z=data['validation'];mid=(int(z['start'].min())+int(z['end'].max()))//2;early=z['start']<mid;late=~early
temporal=[];curveblocks=[]
for a in ['H0','H1']:
 selected=max(allval[a],key=lambda e:(allval[a][e][early].mean(),-e))
 temporal.append(dict(arm=a,early_selected_epoch=selected,early_mean=float(allval[a][selected][early].mean()),later_mean=float(allval[a][selected][late].mean()),full_selected_epoch={'H0':32,'H1':20}[a]))
 for epoch,x in allval[a].items():curveblocks.append(dict(arm=a,epoch=epoch,early=float(x[early].mean()),late=float(x[late].mean()),full=float(x.mean())))

# Load only SHA-bound CPU checkpoints to stratify target collisions; no model forward.
import torch
assert torch.version.cuda is None;torch.set_num_threads(1)
freeze=read(E/'formal_execution_20261027/FOUR_BEST_FREEZE_20261028.json');collision={}
for a in ['H0','H1']:
 path=R/f'runs/{a}/best.pt';bind(path,freeze['arms'][a]['best']['sha256']);state=torch.load(path,map_location='cpu',weights_only=False)
 codes=state['alignment']['hard_codes'].numpy();buckets=Counter(map(tuple,codes[1:]));size=np.array([0]+[buckets[tuple(x)] for x in codes[1:]])
 collision[a]=dict(unique_sids=len(buckets),splits={})
 for split,z in data.items():
  mask=(size[z['targets']]>1).any(axis=1);collision[a]['splits'][split]=dict(target_occurrences_in_collision_rate=float((size[z['targets']]>1).mean()),samples_with_collided_target=float(mask.mean()),collision_target_group=dict(examples=int(mask.sum()),**means(scores[split][mask])),no_collision_target_group=dict(examples=int((~mask).sum()),**means(scores[split][~mask])))
 del state

result=dict(status='PASS',created_at_utc=datetime.now(timezone.utc).isoformat(),scope='post-hoc read-only CPU data/prediction/checkpoint audit; no inference or training',
 definitions=dict(main_metric='single-list NDCG@5, five unique relevant targets, fixed idealDCG',train_popularity='retained training targets only; rank observed items by count, top10percent/head,bottom50percent/tail',temporal_validation='midpoint in wall-clock time, exploratory and already used by original selection',cold='any target absent from retained training histories+targets'),
 chronology=dict(train_start=date(train_min),train_end=date(train_max),cut80=date(CUT80),cut90=date(CUT90),all_eval_histories_strictly_before_targets=True,all_eval_sessions_within_assigned_split=True),
 splits=split_report,paired=paired,groups=groups,daily=daily,common_users=len(commonusers),
 target_shift=dict(tv_validation_test=tv(cv,ct),tv_train_validation=tv(traincount,cv),tv_train_test=tv(traincount,ct),common_target_items=int(common.sum()),validation_mass_common=float(cv[common].sum()/cv.sum()),test_mass_common=float(ct[common].sum()/ct.sum())),
 cold_mix_decomposition=dict(total_reversal=paired['test']['delta']-paired['validation']['delta'],composition_term=mix,within_group_term=within,note='Exact descriptive decomposition, not causal evidence'),
 item_decomposition=dict(total_reversal=float(reversal.sum()),shared_item_composition_term=itemmix,shared_item_conditional_term=itemwithin,nonshared_support_term=unique,note='Exact additive decomposition of item DCG contribution, no causal attribution; per-item estimates may be sparse'),
 top_negative_reversal_items=itemdetail[:20],top_positive_reversal_items=itemdetail[-20:][::-1],prediction_profile=prediction_profile,collision=collision,
 temporal_validation=dict(early_examples=int(early.sum()),late_examples=int(late.sum()),midpoint=date(mid),selections=temporal,curves=curveblocks),
 limitations=['Post-hoc examination of previously seen validation/test labels; no test-based parameter selection.', 'Only one training seed; conditional user bootstrap excludes checkpoint-selection uncertainty.', 'No H0-R20 or alternative H1 checkpoint test predictions; cannot separate training-epoch effect with current tests.', 'R20 vs R35 code collapse does not causally explain R20 best validation/test reversal.'],sources=refs,script_sha256=sha(Path(__file__)))
save(O/'RESULTS.json',result)
for name,rows in [('daily_best.csv',daily),('item_reversal.csv',itemdetail),('validation_temporal_curves.csv',curveblocks)]:
 with (O/name).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
log(stage='complete',paired=paired,target_shift=result['target_shift'],cold_mix=result['cold_mix_decomposition'],item_decomposition=result['item_decomposition'])
