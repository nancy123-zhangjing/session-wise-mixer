import ast, datetime, hashlib, json, math, pathlib, platform, sys
import numpy as np
import torch
root=pathlib.Path.cwd()
task=root/'experiments/09_kuairec_listwise/h_freeze35_20261029'
out=task/'temporal_validation_r32_20261029/ENCODING_FIXED_CHECK.json'
assert not out.exists()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
prior={str(p):sha(p) for p in out.parent.iterdir() if p.is_file()}
report_path=task/'final_analysis/A/analysis_report.json'
report_sha=sha(report_path)
assert report_sha=='66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937'
report=json.loads(report_path.read_text(encoding='utf-8'))
inputs={str(report_path):report_sha}
points={e:report['models'][f'A_R{e}_validation'] for e in (20,32)}
objects={}
for epoch,p in points.items():
 q=pathlib.Path(p['checkpoint']); observed=sha(q); assert observed==p['checkpoint_sha256']; inputs[str(q)]=observed
 assert (p['epoch'],p['split'],p['branch'])==(epoch,'validation','A')
 objects[epoch]=torch.load(q,map_location='cpu',weights_only=False)
old=objects[20]; envelope=objects[32]; new=envelope['inner']
assert old['format']==new['format']=='kuairec_listwise_v1' and envelope['format']=='stage09_recovery35_v1'
assert envelope['protocol']['branch']=='A' and envelope['inspection_only'] is False
assert (old['r_updates'],new['r_updates'],old['t_updates'],new['t_updates'])==(13860,22176,200,200)
assert old['arm']=='H1' and new['arm']=='H1-Freeze20-R35'
assert not old.get('preflight_only') and not envelope.get('preflight_only') and not new.get('preflight_only')
frozen_root=root/'tmp/h_freeze35_execution_20261029_committed_r1/code'
sources={}
for name in ['scripts/listwise_diffgrm/recovery.py','scripts/listwise_diffgrm/recovery35.py','scripts/listwise_diffgrm/training.py','scripts/listwise_diffgrm/model.py']:
 p=frozen_root/name; observed=sha(p); assert observed==new['binding']['code'][name]; inputs[str(p)]=observed; sources[name]=dict(path=str(p),sha256=observed)
for name in ['genrec/models/DIFF_GRM/cross_pse.py','genrec/models/DIFF_GRM/joint_cross_pse.py']:
 p=root/'external/DiffGRM'/name; observed=sha(p); assert observed==new['binding']['engine'][name]==old['binding']['engine'][name]; inputs[str(p)]=observed; sources[name]=dict(path=str(p),sha256=observed)
fpsource=pathlib.Path(sources['scripts/listwise_diffgrm/recovery.py']['path']).read_text(encoding='utf-8')
fpnode=next(n for n in ast.parse(fpsource).body if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
namespace=dict(torch=torch,np=np,hashlib=hashlib)
exec(compile(ast.Module(body=[fpnode],type_ignores=[]),str(frozen_root/'scripts/listwise_diffgrm/recovery.py'),'exec'),namespace)
fingerprint=namespace['fingerprint']
def bit_equal(a,b):
 return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),b.detach().contiguous().reshape(-1).view(torch.uint8))
def tree_compare(a,b,path=''):
 differences=[]; counts=dict(tensors=0,scalars=0,containers=0)
 def walk(x,y,at):
  if torch.is_tensor(x) or torch.is_tensor(y):
   counts['tensors']+=1
   if not(torch.is_tensor(x) and torch.is_tensor(y) and bit_equal(x,y)): differences.append(at)
  elif type(x) is not type(y): differences.append(at)
  elif isinstance(x,dict):
   counts['containers']+=1
   if set(x)!=set(y): differences.append(at+':keys'); return
   for k in x: walk(x[k],y[k],at+'/'+str(k))
  elif isinstance(x,(tuple,list)):
   counts['containers']+=1
   if len(x)!=len(y): differences.append(at+':length'); return
   for i,(xx,yy) in enumerate(zip(x,y)): walk(xx,yy,at+'/'+str(i))
  else:
   counts['scalars']+=1
   if x!=y: differences.append(at)
 walk(a,b,path)
 return dict(exactly_equal=not differences,leaf_counts=counts,differing_paths=differences)
def tensor_details(a,b):
 assert torch.is_tensor(a) and torch.is_tensor(b) and a.device.type==b.device.type=='cpu'
 same=bit_equal(a,b); record=dict(dtype=str(a.dtype),shape=list(a.shape),numel=a.numel(),r20_sha256=fingerprint(a),r32_sha256=fingerprint(b),exact_bytes_equal=same)
 assert a.dtype==b.dtype and a.shape==b.shape
 if a.is_floating_point(): assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
 record['changed_numeric_elements']=int(torch.count_nonzero(a!=b))
 record['max_abs_delta']=float((b.to(torch.float64)-a.to(torch.float64)).abs().max()) if a.numel() else 0.0
 return record
assert set(old['alignment'])==set(new['alignment']) and set(old['model'])==set(new['model'])
alignment={k:tensor_details(old['alignment'][k],new['alignment'][k]) for k in old['alignment']}
model={k:tensor_details(old['model'][k],new['model'][k]) for k in old['model']}
topt=tree_compare(old['t_optimizer'],new['t_optimizer'],'t_optimizer')
assert all(v['exact_bytes_equal'] for v in alignment.values()) and topt['exactly_equal']
fpchecks={}
for k in ('alignment','t_optimizer'):
 a,b=fingerprint(old[k]),fingerprint(new[k]); anchor=envelope['freeze_hashes'][k]; parent=envelope['parent_hashes'][k]
 assert a==b==anchor==parent
 fpchecks[k]=dict(r20=a,r32=b,r32_saved_freeze_fingerprint=anchor,r32_saved_parent_fingerprint=parent,all_equal=True)
config_checks={k:old['binding'][k]==new['binding'][k] for k in ('config','asset_manifest','data','engine')}
assert all(config_checks.values())
assert all(new['binding']['code'][k]==v for k,v in old['binding']['code'].items())
assert fingerprint(dict(model=new['model'],alignment=new['alignment']))==points[32]['model_fingerprint']
assert fingerprint(new['binding'])==points[32]['execution_fingerprint']
changed=[k for k,v in model.items() if not v['exact_bytes_equal']]
assert changed and not model['backbone.embedding.weight']['exact_bytes_equal']
def optimizer_clock(state):
 steps=[float(v['step']) for v in state['state'].values() if 'step' in v]
 return dict(state_entries=len(state['state']),step_entries=len(steps),unique_steps=sorted(set(steps)),param_group_count=len(state['param_groups']),learning_rates=[g['lr'] for g in state['param_groups']],group_parameter_counts=[len(g['params']) for g in state['param_groups']])
clocks={str(e):dict(r_updates=s['r_updates'],t_updates=s['t_updates'],diger_optimizer_step=int(s['alignment']['diger_optimizer_step']),diger_total_optimizer_steps=int(s['alignment']['diger_total_optimizer_steps']),r_optimizer=optimizer_clock(s['r_optimizer']),t_optimizer=optimizer_clock(s['t_optimizer'])) for e,s in ((20,old),(32,new))}
assert all(c['diger_optimizer_step']==200 and c['diger_total_optimizer_steps']==400 for c in clocks.values())
assert all(c['t_optimizer']['unique_steps']==[200.0] for c in clocks.values())
assert clocks['20']['r_optimizer']['unique_steps']==[13860.0] and clocks['32']['r_optimizer']['unique_steps']==[22176.0]
assert all(sha(p)==v for p,v in inputs.items()) and all(sha(p)==v for p,v in prior.items())
assert not torch.cuda.is_initialized()
result=dict(format='freeze35_encoding_fixed_cpu_readonly_v1',status='passed',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope='Read-only CPU tensor/state comparison of accepted A R20 and A R32 checkpoint files; not a model restore or forward evaluation',
 versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),python_executable=sys.executable,
 sources=dict(accepted_branch_report=dict(path=str(report_path),sha256=report_sha),checkpoints={str(e):dict(path=p['checkpoint'],sha256=p['checkpoint_sha256'],state_selector='$' if e==20 else '$.inner',arm=old['arm'] if e==20 else new['arm'],epoch=e,reference=p['reference']) for e,p in points.items()},frozen_implementation=sources),
 method=dict(load='torch.load(map_location=cpu, weights_only=False)',fingerprint='Exact fingerprint FunctionDef AST extracted from bound frozen recovery.py, without importing training/controller modules',fingerprint_function_source_sha256=hashlib.sha256(ast.get_source_segment(fpsource,fpnode).encode('utf-8')).hexdigest(),tensor_equality='same dtype/shape and torch.equal on contiguous uint8 views; per-tensor canonical SHA also reported',optimizer_equality='recursive exact type/key/scalar checks and tensor raw-byte comparisons; canonical full-state SHA cross-check'),
 envelope=dict(r20_format=old['format'],r32_format=envelope['format'],r32_inner_format=new['format'],r32_direct_parent_sha256=envelope['protocol']['parent_sha256'],r32_direct_parent_description='H1-freeze R26, not the historical R20 checkpoint',r32_freeze_phase=envelope['protocol']['freeze_phase'],r32_frozen_t_clock=envelope['protocol']['frozen_t_clock'],r32_inspection_only=envelope['inspection_only']),
 frozen_fingerprint_checks=fpchecks,alignment=dict(state_tensor_count=len(alignment),all_exact_bytes_equal=True,tensors=alignment,
 base_codebook_key='mixer.base_codebooks',mixer_keys=[k for k in alignment if k.startswith('mixer.') and k!='mixer.base_codebooks'],sid_key='hard_codes',hard_code_shape=list(old['alignment']['hard_codes'].shape),catalog_items_excluding_padding=old['alignment']['hard_codes'].shape[0]-1,changed_sid_items_excluding_padding=int((old['alignment']['hard_codes'][1:]!=new['alignment']['hard_codes'][1:]).any(dim=1).sum())),
 t_optimizer=topt,clocks=clocks,r_update_delta=new['r_updates']-old['r_updates'],t_update_delta=new['t_updates']-old['t_updates'],
 recommender=dict(state_tensor_count=len(model),changed_tensor_count=len(changed),unchanged_tensor_count=len(model)-len(changed),changed_keys=changed,tensors=model,
 sid_embedding_key='backbone.embedding.weight',sid_embedding_is_recommender_trainable_parameter_not_tokenizer=True,
 r20_state_fingerprint=fingerprint(old['model']),r32_state_fingerprint=fingerprint(new['model']),r_optimizer_state_changed=fingerprint(old['r_optimizer'])!=fingerprint(new['r_optimizer']),scheduler_state_changed=fingerprint(old['scheduler'])!=fingerprint(new['scheduler'])),
 scientific_binding_checks=dict(equal_sections=config_checks,all_old_source_sha_preserved=True,r32_model_and_execution_fingerprints_match_accepted_report=True),
 conclusion='Between the accepted R20 and R32 endpoints, all saved Tokenizer/base-codebook/Mixer/item-slot/hard-SID tensors and the entire T optimizer state are exactly unchanged, while the recommender state (including its SID embedding) and R optimizer advanced. This supports continued recommender optimization with fixed encoding over this continuation.',
 limits=['Endpoint saved-state comparison; it does not independently replay every intervening update.','No model instantiation, load_state_dict, optimizer restoration, training, forward pass, or newly computed effective crossed_codebooks.','initial_codebooks and _recommender_codebook_cache are persistent=False and absent from checkpoint state_dict; asset/config/source bindings agree, but unsaved transient cache values were not compared.','Unchanged deterministic encoding function, its saved inputs/parameters, and hard_codes are verified; no claim of newly executing deterministic re-encoding.','This is not final real model-restore acceptance, bitwise full-trajectory replay, causal proof for temporal metric changes, or Goal completion.'],
 input_sha256=inputs,input_bytes_unchanged_after_check=True,existing_temporal_files_sha256=prior,existing_temporal_files_unchanged=True,
 execution=dict(cpu_only=True,all_compared_tensors_cpu=True,cuda_initialized=False,model_instantiated=False,optimizer_restored=False,forward_or_training=False,platform_access=False))
with out.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
print(json.dumps(dict(status=result['status'],output=str(out),sha256=sha(out),alignment_tensors=len(alignment),t_optimizer=topt,changed_model_tensors=len(changed),total_model_tensors=len(model),sid_changed_items=result['alignment']['changed_sid_items_excluding_padding'],r_delta=result['r_update_delta'],t_delta=result['t_update_delta'],sid_embedding=model['backbone.embedding.weight'],clocks=clocks,frozen_hashes=fpchecks),ensure_ascii=False,indent=2))
