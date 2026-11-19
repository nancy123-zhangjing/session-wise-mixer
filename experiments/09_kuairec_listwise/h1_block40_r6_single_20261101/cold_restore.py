"""Actual CPU cold restoration using independently extracted sealed code.

No optimizer updates or inference. CUDA RNG bytes are authenticated and retained
in source files; CPU inspection explicitly omits applying CUDA RNG. This does
not claim bitwise CPU/CUDA trajectory equivalence.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('extracted','parent','dataset','assets','output'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); start=time.monotonic()
    root=a.extracted.resolve(); code=root/'code'; run=root/'runs/H1-Block40-R6-R35'
    verified=read(root/'LOCAL_EXTRACTION_VERIFIED.json')
    if verified['status']!='passed': raise ValueError('Fresh independent extraction not verified')
    manifest=read(code/'E_EXECUTION_MANIFEST.json')
    for name,value in manifest['files'].items():
        if sha(code/name)!=value['sha256']: raise ValueError('Extracted code mismatch')
    if sha(a.parent)!=manifest['parent_sha256']: raise ValueError('Original parent external dependency mismatch')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(code))
    os.environ['DIFFGRM_ENGINE_ROOT']=str(code/'external/DiffGRM')
    import torch
    torch.set_num_threads(4)
    from scripts.listwise_diffgrm.block40_single import build_spread, verify_binding
    from scripts.listwise_diffgrm.recovery import fingerprint
    if not Path(sys.modules['scripts.listwise_diffgrm.block40_single'].__file__).resolve().is_relative_to(code):
        raise ValueError('Cold process imported another checkout')
    c,assets,data=build_spread(a.parent,manifest['parent_sha256'],'cpu',str(a.dataset.resolve()),str(a.assets.resolve()),exact_rng=False)
    from scripts.listwise_diffgrm.runtime import ENGINE
    if ENGINE.resolve()!=(code/'external/DiffGRM').resolve():
        raise ValueError('Cold process selected an engine outside the fresh extraction')
    model_source=Path(sys.modules['_kuairec_listwise_engine.model'].__file__).resolve()
    if not model_source.is_relative_to(ENGINE):
        raise ValueError('Cold process imported another engine model')
    best=read(run/'selection.json')['selected']['epoch']
    all_receipts={read(path)['path']:read(path) for path in sorted((run/'checkpoint_receipts').glob('*.json'))}
    keys={'initial_R05_T000.pt', 'last.pt'}
    for name in all_receipts:
        if name.startswith(('phase05_','phase06_','phase12_','phase39_')): keys.add(name)
    for epoch in {best,35}:
        remote=read(run/'validation_records'/f'R{epoch:02d}.json')['checkpoint']
        keys.add(Path(remote).name)
    complete=read(run/'complete.json')
    records=[]
    for name in sorted(keys):
        path=run/name; expected=complete['last_sha256'] if name=='last.pt' else all_receipts[name]['sha256']
        if sha(path)!=expected: raise ValueError('Cold checkpoint SHA mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        original_hash=fingerprint(state)
        if name!='last.pt' and original_hash!=all_receipts[name]['state_sha256']:
            raise ValueError('Cold checkpoint full state differs from its receipt')
        # Verify permitted relocations on both old and new bindings, then copy
        # only path metadata. Never rewrite source checkpoint bytes.
        verify_binding(state['parent_binding'],state['execution_binding'])
        saved=copy.deepcopy(state['execution_binding']); current=copy.deepcopy(c.execution_binding)
        relocated={}
        for field in ('dataset','assets'):
            relocated[field]=dict(saved=saved['config'].pop(field),local=current['config'].pop(field))
        if saved!=current: raise ValueError('Cold execution differs beyond explicit dependency paths')
        clone=copy.deepcopy(state)
        clone['execution_binding']=copy.deepcopy(c.execution_binding)
        clone['inner']['binding']=copy.deepcopy(c.execution_binding)
        clone['protocol']['source_proof']['paths']=copy.deepcopy(c.protocol['source_proof']['paths'])
        c.restore(clone,exact_rng=False)
        restored=c.checkpoint()
        equal={key:fingerprint(restored['inner'][key])==fingerprint(state['inner'][key])
            for key in ('model','alignment','r_optimizer','t_optimizer','scheduler','r_updates','t_updates','progress')}
        if not all(equal.values()) or not c.inspection_only: raise ValueError('Restored CPU state differs')
        expected_t_rng=copy.deepcopy(state['t_rng'])
        if expected_t_rng is not None:expected_t_rng['cuda']=None
        paired_t_rng_cpu_equal=fingerprint(restored['t_rng'])==fingerprint(expected_t_rng)
        if not paired_t_rng_cpu_equal:raise ValueError('CPU paired T random state differs beyond unapplied CUDA bytes')
        if fingerprint(state)!=original_hash or sha(path)!=expected: raise ValueError('Source checkpoint mutated')
        records.append(dict(name=name,sha256=expected,source_state_fingerprint=original_hash,
            restored_equal=equal,progress=state['inner']['progress'],r_updates=state['inner']['r_updates'],
            t_updates=state['inner']['t_updates'],dependency_path_relocation=relocated,
            saved_cuda_rng_present=state['inner']['rng']['cuda'] is not None,
            paired_t_rng_present=state['t_rng'] is not None,paired_t_rng_cpu_equal=paired_t_rng_cpu_equal))
        print(json.dumps(dict(event='real_CPU_cold_restore',checkpoint=name,passed=True)),flush=True)
        del state,clone,restored
    # Every saved stage must contain the same Tokenizer through each six-R
    # interval. These reads do not load or run a model or repeat old diagnostics.
    previous=None; stages=[]
    for name,receipt in sorted(all_receipts.items(),key=lambda item:(item[1]['progress']['phase'],item[0])):
        if sha(run/name)!=receipt['sha256']: raise ValueError('Boundary checkpoint bytes mismatch')
        state=torch.load(run/name,map_location='cpu',weights_only=False)
        inner=state['inner']
        state_hash=fingerprint(state)
        if (state_hash!=receipt['state_sha256'] or inner['r_updates']!=receipt['r_updates']
                or inner['t_updates']!=receipt['t_updates'] or inner['progress']!=receipt['progress']):
            raise ValueError('Boundary full state or clocks differ from receipt')
        for key in ('model','alignment','r_optimizer','t_optimizer','scheduler','rng','binding'):
            if key not in inner: raise ValueError('Incomplete checkpoint')
        c.validate_clocks(inner)
        align=fingerprint(inner['alignment']); topt=fingerprint(inner['t_optimizer'])
        t_rng=fingerprint(state['t_rng'])
        if previous and receipt['t_updates']==previous['t_updates']:
            if (align,topt,t_rng)!=(previous['alignment'],previous['t_optimizer'],previous['paired_t_rng']):
                raise ValueError('R changed Tokenizer, T optimizer or paired T RNG')
        row=dict(name=name,state_sha256=state_hash,t_updates=inner['t_updates'],r_updates=inner['r_updates'],alignment=align,t_optimizer=topt,
                 paired_t_rng=t_rng)
        stages.append(row); previous=row
        del state
    if len(stages)!=36: raise ValueError('Expected initial plus thirty-five saved R/T phases')
    report=dict(status='passed',scope='real CPU model/Tokenizer/dual optimizers/scheduler load and source state equality; no updates or inference',
        seconds=time.monotonic()-start,checkpoints=records,all_boundary_checkpoint_fields_and_clocks_verified=stages,
        code_manifest_sha256=sha(code/'E_EXECUTION_MANIFEST.json'),parent_sha256=sha(a.parent),
        extracted_engine_root=str(ENGINE),engine_model_import=str(model_source),torch_version=torch.__version__,
        source_script_sha256=sha(__file__),limitations='CPU inspection does not apply CUDA RNG and cannot resume training. Original CUDA RNG states stay intact in authenticated checkpoint files. The single-GPU update/restore implementation is inherited from accepted D; no new E CUDA replay was run under the user-requested reduced startup checks.')
    with a.output.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__': main()
