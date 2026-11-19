"""Bounded one-H100 diagnostic only. Never enters the formal training runner."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
CODE = HERE / 'code'
sys.path.insert(0, str(CODE))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scripts.listwise_diffgrm.run import build_context, training_updates, phase_indices, evaluate
from scripts.listwise_diffgrm.runtime import build_model
from scripts.listwise_diffgrm.alignment import build_alignment
from scripts.listwise_diffgrm.training import AlternatingTrainer, seed_all, rng_state, restore_rng, cpu_copy, atomic_save
from scripts.listwise_diffgrm.execution import TrainingTensorCache

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
STAGE = HERE.parent


def write(name, value):
    p = OUT / name
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def event(**values):
    print(json.dumps(dict(unix=time.time(), **values)), flush=True)


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k, v in x.items(): h.update(str(k).encode()); visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def fingerprints(trainer):
    return {key: digest(value) for key, value in dict(
        model=trainer.model.state_dict(), alignment=trainer.alignment.state_dict(),
        model_grad={n:p.grad for n,p in trainer.model.named_parameters()},
        alignment_grad={n:p.grad for n,p in trainer.alignment.named_parameters()},
        r_optimizer=trainer.r_optimizer.state_dict(), t_optimizer=trainer.t_optimizer.state_dict(),
        scheduler=trainer.scheduler.state_dict(), rng=rng_state()).items()}


def make_trainer(cfg, assets, count, mixer):
    seed_all(cfg['seed'])
    model = build_model(cfg, assets.make_tokenizer()).cuda()
    alignment = build_alignment(assets, mixer, config_overrides=cfg['alignment']).cuda()
    return AlternatingTrainer(model, alignment, cfg, count, precision='bf16')


def profile_step(trainer, batches, kind):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        trainer.update(batches, kind)
        torch.cuda.synchronize()
    rows = []
    for item in prof.key_averages():
        rows.append(dict(name=item.key, count=item.count, cpu_self_ms=item.self_cpu_time_total/1000,
                         device_self_ms=getattr(item, 'self_device_time_total', getattr(item, 'self_cuda_time_total', 0))/1000))
    return dict(cpu_top=sorted(rows,key=lambda x:x['cpu_self_ms'],reverse=True)[:15],
                gpu_top=sorted(rows,key=lambda x:x['device_self_ms'],reverse=True)[:15],
                cpu_self_total_ms=sum(x['cpu_self_ms'] for x in rows),
                gpu_self_total_ms=sum(x['device_self_ms'] for x in rows))


def run_path(cfg, data, assets, mixer, optimized, label):
    cfg = copy.deepcopy(cfg); cfg['execution']['optimized_io'] = True
    cfg['execution']['optimized_model_checks'] = optimized
    trainer = make_trainer(cfg, assets, len(data), mixer)
    initial = fingerprints(trainer)
    result = dict(optimized=optimized, initial=initial, phases={})
    retained_batches = {}
    for kind, phase_number in [('R', 0), ('T', 5)]:
        # Eight real global batches: first warms up, remaining seven are timed.
        order = phase_indices(cfg, len(data), phase_number, {'kind':kind,'updates':8})[:8192]
        seed_all(cfg['seed'] + 100003*(phase_number+1))
        torch.cuda.reset_peak_memory_stats()
        rows = []
        with training_updates(data, order, cfg, 0, 'cuda:0') as updates:
            it = iter(updates)
            for u in range(8):
                torch.cuda.synchronize(); started = time.perf_counter()
                actual, batches = next(it)
                batches = list(batches)
                torch.cuda.synchronize(); ready = time.perf_counter()
                assert actual == u and len(batches) == 4 and all(len(b['target_item_ids'])==256 for b in batches)
                values = trainer.update(batches, kind)
                torch.cuda.synchronize(); ended = time.perf_counter()
                rows.append(dict(data_wait_s=ready-started, update_s=ended-ready,
                                 total_s=ended-started, values=values, rng=digest(rng_state())))
                retained_batches[kind] = batches
        result['phases'][kind] = dict(rows=rows, mean_update_s=statistics.mean(r['update_s'] for r in rows[1:]),
            mean_total_s=statistics.mean(r['total_s'] for r in rows[1:]),
            lists_per_second=1024/statistics.mean(r['total_s'] for r in rows[1:]),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            fingerprint=fingerprints(trainer))
        event(event='phase_done', variant=label, optimized=optimized, phase=kind,
              seconds=result['phases'][kind]['mean_total_s'], memory=result['phases'][kind]['peak_allocated_gib'])
    trainer.refresh_catalog()
    torch.cuda.synchronize(); started=time.perf_counter()
    ev=evaluate(trainer,assets,split='validation',destination=OUT/f'{label}_{optimized}_eval',limit=32)
    torch.cuda.synchronize()
    result['evaluation']=dict(seconds=time.perf_counter()-started, metrics=ev['metrics'],examples=ev['examples'],beam=32)
    import gzip
    with gzip.open(OUT/f'{label}_{optimized}_eval/per_sample.jsonl.gz','rt') as f:
        result['prediction_sha']=digest([json.loads(line) for line in f])
    if optimized:
        arm='H1' if mixer else 'H0'
        binding={'diagnostic':True,'config':cfg}
        torch.cuda.synchronize(); started=time.perf_counter()
        snapshot=trainer.checkpoint(binding,{'phase':0,'update':0,'preflight_only':True},arm)
        result['checkpoint_cpu_copy_s']=time.perf_counter()-started
        started=time.perf_counter(); cp=OUT/f'{label}_diagnostic_only.pt'
        atomic_save(cp,snapshot)
        result['checkpoint_write_s']=time.perf_counter()-started
        result['checkpoint_bytes']=cp.stat().st_size
        # Exact next-update check after restoring all state. Never a formal fork.
        before_rng=rng_state()
        expected_values=trainer.update(retained_batches['R'],'R')
        expected=fingerprints(trainer)
        trainer.restore(snapshot,binding,arm=arm,allow_preflight=True)
        self_values=trainer.update(retained_batches['R'],'R')
        result['restore_exact']=(expected_values==self_values and expected==fingerprints(trainer))
        result['profiles']={kind:profile_step(trainer,retained_batches[kind],kind) for kind in ('R','T')}
        del snapshot
    write(f'{label}_{"fast" if optimized else "reference"}.json',result)
    del trainer, retained_batches
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    assert torch.cuda.device_count()==1, 'Only one GPU is authorized'
    assert 'H100' in torch.cuda.get_device_name(), 'H100 required'
    torch.set_num_threads(4)
    cfg=json.loads((CODE/'configs/09_kuairec_listwise/h0_prefix_asymmetric.json').read_text())
    cfg['dataset']=str(STAGE/'goal_20261027_r1/inputs/dataset')
    cfg['assets']=str(STAGE/'goal_20261027_r1/assets/pse_v1')
    started=time.perf_counter()
    bootstrap, assets, data, _=build_context(cfg,'cuda:0','bf16',False,prepare_training=False)
    chosen=np.unique(np.concatenate([phase_indices(cfg,len(data),p,{'kind':k,'updates':8})[:8192]
                                    for k,p in [('R',0),('T',5)]]))
    class SelectedRows:
        def __len__(self):return len(chosen)
        def __getitem__(self,i):return data[int(chosen[i])]
        def collator(self):return data.collator()
    subset_cache=TrainingTensorCache(SelectedRows())
    class RemappedCache:
        def select(self,indices,*,pin=False):
            local=np.searchsorted(chosen,indices)
            assert np.array_equal(chosen[local],indices)
            return subset_cache.select(local,pin=pin)
    data._training_tensor_cache=RemappedCache()
    setup_s=time.perf_counter()-started
    del bootstrap; gc.collect(); torch.cuda.empty_cache()
    report=dict(device=torch.cuda.get_device_name(),total_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
        torch=torch.__version__,cuda=torch.version.cuda,train_rows=len(data),cached_diagnostic_rows=len(chosen),
        setup_and_cache_s=setup_s,diagnostic_row_sha=digest(chosen),
        effective_batch=1024,microbatch=256,timed_steps_per_phase=7,warmup_steps_per_phase=1,variants={})
    write('PROGRESS.json',report)
    event(event='setup_done',seconds=setup_s,rows=len(data),device=report['device'])
    for mode in ('asymmetric','cross_attention'):
        for mixer in (False,True):
            label=('H1' if mixer else 'H0')+('-A' if mode=='asymmetric' else '-B')
            cfg['model']['h_prefix_mode']=mode
            try:
                ref=run_path(cfg,data,assets,mixer,False,label)
                fast=run_path(cfg,data,assets,mixer,True,label)
                checks=dict(initial=ref['initial']==fast['initial'],prediction=ref['prediction_sha']==fast['prediction_sha'],
                    restore=fast['restore_exact'])
                for kind in ('R','T'):
                    checks[kind+'_state']=ref['phases'][kind]['fingerprint']==fast['phases'][kind]['fingerprint']
                    checks[kind+'_values_rng']=all(a['values']==b['values'] and a['rng']==b['rng']
                        for a,b in zip(ref['phases'][kind]['rows'],fast['phases'][kind]['rows']))
                report['variants'][label]=dict(checks=checks,exact=all(checks.values()),oom=False,
                    reference=ref,optimized=fast)
            except torch.cuda.OutOfMemoryError:
                report['variants'][label]=dict(oom=True,error=traceback.format_exc())
                gc.collect();torch.cuda.empty_cache()
            write('PROGRESS.json',report)
            event(event='variant_done',variant=label,exact=report['variants'][label].get('exact'),oom=report['variants'][label]['oom'])
    report['all_exact']=all(v.get('exact',False) for v in report['variants'].values())
    write('COMPLETE.json',report)
    event(event='complete',all_exact=report['all_exact'])


if __name__=='__main__':
    signal.alarm(1500)
    monitor_file=(OUT/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw',
                               '--format=csv','-l','2'],stdout=monitor_file,stderr=subprocess.STDOUT)
    try:
        main()
    except BaseException:
        write('FAILED.json',{'traceback':traceback.format_exc()});raise
    finally:
        monitor.terminate();monitor.wait(timeout=10);monitor_file.close()
