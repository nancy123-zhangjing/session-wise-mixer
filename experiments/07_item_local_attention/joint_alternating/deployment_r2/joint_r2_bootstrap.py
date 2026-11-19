"""Exclusive r2 CPU deployment/preflight; no training, evaluation or GPU work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import sys

BASE = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07')
OLD, NEW = BASE / 'goal_20261020_item_local_joint_r1', BASE / 'goal_20261020_item_local_joint_r2'
PROJECT = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
OLD_CODE = 'a6236b7ad7bdead84869e83221049268d3a123ab'
ENGINE = '1c879252f6395d6e64462511823c3929b9c42c00'
CAMPAIGN = 'stage07_item_local_alternating_joint_seed2024_20261020_r1'
CPU_RECEIPT = 'experiments/07_item_local_attention/joint_alternating/cpu_acceptance_20261020/ACCEPTANCE.json'

def need(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(path, *args): return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
def now(): return datetime.now(timezone.utc).isoformat()

def write(path, value, first=False):
    temporary = path if first else path.with_suffix('.json.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    if not first: temporary.replace(path)

def old_evidence():
    for name, commit in (('code', OLD_CODE), ('engine', ENGINE)):
        need(git(OLD / name, 'rev-parse', 'HEAD') == commit and not git(OLD / name, 'status', '--porcelain'), 'Old checkout is not the frozen clean source: ' + name)
    preparation = read(OLD / 'delivery/JOINT_PREPARATION.json')
    need(preparation['status'] == 'CPU_PREPARATION_PASS' and preparation['tests_passed'] == 20
         and preparation['root_commit'] == OLD_CODE and preparation['engine_commit'] == ENGINE, 'Missing passed old CPU20 preparation')
    cpu = read(OLD / 'code' / CPU_RECEIPT)
    need(cpu['status'] == 'CPU_GATES_PASS' and cpu['total_tests'] == 20
         and len(cpu['checks']) == 4 and all(row['passed'] and row['returncode'] == 0 for row in cpu['checks']), 'CPU20 source receipt is incomplete')
    identities = [read(OLD / 'runs' / run_id(v) / 'code_identity.json')['engine_sources'] for v in ('control', 'item_local')]
    need(identities[0] == identities[1] and len(identities[0]) == 13, 'Old prepared arms lack matching thirteen-source identities')
    for name, digest in identities[0].items():
        need(sha(OLD / 'engine' / name) == digest, 'Old engine source bytes differ: ' + name)
    for name, digest in cpu['source_files'].items():
        if name.startswith('external/DiffGRM/'):
            need(identities[0][name.removeprefix('external/DiffGRM/')] == digest, 'CPU20 source hash differs: ' + name)
    return identities[0]

def run_id(variant): return f'stage07_item_local_joint_{variant}_seed2024_20261020_r1'

PREPARE = r'''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in {'torch','cuda','cupy','pycuda'}: raise RuntimeError('Prepare imported Torch/CUDA: '+name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
if any(x.split('.')[0] in {'torch','cuda','cupy','pycuda'} for x in sys.modules): raise RuntimeError('Import guard escaped')
print('PREPARE_CPU_IMPORT_GUARD_PASS')
'''

PREFLIGHT = r'''
import importlib, importlib.util, json, os, sys
from pathlib import Path
root, variant, target = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('joint_runner', root/'code/scripts/train/run_item_local_joint.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
work = root/'runs'/runner.run_id(variant)
config = runner.read_json(work/'resolved_config.json')
config.update(run_id=config['run_id']+'_CPU_preflight', resume_training_state=False,
              training_deadline_unix_seconds=None, training_deadline_reserve_seconds=0)
config['run_local_time'] = config['run_id']
for key, sub in {'log_dir':'logs','tensorboard_log_dir':'tensorboard','ckpt_dir':'checkpoints',
    'run_metadata_dir':'run_metadata','diffusion_diagnostics_log_dir':'diagnostics',
    'position_loss_log_dir':'position_loss','training_state_dir':'training_state'}.items(): config[key] = str(target/sub)
runner.write_json(target/'resolved_config.json', config)
(target/'runtime').mkdir(); os.chdir(target/'runtime')
os.environ.update(HF_HOME=str(work/'workspace/huggingface'), HF_DATASETS_CACHE=str(work/'workspace/huggingface/datasets'))
sys.path.insert(0, str(root/'engine'))
import torch
from torch.utils.data import DataLoader
import genrec.utils as utils
utils.get_local_time = lambda: config['run_id']
trainer_module = importlib.import_module('genrec.models.DIFF_GRM.trainer')
from genrec.models.DIFF_GRM.training_state import capture_rng
from genrec.pipeline import Pipeline
def forbidden(*args, **kwargs): raise RuntimeError('CPU preflight may not fit/evaluate')
trainer_module.DIFF_GRMTrainer.fit = forbidden; trainer_module.DIFF_GRMTrainer.evaluate = forbidden
pipeline = Pipeline('DIFF_GRM', 'AmazonReviews2014', config_dict=config)
model = pipeline.model
runner.need(not torch.cuda.is_available() and not torch.cuda.is_initialized() and
    str(pipeline.config['device']) == 'cpu' and pipeline.accelerator.device.type == 'cpu' and
    pipeline.accelerator.num_processes == 1 and not pipeline.config['use_ddp'], 'CPU/single-process contract failed')
runner.need(pipeline.config['mixed_precision'] == 'bf16' and pipeline.accelerator.mixed_precision == 'bf16' and
    pipeline.config['gradient_accumulation_steps'] == pipeline.accelerator.gradient_accumulation_steps == 16, 'BF16/16 accumulation changed')
runner.need(model.joint_crosspse_enabled and model.item_local_attention_enabled == (variant == 'item_local') and
    not model.joint_alignment.mixer.mixer_enabled and model.joint_alignment.gradient_mode == 'diger_st' and
    pipeline.config['crosspse_train_base_codebooks'] is True and model.joint_alignment.mixer.train_base_codebooks and
    isinstance(model.joint_alignment.mixer.base_codebooks, torch.nn.Parameter), 'Joint/local/codebook configuration changed')
runner.need(Path(pipeline.tokenizer.semantic_ids_path).resolve() == Path(config['precomputed_semantic_ids_path']).resolve(), 'Consumed SID differs')
records = runner.read_json(work/'input_identities.json')['artifacts']
runner.need(len(records) == 13, 'Expected thirteen inputs')
for row in records:
    path = runner.stage05.under(work/'workspace', row['path'])
    runner.need(path.stat().st_size == row['size_bytes'] and runner.sha(path) == row['sha256'], 'Input SHA mismatch: '+row['path'])
counts = {split:len(pipeline.tokenized_datasets[split]) for split in ('train','val','test')}
runner.need(counts == dict(train=105668,val=22363,test=22363), 'Actual tokenized dataset counts differ')
loader = DataLoader(pipeline.tokenized_datasets['train'], batch_size=config['train_batch_size'], shuffle=True, collate_fn=pipeline.tokenizer.collate_fn['train'])
model.to(pipeline.accelerator.device)
state = {name:value for name,value in model.state_dict().items() if not name.startswith('item_local_attention.')}
receipt = dict(status='CPU_PIPELINE_CONSTRUCTION_PASS', variant=variant, counts=counts, device='cpu',
    initial_base_state_sha256=runner.json_sha(runner.nested_summary(state)),
    initial_rng_sha256=runner.json_sha(runner.nested_summary(capture_rng(loader))),
    input_sha_verified=13, trainer_import_module=trainer_module.__name__, fit_called=False, evaluate_called=False,
    gpu_started=False, mixed_precision='bf16', gradient_accumulation_steps=16,
    rng_scope='CPU two-arm equality only; CUDA RNG is None, not a GPU replay proof')
runner.write_json(target/'parsed_config.json', {k:str(v) if k=='device' else v for k,v in pipeline.config.items() if k!='accelerator'})
runner.write_json(target/'receipt.json', receipt)
print(json.dumps(receipt, sort_keys=True))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--root-commit'); parser.add_argument('--root-bundle-sha')
    args = parser.parse_args()
    sources = old_evidence()
    original_assets = OLD / 'delivery/INPUT_ASSET_CHECK.json'
    assets = read(original_assets)
    need(assets['status'] == 'JOINT_INPUT_ASSETS_VERIFIED' and len(assets['files']) == 13
         and len({row['path'] for row in assets['files']}) == 13, 'Invalid old thirteen-input receipt')
    for row in assets['files']:
        path = PurePosixPath(row['path'])
        need(not path.is_absolute() and '..' not in path.parts and row['passed'] is True and row['exists'] is True
             and row['observed_sha256'] == row['sha256'] and row['absolute_path'] == str(PROJECT/path), 'Invalid input record')
    delivery = NEW / 'delivery'
    if args.init:
        need(not NEW.exists() and not NEW.is_symlink(), 'Refuse existing r2 root, including partial initialization')
        NEW.mkdir(); delivery.mkdir()
        shutil.copyfile(Path(__file__).resolve(), delivery/'joint_r2_bootstrap.py')
        shutil.copyfile(original_assets, delivery/'INPUT_ASSET_CHECK.json')
        write(delivery/'R2_INITIALIZATION.json', dict(status='R2_INITIALIZED_CPU_ONLY', utc=now(), new_root=str(NEW),
            preserved_old_root=str(OLD), input_receipt_sha256=sha(original_assets), bootstrap_sha256=sha(__file__), gpu_started=False), True)
        print('R2_INITIALIZED_CPU_ONLY ' + str(NEW)); return 0
    need(re.fullmatch('[0-9a-f]{40}', args.root_commit or '') and re.fullmatch('[0-9a-f]{64}', args.root_bundle_sha or ''), 'Deploy requires full --root-commit and --root-bundle-sha')
    need(NEW.is_dir() and not NEW.is_symlink() and delivery.is_dir() and not delivery.is_symlink(), 'Run exclusive --init first')
    initialized = read(delivery/'R2_INITIALIZATION.json')
    need(initialized['bootstrap_sha256'] == sha(__file__) and initialized['input_receipt_sha256'] == sha(delivery/'INPUT_ASSET_CHECK.json') == sha(original_assets), 'Initialization/source/asset receipt changed')
    for name in ('code','engine','cpu_preflight'):
        path = NEW/name if name != 'cpu_preflight' else delivery/name
        need(not path.exists() and not path.is_symlink(), 'Refuse existing/partial deployment: '+str(path))
    bundle = delivery/'code_joint_r2.bundle'
    need(bundle.is_file() and not bundle.is_symlink() and sha(bundle) == args.root_bundle_sha, 'Code bundle transfer SHA mismatch')
    path = delivery/'JOINT_R2_PREPARATION.json'
    record = dict(schema='stage07_joint_r2_CPU_preparation_v1', status='CPU_PREPARATION_RUNNING', started_utc=now(),
        root_commit=args.root_commit, engine_commit=ENGINE, bootstrap_sha256=sha(__file__), checks=[], sources=sources,
        engine_tests_reused_from=str(OLD/'delivery/JOINT_PREPARATION.json'), old_CPU20_receipt_sha256=sha(OLD/'code'/CPU_RECEIPT), gpu_started=False, goal_complete=False)
    write(path, record, True)
    environment, record['platform_launcher_env'] = runpy.run_path(str(OLD/'code/scripts/train/item_local_gpu_job.py'))['single_process_environment'](os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
        ACCELERATE_USE_CPU='true', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='disabled', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TORCHDYNAMO_DISABLE='1')
    def command(label, argv):
        log = delivery/f'r2_{len(record["checks"]):02d}_{label}.log'
        check = dict(label=label, argv=list(map(str,argv)), log=str(log)); record['checks'].append(check)
        try:
            with log.open('x', encoding='utf-8') as handle:
                check['returncode'] = subprocess.run(check['argv'], env=environment, cwd=delivery, stdout=handle, stderr=subprocess.STDOUT, timeout=1800).returncode
        finally:
            if log.exists(): check['sha256'] = sha(log)
            write(path, record)
        need(check['returncode'] == 0, 'CPU step failed; preserve log/partial deployment: '+str(log))
        return log.read_text(encoding='utf-8', errors='replace')
    try:
        for name in ('code','engine'):
            command('clone_'+name, ['git','clone','--quiet','--no-hardlinks','--no-checkout',str(OLD/name),str(NEW/name)])
            need(not (NEW/name/'.git/objects/info/alternates').exists(), 'Clone shares alternate objects')
        command('verify_bundle', ['git','-C',str(NEW/'code'),'bundle','verify',str(bundle)])
        heads = command('bundle_heads', ['git','bundle','list-heads',str(bundle)])
        refs = [line.split(maxsplit=1)[1] for line in heads.splitlines() if line.split(maxsplit=1)[0] == args.root_commit]
        need(refs, 'Bundle does not advertise expected r2 commit')
        command('fetch_code', ['git','-C',str(NEW/'code'),'fetch',str(bundle),sorted(refs)[0]])
        need(git(NEW/'code','rev-parse','FETCH_HEAD') == args.root_commit, 'Fetched commit differs')
        for name, commit in (('code',args.root_commit),('engine',ENGINE)):
            command('checkout_'+name, ['git','-C',str(NEW/name),'checkout','--detach',commit])
            need(git(NEW/name,'rev-parse','HEAD') == commit and not git(NEW/name,'status','--porcelain'), 'Deployed source differs/is dirty')
        need(git(NEW/'code','merge-base',OLD_CODE,args.root_commit) == OLD_CODE, 'r2 code is not based on a623')
        for name,digest in sources.items(): need(sha(NEW/'engine'/name) == digest, 'New thirteen-source SHA mismatch: '+name)
        link = NEW/'code/external/DiffGRM'; need(not link.exists() and not link.is_symlink(), 'Engine link already exists')
        link.parent.mkdir(exist_ok=True); link.symlink_to(NEW/'engine', target_is_directory=True)
        command('ignored_engine_link', ['git','-C',str(NEW/'code'),'check-ignore','external/DiffGRM'])
        preflight = delivery/'cpu_preflight'; preflight.mkdir()
        for variant in ('control','item_local'):
            work = NEW/'runs'/run_id(variant)
            output = command('prepare_'+variant, [sys.executable,'-B','-c',PREPARE,str(NEW/'code/scripts/train/run_item_local_joint.py'),
                '--mode','prepare','--config',str(NEW/'code/configs/07_item_local_attention/joint_alternating_seed2024.json'),
                '--variant',variant,'--output',str(work),'--project-root',str(PROJECT),'--engine-root',str(NEW/'engine'),'--expected-engine-commit',ENGINE])
            need('PREPARE_CPU_IMPORT_GUARD_PASS' in output, 'Prepare import guard did not pass')
            need(read(work/'identity.json')['engine_source_sha256'] == sources, 'Prepared engine source identity differs')
            before = {str(p.relative_to(work)):sha(p) for p in work.rglob('*') if p.is_file() and 'workspace' not in p.relative_to(work).parts}
            target = preflight/variant; target.mkdir()
            command('production_pipeline_'+variant, [sys.executable,'-B','-c',PREFLIGHT,str(NEW),variant,str(target)])
            after = {str(p.relative_to(work)):sha(p) for p in work.rglob('*') if p.is_file() and 'workspace' not in p.relative_to(work).parts}
            need(before == after, 'CPU Pipeline wrote into formal artifact directories')
        receipts = [read(preflight/v/'receipt.json') for v in ('control','item_local')]
        for key in ('initial_base_state_sha256','initial_rng_sha256'): need(receipts[0][key] == receipts[1][key], 'CPU two-arm initialization differs: '+key)
        old_evidence()
        for name,commit in (('code',args.root_commit),('engine',ENGINE)): need(git(NEW/name,'rev-parse','HEAD') == commit and not git(NEW/name,'status','--porcelain'), 'Preparation changed source')
        need(sha(bundle) == args.root_bundle_sha, 'Bundle changed during deployment')
        record.update(status='CPU_PREPARATION_PASS', prepare_torch_imports_blocked=True, engine_tests_rerun=False,
            CPU_pipeline_preflights=receipts, formal_artifacts_unchanged=True, test_data_evaluated=False)
    except BaseException as error:
        record.update(status='CPU_PREPARATION_FAILED', error=dict(type=type(error).__name__,message=str(error))); raise
    finally:
        record['ended_utc'] = now(); write(path, record); print(json.dumps(record, ensure_ascii=False,indent=2))
    return 0

if __name__ == '__main__': raise SystemExit(main())
