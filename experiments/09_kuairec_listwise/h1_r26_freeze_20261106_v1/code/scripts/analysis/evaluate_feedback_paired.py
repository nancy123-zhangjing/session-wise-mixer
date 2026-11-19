"""Replay final feedback-pilot validation and capture paired per-example SIDs.

Inference only: no optimizer, no fit, no SID refresh, no test-set selection.
The original Trainer.evaluate and its RNG-consuming diagnostics are retained.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('frozen', 'cross_recon', 'cross_rec')
PER_USER_METRICS = ('hit_rate@5', 'hit_rate@10', 'recall@5', 'recall@10',
                    'ndcg@5', 'ndcg@10', 'weighted_score', 'legal@10', 'duplicate@10')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


class ObservingLoader:
    """Observe metadata only; yield the exact original batch, with no RNG calls."""
    def __init__(self, loader, observer):
        self.loader, self.observer = loader, observer

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.observer.begin_batch(batch)
            yield batch
            self.observer.end_batch()


class SingleModeCollector:
    def __init__(self, codes, destination):
        self.codes = codes
        self.destination = destination
        self.rows = []
        self.pending = None
        self.consumed = False

    def begin_batch(self, batch):
        if self.pending is not None:
            raise RuntimeError('Unconsumed previous batch')
        targets = batch['target_item_id'].detach().cpu().tolist()
        history = batch['history_item_ids'].detach().cpu().tolist()
        if len(targets) != len(history):
            raise ValueError('Metadata length mismatch')
        self.pending = list(zip(targets, history))
        self.consumed = False

    def end_batch(self):
        if not self.consumed:
            raise RuntimeError('No confidence metrics captured for batch')
        self.pending = None

    def capture(self, predictions, labels, values, suffix=''):
        if suffix:
            return
        if self.pending is None or self.consumed:
            raise RuntimeError('Missing metadata or duplicate confidence call')
        preds = predictions.detach().cpu().tolist()
        target_sids = labels.detach().cpu().tolist()
        if len(preds) != len(self.pending) or len(target_sids) != len(preds):
            raise ValueError('Prediction/metadata shape mismatch')
        metrics = {key: values[key].detach().cpu().tolist() for key in PER_USER_METRICS}
        if any(not isinstance(items, list) or len(items) != len(preds) for items in metrics.values()):
            raise ValueError('A batch metric was mistaken for a per-user metric')
        for i, ((target, history), sid, top10) in enumerate(zip(self.pending, target_sids, preds)):
            target = int(target)
            if not 1 <= target <= len(self.codes) or sid != self.codes[target - 1]:
                raise ValueError('Evaluator label differs from official target SID')
            if len(top10) != 10 or any(len(row) != 4 for row in top10):
                raise ValueError('Expected unmodified Top-10 four-digit SIDs')
            if any(not 0 <= int(code) < 256 for row in top10 for code in row):
                raise ValueError('Prediction code outside official codebook range')
            first = next((rank for rank, row in enumerate(top10, 1) if row == sid), 0)
            row = {'sample_index': len(self.rows), 'target_item_id': target,
                   'history_item_ids': [int(item) for item in history if int(item) != 0],
                   'target_sid': sid, 'top10_sids': top10, 'first_hit_rank': first,
                   'metrics': {key: float(items[i]) for key, items in metrics.items()}}
            if not all(math.isfinite(value) for value in row['metrics'].values()):
                raise FloatingPointError('Nonfinite per-user metric')
            self.destination.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            self.rows.append(row)
        self.consumed = True


def restore_rng(torch, np, rng, device):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_cuda'] is None:
        raise ValueError('GPU replay requires the original CUDA RNG state')
    torch.cuda.set_rng_state(rng['torch_cuda'], device)


def validate_output(root, candidate, output_base=None):
    parent = root / 'outputs' / 'analysis' / 'stage05_feedback'
    if output_base is not None:
        # An explicit alternate is confined to this project's own approved SSD
        # analysis area. Never create links into arbitrary/shared result trees.
        inspire_root = Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized')
        approved = Path('/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis/outputs/analysis/stage05_feedback')
        if root != inspire_root or output_base.resolve() != approved:
            raise ValueError('Alternate output must be the explicitly approved own SSD analysis area')
        parent = approved
    raw = candidate if candidate.is_absolute() else root / candidate
    resolved = raw.resolve()
    if resolved.parent != parent or raw.is_symlink():
        raise ValueError('Use outputs/analysis/stage05_feedback/<new_unique_name>')
    for current in (parent, *parent.parents):
        if current.is_symlink():
            raise ValueError('Output parent must not be a symlink')
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def run(args):
    root = args.project_root.resolve()
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    expected_parent = root / 'outputs/pilots/stage05_feedback'
    if run_dir.parent != expected_parent or run_dir.is_symlink():
        raise ValueError('Input must be a completed stage05_feedback pilot')
    output = validate_output(root, args.output, args.output_base)
    summary_path = run_dir / 'summary.json'
    original = json.loads(summary_path.read_text(encoding='utf-8'))
    receipt = json.loads((run_dir / 'COMPLETE.json').read_text(encoding='utf-8'))
    if (receipt['summary_sha256'] != sha256(summary_path) or original['mode'] != 'pilot'
            or original['status'] != 'complete' or not original['all_protocol_arms_complete']
            or receipt['arms'] != list(ARMS) or not original['no_test_evaluation']):
        raise ValueError('Completed three-arm formal input receipt required')
    protocol = json.loads((run_dir / 'protocol.json').read_text(encoding='utf-8'))
    source = json.loads((run_dir / 'provenance.json').read_text(encoding='utf-8'))
    # Guard evaluation semantics and source against the actual training receipt.
    for relative, expected in source['source']['source_sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Historical runtime source changed: {relative}')
    for asset in source['assets']:
        if sha256(root / asset['path']) != asset['sha256']:
            raise ValueError(f'Bound input changed: {asset["path"]}')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Only a single GPU/process is supported')
    args.work_handle = tempfile.TemporaryDirectory(prefix='nancy_feedback_paired_')
    os.environ['HF_DATASETS_CACHE'] = str(Path(args.work_handle.name) / 'hf_datasets_cache')
    import numpy as np
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 GPU is required for validation replay')
    if torch.__version__ != source['torch'] or np.__version__ != source['numpy']:
        raise RuntimeError('Replay requires the original Torch and NumPy versions')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    sys.path.insert(0, str(root / 'external/DiffGRM'))
    sys.path.insert(0, str(root))
    runtime = load_module('_paired_runtime', root / 'scripts/train/run_diger_noise_pilot.py')
    helper = load_module('_paired_protocol', root / 'scripts/checks/diger_feedback_pilot_protocol.py')
    runtime.torch, runtime.np = torch, np
    runtime.phase_seed = helper.phase_seed
    from genrec.pipeline import Pipeline
    from genrec.utils import get_config
    from genrec.models.DIFF_GRM.evaluator import DIFF_GRMEvaluator
    from datasets import disable_caching
    disable_caching()
    results_by_arm = {row['arm']: row for row in original['arms']}
    if list(results_by_arm) != list(ARMS):
        raise ValueError('Unexpected arm order')
    snapshot_rows = {}
    input_files = {str(summary_path): sha256(summary_path)}
    for arm in ARMS:
        arm_row = results_by_arm[arm]
        row = next(row for row in arm_row['snapshots'] if row['label'] == 'decoder_3')
        if arm_row['final_snapshot'] != 'snapshots/decoder_3' or row['recommender_epochs'] != 3:
            raise ValueError('Replay only matched final decoder_3 snapshots')
        path = run_dir / arm / 'snapshots/decoder_3'
        for name, expected in row['files'].items():
            current = path / name
            if sha256(current) != expected:
                raise ValueError(f'Snapshot hash differs: {current}')
            input_files[str(current)] = expected
        snapshot_rows[arm] = row
    initial_arrays = []
    for arm in ARMS:
        initial_path = run_dir / arm / 'snapshots/initial/semantic_ids.npy'
        row = next(row for row in results_by_arm[arm]['snapshots'] if row['label'] == 'initial')
        if sha256(initial_path) != row['files']['semantic_ids.npy']:
            raise ValueError('Initial SID hash mismatch')
        initial_arrays.append(np.load(initial_path, allow_pickle=False))
        input_files[str(initial_path)] = row['files']['semantic_ids.npy']
    if not all(np.array_equal(initial_arrays[0], codes) for codes in initial_arrays[1:]):
        raise ValueError('Arms do not share initial SID')
    if len({results_by_arm[arm]['snapshots'][0]['model_tensor_state_sha256'] for arm in ARMS}) != 1:
        raise ValueError('Initial model tensors differ')
    official = {arm: np.load(run_dir / arm / 'snapshots/decoder_3/semantic_ids.npy', allow_pickle=False).tolist() for arm in ARMS}
    if initial_arrays[0].shape != (12101, 4):
        raise ValueError('Unexpected catalog size/digits')
    output.mkdir(parents=True, exist_ok=False)
    args.owned_output = output
    write_json(output / 'catalog_ids.json', {'initial': initial_arrays[0].tolist(), 'arms': official})
    started = time.perf_counter()
    summary = {'analysis_protocol': 'feedback_paired_validation_replay_v1', 'inference_only': True,
               'input_run': str(run_dir), 'input_summary_sha256': sha256(summary_path),
               'checkpoint_label': 'decoder_3', 'evaluation_level': 'sid', 'no_test_evaluation': True,
               'smoke_not_for_research': args.smoke, 'executed_script_sha256': sha256(__file__),
               'executed_analysis_sha256': sha256(Path(__file__).with_name('feedback_paired_analysis.py')),
               'executed_code_git_head': subprocess.check_output(
                   ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip(),
               'source': helper.source_receipt(root), 'inputs': input_files,
               'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'numpy': np.__version__,
               'arms': {}, 'rng_replay': 'saved pre-evaluation RNG; original Trainer.evaluate unchanged'}
    records = {}
    # Large mutable cache work goes to this job's temporary filesystem. Only
    # lightweight records/diagnostics are persisted to the unique project output.
    with args.work_handle as work:
        work = Path(work)
        os.environ['HF_DATASETS_CACHE'] = str(work / 'hf_datasets_cache')
        base = get_config('DIFF_GRM', 'AmazonReviews2014', [str(root / p) for p in protocol['config_layers']], {})
        base = runtime.absolute_engine_paths(base, root)
        base.update({'rand_seed': protocol['seed'], 'mixed_precision': 'no',
                     'joint_recommender_checkpoint': None, 'joint_crosspse_enabled': True,
                     'joint_gradient_mode': 'diger_st', 'joint_recommender_sia_weight': 0.0,
                     'joint_tokenizer_sia_weight': 0.0, 'joint_diger_recommendation_weight': 1.0,
                     'joint_diger_assignment_temperature': protocol['assignment_temperature'],
                     'joint_diger_softmax_temperature_start': protocol['softmax_temperature'],
                     'joint_diger_softmax_temperature_end': protocol['softmax_temperature'],
                     'joint_diger_gumbel_noise_start': 0.0, 'joint_diger_gumbel_noise_end': 0.0,
                     'joint_diger_uncertainty_decay': 'cosine', 'force_regenerate_opq': False,
                     'run_id': 'feedback_paired_replay', 'num_proc': 1,
                     'max_eval_batches': 2 if args.smoke else None})
        runtime.verify_controlled_settings(base, protocol)
        if (base.get('evaluation_level') != 'sid' or base.get('eval_expand_sid_to_items', False)
                or base.get('beam_search_modes') != ['confidence']):
            raise ValueError('Replay requires unchanged confidence SID evaluation')
        base = runtime.configure_paths(base, output / 'diagnostics')
        cache, cache_receipts = runtime.shadow_dataset_cache(base['cache_dir'], work, base['category'])
        base['cache_dir'] = str(cache)
        summary['isolated_cache'] = cache_receipts
        # Constructor initialization is discarded when final complete state is
        # loaded. No optimizer or training entrypoint is ever constructed/called.
        with runtime.working_directory(work):
            pipeline = Pipeline('DIFF_GRM', 'AmazonReviews2014', config_dict=base)
        if pipeline.accelerator.num_processes != 1:
            raise RuntimeError('Expected one Accelerator process')
        pipeline.model.to(device)
        if len(pipeline.tokenized_datasets['val']) != 22363:
            raise ValueError('Validation sample count differs from formal run')
        loader = torch.utils.data.DataLoader(
            pipeline.tokenized_datasets['val'], batch_size=base['eval_batch_size'], shuffle=False,
            collate_fn=pipeline.tokenizer.collate_fn['val'], num_workers=0,
            generator=torch.Generator().manual_seed(helper.phase_seed(protocol['seed'], 'evaluation', 0)))
        loader = pipeline.accelerator.prepare_data_loader(loader)
        original_calculate = DIFF_GRMEvaluator.calculate_metrics
        try:
            for arm in ARMS:
                arm_dir = output / arm
                arm_dir.mkdir()
                snapshot = run_dir / arm / 'snapshots/decoder_3'
                state = torch.load(snapshot / 'model.pt', map_location='cpu', weights_only=False)
                if runtime.tensor_state_hash(state) != snapshot_rows[arm]['model_tensor_state_sha256']:
                    raise ValueError('Saved model tensor hash mismatch')
                pipeline.model.load_state_dict(state, strict=True)
                if pipeline.model.joint_alignment.hard_codes[1:].cpu().tolist() != official[arm]:
                    raise ValueError('Checkpoint hard codes differ from official SID file')
                pipeline.model.sync_joint_tokenizer_mapping()
                runtime.configure_arm(pipeline, results_by_arm[arm]['settings'])
                pipeline.model.set_joint_phase('recommender')
                pipeline.model.generate_w_decoding_graph = True
                before = runtime.tensor_state_hash(pipeline.model.state_dict())
                training_state = torch.load(snapshot / 'training_state.pt', map_location='cpu', weights_only=False)
                if training_state['markers']['recommender_epochs'] != 3:
                    raise ValueError('Wrong pre-evaluation RNG snapshot')
                with gzip.open(arm_dir / 'records.jsonl.gz', 'xt', encoding='utf-8') as destination:
                    collector = SingleModeCollector(official[arm], destination)

                    def captured_calculate(evaluator, preds, labels, suffix=''):
                        values = original_calculate(evaluator, preds, labels, suffix=suffix)
                        collector.capture(preds, labels, values, suffix=suffix)
                        return values

                    DIFF_GRMEvaluator.calculate_metrics = captured_calculate
                    restore_rng(torch, np, training_state['rng'], device)
                    replayed = runtime.validate_snapshot(pipeline, ObservingLoader(loader, collector), 'decoder_3', 4, device)
                DIFF_GRMEvaluator.calculate_metrics = original_calculate
                after = runtime.tensor_state_hash(pipeline.model.state_dict())
                if before != after:
                    raise RuntimeError('Model/buffers changed during inference')
                expected = next(row['metrics'] for row in results_by_arm[arm]['validation'] if row['label'] == 'decoder_3')
                differences = {key: replayed['metrics'][key] - expected[key] for key in expected}
                if not args.smoke and any(abs(value) > 1e-7 for value in differences.values()):
                    write_json(arm_dir / 'REPLAY_MISMATCH.json', {'expected': expected, 'replayed': replayed, 'differences': differences})
                    raise RuntimeError(f'Overall metrics did not reproduce for {arm}; no subgroup conclusions permitted')
                expected_count = 16 if args.smoke else 22363
                if len(collector.rows) != expected_count:
                    raise ValueError('Incomplete per-example capture')
                for key in PER_USER_METRICS:
                    average = sum(row['metrics'][key] for row in collector.rows) / expected_count
                    if abs(average - replayed['metrics'][key]) > 1e-10:
                        raise RuntimeError('Per-user mean differs from original evaluator aggregate')
                records[arm] = collector.rows
                summary['arms'][arm] = {'metrics': replayed['metrics'], 'expected_metrics': expected,
                    'metric_differences': differences, 'examples': len(collector.rows),
                    'reproduced': not args.smoke, 'model_unchanged': before == after,
                    'records_sha256': sha256(arm_dir / 'records.jsonl.gz'), 'elapsed_seconds': replayed['elapsed_seconds']}
                write_json(arm_dir / 'replay.json', summary['arms'][arm])
                del state, training_state
                torch.cuda.empty_cache()
            analysis = load_module('_paired_analysis', Path(__file__).with_name('feedback_paired_analysis.py'))
            paired = analysis.analyze_records(records, initial_arrays[0].tolist(), official, 16 if args.smoke else 22363)
            write_json(output / 'paired_summary.json', paired)
            report = analysis.markdown_report(paired)
            if args.smoke:
                report = '# SMOKE ONLY: not research results\n\n' + report.replace('完整验证集', '仅前两个 batch')
            with (output / 'README.md').open('x', encoding='utf-8') as handle:
                handle.write(report)
        finally:
            DIFF_GRMEvaluator.calculate_metrics = original_calculate
            pipeline.accelerator.end_training()
    for path, expected in input_files.items():
        if sha256(path) != expected:
            raise RuntimeError(f'Read-only input changed: {path}')
    for asset in source['assets']:
        if sha256(root / asset['path']) != asset['sha256']:
            raise RuntimeError('Historical input changed during replay')
    summary.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                   inputs_unchanged=True, all_metrics_reproduced=not args.smoke)
    write_json(output / 'summary.json', summary)
    write_json(output / 'COMPLETE.json', {'summary_sha256': sha256(output / 'summary.json'),
               'paired_summary_sha256': sha256(output / 'paired_summary.json'), 'smoke': args.smoke})
    print(f'[PAIRED DONE] {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--output-base', type=Path, help='Explicitly approved own SSD analysis parent only')
    parser.add_argument('--smoke', action='store_true', help='First two batches; not for research comparisons')
    args = parser.parse_args(argv)
    args.owned_output = None
    args.work_handle = None
    try:
        run(args)
    except Exception as exc:
        if args.owned_output is not None:
            write_json(args.owned_output / 'FAILED.json', {'error': type(exc).__name__, 'message': str(exc), 'complete': False})
        raise
    finally:
        if args.work_handle is not None:
            args.work_handle.cleanup()


if __name__ == '__main__':
    main()
