"""Load real engine modules without importing its unrelated single-item pipeline."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path(os.environ.get('DIFFGRM_ENGINE_ROOT', ROOT / 'external/DiffGRM')).resolve()
PACKAGE = '_kuairec_listwise_engine'


def engine_module(name):
    source = ENGINE / 'genrec/models/DIFF_GRM'
    if not (source / 'model.py').is_file():
        raise FileNotFoundError(f'DiffGRM engine missing: {source}')
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(source)]
        sys.modules[PACKAGE] = package
    # No fake model/data classes: genrec.model and its dependencies are real.
    return importlib.import_module(f'{PACKAGE}.{name}')


def build_model(config, tokenizer):
    from .model import ListwiseDiffGRM
    from .execution import enabled
    cfg = dict(config['model'])
    cfg['reuse_static_tensors'] = enabled(config)
    cfg['optimized_model_checks'] = config.get('execution', {}).get('optimized_model_checks', False)
    cfg.update(architecture=config['architecture'], list_length=config['list_length'],
               observed_history_max=config['history_max'],
               max_history_len=config['history_max'] + config['list_length'] - 1,
               joint_crosspse_enabled=False)
    backbone = engine_module('model').DIFF_GRM(cfg, None, tokenizer)
    return ListwiseDiffGRM(cfg, backbone)


def read_config(path, architecture=None, seed=None):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if architecture is not None:
        cfg['architecture'] = architecture
    if seed is not None:
        cfg['seed'] = seed
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if type(cfg.get('execution', {}).get('optimized_io', False)) is not bool:
        raise ValueError('execution.optimized_io must be a JSON boolean')
    from .prefix_decoder import MODES
    prefix_mode = cfg.get('model', {}).get('h_prefix_mode', 'history')
    if prefix_mode not in MODES or (prefix_mode != 'history' and cfg['architecture'] != 'H'):
        raise ValueError('Invalid h_prefix_mode, or decoder prefix used outside H')
    if type(cfg.get('alignment', {}).get('crosspse_code_embedding_enabled', True)) is not bool:
        raise ValueError('crosspse_code_embedding_enabled must be a JSON boolean')
    if cfg['architecture'] not in ('J', 'H'):
        raise ValueError('architecture must be J or H')
    if cfg['list_length'] != 5 or cfg['model']['n_digit'] != 4:
        raise ValueError('Formal KuaiRec protocol requires five items, four SID digits each')
    train = cfg['training']
    if train['effective_batch_size'] % train['microbatch_size']:
        raise ValueError('effective list batch must be divisible by microbatch size')
    if any(train[k] <= 0 for k in ('effective_batch_size', 'microbatch_size',
                                  'warmup_epochs', 'cycles', 'r_epochs_per_cycle',
                                  't_updates_per_cycle')):
        raise ValueError('Training sizes and budgets must be positive')
    if cfg['evaluation']['val_beam'] != cfg['evaluation']['test_beam']:
        raise ValueError('Validation/test beam must match in the primary protocol')
    evaluation = cfg['evaluation']
    for name, expected in {
        'tie_break': 'earlier',
        'collision_rule': 'reconstruction_error_then_original_id',
        'aggregation': 'list_score_order_then_item_position_first_occurrence',
    }.items():
        if evaluation[name] != expected:
            raise ValueError(f'Unsupported evaluation policy: {name}')
    if type(evaluation.get('filter_history')) is not bool or (
            evaluation['filter_history'] and evaluation.get('protocol') != 'l5o'):
        raise ValueError('History filtering requires the L5O evaluation protocol')
    selection = 'ev_ndcg@5' if evaluation.get('protocol') == 'l5o' else 'ndcg@5'
    if evaluation['selection'] != selection:
        raise ValueError('Selection metric must match the evaluation protocol')
    if 'r_updates_per_unit' in train and (type(train['r_updates_per_unit']) is not int or train['r_updates_per_unit'] <= 0):
        raise ValueError('r_updates_per_unit must be a positive integer')
    if any(evaluation[k] <= 0 for k in ('val_beam', 'test_beam', 'candidate_lists', 'batch_size')):
        raise ValueError('Evaluation sizes must be positive')
    if evaluation['candidate_lists'] > evaluation['val_beam']:
        raise ValueError('Candidate-list limit cannot exceed the primary beam')
    if train['sia_weight'] != 0:
        raise ValueError('Listwise DIGER-ST protocol does not use SIA')
    if train['checkpoint_every_updates'] <= 0:
        raise ValueError('Checkpoint interval must be positive')
    if cfg['model']['item_local_attention']['enabled']:
        raise ValueError('This experiment varies global Full Mixer only')
    if cfg['model']['diffusion_loss_reduction'] != 'per_view':
        raise ValueError('List protocol requires per-view normalization')
