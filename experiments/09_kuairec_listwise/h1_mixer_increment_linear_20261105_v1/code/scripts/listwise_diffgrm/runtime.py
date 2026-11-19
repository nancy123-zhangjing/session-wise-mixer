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
    cfg = dict(config['model'])
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
        'selection': 'ndcg@5', 'tie_break': 'earlier', 'filter_history': False,
        'collision_rule': 'reconstruction_error_then_original_id',
        'aggregation': 'list_score_order_then_item_position_first_occurrence',
    }.items():
        if evaluation[name] != expected:
            raise ValueError(f'Unsupported evaluation policy: {name}')
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
