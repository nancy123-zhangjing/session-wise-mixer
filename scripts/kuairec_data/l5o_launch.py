"""Build first-wave settings from a completed SASRec calibration receipt."""
import copy
import hashlib
import json
import math
from pathlib import Path


def wave_config(base, receipt_path, family):
    payload = Path(receipt_path).read_bytes()
    receipt = json.loads(payload)
    metrics = receipt['test']['1.0']
    selection = receipt['selection']
    if (receipt['passed'] is not True or metrics['samples'] != 7176 or
            not .25 <= metrics['imp_hit@5'] <= .45 or
            not .05 <= metrics['ev_ndcg@5'] <= .12 or
            type(selection['filter_history']) is not bool):
        raise ValueError('SASRec full-test calibration did not pass')
    cfg = copy.deepcopy(base)
    cfg['seed'] = 2024
    cfg['history_max'] = 128
    cfg['evaluation']['filter_history'] = selection['filter_history']
    cfg['calibration_sha256'] = hashlib.sha256(payload).hexdigest()
    if family in ('cf', 'tiger'):
        old_count = cfg['split_counts']['train']
        cfg['dataset_protocol'] = 'kuairec_l5o_records_v1'
        cfg['split_counts'] = dict(train=2288120, validation=7176, test=7176)
        cfg['evaluation']['metric'] = 'ev_ndcg@5'
        train = cfg['training']
        if family == 'tiger':
            train['validation_interval_updates'] = math.ceil(old_count * cfg['list_length'] / train['batch_size'])
        else:
            train['r_updates_per_unit'] = math.ceil(old_count / train['effective_batch'])
            cfg['evaluation']['epochs'] = list(range(1, train['epochs'] + 1))
            cfg['teacher']['validation_metric'] = 'ev_ndcg@5'
            cfg['calibrated_teacher'] = dict(items_sha256=receipt['items_sha256'],
                source=selection['binding']['source'], calibration_source=selection['binding']['calibration'])
    elif family != 'h':
        raise ValueError(f'Unknown first-wave family: {family}')
    return cfg
