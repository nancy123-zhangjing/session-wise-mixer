"""Fixed-budget test of the verified H0 replay at R14/T120."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
FREEZE = STAGE / 'h_freeze35_20261029_r1'
CODE = FREEZE / 'code'
CHECKPOINT = FREEZE / 'runs/H0-Freeze20-R35/phase16_R_R14_T120.pt'
PARENT = STAGE / 'goal_20261027_r1/runs/H0/initial.pt'
EXPECTED_SHA = '2b919b71c24579fb01070a7f25b63bbcf5c1cae087cdcd2bcae14a4868d434a3'
COMPARATOR = STAGE / 'h1_no_code_tag_20261108_v1/runs/H1-NoCodeTag/test_best/summary.json'
METRICS = ('list/ndcg@5', 'list/hr@5', 'list/recall@5')


def main():
    os.environ['DIFFGRM_ENGINE_ROOT'] = str(CODE / 'external/DiffGRM')
    sys.path.insert(0, str(CODE))
    import torch
    from scripts.listwise_diffgrm.recovery35 import build_recovery35
    from scripts.listwise_diffgrm.recovery35_run import evaluation
    from scripts.listwise_diffgrm.training import sha256

    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('Expected exactly one H100')
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError('H0 R14 replay checkpoint SHA mismatch')
    other = json.loads(COMPARATOR.read_text(encoding='utf-8'))
    expected = ('test', 53194, True, 32, 'fp32', 'H', 2024, 9702, 120)
    fields = ('split', 'examples', 'full_split', 'beam', 'inference_precision', 'architecture',
              'seed', 'r_updates', 't_updates')
    if tuple(other[k] for k in fields) != expected:
        raise ValueError('H1 comparator does not match the fixed R14 test protocol')
    controller, assets, _ = build_recovery35(PARENT, 'C', 'cuda:0')
    controller.restore(torch.load(CHECKPOINT, map_location='cpu', weights_only=False))
    if (controller.trainer.r_updates, controller.trainer.t_updates, controller.frozen) != (9702, 120, False):
        raise ValueError('Checkpoint is not the pre-freeze R14/T120 H0 replay')
    print('H0 replay checkpoint verified and restored at R14/T120', flush=True)
    result = evaluation(controller, assets, ROOT, 'test', 14, checkpoint=CHECKPOINT)
    ours = json.loads(Path(result['summary']).read_text(encoding='utf-8'))
    if tuple(ours[k] for k in fields) != expected:
        raise ValueError('H0 evaluation does not match the fixed R14 test protocol')
    for key in ('dataset_binding', 'metric_protocol'):
        if ours[key] != other[key]:
            raise ValueError(f'Comparator differs in {key}')
    metrics = {key: dict(h0=ours['metrics'][key], no_code_tag=other['metrics'][key],
                         absolute_delta=other['metrics'][key] - ours['metrics'][key],
                         relative_pct=100 * (other['metrics'][key] / ours['metrics'][key] - 1))
               for key in METRICS}
    payload = dict(status='passed', **result, examples=ours['examples'], beam=ours['beam'],
                   inference_precision=ours['inference_precision'], comparator_sha256=sha256(COMPARATOR),
                   metrics=metrics)
    temp = ROOT / 'result.tmp'
    temp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temp.replace(ROOT / 'RESULT.json')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
