"""Bind the two authorized C validation points after the main C audit passes.

This helper only reads local files and creates a new extra-model manifest. Run
the unchanged analyze.py and verify.py separately, with new output paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
PINNED = {
    'analyze.py': '3270debadddbd528a5b9b423a5b6ccae89246ab1351ac02d27279eca34ea7267',
    'verify.py': 'b51c6d3acf597aa481a2b1b90bd371af011d77465bdd302782f57319413cd33b',
    'REPORT.md': '950b708f40896d906780cb99598fc34b4a4cc8f1e27ebe0c49b1b87bde557090',
    'initial_r1/COMPLETE.json': 'b2fdc9321554a986eec0b7342dcd60d877a708635849f43ca304d458d2aa0134',
}
POINTS = {
    'C_freeze_R32': (32, {
        'predictions_sha256': '2405d77c2f247b8bd5fe99309ea1ba3c87ec4c8cfdc56ed029564b231a2ac9bb',
        'summary_sha256': '9a1671fd2193813dd24ca8ed92b61f1496c6974b896418e181d98c1e79255b70',
        'checkpoint_sha256': 'a1cba4ea70e62565a929200e86d79ea2718092fb9c80ca4ac5f2959df381a22e',
    }),
    'C_best_R23': (23, {
        'predictions_sha256': '5ce11e9c75b1992307dc9ac276b1d6dea8d84455a49cbf73f1b4c6f9ec37e223',
        'summary_sha256': '9fd8a895679b558a0fb5b94b4e7982e2bd908434c5b124b0811c8c5928c59cc5',
        'checkpoint_sha256': '04f0b7823cff611f00fe8aa0f51ab564c2c988502dba101a50d8a3d44781e1c7',
    }),
}
SELECTION_SHA = 'c2b0f889c94d761dfae8a3a3965ff2ec50ca2e7669b2c95b4b8dba430e9be3e4'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-report', type=Path,
                        default=TASK / 'final_analysis/C/analysis_report.json')
    parser.add_argument('--output', type=Path, default=HERE / 'C_EXTRA_MODELS.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'New manifest path required; no overwrite')
    for name, digest in PINNED.items():
        require(sha(HERE / name) == digest, 'Original artifact changed: ' + name)
    source = read(args.branch_report)
    require(source.get('status') == 'passed' and source.get('branch') == 'C',
            'Main complete C branch audit must pass before this helper runs')
    require(source.get('selected_epoch') == 23 and source.get('inputs_unchanged') is True,
            'C original selection or source immutability verification changed')
    require(source.get('torch_imported') is False and source.get('gpu_evaluation_performed') is False,
            'Main C analysis must remain CPU statistics only')
    selections = [(path, digest) for path, digest in source['input_and_implementation_sha256'].items()
                  if Path(path).name == 'selection.json' and 'H0-Freeze20-R35' in Path(path).parts]
    require(len(selections) == 1 and selections[0][1] == SELECTION_SHA,
            'Main report does not bind the original locked C selection')
    require(sha(selections[0][0]) == SELECTION_SHA, 'Locked C selection changed')
    points = {}
    for name, (epoch, expected) in POINTS.items():
        point = source['models'][f'C_R{epoch}_validation']
        require((point['branch'], point['epoch'], point['split']) == ('C', epoch, 'validation'),
                'Incorrect validation model identity')
        for field, digest in expected.items():
            require(point[field] == digest, 'Model differs from accepted completion evidence: ' + name)
        for field in ('predictions', 'summary', 'checkpoint', 'audit_path'):
            digest = point['audit_sha256'] if field == 'audit_path' else point[field + '_sha256']
            require(sha(point[field]) == digest, 'Bound local source changed: ' + point[field])
        meta, audit = read(point['summary']), read(point['audit_path'])
        require(meta['r_updates'] == epoch * 693 and meta['t_updates'] == 200,
                'Unexpected C R/T update clocks')
        require(meta['split'] == 'validation' and meta['full_split'] is True and meta['examples'] == 71657,
                'Only saved complete validation predictions are authorized')
        require(audit['passed'] is True and audit['source_labels_verified'] is True
                and audit['source_cold_groups_verified'] is True and audit['mismatches'] == 0,
                'Independent full-sample C verification did not pass')
        require(audit['metrics'] == point['metrics']
                and audit['predictions_sha256'] == point['predictions_sha256']
                and audit['summary_sha256'] == point['summary_sha256'],
                'Independent C audit does not bind this point')
        points[name] = point
    require(points['C_freeze_R32']['checkpoint_sha256'] != points['C_best_R23']['checkpoint_sha256'],
            'Duplicate C model must not be analyzed twice')
    result = dict(
        format='stage09_c_cold_popularity_supplement_inputs_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        branch_report=str(args.branch_report.resolve()), branch_report_sha256=sha(args.branch_report),
        selection=str(Path(selections[0][0]).resolve()), selection_sha256=SELECTION_SHA,
        initial_artifacts_unchanged=PINNED, points=points,
        only_cpu_statistics=True, training_or_inference=False, additional_gpu_hours=0,
        fixed_comparison='H1_freeze_R32 minus C_freeze_R32: R32/T200, common freeze at R20',
        best_comparison='H1_freeze_R32 minus C_best_R23: original selected best; unequal R updates',
        script_sha256=sha(__file__),
    )
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='passed', output=str(args.output.resolve()),
                          sha256=sha(args.output), models=list(points)), ensure_ascii=False))


if __name__ == '__main__':
    main()
