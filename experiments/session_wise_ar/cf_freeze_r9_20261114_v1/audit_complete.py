"""Post-completion, read-only audit. Reuses existing structural and freeze checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'five_arm_20261113_v1'))
from audit_cf_nodes import audit as structural_audit, sha, require
from audit_frozen import audit as freeze_audit


def audit(root):
    branch = json.loads((root/'BRANCH.json').read_text())
    parent = Path(branch['parent_checkpoint'])
    require(sha(parent) == branch['parent_sha256'], 'parent SHA')
    inherited = json.loads((root/'INHERITED_VALIDATION.json').read_text())
    selection = inherited['selection']
    require(selection['R'] == 9 and selection['T'] == 40, 'parent boundary')
    require(selection['sha256'] == branch['parent_sha256'], 'inherited selection identity')
    parent_validation = json.loads(Path(inherited['source']).read_text())
    require(parent_validation['identity']['checkpoint_sha256'] == branch['parent_sha256'] and parent_validation['count'] == 71657, 'original parent validation')
    require(parent_validation['metrics']['ndcg@5'] == selection['validation_ndcg@5'], 'inherited score')
    require(sha(root/'checkpoints/R09_T040.pt') == branch['parent_sha256'], 'copied parent')
    structural = structural_audit(root, expected_epochs=list(range(10, 36)))
    frozen = []
    scores = {9: selection['validation_ndcg@5']}
    for row in structural['nodes']:
        result = freeze_audit(parent, Path(row['path']))
        require(result['passed'], 'frozen state changed at R'+str(row['R']))
        frozen.append(result)
        summary = json.loads((root/f"validation/R{row['R']:02d}/summary.json").read_text())
        scores[row['R']] = summary['metrics']['ndcg@5']
    best = json.loads((root/'BEST.json').read_text())
    expected_best = max(scores, key=scores.get)
    require(best['R'] == expected_best and best['validation_ndcg@5'] == scores[expected_best], 'best selection including R9')
    selected_path = root/f"checkpoints/R{best['R']:02d}_T040.pt"
    require((root/best['checkpoint']).resolve() == selected_path.resolve() and sha(selected_path) == best['sha256'], 'best checkpoint')
    done = json.loads((root/'TRAINING_COMPLETE.json').read_text())
    require(done['R'] == 35 and done['T'] == 40 and done['R_updates'] == 24255, 'final budget')
    tests = json.loads((root/'TEST_COMPLETE.json').read_text())
    binding = json.loads((root/'binding.json').read_text())
    for label, expected, epoch in [('best', best['sha256'], best['R']), ('final', done['final_sha256'], 35)]:
        v = tests[label]
        require(v['count'] == 53194, 'full test scope')
        i = v['identity']
        require(i['checkpoint_sha256'] == expected and i['R'] == epoch and i['T'] == 40, 'test checkpoint identity')
        require(i['binding'] == binding and i['split'] == 'test', 'test binding')
        folder = 'best' if v.get('reused_same_checkpoint') else label
        require(sha(root/f'test/{folder}/predictions.jsonl.gz') == v['prediction_sha256'], 'test predictions')
    return {'status': 'passed_cpu_artifact_audit', 'structure': structural, 'frozen_nodes': frozen,
            'best_R': best['R'], 'final_R': 35,
            'boundary': 'No extra inference or training. Resource release and original GPU restore evidence audited separately.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    print(json.dumps(audit(p.parse_args().run), indent=2))
