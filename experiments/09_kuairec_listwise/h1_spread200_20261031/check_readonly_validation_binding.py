"""Bind downloaded async validation receipts; does not recompute predictions."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--epoch', type=int, required=True)
a = p.parse_args()
assert a.epoch in (26, 29, 32, 35)
directory = Path(__file__).parent / f'saved_validation_R{a.epoch:02d}'
read = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
r = read('record.json')
c = read('checkpoint_receipt.json')
d = read('readonly_done.json')
e = read('evaluation_complete.json')
s = read('diagnostic_complete.json')
assert d['status'] == 'completed_readonly_task'
assert r['checkpoint_sha256'] == c['sha256'] == d['checkpoint_sha256']
assert d['state_before'] == d['state_after'] == c['state_sha256']
assert d['runtime_sha256'] == '7c35eaebb242fe06291707b4c71a038bc79b4c453193dd526ddbb1857bf2d0f6'
assert s['result']['examples'] == 8192
assert s['reused'] == r['predictions']
assert s['source_sha256'] == e['predictions_sha256'] == r['predictions_sha256']
assert e['summary_sha256'] == r['summary_sha256']
assert e['binding']['epoch'] == a.epoch
assert e['binding']['r_updates'] == a.epoch * 693
assert e['binding']['t_updates'] == (a.epoch - 5) // 3 * 20
assert e['binding']['split'] == 'validation' and e['binding']['branch'] == 'D'
assert s['state'] == e['binding']['model']
result = dict(
    status=f'passed_R{a.epoch}_full_validation_checkpoint_and_readonly_binding',
    checkpoint_sha256=r['checkpoint_sha256'],
    complete_state_unchanged=True,
    source_state_fingerprint=d['state_before'],
    subset_reused_from_same_full_validation=True,
    predictions_sha256=r['predictions_sha256'],
    summary_sha256=r['summary_sha256'],
    auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitation='Artifact/state receipt binding only; actual checkpoint SHA and independent metrics are accepted separately.',
)
with (directory / 'READONLY_BINDING_ACCEPTED.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result))
