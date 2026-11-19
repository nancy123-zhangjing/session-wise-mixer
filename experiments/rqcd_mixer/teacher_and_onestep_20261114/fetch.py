"""Fetch COMPLETE.json from the remote result dir and print a summary."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'session_wise_ar/cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

REMOTE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_teacher_onestep_20261114_v1/results/COMPLETE.json'
text = jexec.python(f"import json; print('@@' + json.dumps(json.load(open({REMOTE!r}))))", timeout=120)
d = json.loads([l for l in text.splitlines() if l.startswith('@@')][-1][2:])
(HERE / 'COMPLETE.json').write_text(json.dumps(d, indent=2), encoding='utf-8')
print('teacher', json.dumps(d['teacher'], indent=1))
for name, r in d['cf'].items():
    for s in ('validation', 'test'):
        row = r[s]
        print(name, s, 'stored20=%.6f' % row['stored_20']['ndcg@5'])
        for k in ('steps_1', 'steps_2', 'steps_5', 'steps_20'):
            if k in row:
                e = row[k]
                print('   ', k, '%.6f' % e['metrics']['ndcg@5'], 'diff=%+.6f' % e['ndcg_minus_stored20'],
                      'ci=', [round(x, 6) for x in e['ci95_vs_stored20']], 'distinct', e['distinct_items'],
                      'rawvalid=%.4f' % (sum(e['raw_valid_by_position']) / 5), 'chg', [round(x, 3) for x in e['changed_by_level']])
