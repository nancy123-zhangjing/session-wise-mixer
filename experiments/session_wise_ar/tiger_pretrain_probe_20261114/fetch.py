"""Copy small result JSONs from the remote probe directory."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

R = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/tiger_pretrain_probe_20261114_v1/results/'
for name in ('PROTOCOL.json', 'STEP1.json', 'STEP2.json', 'COMPLETE.json'):
    text = jexec.python(f"import json; print('@@' + json.dumps(json.load(open({R + name!r}))))", timeout=180)
    data = json.loads([l for l in text.splitlines() if l.startswith('@@')][-1][2:])
    (HERE / name).write_text(json.dumps(data, indent=1), encoding='utf-8')
    print(name, 'ok')
log = jexec.run(f"cat {R}../probe.log", 120)
(HERE / 'probe.log').write_text(log, encoding='utf-8')
