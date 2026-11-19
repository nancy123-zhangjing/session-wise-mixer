"""Read-only probe: locate per-sample prediction files for the four CF arms."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jexec  # noqa: E402

BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
ROOTS = {
    'M0': BASE + '/session_wise_ar_five_20261113_v1/runs/CF-M0/train',
    'M1': BASE + '/rqcd_targets_formal_20261113_v1/cf',
    'M0F': BASE + '/cf_freeze_r9_20261114_v1/runs/M0/train',
    'M1F': BASE + '/cf_freeze_r9_20261114_v1/runs/M1/train',
}


def main():
    source = f'''from pathlib import Path
import json
roots={ROOTS!r}; out={{}}
for k,r in roots.items():
    r=Path(r); row={{'exists':r.exists()}}
    row['test']=sorted(str(p.relative_to(r)) for p in (r/'test').glob('*/predictions.jsonl.gz')) if (r/'test').exists() else []
    row['val_R35']=[str(p.relative_to(r)) for p in (r/'validation').glob('R35*/predictions.jsonl.gz')] if (r/'validation').exists() else []
    row['val_count']=len(list((r/'validation').glob('R*/predictions.jsonl.gz'))) if (r/'validation').exists() else 0
    row['test_complete']=json.loads((r/'TEST_COMPLETE.json').read_text()).get('best',{{}}).get('reused_same_checkpoint') if (r/'TEST_COMPLETE.json').exists() else None
    out[k]=row
print(json.dumps(out))
'''
    text = jexec.python(source, timeout=120)
    print(text)


if __name__ == '__main__':
    main()
