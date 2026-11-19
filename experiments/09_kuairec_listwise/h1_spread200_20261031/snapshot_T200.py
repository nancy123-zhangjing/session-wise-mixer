"""Snapshot completed D T rows and sample orders without model execution."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if root!=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1'):
        raise ValueError('Wrong owned root')
    run=root/'runs/H1-Spread200-R35';rows=[];orders=[]
    with (run/'training.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['phase']=='T':rows.append(row)
    if len(rows)!=200 or rows[-1]['t_updates']!=200:raise ValueError('T200 not complete')
    for phase in range(5,42,4):orders.append(json.loads((run/'sample_orders'/f'phase{phase:02d}.json').read_text()))
    destination=root/'recovery_async_r2/ALL_T200_SNAPSHOT.json'
    with destination.open('x',encoding='utf-8') as f:
        json.dump(dict(scope='all_real_D_T_updates',rows=rows,orders=orders,model_execution=False),f,indent=2);f.write('\n')
    print(json.dumps(dict(status='snapshotted_all_200_actual_T_rows',sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),rows=len(rows))))


if __name__=='__main__':main()
