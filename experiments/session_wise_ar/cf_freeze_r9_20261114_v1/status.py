"""Read-only snapshot of the two authorized freeze branches."""
import json
from pathlib import Path
import deploy as d


def main():
    d.c.HERE = d.HERE
    jobs = d.c.cli(['job', 'list', '--workspace', '分布式训练空间',
                    '--keyword', 'nancy-cf-freeze-', '--all'], 'CURRENT_JOBS.json')
    states = [(r['name'], r['status']) for r in jobs['items'] if r['name'].endswith('1114-v1')]
    print(states, flush=True)
    source = f'''from pathlib import Path
import json,time,torch
r=Path({d.REMOTE!r});out={{}}
for arm in ('M0','M1'):
 p=r/'runs'/arm/'train';row={{}}
 for name in ('RESTORE_CHECK.json','TRAINING_COMPLETE.json','TEST_COMPLETE.json','BEST.json'):
  if (p/name).exists():row[name]=json.loads((p/name).read_text())
 row['validation']=[json.loads(f.read_text()) for f in sorted((p/'validation').glob('R*/summary.json'))]
 if (p/'last.pt').exists():
  s=torch.load(p/'last.pt',map_location='cpu',weights_only=False,mmap=True)
  row['progress']={{k:s[k] for k in ('r_epoch','r_updates','t_updates','phase','best_epoch','best_score')}}
  row['freeze_policy']=s.get('freeze_policy')
  row['checkpoint_mtime']=(p/'last.pt').stat().st_mtime
 if (r/(arm+'.log')).exists():row['log_tail']=(r/(arm+'.log')).read_text()[-1800:]
 out[arm]=row
print(json.dumps({{'unix':time.time(),'arms':out}}))
'''
    result = json.loads(d.c.python(source, 'STATUS_RAW.json', timeout=60))
    result['jobs'] = states
    for arm, row in result['arms'].items():
        for summary in row['validation']:
            identity = summary['identity']
            assert summary['count'] == 71657 and identity['split'] == 'validation'
            assert identity['T'] == 40 and 10 <= identity['R'] <= 35
            receipt = d.HERE / 'nodes' / f"{arm}_R{identity['R']:02d}.json"
            receipt.parent.mkdir(exist_ok=True)
            if receipt.exists():
                assert json.loads(receipt.read_text()) == summary, 'Immutable validation receipt changed'
            else:
                receipt.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (d.HERE/'STATUS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
