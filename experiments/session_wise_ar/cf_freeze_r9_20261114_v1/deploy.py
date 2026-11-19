"""Explicit authorized two-job deployment; no action on import."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'five_arm_20261113_v1'))
import control as c

BASE = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09'
REMOTE = BASE + '/cf_freeze_r9_20261114_v1'
SOURCE = BASE + '/session_wise_ar_five_20261113_v1'
DATA = BASE + '/goal_20261027_r1/inputs/dataset'
PREPARED = BASE + '/rqcd_targets_formal_20261113_v1/cf_prepared'
ARMS = {
    'M0': ('B0', SOURCE + '/runs/CF-M0/train', '78718413b7b4588d49c1d8c185ec51000ea33a1ef8c628bf001068b9543ae20e'),
    'M1': ('B1', BASE + '/rqcd_targets_formal_20261113_v1/cf', 'c9ba73da3bb3b56e73b98d13b00d797fca8eb3107ff842476c3707a43f3dc147'),
}


def main():
    c.HERE = HERE
    print(c.python(f'''from pathlib import Path
import shutil
s=Path({SOURCE!r}); d=Path({REMOTE!r})
if not d.exists():
 d.mkdir()
 for name in ('scripts','configs','docs'):
  shutil.copytree(s/name,d/name,ignore=shutil.ignore_patterns('__pycache__'))
 ref='experiments/rqcd_mixer/targets_formal_20261113_v1'
 shutil.copytree(s/ref,d/ref)
print(str(d))
''', 'SOURCE_CLONE.json'))
    wrapper = ROOT / 'scripts/session_wise_ar/cf_freeze.py'
    expected = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    c.cli(['notebook', 'scp', c.CPU, str(wrapper), REMOTE + '/scripts/session_wise_ar/cf_freeze.py',
           '--workspace', 'CPU资源空间'], 'WRAPPER_UPLOAD.json')
    check_source = f'''import hashlib,json,sys
from pathlib import Path
r=Path({REMOTE!r});sys.path.insert(0,str(r))
from scripts.session_wise_ar.cf import verify_parent
assert hashlib.sha256((r/'scripts/session_wise_ar/cf_freeze.py').read_bytes()).hexdigest()=={expected!r}
verify_parent({PREPARED!r},r/'experiments/rqcd_mixer/targets_formal_20261113_v1/runtime.locked.json')
for name,(arm,source,digest) in {ARMS!r}.items():
 p=Path(source)/'checkpoints/R09_T040.pt'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 assert h.hexdigest()==digest,(name,'parent checksum')
 assert json.loads((Path(source)/'BEST.json').read_text())['R']==9
print('source, prepared and both R9 checksums verified')
'''
    check = HERE / 'source_parent_check.py'
    check.write_text(check_source, encoding='utf-8')
    c.cli(['notebook', 'scp', c.CPU, str(check), REMOTE+'/source_parent_check.py',
           '--workspace', 'CPU资源空间'], 'CHECK_UPLOAD.json')
    print(c.remote(c.PYTHON+' -B '+shlex.quote(REMOTE+'/source_parent_check.py'),
                   'SOURCE_PARENT_CHECK.json', timeout=90))
    manifest = {'remote': REMOTE, 'wrapper_sha256': expected, 'arms': {}}
    for label, (arm, source, digest) in ARMS.items():
        name = 'nancy-cf-freeze-' + label.lower() + '-1114-v1'
        found = c.cli(['job', 'list', '--workspace', '分布式训练空间', '--keyword', name, '--all'], label+'_LOOKUP.json')
        matches = [row for row in found['items'] if row['name'] == name]
        if matches:
            if len(matches) != 1:
                raise RuntimeError('Ambiguous job name')
            manifest['arms'][label] = {'name': name, 'status': matches[0]['status'], 'existing': True}
            continue
        intent = HERE / (label+'_SUBMISSION_INTENT.json')
        if intent.exists():
            raise RuntimeError('Resolve prior submission before retry')
        command = [c.PYTHON, '-B', '-u', '-m', 'scripts.session_wise_ar.cf_freeze',
                   '--arm', arm, '--parent-run', source, '--parent-sha', digest,
                   '--pretrained', (SOURCE+'/runs/CF-M0/pretrain' if label=='M0' else BASE+'/rqcd_targets_formal_20261113_v1/cf_rq'), '--prepared', PREPARED,
                   '--dataset', DATA, '--output', REMOTE+'/runs/'+label+'/train']
        shell = ('cd '+shlex.quote(REMOTE)+' && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
                 'PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && '+shlex.join(command)+
                 ' > '+shlex.quote(REMOTE+'/'+label+'.log')+' 2>&1')
        request = ['job', 'create', '--name', name, '--workspace', '分布式训练空间',
                   '--project', '公共科研项目', '--group', '开发区-H100-cuda12.8版本-183核',
                   '--quota', '1,20,200', '--image', 'ngc-pytorch:25.02-cuda12.8.0-py3',
                   '--nodes', '1', '--shm-size', '32', '--priority', '4',
                   '--no-auto-fault-tolerance', '--no-enable-notification',
                   '--command', shlex.join(['bash', '-c', shell])]
        intent.write_text(json.dumps({'name': name, 'request': request}, ensure_ascii=False, indent=2), encoding='utf-8')
        result = c.cli(request, label+'_SUBMITTED.json')
        manifest['arms'][label] = {'name': name, 'submitted': True, 'parent_sha256': digest}
        print(label, result)
        (HERE/'DEPLOYMENT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
