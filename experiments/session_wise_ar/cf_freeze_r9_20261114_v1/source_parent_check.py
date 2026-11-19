import hashlib,json,sys
from pathlib import Path
r=Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/cf_freeze_r9_20261114_v1');sys.path.insert(0,str(r))
from scripts.session_wise_ar.cf import verify_parent
assert hashlib.sha256((r/'scripts/session_wise_ar/cf_freeze.py').read_bytes()).hexdigest()=='04edfece67b17b4c9e34a4a5408ed2cc201b23ef0d64236145d6b65db59944ae'
verify_parent('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1/cf_prepared',r/'experiments/rqcd_mixer/targets_formal_20261113_v1/runtime.locked.json')
for name,(arm,source,digest) in {'M0': ('B0', '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/session_wise_ar_five_20261113_v1/runs/CF-M0/train', '78718413b7b4588d49c1d8c185ec51000ea33a1ef8c628bf001068b9543ae20e'), 'M1': ('B1', '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1/cf', 'c9ba73da3bb3b56e73b98d13b00d797fca8eb3107ff842476c3707a43f3dc147')}.items():
 p=Path(source)/'checkpoints/R09_T040.pt'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 assert h.hexdigest()==digest,(name,'parent checksum')
 assert json.loads((Path(source)/'BEST.json').read_text())['R']==9
print('source, prepared and both R9 checksums verified')
