import json,shutil
from pathlib import Path
from scripts.rqcd_mixer.common import sha256,source_binding,write_json
from scripts.rqcd_mixer.config import load_config
root=Path.cwd();old=root.parent/'rqcd_mixer_benchmark_20261111_v1/prepared';new=root/'prepared'
meta=json.loads((old/'PREPARED.json').read_text());cfg=load_config();compare=json.loads(json.dumps(cfg))
for k in ('protocol_id','design_sha256'):compare[k]=meta['config'][k]
compare['evaluation']['epochs']=meta['config']['evaluation']['epochs']
assert compare==meta['config']
assert sha256(old/'initial.pt')==meta['initial_sha256']
new.mkdir(exist_ok=False)
for name in ('initial.pt','validation_noise.npy','validation_noise.json','test_noise.npy','test_noise.json'):
 shutil.copyfile(old/name,new/name)
 assert sha256(old/name)==sha256(new/name)
meta['migration']={'source':str(old),'source_manifest_sha256':sha256(old/'PREPARED.json'),'change':'eval31 only; initial tensors byte-identical before benchmark updates'}
meta['config']=cfg;meta['source']=source_binding();write_json(new/'PREPARED.json',meta)
print(json.dumps({'prepared_sha256':sha256(new/'PREPARED.json'),'initial_sha256':sha256(new/'initial.pt')}))
