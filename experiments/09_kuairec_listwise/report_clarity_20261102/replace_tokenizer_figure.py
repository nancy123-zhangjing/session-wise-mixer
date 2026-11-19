import json, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
CLI='C:/Users/nancy/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js'
DOC='<feishu-id>'
def call(args,name):
 p=subprocess.run(['node',CLI,'docs',*args,'--doc',DOC,'--as','user','--format','json'],cwd=ROOT,capture_output=True,encoding='utf-8')
 (OUT/(name+'.json')).write_text(p.stdout or p.stderr,encoding='utf-8')
 r=json.loads(p.stdout or p.stderr)
 if not r.get('ok'):raise RuntimeError(r)
 return r
before=call(['+fetch','--detail','full'],'SPLIT_BEFORE')['data']['document']
tree=ET.fromstring('<root>'+before['content']+'</root>')
targets=[n for n in tree.iter('img') if n.get('name')=='tokenizer_loss.png']
assert len(targets)==1,[(n.attrib) for n in tree.iter('img')]
target=targets[0]
call(['+update','--command','block_replace','--block-id',target.get('id'),'--revision-id',str(before['revision_id']),'--content','@./experiments/09_kuairec_listwise/report_clarity_20261102/split_figures.xml'],'SPLIT_WRITE')
after=call(['+fetch','--detail','full'],'SPLIT_READBACK')['data']['document']
newtree=ET.fromstring('<root>'+after['content']+'</root>')
oldimgs=list(tree.iter('img'));newimgs=list(newtree.iter('img'))
assert len(newimgs)==len(oldimgs)+3
for field in ['loss','recommendation','reconstruction','balance']:
 assert any(n.get('name')==f'tokenizer_{field}_large.png' for n in newimgs)
assert ''.join(tree.itertext())==''.join(newtree.itertext()),'Unexpected text change'
(OUT/'SPLIT_VERIFIED.json').write_text(json.dumps({'revision':after['revision_id'],'four_images_verified':True,'body_text_unchanged':True},indent=2),encoding='utf-8')
print('Verified four figures, unchanged text; revision',after['revision_id'])
