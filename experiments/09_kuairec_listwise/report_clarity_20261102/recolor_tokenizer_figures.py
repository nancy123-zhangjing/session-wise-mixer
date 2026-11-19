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
 if not r.get('ok'): raise RuntimeError(r)
 return r
before=call(['+fetch','--detail','full'],'COLOR_BEFORE')['data']['document']
tree=ET.fromstring('<root>'+before['content']+'</root>')
names=[f'tokenizer_{f}_large.png' for f in ['loss','recommendation','reconstruction','balance']]
nodes=list(tree)
targets=[n for n in nodes if n.tag=='img' and n.get('name') in names]
assert [n.get('name') for n in targets]==names
indices=[nodes.index(n) for n in targets]
assert indices==list(range(indices[0],indices[0]+4)), 'Images no longer consecutive'
call(['+update','--command','block_replace','--start-block-id',targets[0].get('id'),'--end-block-id',targets[-1].get('id'),'--revision-id',str(before['revision_id']),'--content','@./experiments/09_kuairec_listwise/report_clarity_20261102/split_figures.xml'],'COLOR_WRITE')
after=call(['+fetch','--detail','full'],'COLOR_READBACK')['data']['document']
newtree=ET.fromstring('<root>'+after['content']+'</root>')
assert ''.join(tree.itertext())==''.join(newtree.itertext())
newimgs=list(newtree.iter('img'))
assert len(newimgs)==len(list(tree.iter('img')))
assert all(any(n.get('name')==name for n in newimgs) for name in names)
assert not set(n.get('src') for n in targets)&set(n.get('src') for n in newimgs if n.get('name') in names)
(OUT/'COLOR_VERIFIED.json').write_text(json.dumps({'revision':after['revision_id'],'four_new_images':True,'text_unchanged':True,'D':'#0072B2 solid','E':'#B2188B solid'},indent=2),encoding='utf-8')
print('Verified revision',after['revision_id'])
