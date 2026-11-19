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
def fetch(name):return call(['+fetch','--detail','full'],name)['data']['document']
d=fetch('LIVE_BEFORE')
root=ET.fromstring('<root>'+d['content']+'</root>')
assert root[1].tag=='h1' and ''.join(root[1].itertext())=='三组H1实验'
assert root[2].tag=='table'
call(['+update','--command','block_replace','--start-block-id',root[1].get('id'),'--end-block-id',root[2].get('id'),'--revision-id',str(d['revision_id']),'--content','@./experiments/09_kuairec_listwise/report_clarity_20261102/front.xml'],'WRITE_FRONT')
d=fetch('AFTER_FRONT')
for i,(old,new) in enumerate([
 ('H1 20T＋3R（D）','D：每次编码更新步数减半'),
 ('H1 40T＋6R（E）','E：推荐器连续训练轮数加倍'),
 ('原H1（40T＋3R）','原H1'),
 ('D：20T＋3R最终','D最终（每次编码更新步数减半）'),
 ('E：40T＋6R最终','E最终（推荐器连续训练轮数加倍）'),
 ('40T＋6R提供了','E组（推荐器连续训练轮数加倍）提供了'),
]):
 if old not in d['content']:continue
 call(['+update','--command','str_replace','--pattern',old,'--content',new,'--revision-id',str(d['revision_id'])],f'RENAME_{i}')
 d=fetch(f'AFTER_RENAME_{i}')
call(['+update','--command','append','--content','@./experiments/09_kuairec_listwise/report_clarity_20261102/loss.xml','--revision-id',str(d['revision_id'])],'WRITE_LOSS')
d=fetch('READBACK')
assert all(s in d['content'] for s in ['模型结构与研究问题','Tokenizer更新20步','训练loss与收敛情况','3.34041','阶段性结论'])
assert d['content'].count('<img')==3
# Preserve user-authored observations and all existing result numbers.
assert '发现减少了T的更新频次之后' in d['content']
assert 'D组的验证结果差于原H1' in d['content']
assert '0.007067' in d['content'] and '91.41%' in d['content']
(OUT/'VERIFIED.json').write_text(json.dumps({'status':'passed','revision':d['revision_id'],'images':3,'user_paragraphs_preserved':True,'operations':'front section replacement; descriptive labels; loss appendix','document':DOC},ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified revision',d['revision_id'])
