"""Read the fixed wave prerequisites from the Inspire shared disk."""
import hashlib,json,pathlib
b=pathlib.Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
w=b/'l5o_wave_20261117_v1'; c=w/'code'
paths={'source':w/'SOURCE_REVISION','calibration':b/'l5o_sasrec_20261117_v1/run/COMPLETE.json','h_warmup':w/'parents/h_warmup/complete.json','cf_rq':w/'parents/cf_rq/COMPLETE.json','tiger':w/'parents/tiger/manifest.json'}
result={k:p.is_file() for k,p in paths.items()}
if result['source']:
 assert paths['source'].read_text()=='d8d828244db6575542566e8ffb44e489e94809ba'
 print('Source revision verified')
h=w/'H_SOURCE_REVISION'
result['h_source']=h.is_file() and h.read_text()=='c10c357f837e947b92397f4e28b4f3548440860e' and (w/'code_c10c357/scripts/listwise_diffgrm/run.py').is_file()
if result['calibration']:
 receipt=json.loads(paths['calibration'].read_text())
 run=paths['calibration'].parent
 def sha(path):
  with path.open('rb') as f:
   h=hashlib.sha256()
   for block in iter(lambda:f.read(4194304),b''):h.update(block)
  return h.hexdigest()
 assert sha(run/'best.pt')==receipt['selection']['best_sha256']
 assert sha(run/'items.pt')==receipt['items_sha256']
 assert json.loads((run/'SELECTION.json').read_text())==receipt['selection']
 result['sasrec']=receipt
 result['calibration_artifacts_verified']=True
print(json.dumps(result))
