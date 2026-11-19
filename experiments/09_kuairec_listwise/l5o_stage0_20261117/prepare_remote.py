"""CPU-only preparation; verify source events and reuse static PSE assets."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'code'
sys.path.insert(0,str(CODE))
from scripts.kuairec_data.l5o import build,finalize,dump,sha
from scripts.listwise_diffgrm.rebind_static_assets import rebind


def main():
    stage=ROOT.parent
    candidates=[ROOT/'source_dataset',stage/'goal_20261027_r1/inputs/dataset',
        stage.parent/'experiment1/outputs/kuairec_listwise_v1/run1',
        Path('/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1/outputs/kuairec_listwise_v1/run1')]
    valid=[p for p in candidates if (p/'events').is_dir() and (p/'manifest.json').is_file()]
    if not valid:
        raise RuntimeError('Clean event store unavailable at documented candidates: '+str([(str(p),p.exists()) for p in candidates]))
    source=valid[0]
    expected=json.loads((ROOT/'LOCAL_DATA_AUDIT.json').read_text())
    actual={p.name:sha(p) for p in sorted((source/'events').glob('*.parquet'))}
    assert actual==expected['source_event_sha256'], 'Local/remote source events differ'
    dataset=ROOT/'dataset'
    cfg=json.loads((CODE/'configs/kuairec_l5o_v1.json').read_text())
    cfg['source_dataset']=str(source)
    if not (dataset/'manifest.json').exists():
        dataset.mkdir(exist_ok=False)
        dump(dataset/'config.json',cfg)
        result=build(cfg,dataset)
        finalize(cfg,dataset,*result)
    construction=json.loads((dataset/'construction.json').read_text())
    assert construction['normalized_sample_hashes']==expected['normalized_sample_hashes'], 'Local/remote L5O rows differ'
    assets=ROOT/'assets'
    if not assets.exists():
        rebind(stage/'goal_20261027_r1/assets/pse_v1',dataset,assets)
    c=json.loads((CODE/'configs/09_kuairec_listwise/l5o.json').read_text())
    c.update(dataset=str(dataset),assets=str(assets))
    dump(ROOT/'resolved_config.json',c)
    dump(ROOT/'PREPARED.json',dict(passed=True,source=str(source),dataset=str(dataset),assets=str(assets),
        normalized_sample_hashes=construction['normalized_sample_hashes'],counts=construction['counts']))
    print('L5O_PREPARED '+(ROOT/'PREPARED.json').read_text(),flush=True)


if __name__=='__main__':
    main()
