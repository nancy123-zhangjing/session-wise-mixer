"""Required R14 evidence and GPU restore gates; no optimizer updates."""
import argparse, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPLAY=ROOT.parent/'h1_replay_r5_r20_20261105_r1'
spec=importlib.util.spec_from_file_location('replay_evidence', REPLAY/'replay.py')
replay=importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)
def main(mode):
    inputs=json.loads((ROOT/'INPUTS.json').read_text())
    parent=Path(inputs['parent'])
    receipt=json.loads((REPLAY/'runs/H1-Replay-R20/checkpoints/R14.json').read_text())
    expected='3cd65569e89cc5f8f1d3215cdc403e3ae0d5e6954d7e8818a891c12ed737f51f'
    assert replay.sha256(parent)==receipt['checkpoint_sha256']==expected
    if mode=='evidence':
        a=Path(inputs['r14_reference']).parent
        b=Path(receipt['validation'])
        left=json.loads((a/'summary.json').read_text()); right=json.loads((b/'summary.json').read_text())
        keys=('split','examples','full_split','dataset_binding','beam','inference_precision',
              'architecture','seed','r_updates','t_updates','metrics','metric_protocol','user_average','groups')
        delta=replay.differences({k:left[k] for k in keys},{k:right[k] for k in keys})
        preds=replay.compare_predictions(a/'per_sample.jsonl.gz',b/'per_sample.jsonl.gz')
        assert not delta and preds['exact'], (delta,preds)
        result=dict(status='passed',parent_sha256=expected,summary_differences=delta,predictions=preds,
                    original_summary_sha256=replay.sha256(a/'summary.json'),
                    replay_summary_sha256=replay.sha256(b/'summary.json'))
    else:
        evidence=json.loads((ROOT/'R14_EVIDENCE.json').read_text())
        assert evidence['status']=='passed' and evidence['parent_sha256']==expected
        sys.path.insert(0,str(ROOT/'code'))
        os.environ['DIFFGRM_ENGINE_ROOT']=str(ROOT/'code/external/DiffGRM')
        import torch
        from scripts.listwise_diffgrm.recovery_r14_g01 import build_g01
        assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
        ctrl,assets,data=build_g01(parent,'cuda:0',inputs['dataset'],inputs['assets'])
        original=torch.load(parent,map_location='cpu',weights_only=False)
        saved=ctrl.checkpoint()
        fields=('model','alignment','r_optimizer','t_optimizer','scheduler','rng','r_updates','t_updates','precision')
        delta=replay.differences({k:original[k] for k in fields},{k:saved['inner'][k] for k in fields})
        assert not delta, delta
        mixer=ctrl.trainer.alignment.mixer
        with torch.no_grad():
            raw,_=mixer._g01_original_crossed_codebooks(False)
            eff,_=mixer.crossed_codebooks(False)
            error=float((eff-raw).abs().max())
            assert torch.allclose(eff,raw,rtol=0,atol=1e-7),error
            codes=ctrl.trainer.alignment.hard_codes.clone()
            ctrl.trainer.refresh_catalog()
            assert torch.equal(codes,ctrl.trainer.alignment.hard_codes),'Initial SID changed'
        ctrl.restore(saved)
        again=ctrl.checkpoint()
        delta2=replay.differences(saved,again)
        assert not delta2,delta2
        result=dict(status='passed',parent_sha256=expected,full_state_restore_exact=True,
                    cuda_rng_restore_exact=True,gate_checkpoint_roundtrip_exact=True,
                    initial_codebook_max_abs_error=error,initial_sid_exact=True,
                    r_updates=9702,t_updates=120,total_T=ctrl.total_t,total_phases=len(ctrl.trainer.plan))
    replay.write_json(ROOT/('R14_EVIDENCE.json' if mode=='evidence' else 'GPU_GATE.json'),result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    main(sys.argv[1])
