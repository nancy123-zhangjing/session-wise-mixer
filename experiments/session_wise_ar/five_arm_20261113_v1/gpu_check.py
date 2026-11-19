"""One combined GPU gate on real shared parents; outputs never seed formal runs."""
import json,time,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.session_wise_ar.config import load
from scripts.session_wise_ar.runtime import load_parent,configure_device,recommendation_optimizers,finite_step
from scripts.session_wise_ar.model import ARRecommender
from scripts.session_wise_ar.tokenizer import make_tokenizer,joint_loss
from scripts.session_wise_ar.data import SessionTasks,PackedSplit
from scripts.session_wise_ar.catalog import add_suffix,Catalog
from scripts.rqcd_mixer.data import load_content
from scripts.rqcd_mixer.common import seed_all,cpu_copy,rng_state,restore_rng,atomic_save,read_state,sha256,write_json
DATA=ROOT.parent/'goal_20261027_r1/inputs/dataset'
CONTENT=ROOT.parent/'goal_20261027_r1/assets/pse_v1/content_features.npy'


def main():
    configure_device('cuda');torch.set_num_threads(4)
    out=ROOT/'gpu_check_r2';out.mkdir(exist_ok=False)
    results={}
    for arm in ('AR-T1','AR-D1'):
        seed_all(2024)
        cfg=load(arm);parent,meta=load_parent(cfg,ROOT/'parents_devicefix'/cfg['family'])
        pre=make_tokenizer(cfg).cuda();pre.load_state_dict(parent['tokenizers']['intra'])
        content=load_content(CONTENT,cfg,'cuda')
        # Check pretraining forward/backward once, then discard all these changes.
        pre.train();loss,_=pre.loss(content[:1024]);loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in pre.parameters() if p.grad is not None)
        pre.zero_grad(set_to_none=True)
        tok=pre
        if cfg['family']=='diger':
            tok=make_tokenizer(cfg,joint=True).cuda();tok.load_pretrained(pre)
        else: tok.requires_grad_(False)
        tok.eval();sid=add_suffix(tok.indices(content),256)
        rec=ARRecommender(cfg).cuda();rec.load_state_dict(parent['recommender'])
        tasks=SessionTasks(PackedSplit(DATA,'train',cfg))
        batch=tasks.batch(torch.arange(256),'cuda')
        small={k:v[:4] for k,v in batch.items()}
        rec.eval()
        with torch.no_grad(): gpu_loss=rec.hard_loss(small,sid).item()
        cpu=ARRecommender(cfg);cpu.load_state_dict(cpu_copy(rec.state_dict()));cpu.eval()
        with torch.no_grad(): cpu_loss=cpu.hard_loss({k:v.cpu() for k,v in small.items()},sid.cpu()).item()
        assert abs(cpu_loss-gpu_loss)<=3e-4*max(1,abs(cpu_loss)),(cpu_loss,gpu_loss)
        del cpu
        optimizers,schedules=recommendation_optimizers(rec,tok,cfg,13857)
        def step():
            rec.train();tok.train(cfg['family']=='diger')
            loss=joint_loss(rec,tok,batch,sid)[0] if cfg['family']=='diger' else rec.hard_loss(batch,sid)
            finite_step(loss,[rec,tok] if cfg['family']=='diger' else [rec],optimizers,schedules,cfg['training']['clip_norm'])
            return loss.detach().item()
        step()
        checkpoint=out/(arm+'.pt')
        atomic_save(checkpoint,dict(rec=rec.state_dict(),tok=tok.state_dict(),rng=rng_state(),
                    optim=[o.state_dict() for o in optimizers],schedule=[s.state_dict() for s in schedules]))
        torch.cuda.synchronize();start=time.perf_counter();a=step();torch.cuda.synchronize();seconds=time.perf_counter()-start
        expected={'rec':cpu_copy(rec.state_dict()),'tok':cpu_copy(tok.state_dict())}
        saved=read_state(checkpoint);rec.load_state_dict(saved['rec']);tok.load_state_dict(saved['tok'])
        for o,s in zip(optimizers,saved['optim']):o.load_state_dict(s)
        for o,s in zip(schedules,saved['schedule']):o.load_state_dict(s)
        restore_rng(saved['rng']);b=step()
        max_error=0.0
        for name,model in (('rec',rec),('tok',tok)):
            for key,value in model.state_dict().items():
                actual=value.cpu();target=expected[name][key]
                torch.testing.assert_close(actual,target,rtol=1e-6,atol=1e-7)
                if actual.is_floating_point():max_error=max(max_error,(actual-target).abs().max().item())
        rec.eval()
        pred=Catalog(sid,256,parent['original_ids']).generate(rec,batch['context'][:2],batch['users'][:2],5,20)
        assert all(len(set(row))==5 for row in pred.tolist())
        results[arm]={'cpu_loss':cpu_loss,'gpu_loss':gpu_loss,'continued_loss':a,'restored_loss':b,
                      'restore_max_abs':max_error,'batch256_step_seconds':seconds,'checkpoint_sha256':sha256(checkpoint),
                      'parent_sha256':meta['initial_sha256'],'complete_unique_lists':True}
        write_json(out/'progress.json',results)
        del rec,tok,pre,optimizers,schedules,expected,saved,tasks,batch;torch.cuda.empty_cache()
    write_json(out/'COMPLETE.json',{'passed':True,'scope':'full architectures, real parents, batch256 FP32; not quality evaluation','results':results})


if __name__=='__main__':main()
