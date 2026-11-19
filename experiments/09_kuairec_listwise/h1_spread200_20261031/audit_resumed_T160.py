"""Independently check the first resumed 20T block; no model execution."""
import hashlib
import json
import math
from pathlib import Path
import tarfile
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    root=Path(__file__).resolve().parent;archive=root/'T160_boundary_evidence.tar.gz'
    assert sha(archive)=='f25fa1708b56f2041327cf0b1fa67f66b58279fc008efa458bc484d640b9854e'
    out=root/'T160_boundary_local_r1';out.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:tf.extractall(out,filter='data')
    run=out/'runs/H1-Spread200-R35';jobs=out/'jobs/formal_async_r2/tasks'
    rows=[json.loads(s) for s in (out/'recovery_async_r2/T160_training_slice.jsonl').read_text().splitlines()]
    assert len(rows)==20
    original=np.random.default_rng(np.random.SeedSequence([2024,7919,17])).permutation(709443)
    selected=original[20480:40960]
    block_sha=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
    expected=next(x for x in read(root/'CPU_INPUT_ACCEPTANCE.json')['t_blocks'] if x['phase']==33)
    order=read(run/'sample_orders/phase33.json')
    assert block_sha==expected['sha256']==order['indices_sha256_int64_le']
    assert order['original_t_phase']==17 and order['original_t_half']==1 and order['lists']==20480
    r_lr=rows[0]['learning_rates_used']['R']
    for i,row in enumerate(rows):
        assert (row['phase'],row['phase_index'],row['epoch'],row['update'])==('T',33,26,i+1)
        assert row['r_updates']==18018 and row['t_updates']==141+i and row['lists']==1024
        assert row['learning_rates_used']==dict(R=r_lr,codebook=1e-5,mixer=1e-4)
        noise=row['noise_used'];assert noise['optimizer_step']==140+i and noise['total_optimizer_steps']==400
        assert noise['softmax_temperature']==1.0
        assert abs(noise['gumbel_noise_scale']-(.2+.4*(1+math.cos(math.pi*(140+i)/400))))<1e-14
        batch=selected[i*1024:(i+1)*1024]
        assert row['sample_indices_sha256']==hashlib.sha256(batch.astype('<i8').tobytes()).hexdigest()
    boundaries=[]
    for phase,label,t,kind in ((32,'phase32_R_R26_T140',140,'validation'),(33,'phase33_T_R26_T160',160,'diagnostic')):
        receipt=read(run/'checkpoint_receipts'/(label+'.pt.json'));task=read(jobs/f'phase{phase:02d}.json')
        assert (task['kind'],task['epoch'],task['label'])==(kind,26,label)
        assert task['checkpoint_sha256']==receipt['sha256']
        assert receipt['r_updates']==18018 and receipt['t_updates']==t and all(receipt['fields'].values())
        assert receipt['progress']==dict(phase=phase+1,update=0,boundary_prepared=False,rng_prepared=False)
        boundaries.append(dict(phase=phase,t_updates=t,checkpoint_sha256=receipt['sha256'],state_sha256=receipt['state_sha256']))
    report=dict(status='passed_resumed_20T_samples_and_400T_clock',updates=20,new_T_start=141,new_T_end=160,
        R_updates_unchanged=18018,original_T_phase=17,original_T_half=1,ordered_block_sha256=block_sha,
        exact_all_batch_hashes=True,noise_total_steps=400,T_learning_rates=dict(codebook=1e-5,mixer=1e-4),
        R_learning_rate_constant_during_T=r_lr,stage_metadata_bindings=boundaries,archive_sha256=sha(archive),
        limitation='Sample/order/noise/LR/counter and receipt checks for this block only. No model-state cold restore or metric replay; complete trajectory and final restoration remain required.')
    (root/'RESUMED_T160_ACCEPTED.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
