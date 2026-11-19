"""Independently audit the completed 200T budget, order, LR and 400T clock."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(indices):return hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;path=root/'ALL_T200_SNAPSHOT.json'
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha=='9c301e7459c4c9409dba5f860117b49cadeb73990de1e6726a8932b43b5eb857'
    data=json.loads(path.read_text());accepted=json.loads((root/'CPU_INPUT_ACCEPTANCE.json').read_text())
    expected={r['phase']:r for r in accepted['t_blocks']}
    rows=data['rows'];orders=data['orders'];assert len(rows)==200 and len(orders)==10
    stream=[];blocks=[]
    for block in range(10):
        phase=5+4*block;source=5+4*(block//2);half=block%2;epoch=5+3*block
        original=np.random.default_rng(np.random.SeedSequence([2024,7919,source])).permutation(709443)
        indices=original[half*20480:(half+1)*20480]
        order=orders[block]
        assert order['phase']==phase and order['kind']=='T' and order['epoch']==epoch and order['lists']==20480
        assert order['original_t_phase']==source and order['original_t_half']==half
        assert digest(indices)==order['indices_sha256_int64_le']==expected[phase]['sha256']
        r=epoch*693
        lr=.001*.5*(1+math.cos(math.pi*((r-500)/(24255-500))))
        for offset,row in enumerate(rows[block*20:(block+1)*20]):
            t=block*20+offset
            assert (row['phase'],row['phase_index'],row['epoch'],row['update'])==('T',phase,epoch,offset+1)
            assert (row['r_updates'],row['t_updates'],row['lists'])==(r,t+1,1024)
            assert row['sample_indices_sha256']==digest(indices[offset*1024:(offset+1)*1024])
            assert row['learning_rates_used']['codebook']==1e-5 and row['learning_rates_used']['mixer']==1e-4
            assert abs(row['learning_rates_used']['R']-lr)<1e-14
            noise=row['noise_used']
            assert noise['optimizer_step']==t and noise['total_optimizer_steps']==400 and noise['softmax_temperature']==1
            assert abs(noise['gumbel_noise_scale']-(.2+.4*(1+math.cos(math.pi*t/400))))<1e-14
        stream.append(indices);blocks.append(dict(phase=phase,original_phase=source,half=half,sha256=digest(indices)))
    whole=np.concatenate(stream)
    original=np.concatenate([np.random.default_rng(np.random.SeedSequence([2024,7919,p])).permutation(709443)[:40960]
                             for p in (5,9,13,17,21)])
    assert len(whole)==204800 and np.array_equal(whole,original)
    assert digest(whole)==accepted['original_t_stream_sha256']==accepted['spread_t_stream_sha256']
    result=dict(status='passed_all_200_actual_T_updates',T_updates=200,T_blocks=10,
        sample_occurrences=204800,all_batch_hashes_exact=True,ordered_stream_matches_original_first200T=True,
        ordered_stream_sha256=digest(whole),R_updates_constant_within_each_T_block=True,
        original_noise_total_steps=400,codebook_lr=1e-5,mixer_lr=1e-4,original_35R_cosine_lr_preserved_at_every_T=True,
        blocks=blocks,snapshot_sha256=sha,
        cpu_input_acceptance_sha256=hashlib.sha256((root/'CPU_INPUT_ACCEPTANCE.json').read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Completed T budget/order/noise/LR/counters only. All R updates, complete scientific endpoints, predictions, RNG persistence across R, final cold restore and resource closure remain separate requirements.')
    (root/'ALL_T200_ACCEPTED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
