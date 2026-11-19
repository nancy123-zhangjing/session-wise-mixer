"""Audit only the newly resumed 30 updates against the isolated CUDA oracle."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    snapshot=root/'async_first30_training.jsonl'
    lines=snapshot.read_bytes().splitlines(keepends=True)
    assert len(lines)==13044
    prefix=b''.join(lines[:13014])
    assert hashlib.sha256(prefix).hexdigest()=='348e7a914323ae4fa7e587d0578406c35357ab67c3c005e212a7c7eb65186b1e'
    oracle=json.loads((root/'concurrency_evidence_local_r1/concurrency_preflight_r1/train/RESULT.json').read_text())
    indices=np.random.default_rng(np.random.SeedSequence([2024,7919,30])).permutation(709443)
    for offset,line in enumerate(lines[13014:]):
        row=json.loads(line)
        assert row['phase']=='R' and row['epoch']==24 and row['phase_index']==30
        assert row['update']==401+offset and row['r_updates']==16340+offset and row['t_updates']==140
        assert row['loss']==oracle['losses'][offset] and row['lists']==1024
        selected=indices[(400+offset)*1024:(401+offset)*1024]
        assert hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()==row['sample_indices_sha256']
        assert row['noise_used']['optimizer_step']==140 and row['noise_used']['total_optimizer_steps']==400
        assert row['learning_rates_used']['codebook']==1e-5 and row['learning_rates_used']['mixer']==1e-4
    result=dict(status='passed_first30_real_formal_continuation',preserved_prefix_rows=13014,
        prefix_sha256=hashlib.sha256(prefix).hexdigest(),new_R_updates=30,first_update=401,last_update=430,
        exact_loss_match_to_isolated_CUDA_oracle=True,ordered_batch_hashes_match=True,
        R_updates_end=16369,T_updates_end=140,training_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        limitation='Only the newly resumed first30 updates; full trajectory, optimizer-state cold restore, remaining evaluations and delivery remain pending.')
    (root/'ASYNC_FIRST30_ACCEPTED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
