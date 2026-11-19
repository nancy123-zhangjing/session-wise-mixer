"""Independent CPU inspection of the first 10,495 new updates; no model execution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
log = HERE / 'midpoint_R20/training.jsonl'
raw = log.read_bytes()
lines = raw.splitlines()
assert len(lines) >= 10495, len(lines)
cpu = json.loads((HERE / 'CPU_INPUT_ACCEPTANCE.json').read_text(encoding='utf-8'))
orders = {x['phase']: x for x in cpu['t_blocks'] + cpu['r_phases']}
r, t, phase, u, indices = 3465, 0, 5, 0, None
counts = {}
for line in lines[:10495]:
    row = json.loads(line)
    kind = 'T' if (phase-5) % 4 == 0 else 'R'
    width = 20 if kind == 'T' else 693
    if indices is None:
        b = (phase-5)//4
        source = 5+4*(b//2) if kind == 'T' else phase
        indices = np.random.default_rng(np.random.SeedSequence([2024,7919,source])).permutation(709443)
        if kind == 'T':
            indices = indices[(b % 2)*20480:((b % 2)+1)*20480]
        assert hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest() == orders[phase]['sha256']
        assert len(indices) == orders[phase]['lists']
    batch = indices[u*1024:(u+1)*1024]
    assert (row['phase'],row['phase_index'],row['update'],row['lists']) == (kind,phase,u+1,len(batch))
    assert row['sample_indices_sha256'] == hashlib.sha256(batch.astype('<i8',copy=False).tobytes()).hexdigest()
    noise = row['noise_used']
    assert (noise['optimizer_step'],noise['total_optimizer_steps'],noise['softmax_temperature']) == (t,400,1)
    assert abs(noise['gumbel_noise_scale']-(.2+.4*(1+math.cos(math.pi*t/400)))) < 1e-14
    lr = row['learning_rates_used']
    assert (lr['codebook'],lr['mixer']) == (1e-5,1e-4)
    assert abs(lr['R']-(.001*.5*(1+math.cos(math.pi*(r-500)/(24255-500))))) < 1e-14
    r += kind == 'R'
    t += kind == 'T'
    assert (row['r_updates'],row['t_updates']) == (r,t)
    counts[phase] = counts.get(phase,0)+1
    u += 1
    if u == width:
        phase += 1
        u, indices = 0, None
assert (r,t,phase,u) == (13860,100,25,0)
result = dict(status='passed_prefix_only', audited_new_updates=10495,
    captured_rows=len(lines), r_updates=r, t_updates=t, new_R=10395, new_T=100,
    paired_T_input_positions=102400, phase_counts=counts,
    all_logged_batch_indices_order_and_sizes_verified=True,
    original_35R_LR_verified=True, original_400T_noise_amplitudes_verified=True,
    tokenizer_LRs_verified=True, source_log_sha256=hashlib.sha256(raw).hexdigest(),
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_acceptance_sha256=hashlib.sha256((HERE/'CPU_INPUT_ACCEPTANCE.json').read_bytes()).hexdigest(),
    scope='Completed training prefix only; validation, tests, final recovery and resources still pending')
with (HERE/'midpoint_R20/ACCEPTED.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps(result))
