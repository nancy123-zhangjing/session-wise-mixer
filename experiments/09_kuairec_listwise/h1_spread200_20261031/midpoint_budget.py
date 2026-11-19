"""Budget forecast from completed new training; no model execution."""
import hashlib
import json
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
log = here/'midpoint_R20/training.jsonl'
rows = [json.loads(x) for x in log.read_text(encoding='utf-8').splitlines()]
receipt = json.loads((here/'midpoint_R20/ACCEPTED.json').read_text(encoding='utf-8'))
assert receipt['source_log_sha256'] == hashlib.sha256(log.read_bytes()).hexdigest()
assert receipt['audited_new_updates'] == 10495
rates = {k:mean(x['seconds'] for x in rows[:10495] if x['phase']==k) for k in ('R','T')}
# R20 validation is counted entirely again for a conservative remaining-work
# forecast. Five future full validations, ten subsets and at most two tests.
remaining_examples = 6*71657+10*8192+2*53194
planning_eval_rate = .04
elapsed = 26540.98647458502  # formal status read at 09:06:02 local
remaining = 10395*rates['R']+100*rates['T']+remaining_examples*planning_eval_rate
projected = elapsed+1.25*remaining+90
result = dict(status='forecast_only', observation_time='2026-10-31T09:06:02+08:00',
    source_log_sha256=receipt['source_log_sha256'], measured_update_mean_seconds=rates,
    formal_elapsed_seconds=elapsed, remaining_R_updates=10395, remaining_T_updates=100,
    remaining_eval_examples_conservative=remaining_examples,
    eval_seconds_per_example_planning_assumption=planning_eval_rate,
    future_work_margin=1.25, save_release_reserve_seconds=90,
    projected_formal_hours_including_future_margin_and_reserve=projected/3600,
    formal_hard_limit_hours=23.9, forecast_within_existing_limit=projected<86040,
    additional_GPU_budget_authorized=False, settlement_verified=False,
    limitation='Planning estimate, not a measured future runtime or bill; includes all current R20 validation again.')
with (here/'MIDPOINT_BUDGET_CHECK.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,indent=2)
    f.write('\n')
print(json.dumps(result))
