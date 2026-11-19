"""Retain the delivered D supervisor, replace its completed-D resume checks by the E start binding."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
s=(ROOT/'outputs/stage09_spread200_20261031_fresh_r1/job_async_r2.py').read_text(encoding='utf-8')
s=s.replace('h1_spread200_20261031_r1','h1_block40_r6_single_20261101_r1')
s=s.replace('H1-Spread200-R35','H1-Block40-R6-R35').replace('formal_async_r2','formal_r1')
begin=s.index("    if gate['status']!=");end=s.index("    inputs=read(root/'INPUTS.json')",begin)
s=s[:begin]+'''    if gate['status']!='accepted_single_e_runtime':raise ValueError('E start not bound')
    if (gate['consumed_gpu_seconds_upper_bound']+gate['platform_max_seconds']+gate['release_allowance_seconds']>86400
            or gate['inner_max_seconds']>=gate['platform_max_seconds']):
        raise ValueError('Cumulative budget does not fit')
    runtime=root/'runtime.py'
    if sha(runtime)!=gate['runtime_sha256']:raise ValueError('Runtime changed')
    if sha(root/'CPU_SCHEDULE_CHECK.json')!=gate['schedule_check_sha256']:
        raise ValueError('Schedule acceptance changed')
    run=root/'runs/H1-Block40-R6-R35'
''' + s[end:]
s=s.replace('scripts.listwise_diffgrm.spread200_run','scripts.listwise_diffgrm.block40_single_run')
(HERE/'job.py').write_text(s,encoding='utf-8')
