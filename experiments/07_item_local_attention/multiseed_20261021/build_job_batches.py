"""Write explicit Inspire Job matrices; never submit jobs."""
from pathlib import Path
import json
import shlex

HERE = Path(__file__).resolve().parent
GOAL = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261021_multiseed_r1'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
PROJECT_ROOT = '/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized'


def main():
    defaults = dict(workspace='分布式训练空间', project='公共科研项目',
                    group='开发区-H100-cuda13.2版本-183核', quota='1,20,200',
                    image='pytorch:25.06-py3:25.06', priority=4, nodes=1,
                    shm_size=32, auto_fault_tolerance=False, enable_notification=False)
    for mode in ('smoke', 'formal'):
        jobs = []
        for seed in (2025, 2026):
            for variant in ((None,) if mode == 'smoke' else ('control', 'item_local')):
                command = (f'cd {GOAL}/code && exec {PYTHON} -B '
                           f'{GOAL}/code/scripts/train/item_local_multiseed_gpu_job.py '
                           f'--mode {mode} --goal-root {GOAL} --seed {seed} '
                           f'--deployment-lock {GOAL}/delivery/DEPLOYMENT_LOCK.json '
                           f'--cpu-gate {GOAL}/delivery/CPU_GATE.json '
                           f'--budget-reservation {GOAL}/delivery/BUDGET_RESERVATION.json '
                           f'--project-root {PROJECT_ROOT} --attempt 1 ')
                if mode == 'smoke':
                    command += '--max-seconds 750 --platform-max-seconds 900'
                else:
                    command += (f'--variant {variant} --max-seconds 10440 --platform-max-seconds 10620 '
                                f'--smoke-gate {GOAL}/seeds/seed{seed}/delivery/smoke_seed{seed}_attempt1/receipt.json')
                arm = '' if variant is None else '-' + variant.replace('_', '-')
                jobs.append(dict(**defaults, name=f'nancy-stage07-multiseed-{seed}-{mode}{arm}-20261021-r1',
                                 command='bash -c ' + shlex.quote(command), max_time=.25 if mode == 'smoke' else 2.95,
                                 description='Stage07 paired joint training; validation beam32/256; no test'))
        path = HERE / (mode.upper() + '_JOBS.json')
        payload = dict(jobs=jobs)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
        assert not (HERE / (mode.upper() + '_SUBMISSION.json')).exists(), 'Submitted Job matrix is immutable'
        path.write_text(text, encoding='utf-8')
        print(f'{path}: {len(jobs)} jobs')


if __name__ == '__main__':
    main()
