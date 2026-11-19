"""Pure schedule rules, shared by runner and CPU checks."""
SEEDS = (2025, 2026)
ARMS = ('H0',)

def skipped(row, arm):
    return arm == 'H0' and row['kind'] == 'T' and row['epoch'] >= 20

def counts(plan, progress, arm):
    phase, update = progress['phase'], progress['update']
    if not 0 <= phase <= len(plan) or update < 0:
        raise ValueError('Invalid cursor')
    if phase == len(plan):
        if update: raise ValueError('Completed cursor has update')
    elif update > (0 if skipped(plan[phase], arm) else plan[phase]['updates']):
        raise ValueError('Invalid update cursor')
    result = dict(R=0, T=0)
    for row in plan[:phase]:
        if not skipped(row, arm): result[row['kind']] += row['updates']
    if phase < len(plan): result[plan[phase]['kind']] += update
    return result
