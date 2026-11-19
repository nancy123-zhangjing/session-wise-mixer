"""Copy the delivered D single-GPU implementation; change only E scheduling and entrypoints."""
import hashlib,json,shutil,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
D=ROOT/'outputs/stage09_spread200_20261031_fresh_r1'
OLD=ROOT/'experiments/09_kuairec_listwise/h1_spread200_20261031'
CODE=HERE/'code'
REMOTE='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1'
def change(s,a,b):
    if a not in s:raise ValueError('Expected source anchor missing: '+a[:80])
    return s.replace(a,b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if CODE.exists():raise FileExistsError('Do not overwrite prepared source')
    shutil.copytree(D/'code',CODE,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source=CODE/'scripts/listwise_diffgrm'
    s=(source/'spread200.py').read_text(encoding='utf-8')
    s=change(s,"FORMAT = 'stage09_h1_spread200_v1'","FORMAT = 'stage09_h1_block40_single_v1'")
    s=change(s,"NAME = 'H1-Spread200-R35'","NAME = 'H1-Block40-R6-R35'")
    s=change(s,'old_block != 2 * new_block','old_block != new_block or new_block != 40')
    s=change(s,'Only halving T block size and explicit path relocation are allowed','Only the E plan override and explicit path relocation are allowed')
    s=change(s,"name.startswith('scripts/listwise_diffgrm/spread200'))", "name.startswith('scripts/listwise_diffgrm/spread200') or\n                name.startswith('scripts/listwise_diffgrm/block40_single'))")
    begin=s.index('def paired_indices(');end=s.index('\ndef index_sha(',begin)
    s=s[:begin]+'''def original_phase(row):
    if row['kind']=='T':return 5+4*(row['cycle']-1)
    epoch=row['epoch']
    return epoch-1 if epoch<=5 else 6+4*((epoch-6)//3)+(epoch-6)%3


def e_plan(config,count):
    from .training import phase_plan
    cfg=copy.deepcopy(config)
    cfg['training'].update(cycles=5,t_updates_per_cycle=40,r_epochs_per_cycle=6)
    return phase_plan(cfg,count)


def paired_indices(config,count,phase,row):
    return phase_indices(config,count,original_phase(row),row)

''' + s[end:]
    s=change(s,"self.branch = 'D'","self.branch = 'E'")
    s=change(s,"['T', 'R', 'R', 'R'] * 10","(['T'] + ['R'] * 6) * 5")
    s=change(s,'Only the ten-cycle H1 schedule is supported','Only five 40T/6R cycles are supported')
    s=change(s,'t_updates=self.width * 10','t_updates=self.width * 5, t_block=40, r_block=6, cycles=5')
    s=change(s,"paired_t_batches='original_first_half_then_second_half'","paired_t_batches='original_first_five_complete_40T_blocks'")
    s=change(s,"seed_all(self.trainer.config['seed'] + 100003 * (phase + 1))","seed_all(self.trainer.config['seed'] + 100003 * (original_phase(self.row) + 1))")
    s=change(s,"    cfg['training']['t_updates_per_cycle'] = 20\n",'')
    s=change(s,'    controller = SpreadController(trainer, parent', '    trainer.plan = e_plan(cfg, len(data))\n    controller = SpreadController(trainer, parent')
    s=change(s,'Explicit R5 fork: pair the original first 200 T batches across ten blocks.','Single-card E: original single-GPU D implementation, five 40T/6R blocks.')
    (source/'block40_single.py').write_text(s,encoding='utf-8')
    s=(source/'spread200_run.py').read_text(encoding='utf-8')
    s=change(s,'from .spread200 import','from .block40_single import')
    s=change(s,'build_spread, index_sha, paired_indices','build_spread, index_sha, paired_indices, original_phase')
    s=change(s,"(5 + 4 * ((phase - 5) // 8)) if row['kind'] == 'T' else None","original_phase(row)")
    s=change(s,"((phase - 5) // 4) % 2 if row['kind'] == 'T' else None",'None')
    s=change(s,"(row['epoch'] - 6) % 3 == 0","(row['epoch'] - 6) % 6 == 0")
    (source/'block40_single_run.py').write_text(s,encoding='utf-8')
    s=(D/'async_resume.py').read_text(encoding='utf-8')
    s=change(s,'h1_spread200_20261031_r1','h1_block40_r6_single_20261101_r1')
    s=change(s,'H1-Spread200-R35','H1-Block40-R6-R35')
    s=change(s,'formal_async_r2','formal_r1')
    s=change(s,'accepted_async_resume','accepted_single_e_runtime')
    s=change(s,'SPREAD200_EXECUTION_MANIFEST.json','E_EXECUTION_MANIFEST.json')
    s=change(s,'from scripts.listwise_diffgrm.spread200 import','from scripts.listwise_diffgrm.block40_single import')
    s=change(s,'paired_indices, index_sha, BEST_EPOCHS','paired_indices, index_sha, BEST_EPOCHS, original_phase')
    s=change(s,"(row['epoch'] - 6) % 3 == 0","(row['epoch'] - 6) % 6 == 0")
    s=change(s,"    p.add_argument('--worker')","    p.add_argument('--worker')\n    p.add_argument('--resume')")
    begin=s.index("    resume = Path(gate['checkpoint'])");end=s.index("    stop = {'requested': False}",begin)
    s=s[:begin]+"    resume = Path(a.resume) if a.resume else None\n    if resume is not None:resume.resolve().relative_to(output)\n"+s[end:]
    s=change(s,'with run_lock(output, resume=True):\n            c.restore(torch.load(resume, map_location=\'cpu\', weights_only=False))', '''with run_lock(output, resume=bool(resume)):
            if resume:
                c.restore(torch.load(resume, map_location='cpu', weights_only=False))
                rows=sum(1 for _ in (output/'training.jsonl').open(encoding='utf-8'))
                if rows != c.trainer.r_updates-3465+c.trainer.t_updates:
                    raise ValueError('Reconcile journal to the saved checkpoint before resuming')
            else:
                write_once(output/'request.json',dict(parent_sha256=gate['parent_sha256'],
                    subset_sha256=sha(inputs['subset']),protocol=c.protocol,
                    execution_binding=c.execution_binding,warmup_predictions_sha256=sha(inputs['warmup_predictions'])))
                save_boundary(c,output,'initial_R05_T000.pt')
                last(c,output)
                catalog(c,output,'initial_R05_T000')
                diagnostic(c,assets,validation,subset,output,'initial_R05_T000',source=inputs['warmup_predictions'])''')
    s=change(s,"(5 + 4 * ((phase - 5) // 8)) if row['kind'] == 'T' else None","original_phase(row)")
    s=change(s,"((phase - 5) // 4) % 2 if row['kind'] == 'T' else None",'None')
    (HERE/'runtime.py').write_text(s,encoding='utf-8')
    inputs=json.loads((D/'INPUTS.json').read_text(encoding='utf-8'))
    inputs['subset']=REMOTE+'/validation_subset_8192.json'
    (HERE/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(D/'validation_subset_8192.json',HERE/'validation_subset_8192.json')
    print(json.dumps({'code':str(CODE),'new_scientific_modules':['block40_single.py','block40_single_run.py'],
        'base':'delivered D single GPU; all preexisting scientific files copied unchanged'}))
if __name__=='__main__':main()
