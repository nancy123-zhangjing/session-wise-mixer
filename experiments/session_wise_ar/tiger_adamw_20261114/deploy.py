"""Create the versioned TIGER-AdamW code root (new dir), patch, verify, and prepare the parent on CPU."""
import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[0] / 'cf_freeze_r9_20261114_v1/paired_bootstrap_20261114'))
import jexec  # noqa: E402

B = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/'
SRC, DST = B + 'session_wise_ar_five_20261113_r2', B + 'session_wise_ar_tiger_adamw_20261114_v1'
PATCHES = {
    'scripts/session_wise_ar/runtime.py': [
        ("    if cfg['family'] == 'tiger':\n        return torch.optim.Adagrad(",
         "    # Default (absent field) keeps the original paper-configured Adagrad exactly.\n"
         "    if cfg['family'] == 'tiger' and c.get('optimizer', 'adagrad') == 'adagrad':\n        return torch.optim.Adagrad("),
        ("            # Original DIGER pretraining trainer clips at norm 1 as well.\n"
         "            finite_step(loss, [tok], [optimizer], [scheduler], 1.0 if cfg['family']=='diger' else None)",
         "            # Original DIGER pretraining trainer clips at norm 1 as well; TIGER clips only\n"
         "            # when its (versioned) config explicitly sets rq.clip_norm.\n"
         "            finite_step(loss, [tok], [optimizer], [scheduler], 1.0 if cfg['family']=='diger' else c.get('clip_norm'))"),
    ],
    'scripts/session_wise_ar/config.py': [
        ('        "AR-D0": ("diger", False), "AR-D1": ("diger", True)}',
         '        "AR-D0": ("diger", False), "AR-D1": ("diger", True),\n'
         '        # 2026-11-14 versioned TIGER rerun: only the RQ pretraining optimizer differs.\n'
         '        "AR-T0A": ("tiger_adamw", False), "AR-T1A": ("tiger_adamw", True)}'),
    ],
}
NEW = {'configs/session_wise_ar/tiger_adamw.json': (REPO / 'configs/session_wise_ar/tiger_adamw.json').read_bytes(),
       'delivery/worker_adamw.py': (HERE / 'worker_adamw.py').read_bytes()}
payload = {k: base64.b64encode(v).decode() for k, v in NEW.items()}
src = f'''
import base64, shutil, subprocess, json, hashlib
from pathlib import Path
s, d = Path({SRC!r}), Path({DST!r})
if not d.exists():
    d.mkdir()
    for name in ('scripts', 'configs', 'docs', 'ar_deps', 'external'):
        shutil.copytree(s / name, d / name, ignore=shutil.ignore_patterns('__pycache__'))
    (d / 'delivery').mkdir()
    for f, reps in {PATCHES!r}.items():
        p = d / f; text = p.read_text()
        for old, new in reps:
            assert text.count(old) == 1, (f, old[:60]); text = text.replace(old, new)
        p.write_text(text)
    for name, data in {payload!r}.items():
        (d / name).write_bytes(base64.b64decode(data))
print({{f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in list({list(PATCHES)!r}) + list({list(NEW)!r})}})
check = """
import sys; sys.path.insert(0, '.')
from scripts.session_wise_ar.config import load
from scripts.session_wise_ar import runtime
import torch
for arm in ('AR-T0', 'AR-T0A', 'AR-T1A'):
    cfg = load(arm); m = torch.nn.Linear(2, 2)
    opt, sch = runtime.pretrain_optimizer(m, cfg, 11)
    print(arm, cfg['family'], cfg['mixer_enabled'], type(opt).__name__, opt.defaults['lr'], sch is not None, cfg['rq'].get('clip_norm'), cfg['rq']['epochs'])
"""
env = dict(PYTHONPATH=str(d / 'ar_deps') + ':' + str(d), PATH='/usr/bin:/bin', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
r = subprocess.run([{jexec.PYTHON!r}, '-B', '-c', check], cwd=d, env=env, capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
r = subprocess.run([{jexec.PYTHON!r}, '-B', '-u', 'delivery/worker_adamw.py', 'prepare'], cwd=d, env=env, capture_output=True, text=True)
print('PREPARE rc', r.returncode, r.stdout[-800:], r.stderr[-1500:])
print(sorted(p.name for p in (d / 'parents' / 'tiger_adamw').iterdir()) if (d / 'parents' / 'tiger_adamw').exists() else 'no parent')
'''
print(jexec.python(src, timeout=1500))
