"""One GPU per target; CF first produces the shared teacher. No model overrides."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
DATA = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/inputs/dataset')
CONTENT = DATA.parent.parent / 'assets/pse_v1/content_features.npy'
sys.path.insert(0, str(ROOT))


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = read(ROOT/'SOURCE_MANIFEST.json')
    for name, expected in manifest['files'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('Deployed source changed: '+name)
    if not DATA.is_dir() or not CONTENT.is_file():
        raise RuntimeError('Original dataset/content not mounted')
    return manifest


def snapshot():
    result = {'unix': time.time(), 'teacher': read(ROOT/'teacher/COMPLETE.json'),
              'teacher_progress': read(ROOT/'teacher/PROGRESS.json')}
    if result['teacher']:
        result['teacher_sha_verified'] = digest(ROOT/'teacher/items.pt') == result['teacher']['items_sha256']
    result['arms'] = {}
    for mode in ('cf', 'hybrid'):
        result['arms'][mode] = {'state': read(ROOT/f'{mode}_STATE.json')}
        for folder, receipt in ((f'{mode}_rq', 'COMPLETE.json'), (mode, 'TRAINING_COMPLETE.json'),
                                (mode, 'TEST_COMPLETE.json')):
            result['arms'][mode][receipt] = read(ROOT/folder/receipt)
        events = ROOT/mode/'events.jsonl'
        if events.exists():
            with events.open('rb') as stream:
                stream.seek(max(0, events.stat().st_size-16000))
                lines = stream.read().decode('utf-8', errors='replace').splitlines()
            result['arms'][mode]['last_events'] = [json.loads(s) for s in lines[-2:] if s.startswith('{')]
    return result


def run(mode):
    verify()
    os.chdir(ROOT)
    state = ROOT/f'{mode}_STATE.json'

    def phase(name, **kwargs):
        value = dict(mode=mode, phase=name, unix=time.time(), pid=os.getpid(), **kwargs)
        write(state, value)
        print(json.dumps(value), flush=True)

    def command(stage, args):
        phase(stage)
        subprocess.run([sys.executable, '-B', '-u', '-m', 'scripts.rqcd_mixer.run',
                        '--config', f'configs/rqcd_mixer/v2_{mode}.json', stage,
                        '--runtime', 'runtime.locked.json', '--dataset', str(DATA), *args], check=True)

    try:
        if mode == 'cf' and not (ROOT/'teacher/COMPLETE.json').exists():
            phase('shared_teacher')
            from scripts.rqcd_mixer import teacher
            from scripts.rqcd_mixer.config import load_config
            original_advance = teacher.TeacherTrainer.advance
            started = time.monotonic()
            first_update = [None]

            def logged_advance(trainer):
                event = original_advance(trainer)
                if first_update[0] is None:
                    first_update[0] = trainer.updates-1
                if trainer.updates % 200 == 0 or event['kind'] == 'teacher_epoch':
                    value = dict(event, unix=time.time(), elapsed_seconds=time.monotonic()-started,
                                 updates_this_process=trainer.updates-first_update[0])
                    write(ROOT/'teacher/PROGRESS.json', value)
                    print(json.dumps(value), flush=True)
                return event

            # Observation only: this wrapper does not change tensors, RNG or optimizer steps.
            teacher.TeacherTrainer.advance = logged_advance
            try:
                teacher.train_teacher(argparse.Namespace(
                    runtime=ROOT/'runtime.locked.json', device='cuda', dataset=DATA,
                    output=ROOT/'teacher', resume=(ROOT/'teacher/last.pt').exists()),
                    load_config(ROOT/'configs/rqcd_mixer/v2_cf.json'))
            finally:
                teacher.TeacherTrainer.advance = original_advance
        complete = read(ROOT/'teacher/COMPLETE.json')
        if not complete or digest(ROOT/'teacher/items.pt') != complete['items_sha256']:
            raise RuntimeError('Shared teacher is not complete and SHA-verified')
        prepared = f'{mode}_prepared'
        if not (ROOT/prepared/'PREPARED.json').exists():
            command('prepare', ['--content', str(CONTENT), '--teacher', str(ROOT/'teacher/items.pt'),
                                '--output', prepared])
        common = ['--prepared', prepared, '--arm', 'B1']
        if not (ROOT/f'{mode}_rq/COMPLETE.json').exists():
            resume = ['--resume'] if (ROOT/f'{mode}_rq/last.pt').exists() else []
            command('pretrain', common+['--output', f'{mode}_rq', *resume])
        if not (ROOT/mode/'TRAINING_COMPLETE.json').exists():
            resume = ['--resume'] if (ROOT/mode/'last.pt').exists() else []
            command('train', common+['--pretrained', f'{mode}_rq', '--output', mode, *resume])
        if not (ROOT/mode/'TEST_COMPLETE.json').exists():
            command('test', common+['--pretrained', f'{mode}_rq', '--run', mode])
        phase('complete')
    except BaseException as exc:
        phase('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    if sys.argv[1] == 'snapshot':
        print(json.dumps(snapshot(), ensure_ascii=False))
    elif sys.argv[1] == 'verify':
        print(json.dumps({'source_verified': True, 'git_commit': verify()['git_commit']}))
    else:
        if sys.argv[1] not in ('cf', 'hybrid'):
            raise SystemExit('Expected cf, hybrid, snapshot or verify')
        run(sys.argv[1])
