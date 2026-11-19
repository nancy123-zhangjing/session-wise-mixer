"""Run a single first-wave worker after the full SASRec gate has passed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .l5o_launch import wave_config


ARMS = ('H0', 'H1', 'T0A', 'T1A', 'CF-M0', 'CF-M1')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def invoke(module, *args):
    subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)], check=True)


def h_worker(arm, root, cfg_path):
    module = 'scripts.listwise_diffgrm.run'
    base = ('--config', cfg_path)
    gpu = ('--device', 'cuda', '--precision', 'bf16')
    warmup = root/'parents/h_warmup'
    output = root/'runs'/arm
    if arm == 'H0' and not (warmup/'complete.json').is_file():
        resume = ('--resume', warmup/'last.pt') if (warmup/'last.pt').is_file() else ()
        invoke(module, *base, 'warmup', *gpu, '--output', warmup, *resume)
    if not (warmup/'complete.json').is_file():
        raise ValueError('H1 requires the completed shared H0 warmup')
    if not (output/'complete.json').is_file():
        start = (('--resume', output/'last.pt') if (output/'last.pt').is_file()
                 else ('--shared-warmup', warmup/'last.pt'))
        invoke(module, *base, 'train', *gpu, '--arm', arm, '--output', output, *start)
    for name, checkpoint in (('best', 'best.pt'), ('final', 'last.pt')):
        destination = output/'test'/name
        if not (destination/'summary.json').is_file():
            invoke(module, *base, 'evaluate', *gpu, '--checkpoint', output/checkpoint,
                   '--split', 'test', '--output', destination)


def tiger_worker(arm, root, receipt, dataset, content):
    module = 'scripts.session_wise_ar.run'
    base = ('--arm', f'AR-{arm}', '--l5o-calibration', receipt)
    common = ('--dataset', dataset, '--content', content)
    parent = root/'parents/tiger'
    output = root/'runs'/arm
    if not (parent/'manifest.json').is_file():
        raise ValueError('Prepare the shared TIGER parent on CPU before submitting either arm')
    for phase in ('pretrain', 'train', 'test'):
        target = output/phase
        if (target/'complete.json').is_file():
            continue
        dependencies = ('--pretrain', output/'pretrain') if phase != 'pretrain' else ()
        if phase == 'test':
            dependencies += ('--training', output/'train')
        elif (target/'last.pt').is_file():
            dependencies += ('--resume',)
        invoke(module, *base, phase, *common, '--prepared', parent, '--device', 'cuda',
               '--output', target, *dependencies)


def cf_worker(arm, root, receipt, dataset, content, runtime):
    module = 'scripts.rqcd_mixer.run'
    base = ('--config', 'configs/rqcd_mixer/v2_cf.json', '--l5o-calibration', receipt)
    common = ('--runtime', runtime, '--dataset', dataset, '--device', 'cuda')
    parent = root/'parents/cf'
    pretrain = root/'parents/cf_rq'
    output = root/'runs'/arm
    if arm == 'CF-M0':
        if not (parent/'PREPARED.json').is_file():
            invoke(module, *base, 'prepare', *common, '--content', content,
                   '--teacher', receipt.parent/'items.pt', '--output', parent)
        if not (pretrain/'COMPLETE.json').is_file():
            resume = ('--resume',) if (pretrain/'last.pt').is_file() else ()
            invoke(module, *base, 'pretrain', *common, '--prepared', parent,
                   '--arm', 'B0', '--output', pretrain, *resume)
    if not (pretrain/'COMPLETE.json').is_file():
        raise ValueError('CF-M1 requires the completed shared CF-M0 RQ pretraining')
    common += ('--prepared', parent, '--pretrained', pretrain,
               '--arm', 'B0' if arm == 'CF-M0' else 'B1')
    if not (output/'TRAINING_COMPLETE.json').is_file():
        resume = ('--resume',) if (output/'last.pt').is_file() else ()
        invoke(module, *base, 'train', *common, '--output', output, *resume)
    if not (output/'TEST_COMPLETE.json').is_file():
        invoke(module, *base, 'test', *common, '--run', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=(*ARMS, 'prepare-tiger'))
    for name in ('root', 'calibration', 'dataset', 'assets', 'cf-runtime'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = wave_config(read(Path('configs/09_kuairec_listwise/l5o.json')), args.calibration, 'h')
    cfg.update(dataset=str(args.dataset), assets=str(args.assets))
    args.root.mkdir(parents=True, exist_ok=True)
    content = args.assets/'content_features.npy'
    if args.arm.startswith('H'):
        path = args.root/f'{args.arm}_config.json'
        if path.exists() and read(path) != cfg:
            raise ValueError('H configuration changed after locking')
        if not path.exists():
            path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        h_worker(args.arm, args.root, path)
    elif args.arm == 'prepare-tiger':
        parent = args.root/'parents/tiger'
        if not (parent/'manifest.json').is_file():
            invoke('scripts.session_wise_ar.run', '--arm', 'AR-T0A', '--l5o-calibration', args.calibration,
                   'prepare', '--dataset', args.dataset, '--content', content, '--output', parent, '--device', 'cpu')
    elif args.arm.startswith('T'):
        tiger_worker(args.arm, args.root, args.calibration, args.dataset, content)
    else:
        cf_worker(args.arm, args.root, args.calibration, args.dataset, content, args.cf_runtime)


if __name__ == '__main__':
    main()
