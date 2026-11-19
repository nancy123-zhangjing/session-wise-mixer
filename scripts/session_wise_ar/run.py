"""Explicit local/remote entrypoint. Default plan is read-only, never submits jobs."""
import argparse
import json
from .config import ARMS, load, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--l5o-calibration', help='Completed SASRec calibration receipt for the L5O wave')
    parser.add_argument('--resolved-config', help='Protocol v2: a complete per-seed config written by l5o_v2')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('plan')
    for name in ('prepare', 'pretrain', 'train', 'test'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', required=True)
        p.add_argument('--content', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda')
        if name != 'prepare': p.add_argument('--prepared', required=True)
        if name in ('train', 'test'): p.add_argument('--pretrain', required=True)
        if name in ('pretrain', 'train'): p.add_argument('--resume', action='store_true')
        if name == 'test': p.add_argument('--training', required=True)
    args = parser.parse_args()
    cfg = load(args.arm)
    if args.resolved_config:
        if args.l5o_calibration:
            parser.error('--resolved-config already contains the calibrated L5O settings')
        resolved = json.loads(open(args.resolved_config, encoding='utf-8').read())
        if resolved.get('arm') != args.arm or 'protocol_v2' not in resolved:
            parser.error('Resolved config must be a protocol v2 config for this arm')
        cfg = resolved
    elif args.l5o_calibration:
        if args.arm not in ('AR-T0A', 'AR-T1A'):
            parser.error('The L5O wave uses only the TIGER AdamW arms')
        from scripts.kuairec_data.l5o_launch import wave_config
        cfg = wave_config(cfg, args.l5o_calibration, 'tiger')
    if args.action == 'plan':
        print(json.dumps(plan(cfg), ensure_ascii=False, indent=2)); return
    # Delay heavy imports and all stateful work until an explicit stage is given.
    from . import runtime
    common = dict(cfg=cfg, dataset=args.dataset, content_path=args.content, output=args.output, device=args.device)
    if args.action == 'prepare': runtime.prepare(**common)
    elif args.action == 'pretrain': runtime.pretrain(**common, prepared=args.prepared, resume=args.resume)
    elif args.action == 'train': runtime.train(**common, prepared=args.prepared, pretrain_path=args.pretrain, resume=args.resume)
    else: runtime.test(**common, prepared=args.prepared, pretrain_path=args.pretrain, training=args.training)


if __name__ == '__main__':
    main()
