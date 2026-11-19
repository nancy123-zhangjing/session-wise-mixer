"""Authorized 200-epoch diagnostic only; no formal configuration is modified."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.code_root.resolve()
    output = args.output.resolve()
    assert output != root and root not in output.parents
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    import torch
    from scripts.session_wise_ar import runtime
    from scripts.session_wise_ar.config import load
    from scripts.session_wise_ar.catalog import encoding_stats
    from scripts.rqcd_mixer.common import read_state, write_json, sha256, preserve_rng
    from scripts.rqcd_mixer.data import load_content
    torch.set_num_threads(4)
    runtime.configure_device('cuda')
    data = root.parent/'goal_20261027_r1/inputs/dataset'
    content_path = root.parent/'goal_20261027_r1/assets/pse_v1/content_features.npy'
    prepared = root/'parents_devicefix/tiger'
    override = dict(initial_accumulator_value=0.1, diagnostic_epochs=200,
                    diagnostic_updates=2200, formal_recommender_authorized=False)
    write_json(output/'PROTOCOL.json', dict(override=override, source=runtime.source_identity(),
               script_sha256=sha256(Path(__file__)), parent_manifest=json.loads((prepared/'manifest.json').read_text()),
               reason='User approved isolated initial-accumulator diagnostic; all other training settings retained'))
    native_optimizer, native_step, native_save = runtime.pretrain_optimizer, runtime.finite_step, runtime.atomic_save
    started = time.monotonic()
    results = {}
    for arm in ('AR-T0', 'AR-T1'):
        cfg = load(arm)
        assert cfg['rq']['epochs'] == 20000 and cfg['rq']['lr'] == 0.4
        count = 0
        early = []

        def optimizer(model, config, steps_per_epoch):
            assert steps_per_epoch == 11 and config['family'] == 'tiger'
            opt, schedule = native_optimizer(model, config, steps_per_epoch)
            # Native Adagrad initializes state sums eagerly. Change both the
            # recorded group setting and these accumulators before any update.
            for group in opt.param_groups:
                group['initial_accumulator_value'] = 0.1
                for parameter in group['params']:
                    opt.state[parameter]['sum'].fill_(0.1)
            return opt, schedule

        def step(loss, models, optimizers, schedulers, clip):
            nonlocal count
            count += 1
            assert count <= 2200
            capture = count in (1, 2, 10, 100, 2200)
            before = [p.detach().clone() for p in models[0].parameters()] if count == 1 else None
            value = native_step(loss, models, optimizers, schedulers, clip)
            if capture:
                with torch.no_grad():
                    model = models[0]
                    row = dict(step=count, batch_loss=float(loss.detach()),
                               base_rms=model.base_books().square().mean((-2,-1)).sqrt().tolist(),
                               effective_rms=model.effective_books().square().mean((-2,-1)).sqrt().tolist())
                    if before is not None:
                        row['first_update_max_abs'] = max(float((p.detach()-old).abs().max())
                                                         for p, old in zip(model.parameters(), before))
                    early.append(row)
                    write_json(output/(arm+'_EARLY.json'), early)
            return value

        def save(path, state):
            state = dict(state, diagnostic_override=override)
            return native_save(path, state)

        runtime.pretrain_optimizer, runtime.finite_step, runtime.atomic_save = optimizer, step, save
        progress = runtime.pretrain(cfg, prepared, data, content_path, output/arm, 'cuda', _stop_after=2200)
        assert count == progress['step'] == 2200
        # _stop_after returns before the final diagnostic boundary: evaluate
        # that same saved state without taking an extra training update.
        state = read_state(output/arm/'last.pt')
        assert state['diagnostic_override'] == override
        assert all(g['initial_accumulator_value'] == 0.1 for g in state['optimizers'][0]['param_groups'])
        tok = runtime.make_tokenizer(cfg).cuda()
        tok.load_state_dict(state['tokenizer'])
        tok.eval()
        content = load_content(content_path, cfg, 'cuda')
        with preserve_rng(), torch.no_grad():
            semantic = tok.indices(content)
            sid = torch.cat((semantic.new_full((1,3),-1), semantic), dim=0)
            stats = encoding_stats(tok, sid)
            maximum = int(torch.unique(semantic, dim=0, return_counts=True)[1].max())
            stats.update(max_collision_bucket=maximum, suffix_capacity_ok=maximum <= cfg['rq']['codes'])
        final = dict(epoch=200, step=2200, encoding=stats,
                     loss={k:v/progress['loss_count'] for k,v in progress['loss_sum'].items()})
        results[arm] = dict(records=progress['records']+[final], early=early,
                            checkpoint_sha256=sha256(output/arm/'last.pt'),
                            parent_sha256=state['parent_sha256'], updates=count)
        write_json(output/(arm+'_COMPLETE.json'), results[arm])
        del tok, content, state
        torch.cuda.empty_cache()
    write_json(output/'COMPLETE.json', dict(diagnostic_complete=True, results=results,
               script_elapsed_seconds=time.monotonic()-started, override=override))


if __name__ == '__main__':
    main()
