"""CPU analysis of the decision-rule run: behaviour profile and paired user bootstrap.

Profile definitions are imported unchanged from the 2026-11-17 deep profile
(experiments/09_kuairec_listwise/l5o_deep_profile_20261117/analyze.py): distinct items,
share of recommendations in the target day's +/-1-day top 100, per-slot hit rates and
item-class shares. Paired intervals: per-user EV-NDCG@5 differences over users with at
least one effective view, 2,000 bootstrap draws, seed 2024.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from scripts.kuairec_data.l5o_metrics import load_predictions

PROFILE = Path(__file__).resolve().parents[2]/'experiments/09_kuairec_listwise/l5o_deep_profile_20261117/analyze.py'


def profile_module():
    spec = importlib.util.spec_from_file_location('l5o_deep_profile', PROFILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True,
                   help='one or more run directories; a rule present in several must have identical predictions')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-label', help='screening: also report label minus this label (same R, same rule)')
    a = p.parse_args()
    m = profile_module()
    S = m.train_stats(a.dataset/'train.parquet')
    E = {s: m.load_eval(a.dataset/f'{s}.parquet') for s in ('validation', 'test')}
    # Same day origin as the deep profile (train and both evaluation splits).
    t0 = min(int(S['ts'].min()), *(int(e['tts'][e['tts'] > 0].min()) for e in E.values()))
    D = m.Days(S, t0)
    e = E['validation']
    nodes, per_user, has, seen = {}, {}, None, {}
    summaries = [path for run in a.runs for path in sorted(run.glob('*/R*/*/summary.json'))]
    for summary_path in summaries:
        rule_dir = summary_path.parent
        arm, epoch, rule = rule_dir.parent.parent.name, rule_dir.parent.name, rule_dir.name
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        pred = load_predictions(rule_dir/'predictions.jsonl.gz', e['ids'])
        key = (arm, epoch, rule)
        if key in seen:
            if not np.array_equal(seen[key], pred):
                raise ValueError(f'{key} differs between run directories')
            continue
        seen[key] = pred
        C = m.class_contrib(pred, e, S)
        prof = m.model_profile(pred, e, S, D, C, {})
        if abs(prof['ev_ndcg5'] - summary['metrics']['ev_ndcg@5']) > 1e-12:
            raise ValueError(f'Profile and run summary disagree for {arm} {epoch} {rule}')
        has = C['has'] if has is None else has
        per_user[(arm, epoch, rule)] = C['contrib'].sum(1)
        nodes.setdefault(arm, {}).setdefault(epoch, {})[rule] = {
            **{k: prof[k] for k in ('ev_ndcg5', 'imp_hit5', 'distinct_items', 'top10_share',
                                    'rec_in_day_window_top100', 'rec_in_global_top100', 'slot_hit_imp',
                                    'slot_hit_ev', 'rec_share')},
            'ev_hit5': summary['metrics']['ev_hit@5'], 'seconds': summary['seconds'],
            **({'chosen_index_counts': summary['chosen_index_counts']} if 'chosen_index_counts' in summary else {})}
    rng = np.random.default_rng(2024)
    versus_a, mixer = {}, {}
    for (arm, epoch, rule), values in sorted(per_user.items()):
        if rule != 'A' and (arm, epoch, 'A') in per_user:
            diff = values - per_user[(arm, epoch, 'A')]
            low, high = m.boot(diff, has, rng)
            versus_a.setdefault(arm, {}).setdefault(epoch, {})[rule] = {
                'delta': float(diff[has].mean()), 'ci95': [float(low), float(high)],
                'relative': float(diff[has].mean() / max(values[has].mean() - diff[has].mean(), 1e-12))}
        if arm == 'CF-M1' and ('CF-M0', epoch, rule) in per_user:
            diff = values - per_user[('CF-M0', epoch, rule)]
            low, high = m.boot(diff, has, rng)
            mixer.setdefault(epoch, {})[rule] = {'delta': float(diff[has].mean()), 'ci95': [float(low), float(high)]}
    versus_reference = {}
    if a.reference_label:
        for (label, epoch, rule), values in sorted(per_user.items()):
            if label != a.reference_label and (a.reference_label, epoch, rule) in per_user:
                diff = values - per_user[(a.reference_label, epoch, rule)]
                low, high = m.boot(diff, has, rng)
                versus_reference.setdefault(label, {}).setdefault(epoch, {})[rule] = {
                    'delta': float(diff[has].mean()), 'ci95': [float(low), float(high)]}
    result = {'split': 'validation', 'users': int(len(has)), 'ev_users': int(has.sum()), 'draws': m.DRAWS,
              'bootstrap_seed': 2024, 'profile_source': str(PROFILE), 'nodes': nodes,
              'rule_minus_A': versus_a, 'M1_minus_M0': mixer,
              'label_minus_reference': versus_reference, 'reference_label': a.reference_label}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for arm, epochs in nodes.items():
        for epoch, rules in epochs.items():
            for rule, row in rules.items():
                d = versus_a.get(arm, {}).get(epoch, {}).get(rule)
                print(f"{arm} {epoch} {rule:6s} ev={row['ev_ndcg5']:.4f} distinct={row['distinct_items']:5d} "
                      f"day100={row['rec_in_day_window_top100']:.3f} "
                      + (f"dA={d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]" if d else ''))


if __name__ == '__main__':
    main()
