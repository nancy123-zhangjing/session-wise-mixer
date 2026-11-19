"""Render an archived SASRec calibration receipt without running evaluation."""
import argparse
import json
from pathlib import Path


def render(receipt):
    selection = receipt['selection']
    validation = selection['validation']
    test = receipt['test']['1.0']
    sensitivity = receipt['test']['2.0']
    if test['samples'] != 7176 or sensitivity['samples'] != 7176:
        raise ValueError('Calibration report requires the full L5O test split')
    passed = .25 <= test['imp_hit@5'] <= .45 and .05 <= test['ev_ndcg@5'] <= .12
    if receipt['passed'] is not passed:
        raise ValueError('Calibration gate differs from the recorded result')
    lines = ['# KuaiRec 留五划分 SASRec 校准', '',
             f'全量测试校准：**{"通过" if passed else "未通过"}**。', '',
             f'按验证 EV-NDCG@5 选中第 {selection["epoch"]} 轮，'
             f'历史过滤{"开启" if selection["filter_history"] else "关闭"}。'
             f'共训练 {receipt["completed_epochs"]} 轮。', '',
             '| 指标 | 验证关闭过滤 | 验证开启过滤 | 全量测试 EV≥1.0 | 全量测试 EV≥2.0 |',
             '|---|---:|---:|---:|---:|']
    for key in ('imp_hit@5', 'imp_recall@5', 'imp_ndcg@5', 'ev_hit@5', 'ev_recall@5', 'ev_ndcg@5',
                'legacy_fixed5_ndcg@5', 'samples', 'ev_samples'):
        values = [validation['False']['1.0'][key], validation['True']['1.0'][key], test[key], sensitivity[key]]
        formatted = [str(x) if isinstance(x, int) else f'{x:.6f}' for x in values]
        lines.append(f'| {key} | ' + ' | '.join(formatted) + ' |')
    lines.extend(['', '门槛为测试 Imp Hit@5 处于 0.25–0.45，且 EV-NDCG@5 处于 0.05–0.12。', '',
                  '| 计划记录的论文参照 | Imp Hit@5 | EV-NDCG@5 | 本次减参照 Hit | 本次减参照 NDCG |',
                  '|---|---:|---:|---:|---:|'])
    for name, hit, ndcg in [('OGR', .376, .0878), ('HiGR', .302, .0839)]:
        lines.append(f'| {name} | {hit:.4f} | {ndcg:.4f} | {test["imp_hit@5"]-hit:+.4f} | '
                     f'{test["ev_ndcg@5"]-ndcg:+.4f} |')
    lines.extend(['', f'选定 checkpoint SHA256：`{selection["best_sha256"]}`。',
                  f'协同商品向量 SHA256：`{receipt["items_sha256"]}`。', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    args.output.write_text(render(receipt), encoding='utf-8')


if __name__ == '__main__':
    main()
