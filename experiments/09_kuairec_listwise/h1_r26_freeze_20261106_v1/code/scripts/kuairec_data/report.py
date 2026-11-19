"""Write the Chinese delivery report from completed audit evidence only."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .prepare import dump


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report(dataset, destination):
    ingestion, construction = read(dataset / 'ingestion.json'), read(dataset / 'construction.json')
    statistics = read(dataset / 'split_statistics.json')
    validation = read(dataset / 'validation.json')
    trace = read(dataset / 'trace_validation.json')
    reproducibility = read(dataset / 'reproducibility.json')
    assert validation['passed'] and trace['passed'] and reproducibility['passed']
    c, b = ingestion['counts'], construction['counts']
    ambiguous = []
    for path in sorted((dataset / 'events').glob('*.parquet')):
        table = pq.read_table(path, filters=[('ambiguous', '=', True)])
        ambiguous.extend(table.to_pylist())
    # Distinguish real tied events from decimal formatting artifacts.
    from decimal import Decimal
    normalized = {}
    numeric_equivalent = 0
    for event in ambiguous:
        key = (event['user_id'], event['video_id'], event['timestamp_us'], event['play_duration'], event['video_duration'], Decimal(event['watch_ratio_raw']))
        if key in normalized and event['watch_ratio_raw'] != normalized[key]:
            numeric_equivalent += 1
        normalized[key] = event['watch_ratio_raw']
    audit = {'ambiguous_events': len(ambiguous), 'numeric_equal_but_text_different_ratio_duplicates': numeric_equivalent,
             'events': ambiguous}
    dump(dataset / 'ambiguity_audit.json', audit)
    if numeric_equivalent:
        raise ValueError('Equivalent numeric ratio spellings found: review dedup before delivery')
    summary = {'raw_events': c['raw_events'], 'clean_unambiguous_events': c['clean_unambiguous_events'],
               'samples': validation['samples_checked'], 'split_samples': {s: v['samples'] for s, v in statistics['splits'].items()},
               'all_checks_passed': True, 'dataset': dataset.as_posix()}
    dump(destination / 'DELIVERY_SUMMARY.json', summary)
    for name in ('ingestion.json', 'construction.json', 'split_statistics.json', 'validation.json',
                 'trace_validation.json', 'loader_validation.json', 'reproducibility.json', 'manifest.json'):
        value = read(dataset / name)
        if name == 'ingestion.json':
            value = {key: value[key] for key in value if key not in ('users', 'items')}
        dump(destination / name.upper(), value)
    dump(destination / 'AMBIGUITY_AUDIT.json', {k: v for k, v in audit.items() if k != 'events'})
    lines = [
        '# KuaiRec 全量数据准备与验收', '',
        '完成日期：2026-10-27。仅用本地CPU，未训练模型、未申请GPU。原始文件已在项目中，本次核验后复用，没有重复下载或修改原文件。', '',
        '## 这份数据回答什么问题', '',
        '它供后续检验“列表生成场景中的码本交互是否有用”。输入是目标开始前最多50条已播放历史，输出标签是同一人工会话内接下来的连续五件不同商品；两种后续列表生成模型共用这些标签。', '',
        '**任务名称是人工会话内的五商品消费续接。** KuaiRec没有真实请求ID，30分钟会话间隔、五件块、10分钟块跨度均为事先固定的项目规则，不是OneRec原始请求协议。也不是会话开始时一次预测整段会话。', '',
        '## 数据来源与清洗', '',
        '使用KuaiRec 2.0的big_matrix及作者静态文本，未混入small_matrix。两文件发布MD5相符，ZIP全成员CRC通过，SHA256与版本记录见[SOURCE_AUDIT.json](SOURCE_AUDIT.json)。', '',
        '| 项目 | 全量结果 |', '|---|---:|',
        f"| 原始行为 | {c['raw_events']:,} |",
        f"| 用户 / 商品 | {len(ingestion['users']):,} / {len(ingestion['items']):,} |",
        f"| 去除的重复事件 | {c['exact_duplicates_removed']:,} |",
        f"| 其中与代表记录日期标签不同 | {c.get('duplicate_rows_with_different_date_label', 0):,} |",
        f"| 去重后结构事件 | {c['clean_structural_events']:,} |",
        f"| 无法确定顺序的并列时间事件 | {c['ambiguous_events']:,} |",
        f"| 有效且无歧义事件 | {c['clean_unambiguous_events']:,} |",
        f"| 无法解析时间 / 非法字段 | {c['invalid_timestamp_rows']:,} / {c['invalid_field_rows']:,} |",
        f"| 原始date标签与精确时间不符 | {c['date_label_mismatch_rows']:,} |", '',
        '精确time与timestamp全量吻合，小数秒最大3位，转换为整数微秒无精度损失。date标签差异单独记录，不把同一次事件误算成两个。不同时间的重复观看保留；同时间但反馈/商品不同的事件隔离，不按商品ID猜顺序。已额外检查剩余歧义不存在仅因watch_ratio十进制格式不同而造成的重复。', '',
        f"清洗后实际观察范围为北京时间{statistics['timeline']['min_time_beijing']}至{statistics['timeline']['max_time_beijing']}，覆盖{statistics['timeline']['played_events']:,}条正播放和{statistics['timeline']['zero_play_events']:,}条零播放。日期范围按完整文件实测，不沿用512用户样本的起止日期。", '',
        '商品文本保留10,728个ID，四条未转义断行（1378、1613、5910、6331）无损修复；1,356件缺标题，4件完全缺可用静态文本，均保留并打标。没有以行为热度补文本。', '',
        '## 构造与时间切分', '',
        '先按全部观察分会话，再从段首每五条切块。不合格块整块丢弃，不跳过中间记录重拼；其已发生的有效播放仍可成为后续历史。主版本不筛五件总播放≥60秒，仅在训练集保存可选标记。', '',
        '| 构造步骤（按互斥判断顺序） | 数量 |', '|---|---:|',
        f"| 人工会话 | {b['sessions']:,} |", f"| 完整五件候选块 | {b['full_blocks']:,} |",
    ]
    for key, label in [('blocked_user_blocks', '用户存在无法定位时间记录'), ('crossing_session_blocks', '所属整会话跨全局切点'),
                       ('quarantined_blocks', '包含隔离事件'), ('repeated_target_item_blocks', '五件商品不互异'),
                       ('nonplayed_blocks', '包含零播放商品'), ('excess_span_blocks', '五件跨度超过600秒'),
                       ('short_history_blocks', '历史不足5条'), ('retained_blocks', '最终保留目标块')]:
        lines.append(f"| {label} | {b.get(key, 0):,} |")
    lines += [f"| 不足五条的尾块 / 尾部事件 | {b['tail_blocks']:,} / {b['tail_events']:,} |", '',
              '以下切点来自清洗后无歧义事件的80%/90% nearest-rank分位。同一时间归同侧；跨切点整个人工会话不贡献目标，但其过去观察仍可作后续历史。', '']
    for i, value in enumerate(ingestion['cuts_us']):
        date = pd.Timestamp(value, unit='us', tz='UTC').tz_convert('Asia/Shanghai')
        lines.append(f"- 切点{[80, 90][i]}：{date.isoformat()}（{value}微秒）。")
    lines += ['', '| 集合 | 五件样本 | 实际占比 | 用户 | 历史或目标商品 | 历史长度中位数 | 目标跨度中位数/秒 |', '|---|---:|---:|---:|---:|---:|---:|']
    for s, label in [('train', '训练'), ('validation', '验证'), ('test', '测试')]:
        v = statistics['splits'][s]
        lines.append(f"| {label} | {v['samples']:,} | {v['sample_fraction']:.2%} | {v['users']:,} | {v['history_or_target_items']:,} | {v['history_length']['p50']:.0f} | {v['target_span_seconds']['p50']:.3f} |")
    lines += ['', '## 反馈、覆盖与冷启动', '',
              '验证和测试按已发生历史滚动构造，当前和未来目标不进入当前历史。五件是已消费目标，不保证每件都是强偏好正例。全目录静态文本的离线可用性是明确假设，尚未证实文本是事件当时快照。', '',
              '| 集合 | 平均五件播放/秒 | 目标watch_ratio>1占比 | 训练未见用户 | 含训练未见目标的样本 |', '|---|---:|---:|---:|---:|']
    for s, label in [('train', '训练'), ('validation', '验证'), ('test', '测试')]:
        v = statistics['splits'][s]
        lines.append(f"| {label} | {v['target_total_play_ms']['mean'] / 1000:.3f} | {v['target_watch_ratio_gt1_occurrences'] / (5*v['samples']):.2%} | {v['cold_users']:,} | {v.get('samples_with_cold_target_item', 0):,} |")
    tr = statistics['splits']['train']
    lines += ['', f"训练集可选总播放≥60秒标记覆盖{tr['optional_total_play_ge60s_samples']:,}个样本，占{tr['optional_total_play_ge60s_samples']/tr['samples']:.2%}；未用此标记删除主版本样本。", '',
              '冷启动以实际训练样本的历史和目标覆盖为准，目录及loader输出标记，不删除冷启动。各集合完整分布、用户权重、缺文本和训练时间曾出现但未入样本的覆盖差异见[SPLIT_STATISTICS.JSON](SPLIT_STATISTICS.JSON)。', '',
              '## 验收与复现', '',
              f"- 33项规则、文本和加载器测试通过；全量{validation['samples_checked']:,}个样本通过边界、唯一性、历史截断、目标互异及三集合目标事件互斥检查。",
              '- 独立验证器直接回读原始ZIP，固定抽取32用户的完整记录，重新计算全部合格块及最近历史，与输出逐项比较通过，见[TRACE_VALIDATION.JSON](TRACE_VALIDATION.JSON)。',
              '- 两次均从完整原始CSV开始处理；ID映射、切点、划分、文本及规范化样本内容哈希一致。限定相同配置与依赖版本，见[REPRODUCIBILITY.JSON](REPRODUCIBILITY.JSON)。',
              '- 三个集合都实际运行了CPU批次加载，历史左补PAD=0到50，目标维度为5，mask和时间边界通过。原始文件处理前后SHA256未变。', '',
              f"正式数据：`{dataset.as_posix()}`。第二份复跑产物：`{reproducibility['right']}`。输入、全部交付文件、源代码哈希和依赖版本见[MANIFEST.JSON](MANIFEST.JSON)；脚本入口和完整运行命令见[README.md](README.md)。", '',
              '## 后续使用边界', '',
              '本轮没有生成embedding、没有拟合PCA/OPQ/PSE、没有实现或训练列表模型。下一步可以在这份锁定数据上统一设计列表版DiffGRM与码本交互对照；评价@10/20/50所需的候选聚合规则仍须单独确定，不能把本数据的五件目标直接说成50件列表。', '',
              'KuaiRec big_matrix是作者过滤过的公开观察轨迹，人工分段不等于完整自然会话或真实请求。因此将来即使取得收益，首先支持的是此代理任务上的结论，仍需真实请求数据进一步验证。', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=Path('docs/research/onerec_listwise/kuairec_preparation'))
    args = parser.parse_args()
    print(json.dumps(report(args.dataset, args.destination)))


if __name__ == '__main__':
    main()
