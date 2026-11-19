"""Summarize the completed diagnostic; does not train or contact the platform."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'COMPLETE.json').read_text(encoding='utf-8'))
verified = json.loads((HERE / 'ARTIFACT_VERIFICATION.json').read_text(encoding='utf-8'))
assert report['all_exact'] and verified['source_verified']
assert hashlib.sha256((HERE / 'COMPLETE.json').read_bytes()).hexdigest() == verified['complete_sha256']
rows = []
for label, variant in report['variants'].items():
    for phase in ('R', 'T'):
        ref = variant['reference']['phases'][phase]
        opt = variant['optimized']['phases'][phase]
        rows.append(dict(variant=label, phase=phase, reference_s=ref['mean_total_s'],
                         optimized_s=opt['mean_total_s'], speedup=ref['mean_total_s']/opt['mean_total_s'],
                         reference_wait_s=ref['mean_total_s']-ref['mean_update_s'],
                         optimized_wait_s=opt['mean_total_s']-opt['mean_update_s'],
                         allocated_gib=opt['peak_allocated_gib'], reserved_gib=opt['peak_reserved_gib'],
                         exact=variant['exact'], oom=variant['oom']))
with (HERE / 'SUMMARY.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
table = []
for label in report['variants']:
    r, t = [row for row in rows if row['variant'] == label]
    table.append(f"| {label} | {r['reference_s']:.3f} → {r['optimized_s']:.3f}（{r['speedup']:.2f}×） | "
                 f"{t['reference_s']:.3f} → {t['optimized_s']:.3f}（{t['speedup']:.2f}×） | "
                 f"{max(r['allocated_gib'],t['allocated_gib']):.2f} | "
                 f"{max(r['reserved_gib'],t['reserved_gib']):.2f} | 全部通过 |")
monitor = list(csv.DictReader((HERE / 'gpu.csv').open(encoding='utf-8')))
max_mib = max(int(row[' memory.used [MiB]'].strip().split()[0]) for row in monitor)
text = f"""# 四个 Decoder 前缀版本：H100 一致性与速度短测

日期：2026-11-12。四个版本均通过本次 GPU 逐位一致性与恢复检查，无 OOM；本次仅短测，未启动正式训练。

## 对照定义与测试范围

- A：已生成商品作为四位 SID 前缀，使用非对称自注意力；B：通过独立的前缀交叉注意力读取已生成商品。H0 无 Mixer，H1 有 Mixer。
- 对照的是**同一新模型版本的原执行路径与优化执行路径**，不是新模型和旧 H0/H1 推荐效果的比较。
- 沿用历史 microbatch=256、有效 batch=1024（累计四次），BF16；模型、损失、优化器、随机设置及样本次序一致。四个配置的完整 training 字典已与历史配置核对，见 `docs/research/onerec_listwise/DECODER_PREFIX_BATCH_ALIGNMENT_20261112.json`。
- 使用真实数据、完整模型和码本资产。保留原训练集规模 {report['train_rows']:,} 及日程以构建优化器/退火配置，固定诊断批次涉及 {report['cached_diagnostic_rows']:,} 个训练样本；并未遍历完整训练集。
- 每个版本、每条执行路径分别做 4 次 R 和 4 次 T 更新，首步预热，后三步计时。下表单位为秒/有效更新，包含本步取数等待；不含一次性缓存准备、验证和 checkpoint 保存。
- 每个版本做相同的 32 样本、beam=32 验证，核对逐样本预测；另核对 checkpoint 恢复后的下一步 R 更新。模型为固定 seed 的新初始化，不是原 R5 权重，不用于判断推荐质量或长期收敛。
- 设备：{report['device']}，可见显存 {report['total_memory_gib']:.2f} GiB；现有远端环境 PyTorch {report['torch']} / CUDA {report['cuda']}。同一张卡顺序测试四个版本。

## 速度与显存

| 版本 | R：原执行 → 优化执行 | T：原执行 → 优化执行 | 优化后训练峰值 allocated/GiB | 优化后训练峰值 reserved/GiB | 一致性 |
|---|---:|---:|---:|---:|---|
{chr(10).join(table)}

allocated 是 PyTorch 活跃张量占用；reserved 包含分配器保留空间，两者不是相加关系。整个短测每 2 秒的 nvidia-smi 采样最高 {max_mib:,} MiB（{max_mib/1024:.2f} GiB），含进程/运行时开销；采样值不保证捕获瞬时峰值。当前设置显存余量明显，不需要为容纳模型改动 microbatch。

四个版本全部通过：初始模型/状态一致；R/T 每步损失、梯度范数与 RNG 一致；R/T 结束的模型、Tokenizer、梯度、优化器、调度器与 RNG 指纹一致；验证逐样本预测一致；恢复后下一步状态一致。该结论限于测过的批次与环境，不等同于已证明完整 R35 训练轨迹一致。

## 主要慢在哪里

1. **原执行路径的数据整理/搬运等待显著。** 本次每个有效更新约等待 0.35～0.48 秒；静态 Tensor 与有序预取启用后，计时区间内取批等待降至约 0.2～1.0 毫秒。预取与计算重叠，不表示数据处理工作消失；一次性缓存准备也不包含在这些数字中。
2. **优化后仍有大量小算子及 CPU 调度开销。** 单独 profiler 观察到 A 的一次 R 更新约 36,090 次 CUDA kernel 启动，B 约 45,305 次；矩阵运算、复制、填充和张量分配均有明显开销。它支持“小算子与调度值得继续优化”的判断，尚未量化其占整轮训练的比例。Profiler 不计入上表速度，避免把剖析开销混入计时。
3. **B 在这次短测中比 A 慢，但占用显存更低。** 与 B 额外执行前缀交叉注意力的结构相符。H1/H0 之间的细小耗时差异有短测噪声，不能据此声称 Mixer 有固定耗时增量或能加速。
4. **验证尚不能确定为整体首要瓶颈。** 32 样本验证约 3.3～4.0 秒，只用于功能/预测一致性；没有做完整验证集吞吐测试。保存 checkpoint 的 CPU 拷贝及落盘各不足一秒，本次不是主要耗时，但不能外推繁忙共享盘的长期表现。

## 下一步与边界

可以保留这批执行优化和 microbatch=256。若继续优化，优先把完整静态缓存准备放在 CPU 资源上提前完成，并进一步定位重复常量构造、复制与小算子调度；都需保持原样本次序、随机消耗与更新边界。编译/算子融合可能改变数值或随机路径，不能作为已经验证的无损优化直接开启。

这只是每阶段 3 步的平均值，执行先后及预热状态可能影响速度；1.30～1.61× 是本次稳态批次吞吐比，不能直接套算整次训练总时长。完整缓存的一次性准备及全量验证尚未计时。

## 文件与资源记录

- 成功作业：`nancy-s09-prefix-short-1112-v2`，平台状态 SUCCEEDED，唯一实例 instance_success。
- 远端结果：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/decoder_prefix_h100_short_20261112_v2/results/`。诊断 checkpoint 标记 preflight_only，保留远端，不作正式模型使用。
- `COMPLETE.json` 是原始结果；`SUMMARY.csv` 为数值表；`SOURCE_MANIFEST.json` 和 `ARTIFACT_VERIFICATION.json` 记录 68 个源码/配置文件、benchmark 与结果摘要的 SHA256 校验。
- 初次 v1 把完整静态数据准备放在 GPU 作业中，准备耗时偏长，已停止；没有产生可用计时结论。v2 改用固定真实诊断批次，完成全部检查。两次作业未并行占用两张卡。
- 最终 CPU/GPU 收尾状态以 `FINAL_ACCEPTANCE.json` 及其引用的平台回执为准。
"""
(HERE / 'REPORT.md').write_text(text, encoding='utf-8')
print(json.dumps({'all_exact': report['all_exact'], 'report': str(HERE / 'REPORT.md'), 'rows': len(rows)}, ensure_ascii=False))
