# 只读诊断参数

这些 JSON 只用于运行后的分析，不参与 DiffGRM YAML 训练配置层叠，不改变历史 SID、码本或模型。

`diger_sid_consistency_seed2024.json` 固定了阶段 05 的五份最佳码本快照及其离线初始化。真实噪声状态读取对应 metadata；Cross v2 额外检查 0.2 噪声，明确为同一码本的反事实采样。全部商品均参与，不抽取子目录中的“最新”文件。

在具有正式输入资产的 Linux 项目根目录中执行：

```bash
source runtime_env.sh
python -B tests/test_diger_sid_consistency.py
python -B scripts/analysis/analyze_diger_sid_consistency.py \
  --project-root "$PWD" \
  --config configs/diagnostics/diger_sid_consistency_seed2024.json \
  --output-dir outputs/diagnostics/diger_sid_consistency/new_unique_run
```

输出目录必须不存在；需要重跑时换一个新的名称，脚本拒绝覆盖。脚本只使用 CPU、NumPy、PyTorch，不导入完整训练 pipeline。25 项测试会直接核对当前引擎的距离和采样实现；若采样实现修改了，需要同时更新诊断并注明协议变化。

输出 `summary.json`、`config.json` 和 `checksums.json`。确认正确后，将轻量结果复制到 `experiments/05_crosspse_joint/diagnostics/` 下新的诊断目录，附上本次执行版本和分析，不覆盖历史 `evidence/`。
