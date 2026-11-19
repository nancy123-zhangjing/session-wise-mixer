# 本轮运行记录与范围

核对日：2026-10-23。工作目录：`C:/Users/nancy/Documents/experiment1`。

## 实际使用的环境

既有解释器 `.runtime/continuous_env/Scripts/python.exe`：Python3.12.14、Torch2.6.0+cpu、NumPy2.2.6，CPU线程1。没有安装包、升级环境、读取大模型权重、构建optimizer或执行optimizer.step；未访问创智平台、未申请GPU。作者代码仅阅读，没有执行其训练环境。

## 实际命令

```powershell
& '.\.runtime\continuous_env\Scripts\python.exe' -B 'docs/research/session_wise_codebook/evidence/workspace_audit.py'
& '.\.runtime\continuous_env\Scripts\python.exe' -B 'docs/research/session_wise_codebook/prototypes/verify_cpu.py'
& '.\.runtime\continuous_env\Scripts\python.exe' -B 'docs/research/session_wise_codebook/prototypes/benchmark_cpu.py'
& '.\.runtime\continuous_env\Scripts\python.exe' -B 'docs/research/session_wise_codebook/evidence/workspace_audit.py' --final
& '.\.runtime\continuous_env\Scripts\python.exe' -B 'docs/research/session_wise_codebook/evidence/complete_audit.py'
```

起始状态收据只创建一次；后续如重新验收，不要覆盖INITIAL_STATE。`-B`防止在生产源码目录生成pycache。

## 检查与修复经过

首次合成检查7/9，保留`CPU_VERIFICATION_INITIAL_DIAGNOSTIC.json`：

- 残差权重零和的FP32误差约2.2e-8，原2e-8阈值过紧，改1e-7；并非修改数学目标。
- 当前Torch的stable argsort要求`dim`以关键字传入，已修正。
- 审查另发现leading PAD query全屏蔽风险，改为按query合并因果与有效key mask；只向无可见key的PAD query开放dummy并抹输出，随后验证输入/参数梯度有限。

增加与axial同两次投影的no_context参照，并锁定新概率几何FP32；增加CPU BF16有限反向及跨候选共同初态检查后，最终仍为**9/9类全部通过**，细项和实际值见`CPU_VERIFICATION.json`。诊断激活query头是手动设置非零权重以验证功能，不是训练所得参数。

基准每结构warmup3、连续15次CPU前向，最终版本单独顺序执行。早期在检查和计时并行、以及精度策略修改前的数字不用于最终报告；最终原始样本、四分位、shape及平台信息在`CPU_BENCHMARK.json`。没有根据推荐指标筛选计时结果。

## 不能由本次验证推出的结论

合成Bridge只接了真实05 Mixer和ST，推荐头、target CE、目录穷举是toy实现。未跑完整Pipeline、真实Gumbel/target-ST/辅助损失、生产beam、CUDA/OCN或真实数据无标签generate；未得到HR/NDCG、GPU峰值显存或线上延迟。

原始文献与作者实现阅读证据分别在`sources_session.json`、`sources_grid_loop.json`。PIT/TIGER只做论文分析，不虚构作者代码验证。当前作者HEAD不等于论文实验commit，也不等于项目冻结commit。

本轮产物均在`docs/research/session_wise_codebook/`；`FINAL_STATE.json`比较根仓库、两个独立引擎HEAD/已有tracked状态及25份保护文件。最终内容清单与完成条件见`COMPLETION_AUDIT.json`。
