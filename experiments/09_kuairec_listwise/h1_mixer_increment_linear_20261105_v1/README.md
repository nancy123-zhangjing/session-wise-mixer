# Mixer 新增修正线性收紧版本

状态：仅本地代码版本；未提交或启动新实验。原 G01 目录和运行代码不修改。

## 唯一科学改动

从原 H1 R20/T200 完整 checkpoint 分叉，初始化实际 Mixer 修正为 R20 的修正。
第 k 次新增 T 更新（k=1…200）后：

```text
alpha(k) = 0.1 - 0.09 * (k - 1) / 199
M_effective(k) = M_effective(k-1) + alpha(k) * (M_raw(k) - M_raw(k-1))
C(k) = B(k) + M_effective(k)
```

T201 使用 0.1，T400 使用 0.01。这里的 M_raw 包含原 residual_scale，依赖
当前基础码本和 Mixer 参数；接受的是两次真实 T optimizer step 之间的修正变化。
已经接受的历史增量不会随 alpha 减小而回撤。不是学习率衰减，不是 EMA，
不对相对 R20 的累计偏移重新乘一个不断变小的系数。

基础码本仍正常更新。本版本不保证完整有效码本的位移单调减小，不给绝对位移
设置上限，也不保证 SID churn 按 alpha 缩小；原始 Mixer 增量可能变大。
0.1→0.01 是预先指定的候选方案，不是已验证的最优参数。

## 执行和梯度语义

- 每次 T 更新的所有 microbatch 和辅助损失使用同一个已提交码本、同一个 alpha。
- 沿用局部梯度 `dC = dB + alpha * dM_raw`；不反传历史更新的计算图。
  用零值 autograd 项保留该梯度，使只读求值或 train/eval 切换不会改变已提交码本。
- optimizer.step 后、保存或诊断前提交一次增量；不会在每个 forward、
  每个 microbatch、R 更新、码本刷新或评估时重复累积。
- 为保证 R20 起点恒等，实际前向写作
  `C_R20 + (B-B_R20) + accepted_delta`，初始化 accepted_delta=0。
- 原始增量测量固定使用零 dropout 的 training 路径、FP32、不采样；
  测量后恢复各子模块原来的模式。RNG 不由门控推进。
- checkpoint 保存 R20 锚点、上次原始修正、累计接受增量、最近原始增量、
  200 步时钟和固定调度；校验其哈希、父绑定、优化器时钟及模型对应关系。
  原 G01 checkpoint 不可作为新版本的续训文件。

## 保持的设置

复用 G01 单卡恢复实现与原始模型代码。仍从原 H1 R20/T200 出发，执行
五次 40T+3R，终点 R35/T400；保留阶段编号、样本顺序、seed、RNG 恢复、
Adam 动量、Tokenizer 学习率、推荐器调度、噪声时钟、损失、梯度裁剪、
有效 batch 1024=256×4。

诊断仍用同一 8192 子集，R23/26/29/32/35 全验证，R20 也纳入 best，
平局取早，随后 best/R35 测试。新增日志仅记录接受比例、实际原始/接受增量
及位移；评估缓存额外绑定累计门控状态，因为它不能仅由当前模型参数重建。

## 文件与运行入口

- `code/scripts/listwise_diffgrm/recovery_increment.py`：新门控和恢复控制器。
- `code/scripts/listwise_diffgrm/recovery_increment_run.py`：沿用 G01 的训练、验证、选模、测试流程。
- `job.py`：仅显式调用时运行；输出为当前新根目录下 `runs/IncrementLinear`。
- `INPUTS.json`、`R20_REFERENCE.json`、`validation_subset_8192.json`：复用原输入。
- `SOURCE_MANIFEST.json`：原快照复用文件哈希与新版本文件哈希。

必须从本版本 `code/` 目录使用本版本模块，不能从仓库根目录调用同名包：

```text
python -B -m scripts.listwise_diffgrm.recovery_increment_run --help
python -B -m unittest discover -s tests -p test_listwise_increment.py -v
```

未来正式执行时，将本目录放在一个新的远端根目录，并显式调用：

```text
python job.py --root <新的绝对根目录> --python <远端环境python>
```

此命令会运行训练和测试，不是预览命令；本次未执行。没有复制提交回执、
自动提交脚本或旧训练输出。新代码的本地测试不代替真实父文件/数据、
H100 数值与平台预检，不构成新实验完成验收。

## 比较边界

对同一 M_raw 序列，常数 alpha 的增量求和与固定锚点公式数学等价。
本版本浮点累计和不同调度会改变训练轨迹，不承诺与 G01 逐位等价。
线性调度同时减少了平均接受比例，单条结果不能区分“逐步收紧”与
“总体接受更少”的作用；不自动追加对照实验。
