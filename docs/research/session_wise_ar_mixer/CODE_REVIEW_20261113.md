# 五组实验代码核对与交付

状态：**代码已实现；只完成本地静态及CPU合成检查，未启动真实实验、GPU短测或资源申请。**

对照依据为已批准的 `FIVE_ARM_DESIGN_20261113.md`，SHA256：`57fee9c6e1ee4f3035e38394ebf0984abd2a236553ebd446ca5aa8a34a8c2513`。该文档保持原字节，未为了迁就代码改写方案。

## 逐组核对

| 组 | 已实现的科学差别 | 训练节奏与初始化 |
|---|---|---|
| CF-M0 | 原CF的B1→B0，无Mixer；原引擎源码未改 | 同教师、prepared初始状态、噪声、有效batch1024、5R＋30×(10T＋1R)，R35/T300 |
| AR-T0 | TIGER式session-wise、无Mixer | 同系列共同RQ/推荐器父状态；RQ20,000目录epoch选最终点，推荐200,000步，Tokenizer全冻结 |
| AR-T1 | T0上从RQ预训练加入层内Mixer | 与T0相同预算/初始化/选点；推荐期连同Mixer一起冻结 |
| AR-D0 | 原DIGER修正版＋session-wise、无Mixer | RQ10,000目录epoch选最低碰撞；每batch双优化器联合更新，120轮上限、patience15 |
| AR-D1 | D0上从预训练加入层内Mixer | 推荐期码本和Mixer均更新；RQ Encoder、内容表冻结；不套CF的R/T块日程 |

同系列两份配置展开后仅 `arm` 和 `mixer_enabled` 不同。AR四组均用内容PCA256、3层真实RQ＋1个消歧码，每session五个条件任务、batch256任务；真实历史＋已生成前缀进入Encoder，当前商品内AR，逐件beam20提交。

## 核对覆盖

| 风险点 | 已做检查与结论 |
|---|---|
| 偷改CF | 原CF源码清单SHA逐项通过；入口只选择B0；拒绝写入共同prepared目录 |
| 上游漂移 | DIGER固定 `ec0a9b51d02272be1935e1a6438cdc30bf9fadc1`，20份公开源文件逐文件SHA锁定；外部文件不修改、不提交为自研代码 |
| 前缀泄漏 | 人工短历史紧凑拼接，当前/未来目标不进入Encoder；改变未来码位不影响较早Decoder输出 |
| 词表/模型 | TIGER独立全局SID/用户词表，6×64注意力投影；DIGER原4码位embedding及输出头 |
| Mixer接口 | 零输出初始化与无Mixer前向一致；非零修正的损失与SID导出等价于把有效中心装入原量化器 |
| 梯度 | 单独semantic＋suffix推荐损失能反传到DIGER码本/Mixer，冻结Encoder无梯度；不是以VQ梯度冒充 |
| TIGER冻结 | 两个TIGER组从预训练选点到推荐末端的Tokenizer状态逐字段不变 |
| 任务计数 | 709,443×5=3,547,215条件任务；每遍13,857更新；TIGER200,000步，DIGER最多1,662,840联合步 |
| 落地 | 固定4码位合法目录树；完整五件、无列表内重复；已生成商品确实加入下一次Encoder；suffix容量溢出报错 |
| 中断恢复 | 四AR组人工数据端到端；epoch边界恢复与连续运行参数/SID精确一致；DIGER验证文件写出后中断可复用同节点继续 |
| 结果汇总 | best/final身份核验；CF原收据格式支持；拒绝错误checkpoint及hybrid冒充CF；缺组明确披露 |

16项CPU单元/集成检查通过。主检查环境为PyTorch2.6.0+cpu、Transformers4.57.1（仅在隔离依赖目录安装）；早期组件检查也在现有Transformers4.53.2通过。合成目录不涉及真实训练数据，小目录k-means的重复中心警告已记录为fixture限制。

## 来源与实现边界

- DIGER模型、RQ预训练/联合量化器直接调用固定版本；训练循环因session-wise、统一数据格式及断点要求由适配层组织，**不是直接执行作者main.py**。核心损失、FrQUD、逐batch联合更新、调度和早停规则保留。预训练norm clip1沿原trainer保留。
- TIGER未核实完整官方训练入口。按方案采用论文结构与预算，HF T5/Adafactor等补全明确记录在 `IMPLEMENTATION.md`；不声称逐位复现原TIGER。
- 预训练阶段Mixer参与学习是两条AR路线共同的设计扩展；因此组内对比估计从预训练起加入Mixer的整体效果。
- AR先用FP32，未擅自换BF16、batch或DIGER更新频率。CF继续原BF16执行配置。
- 尚未做GPU数值/恢复核验、真实目录完整性检查或H100吞吐测量；这些不属于本次已经通过的CPU验收。没有结果或耗时估计可供宣称。

代码任务按批提交；运行入口、参数来源、命令模板及恢复边界见 `IMPLEMENTATION.md`。没有重跑原实验、删除历史资产、推送远端或改动其他任务的工作区文件。
