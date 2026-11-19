# 06第一阶段验收：本地数据准备与CPU检查

日期：2026-10-15。结论：**本轮第一阶段完成**。两套数据协议、原始身份核对、CPU测试和双目录确定性均有实际执行证据。未实现或训练推荐模型，未使用GPU，未同步或在线检查服务器。

## 实际结果

| 验收项 | 实际结果 | 证据 |
|---|---|---|
| 原始数据 | 22,363用户、12,101真实商品、198,502交互 | [全量验证收据](acceptance/data_verification.json) |
| 完整协议 | 131,413个训练目标，至少1件历史，输入最近50件 | 同上，逐条对照原始交互位置 |
| 旧协议 | 105,668个训练目标，保留至少2件历史、目标位置上限50 | 同上，直接执行原旧方法作为参照 |
| 留出数据 | 每套验证22,363条、测试22,363条；两套留出文件哈希相同 | [数据manifest](acceptance/data_manifest.json) |
| CPU测试 | **37项通过，0失败、0错误、0跳过** | [原始测试报告](acceptance/unit_tests.json) |
| 两次独立生成 | 各核查326,533条记录，共653,066条；11项内容文件SHA256全部相同 | [比较结果](acceptance/data_verification.json) |
| 既有工作保全 | 开始时909项文件中908项字节不变；唯一允许变动为交接摘要 | [保全收据](acceptance/preservation.json) |
| Git/引擎 | 根HEAD不变、暂存区仍为空；两独立引擎HEAD和状态均不变 | 同上 |

共同数据manifest的SHA256：

```text
6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa
```

两次生成分别位于：

- [data_20261015_a](../../outputs/stage06_continuous/data_20261015_a/manifest.json)
- [data_20261015_b](../../outputs/stage06_continuous/data_20261015_b/manifest.json)

每份输出都单独读取备份、写入原始输入副本并完整生成；不是把第一份结果复制到第二个目录。内容哈希比较覆盖两份输入、配置、候选表、统计和六个划分。运行时间、绝对路径和Git状态等provenance因独立进程而不同，不声称全部运行日志逐字节一致。

## 数据身份与协议核查

仅从本地 `C:/Users/nancy/Downloads/experiment1_migration_20260923_retry/experiment1.tar.gz` 读取两份allowlist普通文件；不使用`extractall`，不下载原数据，不改编号。大小与SHA256逐项匹配原 [01资产锁](../01_ar_baseline_3x256/reproducibility/assets.lock.json)；本轮不重新计算整个迁移包的整体哈希。

两份原始输入哈希、所有产物大小/行数/哈希均保存在 [manifest副本](acceptance/data_manifest.json)，协议全文在 [resolved_protocol.json](acceptance/resolved_protocol.json)。

验证器独立读取原始JSON：完整协议逐条按合法交互位置推导预期记录；旧协议从当前 `external/DiffGRM/genrec/dataset.py` 抽取并执行 `_sliding_train`，以无第三方依赖的`Dataset.from_dict`替身取回真实旧分割结果。还核查候选ID、原始映射、历史上下文、记录顺序、行数、留出相同和统计。它不会仅凭生成器自报的哈希宣布通过。

测试包含重算哈希后的错误目标、历史包含未来交互、遗漏合法目标、重排记录等反例；同时覆盖重复商品、短序列、长序列、PAD、映射双射、计数、输入身份、禁止覆盖和选择性归档读取。CLI在`-S`关闭site-packages的子进程中也通过，表明本轮不依赖Torch或其他第三方训练包。

序列长度中位数6、95分位数21、最大204。训练交互出现12,068件商品；验证/测试分别有51/138条目标未在训练交互出现。保留全部留出样本与原12,101件候选，没有为优化结果过滤这些样本。本轮没有模型指标。

## 实际执行命令

以下命令均在项目根目录执行。准备命令的输出目录现已存在；再次生成时必须换新目录名，工具会拒绝覆盖。

```powershell
python -B tests/test_continuous_data.py --report .runtime/continuous_stage1_20261015/unit_final.json

python -B scripts/prepare/prepare_continuous_data.py --archive 'C:\Users\nancy\Downloads\experiment1_migration_20260923_retry\experiment1.tar.gz' --output outputs/stage06_continuous/data_20261015_a

python -B scripts/prepare/prepare_continuous_data.py --archive 'C:\Users\nancy\Downloads\experiment1_migration_20260923_retry\experiment1.tar.gz' --output outputs/stage06_continuous/data_20261015_b

python -B scripts/checks/verify_continuous_data.py --output outputs/stage06_continuous/data_20261015_a --peer outputs/stage06_continuous/data_20261015_b
```

测试和准备进程的退出码均为0；最终验证子进程退出码与实际参数见 [verification_command.json](acceptance/verification_command.json)。准备进程的命令、环境、来源、时间和producer哈希见 [A来源](acceptance/provenance_a.json)、[B来源](acceptance/provenance_b.json)。

需要重新核查时，可直接运行最后一条只读验证命令。单元测试可用 `python -B tests/test_continuous_data.py` 重跑；不带`--report`即可避免覆盖既有验收报告，测试仅在项目`tmp/continuous_data_tests/`下创建并清理独立临时夹具。

## 代码与保全

根Git仍为 `78d446d365a589cfb476424e732e673795e1e79b`。本次代码尚未提交，不能将该HEAD冒称为包含新代码的执行版本。源文件字节哈希保存在 [source_hashes.json](acceptance/source_hashes.json)，并与测试报告、准备/验证收据逐项核对。

新建：

- [准备脚本](../../scripts/prepare/prepare_continuous_data.py)
- [独立验证器](../../scripts/checks/verify_continuous_data.py)
- [协议配置](../../configs/06_continuous_baseline/data_protocols.json)
- [CPU测试](../../tests/test_continuous_data.py)
- 本目录README、PLAN、ACCEPTANCE和`acceptance/`轻量证据。

按要求更新 [PROJECT_HANDOFF.md](../../docs/guides/PROJECT_HANDOFF.md)。先前README和指南索引的未提交改动保持原字节；无关草稿未读取、移动或提交。旧实验、历史锁和引擎源码未修改，未执行Git提交、暂存或服务器同步。

## 剩余边界

- 本地环境实际为Python3.14.6、Windows11、标准库CPU。启动器会输出`Failed to find real location of C:\Python314\python.exe`，随后脚本正常运行；已保存退出码和实际环境，未改动Python安装。
- 数据本体、`.runtime/`记录和原始运行provenance位于Git忽略的输出区；本目录保留轻量副本。仅克隆主仓库不会自动获得数据和两个独立引擎，需要按锁定输入重新准备。
- 远端当前数据、环境、代码和GPU均未验收。本地数据完成不等同于远端预检通过。
- 连续推荐模型、Loss、HR/NDCG、速度比较、GPU smoke、搜索、多seed训练与Decoder适配编码均尚未开始。下一步只在新的执行请求范围内推进模型实现和CPU检查。
