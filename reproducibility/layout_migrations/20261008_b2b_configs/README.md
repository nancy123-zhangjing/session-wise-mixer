# B2b：按研究阶段归类配置

日期：2026-10-08。迁移基线为 B2a 提交 `07d6a83`。本批尚未提交、未推送；没有删除旧证据或启动训练。

## 1. 变了什么

34 份 YAML 从 `configs/` 根目录移入以下六个目录，文件名和内容保持不变：

| 目录 | 数量 | 内容 |
|---|---:|---|
| `shared/` | 5 | AR/Diff 共用参数、Diff v6 协议、通用 smoke/profile |
| `01_ar_baseline_3x256/` | 4 | AR RQ 基线及早期短测 |
| `02_ar_capacity_ablation/` | 3 | AR PSE、两组不等容量对照 |
| `03_diffgrm_tokenizer_comparison/` | 6 | Diff 三种 Tokenizer 及早期训练设置 |
| `04_crosspse_offline/` | 6 | 离线 Refined/Cross-PSE 的历史与最终版本 |
| `05_crosspse_joint/` | 10 | 联合训练、冻结对照及专用 smoke |

使用说明见 [配置索引](../../../configs/README.md)。共享设置只保留一份，不复制到不同阶段；旧根目录不留第二套 YAML，以免修改错文件。

29 份调用脚本同步了配置地址：10 份 Python（预检查/Attention）、17 份 Bash、2 份 PowerShell。脚本本身未移动，shell 启动命令不变；手动传旧 `--config` 路径时需使用新地址。

## 2. 没变什么

- 所有 YAML 原字节、run_id、seed、训练参数、数据与 checkpoint 路径不变。
- 29 个调用方只改配置地址，分支、命令参数、工作目录、配置加载和覆盖顺序不变。
- YAML 内的 `../../data`、`../../outputs` 继续基于运行时 `external/DiffGRM` 的 cwd，不因配置搬家而改写。
- 五阶段科学证据、结果、配方、原环境和源码快照不变；545 个实验包文件不增删。
- 五份 `package_integrity.json` 及其绑定的复现 helper 原字节不变，无需重建任何科学锁。
- 历史导入的 source/configs、metadata、证据内的旧路径和旧审计收据保留。旧路径不是扫描到就全部替换。
- 原始 `outputs/`、`archive/`、个人草稿与授权图未动；没有安装训练依赖或恢复大文件。

## 3. 验收

| 检查 | 结果 |
|---|---|
| 原文件哈希审计 | 727 份：687 份字节不变，40 份批准的调用地址/导航/测试适配变化 |
| 配置迁移 | 34 份唯一目的地、旧位置退出、SHA256 与迁移前一致 |
| 调用关系 | 29 个调用方、137 组去重引用全部保留；引用顺序一致 |
| 代码变化边界 | 调用方逆映射新路径后与 B2a 基线一致；改学习率、seed、模型名或配置顺序的负例均被拒绝 |
| 本地标准库测试 | 原78项 + B2a 9项 + B2b 12项，共99项通过 |
| 启动脚本语法 | 17个 Bash `-n`、2个 PowerShell parser 检查通过；未运行训练命令 |
| 结果重算 | 临时副本的14份CSV值与字节一致、05 SVG内容一致；原实验包不写入 |
| GPU验收 | 未进行。当前本机无 PyTorch 等训练依赖，不能把目录检查当成 GPU smoke |

回归使用本机 Python 3.14.6；Python 3.10 只验证语法兼容。05 六组在本机仍各缺13项运行输入，历史结果可核验不等于训练条件已齐备。

## 4. 当前检查入口

在项目根目录执行，不启动训练：

```powershell
python tests/run_repro_checks.py
python tests/test_entrypoint_layout.py
python tests/test_config_layout.py
python scripts/audit_config_layout.py verify --check-receipt
```

- [before.json](before.json)：B2a 基线提交、727份已有文件哈希、34条映射、29个调用方的原引用序列。
- [receipt.json](receipt.json)：本批验收收据与40份变更的前后哈希。
- [audit_config_layout.py](../../../scripts/audit_config_layout.py)：拒绝配置/证据改变，拒绝调用方发生超出路径替换的变化，不做 reseal。

A、B1、B2a 审计是各自时点的记录，不通过改写旧收据使其“适配”新目录。原 B2a 测试先撤销已由 B2b 单独验证的配置地址迁移，再做原 tests/docs 变化断言；旧断言没有放宽。

## 5. 下一批

确认并提交本批之后，再分批归类 `scripts/`。优先迁移可用标准库完整验证的独立分析工具；复现工具和训练入口另批处理互相导入、根目录定位、脚本互调与完整性锁。不在本批混入旧包清理或模型算法修改。
