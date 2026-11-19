# 项目结构重构：迁移与清理清单

原盘点日期：2026-10-07；迁移收尾更新：2026-10-09。回退基线：`32f3b4e2039f40129f709646b535742eadd2de53`，整理分支 `codex/project-organization`。

**当前进度：A、B1、B2a、B2b、B2c 均已提交，服务器资产独立迁移与有限运行验收完成；C删除未执行。** A=`737bebf`，B1=`5f4f433`，B2a=`07d6a83`，B2b=`3d41254`，B2c=`fd5faee`，跨平台字节保护=`5cc5ef3`，运行路径兼容=`594b6ad`。当前结构见 [项目地图](PROJECT_MAP.md)，最新服务器与 GPU 短测范围见 [迁移收尾](EXPERIMENT_WORKFLOW.md#六2026-10-09-迁移收尾与验收范围)，B2c 原记录保留为其整理时点的验收。旧服务器目录只改名为退役备份；压缩包、草稿、原证据和历史源码快照继续保留，没有删除文件或启动正式重训。目录整理以“可恢复、能复现、不重复存放”为目标，不通过删除失败实验掩盖研究过程。

迁移前记录及迁移后验收见 [A 批记录](../../reproducibility/layout_migrations/20261007_output_to_experiments/README.md)。下文的原位置、压缩包依赖和删除候选保留重构前的盘点口径，不表示这些对象已清理；B1 后的日常操作以 [工作流说明](../../reproducibility/WORKFLOW_B1.md) 为准。

## 1. 已固定的起点

- 01–05 实验包、辅助脚本和复现锁已进入基线 commit。
- 根目录四个 `draft_*_folder` 和两张授权图未进入 Git。
- `archive/`、`data/`、`outputs/`、`external/` 被忽略，不能假定它们删除后可以通过根仓库 Git 恢复。
- `data/` 本地没有正式数据；大数据、Tokenizer 产物和权重仍依赖原迁移包/服务器。根仓库 commit 不能替代大资产备份。
- 当前上游工作副本：DiffGRM `39e94e6153c9…`，genrec `b6c8b41d1d20…`。`upstreams.lock` 指向原上游基准，不等于包含本项目修改的当前引擎；不能仅重新拉取它就声称恢复了全部训练代码。

## 2. 建议的最终结构

```text
experiment1/
├── README.md                 # 唯一总入口：研究问题、阶段、当前任务
├── experiments/              # A批已迁入，01—05 统一实验包
│   ├── README.md
│   ├── experiment_registry.json
│   └── 01_... 至 05_.../
├── configs/                  # B2b：shared/ + 01–05，保留配置内容及覆盖顺序
├── scripts/                  # 已按repro/train/checks/prepare/tokenizers/analysis/reports/audit/legacy分类
├── tests/                    # B2a 迁入既有测试；检查范围见 tests/README.md
├── docs/                     # B2a 分类；总索引保留原位
│   ├── guides/               # 现用工作流与重构清单
│   ├── methods/              # 8 份方法/协议原文
│   └── history/              # 9 份历史记录原文
├── reproducibility/          # 源码快照、环境、资产清单、恢复规则
├── external/                 # 固定版本训练引擎，不在本轮改算法
├── data/                     # 服务器运行数据，本地可为空，不入 Git
└── outputs/                  # 新 run 的原始结果，不覆盖归档实验包
```

核心区别：`experiments/` 是审核过的实验记录；`outputs/` 是程序产生的运行记录。不再用容易混淆的 `output/` 与 `outputs/` 区分正式材料和原始产物。

过去的个人材料移到用户指定的项目外目录；历史导入材料保留一份经校验的独立备份，不再在项目内重复解压多层。具体项目外存储位置尚未指定，本清单不自行创建。

## 3. 逐项迁移清单

| 原位置 | 目标/处理 | 前置条件与保护 |
|---|---|---|
| `output/01_*` 至 `output/05_*` | 整体迁到 `experiments/`，保留阶段编号 | 更新路径解析、注册表、链接、`.gitattributes` 与测试；不改原证据字节 |
| `output/README.md`、`output/experiment_registry.json` | 迁到 `experiments/` | 基线复用关系、14次独立运行/17条目及05六组身份不变 |
| 各阶段 `evidence/` | 保留为正式轻量证据来源 | hash 必须与迁移前一致，不手工改数据 |
| 各阶段 `HISTORY.md` | 保留版本演进主线 | 明确失败/替代版本、研究目的和关键发现，不要求保留每份重复报告 |
| 各阶段 `previous_README.md`、旧PDF/补充报告 | 暂保留，后续判断是否只在基线 Git 中保留 | 有的已被完整性锁引用；退场前需更新新版本验收规则，不能直接删 |
| `reproducibility/` | 保留；可统一管理跨阶段重复源码快照 | 05目前也有独立快照副本，合并需更新相对路径及锁；不是现在直接删除重复tar |
| `configs/` | B2b 已按shared/与01—05分类，34个YAML原字节移动 | 29个调用方只变地址、不变加载顺序；历史配置证据副本保持不变 |
| `scripts/` | B2c已完成75个根层脚本归类；根层只保留README入口 | 同步互调、Python导入、root推导和5份辅助代码绑定；旧科学锁/快照不变 |
| `scripts/test_*.py` | B2a 将 12 个既有测试迁到 `tests/` | 同步import、调用入口和已锁定helper路径；检查范围见 [tests 说明](../../tests/README.md) |
| `docs/` | B2a 将 19 份文档分为 2 份 guides、8 份 methods、9 份 history；总索引保留原位 | 17 份方法/历史正文原字节保留，旧状态和命令不改写；现用状态写在指南与分类索引 |
| `outputs/` | 保持原运行目录，不本轮整体改名/清空 | 后续逐run确认是否已完整归档；新run使用唯一ID |
| `data/`、`external/`、`patches/` | 先保留原路径 | 数据完整性、补丁和上游版本是运行依赖，不是报告整理垃圾 |

本轮目标是结构重构，不顺便改损失、训练策略、Tokenizer或实验参数。历史 metadata 内的旧绝对路径保留，通过新路径映射解决，不伪造历史运行地点。

## 4. 压缩包逐项清理判断

检查方法：读取7个报告ZIP的所有普通文件，逐成员SHA256与当前Git跟踪的 `output/` 内容比较（545个文件、490种不同内容hash，包含旧README备份）。匹配看内容，不仅看文件名。成员时间戳/ZIP封装本身不属于此次内容等价结论。

| `archive/report_exports/` 下文件 | 文件数 / 精确匹配数 | 建议 |
|---|---:|---|
| `01_ar_baseline_full.zip` | 14 / 14 | 高置信删除候选，正文和图均有已提交副本 |
| `01_ar_baseline_supplement.zip` | 4 / 4 | 高置信删除候选 |
| `03_diffgrm_tokenizers_full.zip` | 18 / 18 | 高置信删除候选 |
| `04_crosspse_offline_report.zip` | 9 / 9 | 高置信删除候选 |
| `01_ar_baseline_compact.zip` | 4 / 2 | 暂保留：README和`position_nll_all_seeds.png`尚无精确匹配 |
| `02_ar_capacity_report.zip` | 10 / 9 | 暂保留：README尚无精确匹配 |
| `03_diffgrm_tokenizers_compact.zip` | 8 / 7 | 暂保留：README尚无精确匹配 |

前四个候选共2,927,618字节，约2.79 MiB。对所检查的 `scripts/`、`configs/`、DiffGRM引擎源码未发现这些报告ZIP的运行引用。删除前仍需重新核对当前文件hash、正式副本和是否需要保留原交付包；**本次没有删除**。其余三个的“未匹配”不等于无用，应先审阅独有内容。

四个候选的当前文件SHA256：

```text
c24b5dd3e0c2f73be9243c1eede8247b9756229a75c77cf74e150ef867606588  01_ar_baseline_full.zip
8aac840a8c51494b7b6e2dcc6a0ec3d72ccfea0f8907bb3124b8509e4b7b113f  01_ar_baseline_supplement.zip
4e5b6db6263d6c4c88c98c1560ee95784351ba3fb63b69012c689483c65f7e5e  03_diffgrm_tokenizers_full.zip
950d026959c52d9e4160dc0cb11a213170e864e8b2882414f4f3e6a0961d34ac  04_crosspse_offline_report.zip
```

### 这些压缩包不能按“重复报告”处理

| 对象 | 当前依赖 | 清理前必须完成 |
|---|---|---|
| `archive/evidence_bundles/` 的6个小tar.gz | 01–04 `build`每次都会读取 | 将历史导入与日常复现分离；保留来源收据、必要独有材料和独立备份 |
| `archive/evidence_bundles/repro_sources_01_04/` | 01–04 `build`读取恢复缓存、原控制台等 | 修正缓存恢复逻辑或新建不依赖旧目录的封装入口；现有build不能靠删缓存自愈 |
| `archive/evidence_bundles/05_joint_source_20261007_192731/` | 05 `build --source`读取 | 证明六组所需原始材料均已绑定且能从新来源恢复 |
| `archive/deprecated_and_temp/.../crosspse_report_20260829/` | 01–04 `build`实际读取04 reports/pilots | 迁出并保留有效来源，更新导入路径；不能看到tmp就整删 |
| 根目录与05复现目录中的源码tar | `verify`强制校验，`run`实际读取代码 | 保留，或迁移/去重后建立新的等价源码锁；不当作废包 |
| Downloads中的4.62GB原迁移包 | `build`、`restore-inputs`使用，含大资产 | 有经hash校验的独立副本、可用资产存储和不依赖待删副本的恢复入口后才能考虑本地副本退场 |

已有 `verify/analyze/run` 不会回读旧原始来源目录，这不等于 `build/restore` 也不依赖它们。仅“报告能打开/verify通过”不足以证明所有旧来源可删。

B1已修正 `archive/README.md` 中“可以整体删除deprecated_and_temp”“archive不参与脚本”的过时说法。上表保留的是历史导入的真实依赖；日常 `build` 已退役，但独立导入器和旧报告生成工具仍要读取相应来源，所以仍不删除。

## 5. 根目录草稿和临时内容

| 路径 | 已查到的用途 | 建议 |
|---|---|---|
| `draft_8246f16f_folder/draft.xml` | DiffGRM×PSE词表训练方式阶段汇报 | 保留研究稿；可归入05沟通材料，不按目录名删除 |
| `draft_97467961_folder/draft.xml` | Cross-PSE×DiffGRM联合训练复盘与后续讨论 | 与上一份正文不同；先保留/核对独有内容 |
| `draft_8b7f2860_folder/` | 个人实习材料 | 正文与下行相同；建议至少保留一份，移到用户指定的项目外个人材料目录 |
| `draft_ca1b3e3c_folder/` | 同一份个人实习材料正文 | XML正文重复，工具状态JSON不同；不能把整个目录称为完全重复 |
| `feishu_docs_read_auth.png`、`lark_inspire_read_auth.png`、8246目录内授权图 | 授权相关图片，不是实验资产 | 不提交Git；确认没有正在使用的授权流程后可清理，未打开/展示图片 |
| `tmp/pdfs/resume_review/page-1.png` | 从路径判断为简历预览 | 未找到精确副本，也未核验源PDF；先确认个人材料源文件，不能假定Git可恢复 |
| `.runtime/`、`__pycache__/`等工具缓存 | 非实验结果 | 与实验重构分开；先确认工具未使用、缓存可重建，不强制改权限删除 |

以上draft/tmp均未进入基线Git。建议研究稿和个人稿分开，避免把个人资料误推到研究仓库。本次没有读取或记录凭据/二维码内容。

## 6. 实施分批与验收

### A：先统一正式入口，不删除

1. `output/`整体迁到`experiments/`，阶段包内部先保持不变。
2. 同步注册表、README链接、脚本PACKAGE/ROOT定位、`.gitattributes`、测试和新版本锁。
3. 保存旧→新路径映射；基线commit仍可回溯旧锁与旧入口，不把新锁当历史运行状态。
4. 核验01–05数值、原证据hash、源码身份及全部测试。此批不改算法，不处理原始资产。

### B：解除旧目录依赖，再归并资料

1. 将一次性“导入旧档案”和日常`verify/analyze/restore/run`分开，明确哪些历史命令退役。
2. 建立可检查的大资产位置/备份映射；本地和服务器路径不写死在实验定义里。
3. B2a 已迁移既有 tests 并归类 docs，B2b 已归类 configs，B2c 已完成 scripts 职责归类和引用更新。沟通稿和个人材料保留原位并忽略，pilot配置/脚本保留、不删除。
4. 测试不依赖计划删除的旧来源也能重建必要结果、恢复输入并生成正确运行计划；训练入口变化还需相应GPU验收。现有AR不支持两batch smoke，不伪称已通过。

### C：按批准清单清理并收尾

1. 再次检查精确目标、hash、副本和依赖；待清理范围有变化就停下。
2. 先清理已核验重复导出；独有材料先保留/迁出，大资产最后处理。
3. 优先可恢复删除；实际删除后记录移除了什么、能从哪里恢复。
4. 重跑校验、检查全部本地链接、提交结构版本。不要把授权图/个人材料顺手`git add .`。

当前不需要重训全部实验；本轮重构的验收重点是数据不变、路径可用、依赖完整、结果可重建。

## 7. 本批交付与下一批边界

A批已执行并提交为 `737bebf`：545 个实验包文件迁入 `experiments/`；注册表23条记录只更新 `package` 路径。保留旧版本 commit 与迁移前哈希记录，不修改历史实验版本。

3 个旧 Stage1 报告工具的路径也已修正，但仍依赖 `archive/evidence_bundles/repro_sources_01_04/` 中的历史输入；本次没有重新生成 PDF/图表。这项依赖留待B处理，不能据此删除 archive。

B1已执行并提交为 `5f4f433`：日常入口拒绝旧build；独立历史导入默认计划且拒绝既有封存目标；迁移包位置由本机映射提供，SHA256身份不变；新增归档隔离、临时重算和安全恢复测试。见 [B1操作与边界](../../reproducibility/WORKFLOW_B1.md) 及 [B1验收](../../reproducibility/layout_migrations/20261007_b1_workflow/README.md)。本次没有真实历史导入、自动解压大数据或启动训练。

B2a 已迁移 12 个既有测试和 19 份文档，并提交为 `07d6a83`。方法/历史原文、科学证据和历史源码快照保持不变，见 [B2a 记录](../../reproducibility/layout_migrations/20261008_b2a_tests_docs/README.md)。

B2b 已将34份YAML按共享设置/阶段归类，并同步29个调用方，已提交为`3d41254`；旧路径退出、内容与顺序保持一致，该批五份完整性锁无需重绑。见 [B2b记录](../../reproducibility/layout_migrations/20261008_b2b_configs/README.md)。标准库基础回归入口为 [run_repro_checks.py](../../tests/run_repro_checks.py)，迁移检查另见 [tests说明](../../tests/README.md)。

B2c已将75个根层脚本全部归类，适配调用/根目录/导入；五份完整性锁只调整辅助代码绑定。完整旧→新映射和检查范围见 [B2c记录](../../reproducibility/layout_migrations/20261008_b2c_scripts/README.md)。本轮不为后续布局改写旧迁移收据。

研究项目的开发/复现/分析入口已完成分类，后续不再小批搬几个脚本。服务器已使用 `experiment1_reorganized`，旧项目退役备份为 `experiment1_retired_20261009`；本地仍使用 `experiment1`。四组主要实验输入已校验，一组 Cross Joint v2 H100 短测通过，可以准备下一轮研究；不要求为整理重训全部阶段。

余下是独立的清理决策：个人材料项目外存放位置、大资产第二份备份、压缩包独有内容审阅和退役目录最终处理。这些不阻塞研究推进。C删除候选单独确认，不把同意结构方案等同于授权清空archive。
