# 创智平台项目资产合同

## 阶段 08 已退役：原单目标历史码字交互

- 用户确认研究目标为OneRec的请求级列表生成，原08代码及模型已按要求清理。`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_session_codeword/`不再是可执行或可恢复资产入口；不得根据旧收据重建旧实验。
- 共享原始数据、01–07模型及项目外`grm-exp`环境保留。历史结果及删除清单见`experiments/08_session_codeword/RETIREMENT_20261026.md`。
- 新09列表生成研究已有本地实现`scripts/listwise_diffgrm/`和固定配置，尚无远端执行目录或GPU作业；后续独立命名，不复用原08的语义。当前只完成CPU代码验收，不能根据本地runner自动启动正式训练。具体入口见`docs/research/onerec_listwise/IMPLEMENTATION.md`。

## 阶段 07 商品内局部注意力的独立执行副本

- Project仍为公共科研项目；CPU资源空间准备与同步，分布式训练空间运行单H100持久Job。
- 本阶段个人共享根目录为 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/`；每轮使用独立Goal子目录，`code/`与`engine/`分别接收根仓库和DiffGRM引擎的锁定Git bundle，`delivery/`保存传输校验和作业收据。
- 原始Beauty数据、PSE量化资产和正式SID从HDD主项目只读校验后复制到该轮`inputs/`；不修改历史主项目，不重新拟合量化器。训练、smoke和缓存分别写独立输出目录。
- 继续使用主项目外已有`grm-exp`环境，不升级共享环境。交替联合训练入口为 `scripts/train/run_item_local_joint.py`，GPU包装器为 `scripts/train/item_local_joint_gpu_job.py`；复用05锁定离线Refined-PSE和配套DiffGRM暖启动，码本通过交替优化更新。旧固定SID入口 `scripts/train/run_item_local_attention.py` 仅用于历史追溯。配置、版本、预算和即时状态保存在本轮实验收据，不把旧作业状态当成当前事实。

## 阶段 06 连续向量基线的独立执行副本

- Project仍为公共科研项目；CPU资源空间同步/验收，分布式训练空间运行GPU Job。独立根目录`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/`，不与05训练副本混用，不代表主项目迁移。
- 每个Goal使用独立子目录：`delivery/`保存Git bundle、传输哈希与平台收据，`inputs/prepared/`和`inputs/cache/`原样保存已验收06输入，`code/`为从bundle克隆的固定版本。
- 06数值修复副本位于`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016_precision_v2/code/`；只读复用`goal_20261016/inputs/`的锁定输入，原`goal_20261016/code/`及输出保留作版本追溯，禁止混入修复版本选模。
- 进程管理修复作为独立部署保存在`goal_20261016_precision_v2/orchestration_v3/`，入口`scripts/train/continuous_queue_supervisor.py`，以`DEPLOYMENT_MANIFEST.json`绑定其独立提交与哈希；训练worker仍从原`code/`冻结版本启动。恢复输出使用`code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3/`；原pipeline保留，完成项和中断状态的继承须有独立CPU核对收据，不能把调度版本当作新模型版本。
- 入口`scripts/train/run_continuous_campaign.py`；输出位于该`code/outputs/stage06_continuous/<run_id>/`。运行期间不更新执行代码；CPU gate与正式作业绑定同一源码和数据哈希。
- 使用主项目外的`grm-exp`环境，不升级共享环境。Job名、实际版本、费用和完成状态写本轮收据，不作为永久资产状态。

## 适用范围与主目录

- Project：公共科研项目。CPU资源空间用于开发与文件处理；分布式训练空间用于 GPU 运行。
- HDD 主项目：`/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized`。
- 本地工作副本：`C:/Users/nancy/Documents/experiment1`。两个名称不同，不代表本地副本已经废弃。
- 旧服务器 `nancy/experiment1` 已退出运行路径，不重建同名目录或指回旧副本的软链接。`experiment1_retired_20261009` 在完整核验后于 2026-10-12 清理；新项目保留独立迁移资产，本地工作副本不随服务器退役而改名。
- 新主目录内的 `data/`、`outputs/`、`external/DiffGRM/saved/` 是独立迁移后的运行资产；不得软链回旧 `experiment1`。
- `archive/legacy_server_20261008/` 保留旧脚本、文档、证据包及旧引擎源码原始字节，仅用于追溯，不作为启动入口。
- 历史原始日志中的机器路径保持原样。开发层通过 `runtime_paths.py` 显式重定位已知旧前缀，不改写历史指标和哈希锁。

## 环境与代码版本

- 模型环境：`/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp`，Python 3.10；它在新旧项目之外，删除旧项目不得删除此环境。
- 严格历史 CSV 重算使用 `/inspire/hdd/global_user/<inspire-account>/miniconda3/bin/python`，已验收 Python 3.14.7。Python 3.10 的 statistics 标准差末位可能不同，不据此改历史结果。
- 根仓库移植基线：`5cc5ef3d2a318e84a58d7906c01bfc900639946d`；迁移路径兼容在此后单独提交。运行时记录实际 `git rev-parse HEAD`；工作区若有改动，另存补丁，不把未提交代码当作该 commit。
- DiffGRM 开发引擎：`external/DiffGRM`，独立 Git，基线 `39e94e6153c97e59877ba02d4ef1d9a202ed5384`。
- GenRec 开发引擎：`external/genrec`，独立 Git，基线 `b6c8b41d1d20a812590dae11408b69b1ed2a8313`。
- 引擎基线是已锁定的可重放源码，不冒充未记录的历史运行 commit。根 Git 不自动跟踪引擎修改，后续必须分别记录版本。
- 离线 Git bundle 不是 GitHub。服务器上引擎 origin 指向新主目录内的离线包；根 origin 是共享盘交付包。未建立自动双向同步。
- 代码真源与同步：本地审查并提交主仓库；服务器主目录接收同一 commit 后运行。引擎修改在独立仓库另行提交并记录，不同时维护两套未同步的修改。迁移验收与收据索引见 `docs/guides/EXPERIMENT_WORKFLOW.md`，不以此文件替代运行证据。

## 使用与保全

- Linux 上从主项目根目录 `source runtime_env.sh` 后使用模型环境，输出仍放新主目录；正式任务使用唯一 run_id。
- 当前路径兼容测试：`python -B tests/test_runtime_paths.py`。历史目录整理审计对应原封存版本，不用于否定随后明确提交的功能改动；历史证据和哈希锁不因迁移而修改。
- 复现历史实验优先使用 `scripts/repro/` 的封存配方，以新主目录作为 `--project-root`，不重新拟合历史 SID。
- 每份历史实验的初始化数据、模型与 Tokenizer 以 `experiments/01*` 至 `05*` 中的 recipe 和 assets.lock 为准。
- 迁移清单、源/目标哈希、退役收据和验收报告存于 `archive/server_migration_20261008/`；2026-10-12 的逐文件清理记录与删除后验收存于其 `cleanup_20261012/` 子目录。本地对应记录在 `archive/server_sync/server_cleanup_20261012/`。
- 原始迁移包保留于本地 `C:/Users/nancy/Downloads/experiment1_migration_20260923_retry/`，完整 SHA256 和 gzip/tar 校验通过后清理了远端重复包。服务器直接使用已迁入的锁定输入；如需从压缩包恢复损坏文件，先从本地备份重新上传。该原包不包含后续新短测试，后增产物不得以“原包已有备份”为由删除。
- `.git`、可重建字节码、编辑器缓存不从旧服务器副本迁入；新仓库及引擎使用已恢复的真实 Git 历史。私人凭据不作为实验资产复制。
- 不删除历史证据、锁文件、配对 SID/权重或环境来释放空间；是否清理需先核对消费者和替代副本。

## 阶段 05 新短对照资产

- 固定一次更新、冻结、恢复的训练入口是 `scripts/train/run_diger_noise_pilot.py`；固定参数和共同起点哈希在 `configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json`。
- 新输出使用主目录下 `outputs/pilots/stage05_noise/<唯一名称>/`，不覆盖历史 `outputs/reports/` 或 `experiments/05_crosspse_joint/evidence/`。
- 同步包、后台 Job 日志与启动收据使用 `outputs/pilots_delivery/<唯一名称>/`。代码通过增量 Git bundle 快进同步；不以逐文件覆盖代替版本同步。
- GPU smoke 的原始 JSON/日志在 `experiments/05_crosspse_joint/pilots/noise_update/preparation/`；完整 GPU 权重保存在对应共享盘输出。轻量证明包不含模型权重，不能单独恢复训练。
- 运行状态与即时资源信息以 Job、启动收据和输出文件为准，不把它们写成此资产合同的永久状态。未保存完整历史优化器状态的共同起点实验不得称为历史精确续训。

## 阶段 05 推荐反馈对照资产

- 入口为 `scripts/train/run_diger_feedback_pilot.py`，固定协议为 `configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json`。三组从同一锁定权重开始，分别不更新 Tokenizer、仅重构等目标更新、额外推荐反馈更新；三组均保留 Cross Mixer。
- 新输出仅使用 `outputs/pilots/stage05_feedback/<唯一名称>/`。GPU smoke 与正式 pilot 分开保存，正式实验只使用验证集，不覆盖既有噪声对照及历史证据。
- 同步包、GPU 启动入口和收据继续存放 `outputs/pilots_delivery/<唯一名称>/`。启动前须通过服务器 `--require-torch` 张量检查、18 项输入绑定检查及三组 GPU smoke；后台 Job 记录实际 commit 和输入哈希，不依赖本地电脑持续登录。

## 阶段 05 只读分析的 SSD 工作副本

- 限定分析根目录：`/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis`，适用于公共科研项目的 CPU/GPU 分析作业。
- `code/` 是按本地已提交版本同步的稀疏分析代码副本，不是 HDD 主项目的完整备份。每次锁定实际 commit、必要配置与测试文件；不要把 SSD 同步称为 HDD 主项目同步。
- `delivery/<唯一名称>/` 保存增量 bundle、校验、启动与完成收据；`outputs/analysis/stage05_feedback/<唯一名称>/` 保存逐样本轻量统计。短测试和正式统计分目录，不覆盖旧产物。
- 数据、初始化及历史快照仍从 HDD 主项目只读加载，使用主项目之外的 `grm-exp` 环境；临时缓存写独立临时目录，新结果只写个人 SSD。
- 固定推荐器诊断入口为 `scripts/analysis/evaluate_feedback_hard_compatibility.py`，独立重算入口为 `scripts/analysis/feedback_hard_compat_analysis.py`。必须保留实际解析的原配置、正式 SID、输入哈希及模型未改变收据。
- 作业运行时不更新 SSD 代码；所有作业结束后再快进同步。原始逐样本压缩记录与轻量归档保留在输出区，研究总结和版本绑定收据按阶段目录归档，不把模型权重加入 Git。

## 阶段 05 交互模块四组对照的独立训练副本

- 已授权本轮代码与新权重使用 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training`。这不是项目主目录迁移；原项目数据、历史模型与初始化仍从 HDD 主项目只读加载。
- `code/` 是提交版本锁定的稀疏执行副本，不作为完整备份；`delivery/<唯一名称>/` 保存同步、短测试和启动收据，`outputs/pilots/stage05_mixer_factorial/<唯一名称>/` 保存四组产物。运行期间不更新执行代码。
- 入口 `scripts/train/run_diger_mixer_factorial.py`；实验定义见 `experiments/05_crosspse_joint/pilots/mixer_factorial/README.md`。共同有效码本 E₀重基、SID、完整推荐器状态、随机种子和恢复预算必须一致。
- 正式启动前执行 `tests/test_diger_mixer_factorial.py --require-torch`、输入哈希核验与四组 GPU smoke，并检查 GPU 内可以读取全局个人目录与实际写入。`DIFFGRM_ENGINE_ROOT` 可显式指向只读历史引擎，禁止因稀疏副本缺引擎而跳过真实张量测试。
