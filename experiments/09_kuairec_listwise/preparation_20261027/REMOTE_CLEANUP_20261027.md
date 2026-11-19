# 06/07远端清理及09存储状态

2026-10-27。用户先要求09只写入`experiment1_reorganized`，随后明确授权清理`global_user/.../nancy/experiment1_stage06`和`experiment1_stage07`的实验模型及运行副本，保留本地备份和轻量报告。清理后原路径仍受配额限制；用户最新又同意09使用独立global_user目录。新许可覆盖此前仅原项目目录的限制，只针对09新执行资产，不迁移原项目。

## 已完成清理

- `experiment1_reorganized/outputs/stage06_continuous/goal_20261016/delivery`中的两份失败上传残留：32768字节及0字节。删除前逐一核验路径、类型、大小、SHA；不是模型。
- `experiment1_stage06`下两份、`experiment1_stage07`下九份明确命名的运行/备份目录全部移除。删除前生成逐文件清单，apply时清单哈希必须保持`d5faa639da970365fccdcbec23b61cb639c3eeb09e3b7437dbc2764fe3b44501`；使用Linux fd安全删除，五个符号链接只删除链接本身。
- 清理前du显示06约34G、07约17G；清理后两个根各剩512字节空目录。全局个人挂载df显示从616G降为567G，这属于显示精度及共享挂载观测，不能宣称差值全部是独占结算或精确释放量。
- 08历史交互目录此前已清空，本次核实仍为空，不重复计入释放空间。

清单记录15272件普通文件的逻辑长度约90.46GB；含重复硬链接/稀疏文件等，不能将其当作物理释放量。完整清单压缩包、实际清理脚本、删除后及写入探针日志在`remote_cleanup/`，文件SHA见其中`MANIFEST.json`。实际删除回执为`OLD_REMOTE_DELETION.json`。

## 保全及证据边界

- 06九卷、07 seed2024两卷、seed2025/2026十卷，本次重新顺序读取19,002,028,271字节，各分卷及07合并包SHA全部匹配。恢复元数据和浅克隆补充文件同时核验；见`OLD_BACKUPS_CURRENT_SHA_AUDIT.json`。
- 正式结果及beam trace/validation九份本地轻量报告保留，绑定见`OLD_REPORTS_PRESERVED.json`。未改旧指标。
- 旧失败版、独立诊断原始运行目录及未纳入正式包的smoke状态不承诺完全恢复；正式备份的既有覆盖边界不因本次删除扩大。07 seed2024包不含共享data/cache，因此05共享原始输入仍保留。
- 删除前平台活跃GPU Job为空，见`BEFORE_CLEANUP_ACTIVE_JOBS.json`。05、09、共享环境、当前DiffGRM源码及其导入依赖、本地大包均未删除。09禁用商品内交互但仍导入相应模块，不能据本次清理删除该源码文件。

## 原路径配额仍受限，09获准改用global_user

新根上的正式共同暖身已于北京时间20:35左右RUNNING：`nancy-stage09-warmup-j-20261027-r1`（`qb-prod-gpu002`、平台上限2.15小时）和`nancy-stage09-warmup-h-20261027-r1`（`qb-prod-gpu146`、2.4小时），各1张H100、无DDP。实际创建、状态及首次日志见`../formal_execution_20261027/`中的`J_CREATE.json`、`H_CREATE.json`、`STATUS_START.json`、`j_LOG_START.json`和`h_LOG_START.json`。TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果；四分支尚未启动。

09当前累计75 GPU小时授权见[BUDGET_APPROVAL.json](BUDGET_APPROVAL.json)及[已批准资源计划](../h100_preflight_r2/APPROVED_RESOURCE_PLAN.json)；J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

清理后再次向用户指定的HDD主项目09目录写入136305字节bundle，仍返回`Disk quota exceeded`。全局个人目录清理没有解除该路径的配额。CLI当前没有存储配额修改命令；网页控制因无法可靠识别Windows浏览器网址而停止，没有执行扩容。

历史证据显示06已遇过相同问题：`experiments/06_continuous_baseline/execution_acceptance/PLATFORM_SUBMISSION_20261016.json`的`storage_note`记录2026-10-16主项目写入报`errno122 Disk quota exceeded`，随后global_user独立根通过1MiB写入、完整输入同步及35项远端CPU检查。07继续使用global_user，因此06/07能完成不代表原主项目配额曾恢复。

原主项目盘点8.2G是目录占用，不是配额上限。清理后`df`显示global_user入口为10T总量、567G已用，project入口为80T总量、56T已用；两者设备名均为`gpfs_hdd`。这些统计和写入表现支持区分两个存储入口，不能据此宣称完全独立配额，也不能确定用户、项目、文件集或inode中哪一层达到限制。原路径扩容尚未发生/验证。

用户最新同意的09执行根为：
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/`

只在这里准备本轮代码副本、输入、依赖、缓存与产物；不移动原`experiment1_reorganized`、不复活已清理的06/07目录。1MiB探针、661708800字节代码/数据包以及BGE四片/整包/11文件远端回读均通过，见`GLOBAL_USER_WRITE_PROBE.json`、`DEPLOYMENT.json`和`../assets_preparation_20261027/BGE_REMOTE_TRANSFER_VERIFIED.json`。单H100内容编码已实际读写新根，完整Goal容量仍待验收；切换目录不等于修复原目录配额。

远端32真实文本CPU probe通过，编码5.4594659秒、5.86138件/秒，保守CPU全量估算40.4383分钟，报告见`../assets_preparation_20261027/ENCODER_CPU_PROBE.json`。随后冻结BGE内容编码Job完成10728件、exit0，按北京时间18:02:14至18:04:29计135 GPU秒；后续H100验收R1为18:21:22至18:25:48、266秒，R2为18:30:22至18:34:37、255秒。三个准备Job均已释放，已终态的准备作业合计656 GPU秒（0.182222 GPU小时），准备阶段原7200秒上限下剩6544秒（历史）。最新已批准累计75 GPU小时（270000秒），包含已终态作业656秒，269344秒为两暖身分配前余额；当前新增占用和实际余额待平台终态事件核算。事件见`BGE_ENCODER_JOB_EVENTS.json`、`H100_PREFLIGHT_JOB_EVENTS.json`、`H100_PREFLIGHT_R2_JOB_EVENTS.json`及上级`EXECUTION.md`。

CPU共同PCA/PSE及独立审计现已完成，见`../assets_preparation_20261027/PSE_MANIFEST.json`及`PSE_ASSET_AUDIT.json`：`toy=false`、10728件、码本`[4,256,64]`，最近邻SID及误差重算一致。9513个唯一SID，1505件商品位于多商品碰撞桶（14.0287%），最大桶84件；这是商品可区分性的限制，不是工程失败。正式阶段按用户最新要求以2卡分别跑J/H共同暖身、4卡分别跑四个独立分支，每模型单卡、无DDP；累计75 GPU小时预算已批准。

R1/R2四臂及supervisor均passed，报告见`../h100_preflight_r1/`和`../h100_preflight_r2/`：真实BF16梯度、R/T隔离、恢复、共同初始化、SID刷新及无目标五商品生成通过。R1的`METRIC_RECOMPUTE.json`仅为beam2小批商品指标检查，不是正式效果。R2的`runtime_efficiency_r2.json`采用微批256、有效batch1024、推理batch32，科学模型/beam/评价定义不变，不复用短验收状态。

远端训练及代表checkpoint恢复继续使用冻结版本`1c423662d9a25a926328ee6466c30c0f2857b4f6`；本地`83a1b9c`新增analysis仅用于离线分析，不同步改变远端训练源码binding。

共同资产18文件、94,105,600字节tar已下载并在`outputs/stage09_restore/goal_20261027_r1/`本地独立恢复PASS，见`../assets_preparation_20261027/GENERATED_ASSETS_LOCAL_RESTORE.json`；它不含正式训练模型。`../h100_preflight_r2/BUDGET_ESTIMATE.json`估后续基础58.1855 GPU小时，含25%余量和已用量累计72.9141 GPU小时；2卡暖身、最多4卡独立分支，墙钟基础15.868小时、含余量19.835小时，不含排队。1点/GPU小时仅作未结算估算，用户已批准累计75 GPU小时上限，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

历史`INPUT_TRANSFER_MANIFEST.json`和归档内旧目标路径保持原字节，不回写已经锁定的传输包；当前部署路径由本轮新的DEPLOYMENT收据单独绑定。

## 所需空间规划

4GB仅接近准备阶段资产量，不能覆盖完整Goal。已确定输入tar+解包payload+BGE权重共约3.368GiB；源码、依赖及正式embedding/PSE完成后，准备常驻约3.7–4.5GiB。

按真实1层Encoder/4层Decoder在meta设备上的参数量计算，推荐器5,604,352参数、Mixer98,944参数；含双Adam及alignment持久buffer的完整状态每份约79.8MB。正式18份initial/best/last约1.25GiB，preflight恢复状态约0.60GiB，四组同时原子保存额外约0.30GiB。

42次验证、4次测试及8次固定子集共3,230,562条预测记录，gzip大小尚未实测；预测/日志先预留2–5GiB，完整最终归档再预留约8–12GiB。建议申请约50GiB新增可用空间以容纳归档、恢复和重试余量；这是规划值，不是已测用量或必须购买的容量。正式预测产生后重新核对实际占用，不据此额外启动模型或实验。
