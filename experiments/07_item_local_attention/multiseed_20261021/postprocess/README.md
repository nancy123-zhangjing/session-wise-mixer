# 本次多 seed 实验的完整备份

这套工具独立于冻结训练代码，全部在 CPU 上运行。上传本目录时放在新 Goal 的独立 `postprocess/` 目录，不要放进已冻结的 `code/` 或 `engine/`。

打包范围：两个 `seeds/seed2025`、`seeds/seed2026` 的完整树和 Goal 的 `delivery/`，包括全部 13 项私有输入的每份实际拷贝、正式及烟测结果、模型权重、优化器/调度器/RNG 状态、日志和验证集逐用户结果；另外保存 code/engine 的完整 HEAD 可达 Git bundle、tracked source tar、clean/ignored 状态审计和本目录工具。忽略文件只记入源码审计；源码仓库的账户设置、共享环境不纳入。任何数据树软链接会使打包停止；已知 `code/external/DiffGRM` 链接只记录，engine 单独保存，不读取链接外目标。

1. 四组正式训练及验证都完成、包装进程退出后，在服务器 CPU 环境打包。`--output` 必须是 Goal **外部**的新目录。完成标志是 `PACK_COMPLETE.json`，单有 `PACKAGE.json` 不算完整打包。

```bash
python -B /path/to/postprocess/multiseed_backup.py pack \
  --goal-root /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261021_multiseed_r1 \
  --output /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/backup_20261021_multiseed_r1 \
  --part-mib 512
```

2. 下载 `PACKAGE.json`、`PACK_COMPLETE.json` 和 PACKAGE 列出的**全部压缩分卷**。先运行本地可信的工具，解包到不存在的新目录。它会先校验分卷、完整压缩流、逐文件 SHA 和路径，再解包、再次验字节；拒绝路径穿越、软/硬链接和文件名冲突，不用系统的直接解压命令。

```powershell
python -B C:/path/to/postprocess/multiseed_backup.py verify --package-dir C:/backup/package --output C:/backup/verified
python -B C:/path/to/postprocess/multiseed_backup.py restore-git --package-dir C:/backup/package --verified-root C:/backup/verified --output C:/backup/restored_git
```

3. 在有 PyTorch、NumPy 的 CPU 环境读回真实 checkpoint。使用 `map_location='cpu'`，CUDA 隐藏；检查所有 `.pt/.bin`、28 个必要联合训练状态、四组固定训练量、Adam 时钟、RNG、最佳模型与 SID 绑定。输出放在解包目录外，不改动备份证据。

```powershell
python -B C:/path/to/postprocess/audit_multiseed_checkpoints.py --package-dir C:/backup/package --verified-root C:/backup/verified --output C:/backup/CHECKPOINT_READBACK.json
```

恢复后的 JSON 保留原服务器绝对路径以保持证据原样。实际迁移续训前仍需单独处理路径重映射及环境安装；CPU 状态读回不等于执行过迁移续训。GPU 释放、平台费用也由另行验收记录证明。工具不会训练、评估测试集、操作平台或自动把 Goal 标为完成。

体积由实际全文件决定，开始打包时会输出输入总字节数，最终压缩体积以 `PACKAGE.json` 为准。模型张量压缩率不保证高。服务器需容纳最终压缩包及分卷写入时的临时副本；本地校验需同时容纳下载包、临时合并流、完整解包文件和两个 Git checkout，可用 `TMPDIR`（Linux）或 `TEMP`（Windows）指定临时合并流所在磁盘。

离线自测（标准库 + Git，不用 GPU 或 PyTorch）：

```bash
python -B /path/to/postprocess/test_multiseed_backup.py -v
```

另有 3 项真实 Torch CPU 自测，构造四组共 28 个小型 checkpoint，验证完整读回、Adam 时钟异常和 NaN 拒绝：

```powershell
C:/Users/nancy/Documents/experiment1/.runtime/continuous_env/Scripts/python.exe -B C:/path/to/postprocess/test_checkpoint_readback.py -v
```

复用来源：旧 `pack_item_local_joint_delivery.py` 的通用字节流程、原样安全路径校验、旧 `verify_joint_backup_checkpoints.py` 的序列化状态校验函数；各辅助文件记录了来源 SHA。新增逻辑只有多 seed 范围、完整输入、冻结源码审计、完成收据和本地恢复绑定。
