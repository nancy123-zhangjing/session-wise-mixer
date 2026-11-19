# 本地冻结包与真实父状态 CPU 审计

2026-10-29，本次检查不使用 GPU、不访问远端、不执行训练更新。本记录对应 `tmp/h_freeze35_execution_20261029_r2` 的 CPU-only draft；不代表正式源码已冻结或 GPU 短验收通过。

## 打包与历史引用

`prepare.py` 从已验收的 `outputs/stage09_restore/h1_recovery_20261028_r2/code` 复制完整 3,029 个历史文件，并同时核对原 goal 的 20 个 code / 20 个 engine 文件绑定。旧 `RECOVERY_EXECUTION_MANIFEST.json`、`RECOVERY_PARENT_BINDING.json` 原字节保留。本次 draft 仅新增 `recovery35.py`、`recovery35_artifacts.py`、`recovery35_run.py`，不覆盖旧科学源或 evaluation。

实际包大小 14,439,877 bytes，3,036 个 tar 成员均读回核验；不含 checkpoint、dataset、content/PSE 或 predictions payload。

- tar SHA256：`37cd8554306d80dc7afad250d269365f7f0dc9cfa7a1d1f70c8053aa82de161c`
- manifest SHA256：`0f01f22570530ca454062e95602d0b4ba5059c9d1f81d2c1667ede29e2a4954a`
- prepare.py SHA256：`4c8f4e0b27602525a82aaa881903586c7b581d82ae7274de4967c6a37f712f6b`

另一次在独立目录完整重打包得到相同 tar.gz 字节与 SHA，见 `DETERMINISTIC_PACKAGE_AUDIT.json`。路径拒绝和 synthetic 压缩测试见 `PACKAGE_TOOL_SMOKE.json`；后者仅是工具测试。

纯标准库 `python -S` 验收核对 3 个父 checkpoint、A 的原 R20 祖先、8 个 dataset 文件、12 个资产文件，以及去重后的 31 个历史结果/模型引用文件；见 `DRAFT_PACKAGE_VERIFY.json`。普通 verify 已实测拒绝 CPU-only draft。第一次候选在源码并发变化时拒绝发布，不存在成功 receipt。

`FREEZE35_REFERENCE_RESULTS.json` 接口为 `validation/test → A/B/C → epoch`，每条给远端绝对路径、SHA、bytes 和 `local` 路径映射。完整验证引用为 A R20/23/26、B R32、C R5/8/11/14/17/20；测试引用为 A R20/26、B R32，C 为空。C R8–R20 无旧 checkpoint，明确 `checkpoint_available=false`，只作为重放预测参考。A 的原 R20 祖先另列于 `FREEZE35_PARENTS.json → ancestors.A`。

## 真实父状态 CPU 加载

在 draft `code` 目录中，使用 CPU-only `.runtime/stage08_restore_env/Scripts/python.exe`，A/B/C 各在两个独立进程执行 `build_recovery35(..., device='cpu', inspection=True)`。加载实际父 checkpoint、真实 dataset/assets，不使用 toy 对象。

| 分支 | Mixer | R updates | T updates | 当前 R LR | 保存的 T LR（codebook / mixer） |
| --- | --- | ---: | ---: | ---: | --- |
| A | 开启 | 18,018 | 200 | 0.00016066336338464165 | 0.00001 / 0.0001 |
| B | 关闭 | 22,176 | 360 | 0.000018780235250824395 | 0.00001 / 0.0001 |
| C | 关闭 | 3,465 | 0 | 0.0009620503978340745 | 0.00001 / 0.0001 |

每组 model、alignment、R optimizer、T optimizer、scheduler 的完整值指纹均等于父状态；CPU/Python/NumPy RNG 与父状态对应部分一致；每组两次独立重建的完整 inspection checkpoint 指纹一致。全部参数在 CPU、梯度为 None、计数未推进，训练数据长度均为 709,443。实际 controller SHA 为 `500d272dc1f7bbc9e2eb920433ded137c80359e1dd4ac5d63283738ac65256a7`。

审计入口 `cpu_restore_audit.py`；逐次记录及总凭据见 `cpu_restore_r2/CPU_RESTORE_AUDIT.json`。inspection checkpoint 被 controller.restore 拒绝的防护已实测。此处验证的是从真实父状态的独立 CPU 重建，不绕过该防护，不声称 CUDA RNG 精确恢复或下一训练步一致；这些属于后续 GPU 短验收。

## 正式包装器检查与修复

初版 `job.py` SHA `f977248321d176d72c472c97f51b706a41fd9f712c7a3023251d5dc6d1587b9e` 的三项问题，经负责人授权后已修复。修复版 SHA 为 `09ca1f54036c1fae6b60d4c28ca241c8d34e5b4bbeb634ca947090697372ea00`。

已核逻辑：仅接受一张完整 H100（数量、型号、容量及非 MIG）；`jobs/<branch>` 独占创建且拒绝既有 output；train 后 test 顺序使用同一环境的 cuda:0；没有自动重试；不删除失败状态或已落盘边界。

修复内容：启动即建立总截止时间；完整源码和外部输入验收改为受同一 deadline 监督的纯标准库子进程，验收日志及失败 `result.json` 保留。正常工作从总授权时长中预留 15 秒终止宽限；收到 SIGTERM/SIGINT 后立即转发，最多 15 秒（且不越过总截止）后向子进程组发送 SIGKILL，不再继续等待剩余整段训练预算。显式绑定包内 `DIFFGRM_ENGINE_ROOT`。已落盘边界保留，但中断不等于立即保存正在执行的未完成 R。

`test_job.py` 的 5 项纯标准库 stub 测试全部通过：train→test 同一可见卡且各执行一次、源 SHA 失败在无子进程时落盘、verifier 失败落盘且不重试、verifier 纳入总 deadline（30 秒场景在 15 秒 TERM / 30 秒 KILL）、外来信号从收到时起最多 15 秒升级终止。测试无 GPU 查询、无真实模型子进程；不替代 Linux/H100 短验收。命令：`.runtime/stage08_restore_env/Scripts/python.exe -B -X utf8 -m unittest discover -s experiments/09_kuairec_listwise/h_freeze35_20261029 -p test_job.py -v`。

## 正式打包前仍需完成

加入最终 preflight/trajectory 模块及 runner 接线、完成其测试和所需审查、提交本轮源码后，在全新目录运行 `prepare.py prepare`，不加 `--draft`。当前 draft 不可用于 GPU 正式执行。新包包含全部符合白名单的 `scripts/listwise_diffgrm/recovery35*.py`，会绑定各文件 SHA 与 workspace commit；只接受本任务源码干净已跟踪状态，无关工作区变更明确忽略。新模块必须不与历史路径冲突。

远端部署应先独立核 tar SHA，再解压到唯一新目录，使用既有 `grm-exp/bin/python3.10` 运行包内 `delivery/prepare.py verify --code-root <root>/code --manifest-sha <锁定SHA> --references remote --verify-results`。这一步是纯标准库 CPU 验收，宜在 GPU 分配前完成；job 内再次校验的时间仍须计入其预算。既有环境路径记录在 `FREEZE35_PARENTS.json`，本次未重新验证远端环境可用性；无需安装或修改共享环境。
