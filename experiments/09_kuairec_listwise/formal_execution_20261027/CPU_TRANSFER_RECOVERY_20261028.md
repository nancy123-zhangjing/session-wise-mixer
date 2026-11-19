# CPU文件连接恢复记录

2026-10-28 01:16—01:32。原CPU Notebook `nancy-stage09-prep-20261027-r1` 继续运行于 `hpc-compute262`，0 GPU。原正式两卡Job始终独立运行，本次没有重启训练或Notebook，没有升级CLI、共享Conda或系统包。

## 已验证的事实

1. 官方SDK取当前原始Jupyter入口：页面、`api/status`、`api/terminals`均200，认证正常。见 `JUPYTER_GET_DIAGNOSTIC_20261027T171604Z.json`。已有40个终端只读取数量，没有删除。
2. `InspireClient(account="sii-shared", allow_browser=False).notebooks.exec(..., transport="jupyter", command="hostname")`成功，临时终端由SDK回收。见 `JUPYTER_HOSTNAME_PROBE_20261028.json`。先前子任务的该探针被自动审批拒绝，未执行；根任务依据当前Goal已授权的远端执行/备份范围重新明确申请，获准后才执行。
3. SSH与rtunnel二进制及旧初始化标记存在，但22222、31337端口关闭。通过官方 `build_rtunnel_setup_commands` 的变量、公钥和已有服务启动片段恢复，**跳过安装/校正包的命令**；随后两个端口均open。见 `restore_notebook_transfer_services.py` 及对应JSON。公钥由原官方加载器读取，未输出密钥或访问URL。
4. 官方CLI refresh仍报xterm未就绪。正确Jupyter `proxy/31337/` 的普通GET为404；源码明确这种返回不能单独判定SSH不通，见 `RTUNNEL_HTTP_DIAGNOSTIC_20261027T172755Z.json`。
5. 以官方当前Jupyter入口构造代理，在内存候选bridge里验证SSH。默认stdin模式返回127，未保存缓存；改用官方 `run_ssh_command(..., pass_stdin=True)` 的argv模式返回正确hostname/exit0，随后仅通过 `save_tunnel_config` 保存sii-shared账号的该Notebook连接，保留其他连接。见 `CANONICAL_SSH_RECOVERY_20261028_R2.json`。Windows文本stdin换行为一个候选原因，未单独复现诊断，不能写成已确认根因。
6. 常规官方CLI `notebook ssh`与`notebook scp`已实际成功：下载本轮 `delivery/SOURCE_LOCK.json` 后与远端SHA一致；新备份helper上传与依赖SHA回读也一致，见 `FINAL_SIDECAR_DEPLOYMENT_20261028.json`。因此文件通道恢复有实际传输证据，不仅是hostname成功。

## 后续使用

继续显式使用 `--account sii-shared`、CPU Notebook原名和 `--workspace CPU资源空间`。只读命令优先官方 `notebook ssh ... --command-timeout 30 --timeout 60 -- <command>`；文件走 `notebook scp`。本机环境 `NO_PROXY=.sii.edu.cn,<institution-host>`。输出过滤访问URL及token，不把连接缓存写进Git。

源码副本供审计；实际工作副本在 `outputs/stage09_delivery/goal_20261027_r1/`，诊断收据使用exclusive-create。不要盲目再次运行恢复脚本覆盖收据；先查Notebook当前状态及已有服务。Notebook将按平台时限自动停止，后续若再次启动需重新验证连接，但不能因此重提GPU训练。

`read_stage09_job_progress.py`提供另一条已实测的只读观察路径：官方SDK `jobs.get/exec`读取原Job日志，输出本地快照，不写远端。它可在CPU Notebook停止时继续使用；Job必须实时RUNNING。日志末条验证进度可能停留在上一轮，判断当前训练轮次以 `training` 和完整summary共同为准。

URL拼接函数对假 `/lab/terminals/1?reset` 的离线检查确有异常，但本次完整失败调用链尚未重现；此记录不把它断言为唯一根因。
