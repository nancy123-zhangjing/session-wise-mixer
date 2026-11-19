# 协同目标两组正式实验

用户于2026-11-13授权启动，每组一张卡。模型代码基点 `cd1f236`；使用 `v2_cf.json` 和 `v2_hybrid.json`，两组均为B1层内Mixer，seed2024。其他设置、R35/T300预算、R5至R35逐轮全验证、best/final测试均按已实现设计保持不变。共同监督为去噪MSE＋0.1×RQ落地损失；T阶段仍有原潜空间与VQ约束。

第一张H100在同一个CF作业中先训练共享SASRec教师，按验证首件NDCG@5选best，再依次完成CF准备、RQ预训练、联合训练和测试。教师上限200轮、耐心10轮，测试集不参与教师选点。教师完成收据与items SHA一致后，依赖控制器才申请第二张H100，执行Hybrid完整流水线。两组不重复训练教师，不占卡等待。CF平台时限24小时、Hybrid 12小时是故障上限，不是完成耗时预测。

执行配置复用旧版已运行的BF16、R/T微批256、有效batch1024、验证batch256；没有声称新落地损失已通过GPU短测。直接正式运行，保留数值检查与每200次更新恢复点。外层worker只增加阶段记录和教师进度日志，不改变训练参数、随机数或梯度。

本地 `control.py stage/cf/watch/snapshot` 负责同步、单次提交、教师完成后提交第二组、查询。watch为本机后台依赖进程；需要本机开机和启智网络可用，暂时断网每60秒重试。同名作业先查询并对账，模糊提交不盲目重复创建。若本机进程中断，重新启动watch可读取远端教师收据接续；不要重提CF。

源码冻结在独立共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1`，逐文件SHA清单在 `SOURCE_MANIFEST.json`。数据与内容只读复用原 `goal_20261027_r1`，大模型和预测结果只留远端。已有Decoder前缀四组不动。不要凭RUNNING宣称完成或验收。

状态与提交响应保留在本地 `raw/`、`LATEST.json`、`LIVE_DEPENDENCY.json`；远端 `teacher/PROGRESS.json`、`cf_STATE.json`、`hybrid_STATE.json` 区分当前阶段。共同教师、共同推荐器初始化和噪声绑定由原入口检查；跨组最终比较还需读回双方prepared身份和完整验证/测试产物。
