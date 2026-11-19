# 第一批本地冗余归档清理

2026-11-05，用户授权：远端大小、SHA256及恢复内容核对通过后删除本地冗余文件。

已删除10个完整归档/分卷，共20,177,206,478字节；另删除D下载残留513,146,880字节。总计20,690,353,358字节，约19.27 GiB（逻辑文件大小，非磁盘物理释放测量）。

证据：CLEANUP_BATCH1_LOCAL_20261105.json、CLEANUP_BATCH1_REMOTE_20261105.json、CLEANUP_BATCH1_DELETED_20261105.json。

远端10文件现读SHA与本地逐字节一致；6个完整归档的所有普通成员已从归档完整读出并计算SHA，清单保存在REMOTE收据。BGE四个分卷亦逐一核对，完整BGE归档可读。恢复验证范围为归档字节及所有普通成员可读性，不表示本轮重新加载了所有模型或重跑实验。所有本地解压结果、模型状态、数据、代码、环境和报告均保留。

D残留路径：experiments/09_kuairec_listwise/h1_spread200_20261031/delivery/stage09_h1_spread200_r1.tar.gz.partial_r1。大小513146880，其SHA256为bd169bec84bc5b80ffbfa98c540832e63e1a96aa07785f4eb947ecf9a0d3f0ba，与远端已验完整D归档前513146880字节相同；删除前本地再次SHA一致，删除后Test-Path为False。

远端共享根：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09；每个文件的具体远端路径见REMOTE与DELETED收据。

Stage06/07未删除：本轮尚未确认新上传的完整位置与完成收据，在nancy global_user根maxdepth6查询两个代表分卷文件名未找到；该有限查询不证明其他目录中没有备份。必须定位新上传目标、确认所有分卷和恢复元数据并核验后才能处理。

通过现有env-zzl CPU入口只读核验，未创建或关闭他人/其他任务资源，未修改远端实验产物。访问时仅为命令绕过失效代理；持久网络配置未更改。
