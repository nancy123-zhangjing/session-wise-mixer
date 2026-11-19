# 2026-11-14 01:26 存储等待观察（未解决）

本记录不是训练验收。未修改配置、未停止或重启作业、未删除资产。

- AR-T0 训练 PID 12286：Dl，wchan=cxiWaitEventWait；syscall=20（x86_64 writev），fd=41；readlink 指向 runs/AR-T0/train/last.pt.tmp。GPU最新采样0%，显存仍占用。
- AR-D0 训练 PID 44499：Dl，同一wchan；syscall=257（x86_64 openat），flags=0x241。未确认具体文件路径。GPU最新采样0%。
- AR-D1 平台最近10分钟GPU平均61.95%，最新65%，尚未观察到同类阻塞。
- CPU读取AR目录连续超时/空响应；CF文件可读。CPU Notebook平台RUNNING；简单true命令成功。
- T0平台事件仅包含启动调度记录，无新的存储诊断事件。

推断：T0/D0停顿与共享文件系统访问有关；具体服务端原因尚未确认，不能断言全平台故障。不可把last.pt.tmp视为有效checkpoint；恢复前须核验最后完整last或验证节点。暂不重启，继续核对存储恢复和其他作业。

证据：raw/T0_METRICS_READ_DELAY.json、D0_METRICS_READ_DELAY.json、D1_STORAGE_METRICS.json、T0_STORAGE_EVENTS.json；本对话只读job shell的ps、/proc/syscall、fd链接输出。内核stack读取被拒绝，未绕过权限。

## 后续观察（01:28）

D0自行恢复：训练进程从Dl变为Rl，新的完整last.pt可读，step=8600，mtime=1794590868.56；未进行重启。D1完整last.pt到13200更新。T0进程仍Dl。CF日志最后为R18后的T块，最新GPU采样0%，因此“CF文件可读”不能证明训练仍推进；需继续核对其进程。读取证据：DIGER_TRAIN_PROGRESS.json、raw/CF_STORAGE_METRICS.json。

## 01:29 更新

T0自行恢复，进程Rl，完整last.pt读取成功，step=39800，mtime=1794590922.82；D0到9200，D1到13857首轮验证边界。未重启任何训练。CF训练子进程8772当前Rl，但日志仍为R18/T140，尚需下一条训练/验证产物确认持续推进。本次可确认T0/D0文件等待已解除，不能保证平台存储不再复发。

## 01:44–01:45 更新：D1 验证预测写入等待

平台重新确认 T0、D0、D1、CF-M0 为 RUNNING，T1 为 FAILED。CF 已完成验证仍到 R19；D1 progress.json 尚不存在，不能声称首轮全验证完成。

只读进入 D1 作业：训练 PID 13837 为 Dl，wchan=cxiWaitEventWait；/proc/13837/syscall 返回 1（x86_64 write），fd=0x29=41；fd/41 指向 runs/AR-D1/train/validation/step_0013857/predictions.jsonl.gz.partial。这证明当前停顿在验证预测文件的共享盘写入，不是仅凭零字节推测缓冲。GPU 10 分钟均值 40.925%，最新 0%，显存占用不变（raw/D1_METRICS_0144.json）。CPU读取此预测临时文件也超时。

未重启或停止作业，未修改配置。首轮验证耗时应区分存储等待与计算，不使用当前墙钟等待直接估算正常验证吞吐。临时文件不能算作完成预测；等待完整 summary 和哈希后才纳入结果。

## 01:49–01:50 阻塞复核

连续三轮检查未解除同一共享盘阻塞。平台仍报告 T0/D0/D1/CF RUNNING；D1 PID 13837 仍为 Dl/cxiWaitEventWait，write(fd41) 参数与上次一致，目标仍为 step_0013857/predictions.jsonl.gz.partial。T0、D0、CF 最近10分钟 GPU 均值及最新值均0；D1最新0。新一次CPU读取D1 progress依旧无有效响应。证据为 raw/*_METRICS_RECHECK.json 和本对话只读进程检查。

目前无法验证新checkpoint或完成验收，等待共享存储恢复或平台排查；不以读取超时为终止证据，不重启作业、不删除文件、不修改模型配置。Goal 标记 blocked 不会停止远端作业，四张GPU和临时CPU仍由现有任务占用，恢复后须先复核平台状态与有效checkpoint再继续。T1科学失败仍另行保留，不擅自修订正式配置。

## 02:11–02:12 恢复Goal后的实证进展

未重启任何作业。D1首轮全验证已写出：71657条，NDCG@5=0.0010131827140747124，预测与checkpoint SHA见 AR_D1_FIRST_VALIDATION.json；验证墙钟1351.61秒含已确认的存储等待，不能直接作为正常吞吐。T0第三轮完整验证NDCG@5=0.003390505215986489，为当前最佳，见 AR_T0_THIRD_VALIDATION.json。D0首轮progress仍未出现，不能推断已完成或已恢复。

CF新增R20/R21完整验证，均71657条，NDCG@5分别0.005471460486597282/0.0055443125244448435；R20验证602.46秒，R21恢复至241.06秒。具体同轮对照见 CF_INTERIM_COMPARISON.json。上述新产物证明部分文件访问和训练已恢复，尚不保证所有挂载点持续正常。未改配置、未新增训练，Goal继续保持active。
