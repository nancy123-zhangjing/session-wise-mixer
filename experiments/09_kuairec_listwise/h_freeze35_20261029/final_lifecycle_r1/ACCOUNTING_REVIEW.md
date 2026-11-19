# 最终 GPU 释放与用量复核

原短测及 A/B/C 四次已知单卡 H100 分配均有 SUCCEEDED、退出码 0 和 SuccessfulDeletePod 证据。
累计保守分配口径为 **78,518 GPU 秒 = 21.810555556 GPU 小时**，含短测 219 秒，低于累计 30 GPU 小时上限；算术余量 8.189444444 小时不构成新作业授权。

最新正式作业状态查询区间：2026-10-30T02:32:17.7910114Z 至 2026-10-30T02:32:19.8891220Z（北京时间 2026-10-30 10:32:19 返回）。事件文本时间按 Asia/Shanghai +08:00 解释。

| 分配 | Scheduled（北京时间） | Pod 删除 | 原次收尾边界 | 计入 GPU 秒 | GPU 小时 |
|---|---|---|---|---:|---:|
| preflight | 2026-10-29 13:55:37 | 2026-10-29 13:59:12 | 2026-10-29 13:59:16 | 219 | 0.060833333 |
| A | 2026-10-29 14:07:30 | 2026-10-29 19:03:01 | 2026-10-29 19:03:01 | 17,731 | 4.925277778 |
| B | 2026-10-29 14:07:30 | 2026-10-29 16:03:28 | 2026-10-29 16:03:38 | 6,968 | 1.935555556 |
| C | 2026-10-29 14:07:30 | 2026-10-30 05:00:43 | 2026-10-30 05:00:50 | 53,600 | 14.888888889 |

口径说明：以保存的真实 Scheduled 事件作为分配起点，保留同一次退出、删 Pod 和收尾的保守边界。短测、A、B 既有秒数保持不变；C 为 53,600 秒。该口径包括 Scheduled 到退出收尾的全部间隔；未因中途网络不可见而减扣耗时。四次分配的证据均只出现一次建 Pod，没有可见的失败重建或重提分配；审计范围是这四个明确命名作业，不是整个平台账号的所有资源。

B 在 10 月 29 日 16:03:19 成功退出，16:03:28 删除 Pod，16:03:38 已完成该次收尾。新查询中又出现 18:27:11 和次日 04:18:00 的 JobTerminated（后者伴随 PodGroup 删除），但没有新的 Pod 创建、Scheduled 或 Started，且 platform finished_at 仍是 16:03:19。它们不证明 GPU 重新分配，因此不把任意最新控制器事件当成计费终点。原台账“LAST JobTerminated”文字现已收紧为“原次同时段收尾完成”，以免误读。

C 在 10 月 30 日 05:00:36 平台标记成功，05:00:43 删除 Pod，05:00:50 同次重复收尾结束。保守核算多计删除后的 7 秒；它与 B 数小时后没有新 Pod 的控制器重报不混为一谈。

| 替代时间口径（四次合计） | 秒 | 小时 |
|---|---:|---:|
| Scheduled 到实际 Pod 删除 | 78,497 | 21.804722222 |
| Scheduled 到原次收尾（主台账） | 78,518 | 21.810555556 |
| Scheduled 到 platform finished_at | 78,478 | 21.799444444 |
| platform created_at 到 finished_at | 78,506 | 21.807222222 |

以上都不是已核实的账单。按先前记录的 1 点/H100 小时仅可估算约 21.810556 点；未获得可归属的实际结算记录，settlement_verified 保持 false。资源释放通过不等于模型、逐样本测试、完整备份或恢复已验收。

原始证据按字节复制到本目录，上传清单为 UPLOAD_MANIFEST.json。RELEASE.json 的 evidence 使用指定远端绝对路径；当前只准备本地文件，根任务必须上传后逐 SHA 核验才能打包。GPU_LEDGER_BEFORE_FINAL.json 保留更新前原字节，最新台账另保留 prior_incomplete_accounting_snapshot。

| 证据文件 | SHA-256 |
|---|---|
| STATUS.json | 268522350068da341b9a349cd6b943726a3b8f9e26910e70be433eb0696cda93 |
| STATUS_QUERY.json | 54db0535749f87e3b3495e3526da753f6066bb9703a788d6fea6cf766aced99d |
| EVENTS.json | 49b643799aa1c04ba75fa252cca3aa605b14446ca273d333fce4971d3ef27fd3 |
| PREFLIGHT_EVENTS.json | d29972bf1b5d56fa1de6378c433f3ff9a59b3d45e0f7337ea67e77bc12a3df26 |
| PREFLIGHT_STATUS.json | b530e8b98a4e40c0bc117d7d048da84e16cf0562a5c8db58cc1c2ad88c10468c |
| FORMAL_ALLOCATION_EVENTS.json | 2dc32bc14ff3015166f337285f82d612a2769f55caf48edc7cc5095e3d27dee2 |
| A_RELEASE_EVENTS.json | 07be8802a7476ca7adc50e4a8ba2a1fdd8617136251b4ec7d8c929c404430cde |
| B_RELEASE_EVENTS.json | 6911fdf5b15e95a0dc40d3df38bf35fde4613fa721f3a43ad20828874e4bef9c |
| STATUS_20261029T1605.json | 96b2c33874ca7e60af49f98868b362c704b717c4bcce554538ec34ff150c1326 |
| STATUS_20261029T190527.json | b52f41bd8076fcdbd4d915648872fb09e1ea32db73082a583d007463db9365a9 |

更新后 GPU_LEDGER.json SHA-256：`61f6659d87ce076bc9630e4ba431a8d11d5e0e6b44a5de66d42da3a630d75580`。
RELEASE.json SHA-256：`93e795708ceb548afdf05960ad5bcd22e79f666f846578bd59a0c360b9fdd81a`。
