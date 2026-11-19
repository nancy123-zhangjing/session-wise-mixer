# 本轮多 seed 资源占用核算

本轮共6次单卡H100分配（2次smoke、4次正式训练），累计 **28,696 GPU秒 = 7.971111 GPU小时**。六次均退出码0，并观察到Pod删除释放；保存的提交与事件记录中没有重试。

按每次 `Scheduled` 到 `SuccessfulDeletePod` 计算；表中时间均为北京时间。包含启动、训练/验证和退出后的释放等待，不包含调度前排队。

| 任务 | 开始分配 | Pod释放 | 分配秒数 | GPU小时 |
|---|---|---|---:|---:|
| 2025 smoke（A/B串行） | 2026-10-21 23:11:49 | 2026-10-21 23:15:29 | 220 | 0.061111 |
| 2026 smoke（A/B串行） | 2026-10-21 23:11:45 | 2026-10-21 23:15:54 | 249 | 0.069167 |
| 2025 A 原版 | 2026-10-21 23:17:28 | 2026-10-22 00:58:42 | 6074 | 1.687222 |
| 2025 B 商品内交互 | 2026-10-21 23:17:21 | 2026-10-22 01:21:20 | 7439 | 2.066389 |
| 2026 A 原版 | 2026-10-21 23:17:24 | 2026-10-22 01:21:29 | 7445 | 2.068056 |
| 2026 B 商品内交互 | 2026-10-21 23:17:32 | 2026-10-22 01:18:41 | 7269 | 2.019167 |

smoke合计 469 秒（0.130278 GPU小时）；正式任务合计 28227 秒（7.840833 GPU小时）。本轮观察值低于14 GPU小时执行上限；旧实验的28小时预算不计入。

**这是观察到的资源占用量，不是实际账单或结算金额。** 本次没有使用平台计费明细或单价。

事件文件中的平台内部标识已脱敏，单任务JSON也不含任务名。任务对应关系来自本次按任务名查询后保存的文件映射，并用具名提交回执、单卡H100状态快照、具名组合事件逐条匹配印证。部分状态快照早于退出，最终完成与释放以随后完整事件中的成功退出和Pod删除为准。没有额外执行平台查询。

完整逐事件依据、查询对应关系及所有输入文件的SHA256见 [RESOURCE_USAGE.json](RESOURCE_USAGE.json)。主要证据：

| 证据文件 | SHA256 |
|---|---|
| [SMOKE_SUBMISSION.json](SMOKE_SUBMISSION.json) | `9a3affc16b1ea60f1243dc2551a678ce62f1ac6eb912396d453e07e1d5af2fa0` |
| [FORMAL_SUBMISSION.json](FORMAL_SUBMISSION.json) | `ff2b891c004bee6297ac5d523baaf56fd063bc3bcdb728604166d9f51d5adbea` |
| [SMOKE_STATUS_2315.json](SMOKE_STATUS_2315.json) | `70e3936cde2f6db13e12dceff1675733cf4589330e9ba32735b28f114c892523` |
| [STATUS_20261022_012112.json](STATUS_20261022_012112.json) | `f4f1b81661d6bdf7924c9dd379997285457344dc9e9c6fbf791e5cdc4c2900fd` |
| [SMOKE_DRY_RUN_LIVE.json](SMOKE_DRY_RUN_LIVE.json) | `c04173decdc089a3e146fa9721e2fc7580a3884f9286ea283611f3ddc1308a66` |
| [FORMAL_DRY_RUN_LIVE.json](FORMAL_DRY_RUN_LIVE.json) | `dc6e2aeb23d68fc070815d3cdad6a3f2c290200b344c2ed14ef4e0b576f30287` |
| [SMOKE_EVENTS_INITIAL.json](SMOKE_EVENTS_INITIAL.json) | `6e468ede4ced0a0ca74e314a91c5939748367da104d7cfb2bfb8152b415b2811` |
| [FORMAL_EVENTS_INITIAL.json](FORMAL_EVENTS_INITIAL.json) | `ec0945a053d17de6742c676fbbc6d7f62de6dd8ea760ebab89914203b261cfba` |
| [SMOKE_EVENTS_TERMINAL.json](SMOKE_EVENTS_TERMINAL.json) | `8c22117a5f2d14e4cb6c6c1bf1738bad1a2200ea9f4e99f8e4d9147ba6ae4bef` |
| [SMOKE_FULL_EVENTS_2025.json](SMOKE_FULL_EVENTS_2025.json) | `f1189c45d5f9988b826d47a913f740e7b7710f82a5c85ef48455cefb1cbd58aa` |
| [SMOKE_FULL_EVENTS_2026.json](SMOKE_FULL_EVENTS_2026.json) | `383610a1790593374645a04fe3bd63fc0a0014887b5247916ff521359c806025` |
| [FORMAL_FINAL_EVENTS_2025_control.json](FORMAL_FINAL_EVENTS_2025_control.json) | `ece5636ec664dfe978d01dc01fd2ccca009189acbb2099ecb986bbb673fd5263` |
| [FORMAL_FINAL_EVENTS_2025_item_local.json](FORMAL_FINAL_EVENTS_2025_item_local.json) | `3a2e7b757a00543dab79cce506a36e92c7a337e05b5aec2f5a6b9879a717fd0e` |
| [FORMAL_FINAL_EVENTS_2026_control.json](FORMAL_FINAL_EVENTS_2026_control.json) | `31c033a82ce64cb1fa2c8e2b704fab6cfb8cc287a0a6a4e0096fa09bfb4a09ef` |
| [FORMAL_FINAL_EVENTS_2026_item_local.json](FORMAL_FINAL_EVENTS_2026_item_local.json) | `c821d453fffc2f2d73e92d85637e8b55c54d14f200d02a3969dc87628f0aa227` |
