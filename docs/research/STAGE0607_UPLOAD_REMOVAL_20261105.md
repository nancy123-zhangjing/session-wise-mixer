# Stage06/07 未完成远端上传清理

2026-11-05，用户要求删除本次未完成的 Stage06/07 远端上传副本，暂停本轮迁移。

远端根：`/inspire/hdd/global_user/<inspire-account>/nancy/`。

已删除 Stage06 `archive_20261105/` 内三个文件：

| 文件 | 字节数 |
|---|---:|
| checkpoints-01.tar.gz.uploading | 1584215040 |
| exports-01.tar.gz | 623246079 |
| evidence-01.tar.gz | 76535462 |

删除后重新列举 Stage06/07 目录，确认无文件残留。Stage07 本次未发现已上传文件。所有本地原件保留，本次未删除本地文件；不再将上述远端位置视为有效备份。其他实验远端结果未改动。
