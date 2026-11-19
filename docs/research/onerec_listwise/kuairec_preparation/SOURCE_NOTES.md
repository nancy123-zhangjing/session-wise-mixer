# KuaiRec 原始来源核验

核验日期：2026-10-27。本轮复用本地两份原始文件，没有重新下载，也没有修改原文件。完整校验记录见 [SOURCE_AUDIT.json](SOURCE_AUDIT.json)。

## 来源与版本

[作者主页](https://kuairec.com/)和[官方仓库](https://github.com/chongminggao/KuaiRec)均直接指向 [Zenodo 18164998](https://zenodo.org/records/18164998)。该记录发布于 2026-01-06，记录版本为 v1；压缩包内部目录为 KuaiRec 2.0。二者是不同层级的版本名，后续应引用记录 DOI 和文件哈希，不能只写“v1”。

| 原始文件 | 字节数 | 与 Zenodo 公布 MD5 相符 |
|---|---:|---|
| KuaiRec.zip | 431,964,858 | 是 |
| kuairec_caption_category.csv | 1,964,578 | 是 |

ZIP 的 15 个成员均通过全量读取和 CRC32 检查。独立文本 CSV 与 ZIP 内同名文件的 SHA256 相同。SHA256、成员路径和可复核的官方下载地址均保存在机器记录中。以上验证的是发布文件一致性；不表示其中每行都没有语义问题。

## 字段与使用边界

行为入口为 `KuaiRec 2.0/data/big_matrix.csv`。实读表头没有额外 CSV 行号，原始行位置需由处理脚本另行保留。官方说明播放时长和视频时长单位均为毫秒，时间戳是带小数的 Unix 秒；观看比例可以超过 1，也没有原生点赞标签。全量小数精度与异常行仍由处理流水线逐行验收。[字段说明](https://kuairec.com/)

作者已在 [Issue #2](https://github.com/chongminggao/KuaiRec/issues/2)回应重复记录问题，并解释保留已有版本是为了避免使用者之间版本不一致。因此去重属于显式预处理，不能把重复数据误报成下载损坏。

[论文](https://hexiangnan.github.io/papers/cikm22-kuairec.pdf)说明 big_matrix 排除了 small_matrix 对应的用户商品交互。这里得到的是公开观察轨迹，不能声称完整自然浏览过程。本项目只用 big_matrix，人工会话与五件块也不等于真实请求 ID。

商品文本是在行为数据之后补发的，未证明是每个行为时刻可见的历史快照。使用完整静态目录时必须保留这一离线假设；不引入全时期热度、播放统计等行为汇总。ZIP 内同时保留 CC BY-SA 4.0 许可文件。
