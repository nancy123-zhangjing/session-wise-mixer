# H1 去除中心编号标记：代码修改说明

状态：2026-11-08，本地代码与合成 CPU 检查完成；没有启动实验、访问远端或改写既有实验封存源码。

## 本次问题

单独检验跨码本共享的中心编号标记是否有正向价值，不将本次消融定义为解决后期崩塌。未来可沿用已经验证的冻结策略保存训练成果；本次尚未确定冻结节点或实验终点。

## 唯一结构变化

原输入 `base[m,k] + book_embedding[m] + code_embedding[k]`，新输入 `base[m,k] + book_embedding[m]`。保留码本身份、基础码本、LayerNorm、四头全注意力、残差、推荐器及原 R/T 训练逻辑。

配置：`configs/09_kuairec_listwise/h1_no_code_tag.json`。

```json
"alignment": {
  "crosspse_code_embedding_enabled": false
}
```

该文件与当前 base.json 只差这一开关；沿用的 batch/路径等仍是本地基础配置，并非新增的正式平台执行配方。原配置缺省为 true，行为不变。新配置需要与 H1（Mixer 开启）配套使用。

实现选择：保留 code_embedding 参数字段及优化器参数组位置以兼容权重布局，但关闭时不读取该参数、不产生梯度，R/T 切换也不会重新启用它。即使加载的权重中此字段非零，也不会影响输出。book_embedding 正常参与 T 学习。有效可训练参数减少 16384 个，但 checkpoint 文件仍保留该占位字段。

## 保存与恢复边界

开关存入现有完整 config/binding，完整 checkpoint 恢复继续严格检查代码、配置、数据和资产绑定，不绕过旧实验校验。单独 mixer.state_dict 不包含 Python 布尔开关，独立加载该权重时必须显式使用对应配置。权重布局兼容不代表可以用新配置直接 resume 旧实验。

未来若从原 H1 R5/T0 做成对分叉，需要显式记录原 checkpoint 来源及本次允许的代码/配置差异；本次没有编写或执行旧 checkpoint 的跨版本迁移。不要在已训练的 R20 上临时关掉标记，就将结果解释为从头不使用标记的消融。

## 本地验证

使用已有 CPU 环境 `.runtime/stage08_restore_env/Scripts/python.exe`；未安装依赖。

- `tests/test_listwise_no_code_tag.py`：4 项通过。覆盖配置全链路、禁止字符串 false、默认开启行为、相同初始化与 RNG、关闭后非零标记不影响输出、无编号标记梯度、其余参数梯度、严格权重加载、R/T 切换、优化器布局。
- `tests/test_cross_pse_module.py`：通过，原 Mixer 的形状、注意力模式、梯度与恒等初始化检查正常。
- 检查只使用合成张量和临时合成资产，未进行真实数据训练、评估或推理。

引擎修改位于独立 external/DiffGRM 工作树，补丁另存 `patches/h1_no_code_tag_20261108.patch`，避免仅保存主仓库时遗漏引擎变更。
