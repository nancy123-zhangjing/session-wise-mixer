# 测试入口

本目录区分复现工具安全回归与模型功能测试。训练、预检和分析入口仍在 [`scripts/`](../scripts/README.md)；训练配置已在 [`configs/`](../configs/README.md) 内按共享设置和阶段分组。

## 本机标准库回归

在项目根目录运行：

```text
python -B tests/run_repro_checks.py
```

也可在任意工作目录用 runner 的绝对路径运行。runner 使用当前 Python，以独立的 `python -B` 子进程依次执行下面六个文件；遇到失败立即返回非零退出码，不会自动加载或执行模型测试。`--list` 只列出入口，`--help` 查看说明。

| 测试 | 检查内容 | B2a 迁移前用例数 |
| --- | --- | ---: |
| [`test_asset_locations.py`](test_asset_locations.py) | 资产位置与锁定身份分离、恢复保护 | 10 |
| [`test_daily_repro.py`](test_daily_repro.py) | 日常复现禁止读取旧归档；临时副本重算一致性 | 2 |
| [`test_legacy_import.py`](test_legacy_import.py) | 旧导入入口保护、禁止隐式重建封存包 | 15 |
| [`test_project_layout.py`](test_project_layout.py) | 实验包布局、索引和受保护字段 | 7 |
| [`test_project_repro.py`](test_project_repro.py) | 01–04 阶段来源、分析与非破坏性回放保护 | 31 |
| [`test_stage05_repro.py`](test_stage05_repro.py) | 05 阶段安全解包、历史结果与回放保护 | 13 |

这六个文件合计原有 78 项用例，只需 Python 标准库和仓库内已封存的实验包；不需要 torch、模型下载或 GPU。真实实验包只读，测试生成物放在临时目录，daily 测试的重算仅作用于临时副本。它们不启动训练，也不证明训练环境、数据输入或 GPU 已就绪，更不证明旧归档可删除。

## 本批目录迁移检查

另有 [`test_entrypoint_layout.py`](test_entrypoint_layout.py) 的 9 项测试/文档迁移检查，以及 [`test_config_layout.py`](test_config_layout.py) 的 12 项配置迁移检查。不混入上述 78 项计数；另加本轮 [`test_script_layout.py`](test_script_layout.py) 检查全部脚本归类、根目录定位和辅助锁绑定：

```text
python -B tests/test_entrypoint_layout.py
python -B tests/test_config_layout.py
python -B tests/test_script_layout.py
python -B scripts/audit/audit_script_layout.py verify --check-receipt
```

本次实际通过114项标准库测试：基础78项、B2a的9项、B2b的12项、B2c的15项。旧检查先撤销本批已独立验证的脚本/配置地址变换，再检查原约束；新检查覆盖路径遗漏、根目录、互调、seal保护等。配置字节、137组配置调用引用及顺序保持不变，错序/参数变化会被拒绝。

当前迁移验收见 [B2c记录](../reproducibility/layout_migrations/20261008_b2c_scripts/README.md)。旧B2a/B2b测试显式使用迁移前快照或撤销批准的路径变化来检验历史约束；当前文件是否只改变了允许内容由B2c另行校验，不能拿旧收据当新布局收据。旧 [B2a记录](../reproducibility/layout_migrations/20261008_b2a_tests_docs/README.md) 和 `audit_entrypoint_layout.py` 保留对应时点，不直接用于验收新配置布局。

新增短实验预算/输入绑定/每批模式保护测试：`python -B tests/test_diger_noise_pilot.py`，40 项标准库测试，不启动训练、不需要 GPU。

推荐反馈短对照另见 [实验说明](../experiments/05_crosspse_joint/pilots/feedback_update/README.md)：`python -B tests/test_diger_feedback_protocol.py` 校验三组协议；`python -B tests/test_diger_feedback_runner.py` 检查运行保护，有 Torch/NumPy 时还执行合成梯度和 SID probe 测试。服务器须加 `--require-torch`，避免依赖缺失时跳过张量验收；该检查不用 GPU，也不能替代真实三组 GPU smoke。

`test_script_layout.py` 与旧 B2a/B2b/B2c 审计绑定目录迁移时点的固定清单和字节，不是持续科研开发的全绿门槛。后续新增入口或算法开发会超出其冻结清单；不要修改历史收据或封存锁来掩盖差异。当前开发应另做新入口测试、历史复现安全回归和真实模型 smoke。

## 模型功能测试（单独运行）

| 测试 | 环境要求 |
| --- | --- |
| [`test_cross_pse_module.py`](test_cross_pse_module.py) | torch；直接加载 Cross-PSE 模块，可用 CPU |
| [`test_joint_cross_pse.py`](test_joint_cross_pse.py) | torch；交替梯度路由检查，可用 CPU |
| [`test_ar_variable_codebooks.py`](test_ar_variable_codebooks.py) | DiffGRM 训练依赖；有 CUDA 时优先使用，否则 CPU |
| [`test_collision_aware_eval.py`](test_collision_aware_eval.py) | DiffGRM 训练依赖；CPU 功能检查 |
| [`test_diff_crosspse_diger.py`](test_diff_crosspse_diger.py) | DiffGRM 训练依赖；CPU 硬前向/软反向梯度检查 |
| [`test_diff_refinedpse_diger.py`](test_diff_refinedpse_diger.py) | DiffGRM 训练依赖；CPU 关闭 mixer 的梯度检查 |
| [`test_diger_sid_consistency.py`](test_diger_sid_consistency.py) | NumPy、torch；25 项 CPU 诊断测试，隔离加载距离和采样模块，不启动训练 |

后四项的包级导入会加载 `genrec` 的模型、tokenizer 和 trainer，不能仅凭安装 torch/numpy 就认定依赖齐备。应使用项目对应的训练环境；不要为整理目录改动训练算法或配置。

模型测试保留各自的脚本入口，例如：

```text
python -B tests/test_cross_pse_module.py
```

多项模型断言位于 `main()` 中，因此 `unittest discover` 不能替代逐文件脚本执行。标准库回归通过不等于这些模型测试通过，也不代表完成 GPU smoke 或正式复现。

## 封存边界

`test_project_repro.py` 和 `test_stage05_repro.py` 是实验包校验绑定的 helper。路径和导入调整通过 B2a/B2c 迁移审计登记；不要通过重建整个封存包消除校验失败。历史来源锁、旧迁移收据及归档中的 `scripts/test_*.py` 路径记录原样保留。
