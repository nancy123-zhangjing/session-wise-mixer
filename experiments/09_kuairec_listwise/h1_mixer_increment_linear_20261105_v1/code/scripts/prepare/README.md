# prepare：脚本清单

用于embedding、fit mask、RQ-VAE输入准备，以及固定上游仓库的bootstrap。准备工具会写数据，bootstrap会拉取/切换仓库和应用patch；普通读报告和校验不需要执行它们。

所有命令从项目根目录执行，例如脚本路径写成 `scripts/prepare/文件名`。历史配方内部的旧源码路径保留原样，不手工改历史锁。

- [bootstrap_upstreams.ps1](bootstrap_upstreams.ps1)
- [bootstrap_upstreams.sh](bootstrap_upstreams.sh)
- [prepare_diff_crosspse_fit_mask.py](prepare_diff_crosspse_fit_mask.py)
- [prepare_diff_rqvae_inputs.py](prepare_diff_rqvae_inputs.py)
- [prepare_formal_embeddings.py](prepare_formal_embeddings.py)

返回 [脚本总入口](../README.md) · [项目地图](../../docs/guides/PROJECT_MAP.md)
