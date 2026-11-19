# 待正式预算批准后使用的命令

状态：尚未执行。不得把本文当作预算授权；先读取 `PROPOSED_RESOURCE_PLAN.json` 和累计 GPU 台账。运行代码继续固定于主仓库 `1c423662d9a25a926328ee6466c30c0f2857b4f6`，不得把短测权重用于正式暖身。

各 Job 都只暴露其获配的1张 H100；两种暖身并行，分支最多4个并行。提交前重查同名Job和实际输出目录，已有状态必须审查后恢复，不能重复启动。沿用已验收的镜像、只读Python环境、离线配置和确定性设置。

```bash
R=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1
PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
CFG="$R/config/efficiency_r2.json"
SUBSET="$R/code/experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json"
cd "$R/code"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$R/tmp" HF_HOME="$R/cache/huggingface"
```

正式使用前核对 CFG SHA `d8e3a40f9c7ac446dbca6f4e8b01811ee4598c0d4d07e18310f524afda9bd2bb`；SUBSET SHA `c74343799024ea0ca0bdb806ab482f362ff227880a2676cf1541a3511a7942af`，后者已随冻结源码部署，不能重新抽样或用前1024条代替。

## 两份共同暖身

`ARCH`分别为`J`、`H`，在两个独立单卡Job执行：

```bash
"$PY" -u -B -m scripts.listwise_diffgrm.run \
  --config "$CFG" --architecture "$ARCH" --seed 2024 \
  warmup --device cuda:0 --precision bf16 \
  --output "$R/runs/warmup_$ARCH"
```

暖身须完整完成5R和R5验证，检查3465次R更新、T=0、checkpoint完整。其对应分支才能启动。

## 四个分支

`ARM`分别为`J0/J1/H0/H1`，`ARCH`为其首字母，各占一张卡：

```bash
"$PY" -u -B -m scripts.listwise_diffgrm.run \
  --config "$CFG" --architecture "$ARCH" --seed 2024 \
  train --device cuda:0 --precision bf16 --arm "$ARM" \
  --shared-warmup "$R/runs/warmup_$ARCH/last.pt" \
  --output "$R/runs/$ARM"
```

检查配对initial的推荐器、有效码本、SID、R优化器/scheduler/RNG/进度一致，T优化器共同为空；每个最终模型累计35R、400T，不能再追加35R。结束后核对所有四组best并冻结，再进行测试。

## 最终评估

每臂分别运行下面的全测试，再在同一best、同一固定验证子集上分别取`BEAM=32`和`256`；不同臂最多4卡并行。

```bash
"$PY" -u -B -m scripts.listwise_diffgrm.run \
  --config "$CFG" --architecture "$ARCH" --seed 2024 \
  evaluate --device cuda:0 --precision bf16 \
  --checkpoint "$R/runs/$ARM/best.pt" --split test --beam 32 \
  --output "$R/evaluations/${ARM}_test_beam32"

"$PY" -u -B -m scripts.listwise_diffgrm.run \
  --config "$CFG" --architecture "$ARCH" --seed 2024 \
  evaluate --device cuda:0 --precision bf16 \
  --checkpoint "$R/runs/$ARM/best.pt" --split validation --beam "$BEAM" \
  --subset-manifest "$SUBSET" \
  --output "$R/evaluations/${ARM}_validation_subset_beam${BEAM}"
```

CLI中的bf16用于匹配checkpoint训练状态；生成实际使用FP32。实际Job需配套日志、平台超时、累计预算和恢复管理；本文件中的命令不提供独立计费保护。首轮完整验证后依据实测重新估算余额，不因某组指标较好而改变训练量。
