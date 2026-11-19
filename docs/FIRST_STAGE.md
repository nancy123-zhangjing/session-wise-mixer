# 第一阶段：Collision-aware AR-Smoke-v2

## 目标

- 保留 4 个 RQ Semantic Code。
- 增加 1 个 Collision Suffix，使完整 SID 与商品一一对应。
- 用真实 `target_item_id` 计算严格的商品级 NDCG 和 Hit Rate。
- 输出 Legal@10、Duplicate@10 和 5 个位置的 SFT Loss。

第五个位置仅用于碰撞消解，分析 Semantic Space 时必须与前四个语义位置分开。

## 服务器运行

在本地 VS Code 执行 `SFTP: Upload Project` 后，在服务器运行：

```bash
cd ~/experiment1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp
mkdir -p outputs/console
set -o pipefail
bash scripts/run_ar_smoke_v2.sh python 2>&1 | \
  tee "outputs/console/ar_smoke_v2_$(date +%Y%m%d_%H%M%S).log"
```

脚本会先运行一个小型接口测试。看到下面这行后，才会开始数据和训练流程：

```text
Collision suffix and strict item evaluator: OK
```

## 完成后分析

```bash
cd ~/experiment1
bash scripts/analyze_ar_smoke_v2.sh python
```

通过标准：

```text
semantic_sid.collision_rate 允许大于 0
full_sid.collision_rate 必须等于 0
full_sid.largest_collision_group 必须等于 1
日志必须包含 strict item-level
日志必须输出 hit_rate@5、hit_rate@10、legal@10、duplicate@10
duplicate@10 必须等于 0；legal@10 应尽量接近 1
位置 Loss 必须包含 pos1 至 pos5
```

其中，Semantic SID仍可能发生碰撞，这是研究对象；加入Collision Suffix后的Full SID必须完全唯一。
