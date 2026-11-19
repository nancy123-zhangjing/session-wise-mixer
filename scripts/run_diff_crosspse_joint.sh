#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
seed="${2:-2024}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/diff_formal_common.yaml"
formal_config="${project_root}/configs/diff_formal_repo_v6.yaml"
joint_config="${project_root}/configs/diff_formal_crosspse_joint_sia.yaml"
smoke_config="${project_root}/configs/diff_joint_smoke.yaml"
console_dir="${project_root}/outputs/console"
run_id="diff_formal_crosspse_joint_sia_4x256_seed2024"
baseline_run_id="diff_formal_crosspse_4x256_v3_catalog_bf16"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled
mkdir -p "${console_dir}"

find_recommender_checkpoint() {
  cd "${project_root}"
  "${python_exe}" scripts/find_diff_checkpoint.py \
    --run-id "${baseline_run_id}" \
    --seed "${seed}"
}

run_one() {
  local kind="$1"
  local extra=()
  local selected_run_id="${run_id}"
  if [[ "${kind}" == "smoke" ]]; then
    extra=(--config "${smoke_config}")
    selected_run_id="diff_smoke_crosspse_joint_sia_4x256_seed${seed}"
  fi
  local log_path="${console_dir}/diff_${kind}_crosspse_joint_sia_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
  local recommender_checkpoint
  recommender_checkpoint="$(find_recommender_checkpoint)"
  echo "[RUN] kind=${kind} seed=${seed} run_id=${selected_run_id}"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${formal_config}" \
    --config "${joint_config}" \
    "${extra[@]}" \
    --joint_recommender_checkpoint="${recommender_checkpoint}" \
    --run_id="${selected_run_id}" \
    --rand_seed="${seed}" \
    2>&1 | tee "${log_path}"
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    recommender_checkpoint="$(find_recommender_checkpoint)"
    "${python_exe}" scripts/test_joint_cross_pse.py
    "${python_exe}" scripts/preflight_diff_crosspse_joint.py \
      --recommender-checkpoint "${recommender_checkpoint}"
    ;;
  smoke|formal)
    run_one "${stage}"
    ;;
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py --run-id "${run_id}"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | smoke | formal | summarize" >&2
    exit 2
    ;;
esac
