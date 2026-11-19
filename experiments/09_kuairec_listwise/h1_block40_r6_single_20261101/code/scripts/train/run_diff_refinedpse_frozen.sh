#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
seed="${2:-2024}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/shared/diff_formal_common.yaml"
formal_config="${project_root}/configs/shared/diff_formal_repo_v6.yaml"
joint_config="${project_root}/configs/05_crosspse_joint/diff_formal_refinedpse_joint_diger_v1.yaml"
control_config="${project_root}/configs/05_crosspse_joint/diff_formal_refinedpse_frozen_decoder_v1.yaml"
smoke_config="${project_root}/configs/05_crosspse_joint/diff_frozen_refinedpse_smoke.yaml"
console_dir="${project_root}/outputs/console"
run_id="diff_formal_refinedpse_frozen_decoder_continuation_4x256_seed2024"
smoke_run_id="diff_smoke_refinedpse_frozen_decoder_continuation_4x256_seed${seed}"
baseline_run_id="diff_formal_refinedpse_4x256_v2_catalog_bf16"

if [[ "${seed}" != "2024" ]]; then
  echo "Frozen Refined-PSE currently uses seed-2024 artifacts; requested seed=${seed}." >&2
  echo "Run seed 2024 for the matched comparison." >&2
  exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled
mkdir -p "${console_dir}"

find_recommender_checkpoint() {
  cd "${project_root}"
  "${python_exe}" scripts/checks/find_diff_checkpoint.py \
    --run-id "${baseline_run_id}" \
    --seed "${seed}"
}

run_one() {
  local kind="$1"
  local extra=()
  local selected_run_id="${run_id}"
  if [[ "${kind}" == "smoke" ]]; then
    extra=(--config "${smoke_config}")
    selected_run_id="${smoke_run_id}"
  fi
  local log_path="${console_dir}/diff_${kind}_refinedpse_frozen_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
  local recommender_checkpoint
  recommender_checkpoint="$(find_recommender_checkpoint)"
  echo "[RUN] kind=${kind} seed=${seed} run_id=${selected_run_id}"
  echo "[CONTROL] Refined-PSE codebooks and hard SIDs frozen; Decoder only"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${formal_config}" \
    --config "${joint_config}" \
    --config "${control_config}" \
    "${extra[@]}" \
    --joint_recommender_checkpoint="${recommender_checkpoint}" \
    --run_id="${selected_run_id}" \
    --rand_seed="${seed}" \
    2>&1 | tee "${log_path}"
  if [[ "${kind}" == "smoke" ]]; then
    cd "${project_root}"
    "${python_exe}" scripts/checks/check_diff_refinedpse_frozen_smoke.py \
      --run-id "${smoke_run_id}" --seed "${seed}"
  fi
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    recommender_checkpoint="$(find_recommender_checkpoint)"
    "${python_exe}" tests/test_diff_refinedpse_diger.py
    "${python_exe}" scripts/checks/preflight_diff_refinedpse_frozen.py \
      --recommender-checkpoint "${recommender_checkpoint}"
    ;;
  smoke|formal)
    run_one "${stage}"
    ;;
  check-smoke)
    cd "${project_root}"
    "${python_exe}" scripts/checks/check_diff_refinedpse_frozen_smoke.py \
      --run-id "${smoke_run_id}" --seed "${seed}"
    ;;
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/summarize_diff_formal.py --run-id "${run_id}"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | smoke | check-smoke | formal | summarize" >&2
    exit 2
    ;;
esac
