#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
seed="${2:-2024}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/diff_formal_common.yaml"
formal_config="${project_root}/configs/diff_formal_repo_v6.yaml"
rqk_config="${project_root}/configs/diff_formal_rqkmeans_v6.yaml"
smoke_config="${project_root}/configs/diff_smoke.yaml"
profile_config="${project_root}/configs/diff_profile.yaml"
console_dir="${project_root}/outputs/console"
run_id="diff_formal_rqkmeans_4x256_v6_bf16"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${console_dir}"

run_one() {
  local run_kind="$1"
  local run_seed="$2"
  local log_path="${console_dir}/diff_${run_kind}_rqkmeans_seed${run_seed}_$(date +%Y%m%d_%H%M%S).log"
  local extra_config=()
  local run_id_override="${run_id}"
  if [[ "${run_kind}" == "smoke" ]]; then
    extra_config=(--config "${smoke_config}")
    run_id_override="diff_smoke_rqkmeans_4x256"
  elif [[ "${run_kind}" == "profile" ]]; then
    extra_config=(--config "${profile_config}")
    run_id_override="diff_profile_rqkmeans_4x256"
  fi

  echo "[RUN] kind=${run_kind} tokenizer=RQ-KMeans seed=${run_seed}"
  echo "[RUN] tokenizer_seed=2024 preprocessing=PCA256"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${formal_config}" \
    --config "${rqk_config}" \
    "${extra_config[@]}" \
    --run_id="${run_id_override}" \
    --rand_seed="${run_seed}" \
    2>&1 | tee "${log_path}"
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    "${python_exe}" scripts/preflight_diff_formal.py --tokenizer rq_kmeans
    ;;
  smoke)
    run_one smoke "${seed}"
    ;;
  profile)
    run_one profile "${seed}"
    ;;
  formal)
    run_one formal_v6 "${seed}"
    ;;
  all-seeds)
    for run_seed in 2024 2025 2026; do
      run_one formal_v6 "${run_seed}"
      cd "${project_root}"
      "${python_exe}" scripts/analyze_diff_cpd_attention.py \
        --seed "${run_seed}" --run-id "${run_id}"
    done
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py \
      --run-id "${run_id}" --require-all-seeds
    ;;
  remaining-seeds)
    for run_seed in 2025 2026; do
      run_one formal_v6 "${run_seed}"
      cd "${project_root}"
      "${python_exe}" scripts/analyze_diff_cpd_attention.py \
        --seed "${run_seed}" --run-id "${run_id}"
    done
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py \
      --run-id "${run_id}" --require-all-seeds
    ;;
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py --run-id "${run_id}"
    ;;
  compare)
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py \
      --run-id diff_formal_pse_4x256_repo_repro_v6_bf16
    "${python_exe}" scripts/summarize_diff_formal.py --run-id "${run_id}"
    "${python_exe}" scripts/compare_diff_tokenizers.py
    ;;
  attention)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id "${run_id}"
    ;;
  attention-smoke)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id "${run_id}" \
      --max-samples 128 --batch-size 32
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | smoke | profile | formal | all-seeds | remaining-seeds | summarize | compare | attention | attention-smoke" >&2
    exit 2
    ;;
esac
