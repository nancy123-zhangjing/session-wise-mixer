#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
seed="${2:-2024}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/diff_formal_common.yaml"
pse_config="${project_root}/configs/diff_formal_pse.yaml"
formal_config="${project_root}/configs/diff_formal_repo_v6.yaml"
smoke_config="${project_root}/configs/diff_smoke.yaml"
profile_config="${project_root}/configs/diff_profile.yaml"
console_dir="${project_root}/outputs/console"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${console_dir}"

run_one() {
  local run_kind="$1"
  local run_seed="$2"
  local log_path="${console_dir}/diff_${run_kind}_pse_seed${run_seed}_$(date +%Y%m%d_%H%M%S).log"
  local extra_config=()
  if [[ "${run_kind}" == "smoke" ]]; then
    extra_config=(--config "${smoke_config}")
  elif [[ "${run_kind}" == "profile" ]]; then
    extra_config=(--config "${profile_config}")
  fi

  echo "[RUN] kind=${run_kind} tokenizer=PSE seed=${run_seed}"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${pse_config}" \
    --config "${formal_config}" \
    "${extra_config[@]}" \
    --rand_seed="${run_seed}" \
    2>&1 | tee "${log_path}"
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    "${python_exe}" scripts/preflight_diff_formal.py
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
      "${python_exe}" scripts/analyze_diff_cpd_attention.py --seed "${run_seed}"
    done
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py --require-all-seeds
    ;;
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py
    ;;
  archive-pilot)
    cd "${project_root}"
    "${python_exe}" scripts/archive_diff_pilot.py
    ;;
  attention)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py --seed "${seed}"
    ;;
  attention-smoke)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed "${seed}" --max-samples 128 --batch-size 32
    ;;
  attention-pilot-smoke)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed 2024 --run-id diff_formal_pse_4x256 \
      --max-samples 128 --batch-size 32
    ;;
  attention-v2)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id diff_formal_pse_4x256_schedfix_v2
    ;;
  attention-v3)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id diff_formal_pse_4x256_schedfix_v3_bf16
    ;;
  attention-v4)
    cd "${project_root}"
    "${python_exe}" scripts/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id diff_formal_pse_4x256_officialbatch_v4_bf16
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | smoke | profile | formal | all-seeds | summarize | archive-pilot | attention | attention-smoke | attention-pilot-smoke | attention-v2 | attention-v3 | attention-v4" >&2
    exit 2
    ;;
esac
