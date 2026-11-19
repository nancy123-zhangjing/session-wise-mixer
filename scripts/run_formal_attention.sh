#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
python_exe="${2:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${project_root}/outputs/attention/formal_3x256"
console_dir="${project_root}/outputs/console"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${output_root}" "${console_dir}"
cd "${project_root}"

run_one() {
  local variant="$1"
  local seed="$2"
  local max_samples="$3"
  local target_root="${4:-${output_root}}"
  local log_path="${console_dir}/attention_${variant}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"

  "${python_exe}" "${project_root}/scripts/analyze_ar_self_attention.py" \
    --variant "${variant}" \
    --seed "${seed}" \
    --split test \
    --batch-size 32 \
    --max-samples "${max_samples}" \
    --output-root "${target_root}" \
    2>&1 | tee "${log_path}"
}

case "${mode}" in
  smoke)
    run_one rq_kmeans 2024 256
    echo "Smoke test complete. Inspect outputs/attention/formal_3x256/rq_kmeans/seed2024/run_info.json"
    ;;
  all)
    for quantizer in rq_kmeans rqvae; do
      for seed in 2024 2025 2026; do
        run_one "${quantizer}" "${seed}" 0
      done
    done
    "${python_exe}" "${project_root}/scripts/summarize_ar_self_attention.py"
    ;;
  internal-smoke)
    internal_root="${project_root}/outputs/attention/ar_internal_study"
    run_one pse_256x256x256 2024 256 "${internal_root}"
    echo "Internal smoke complete. Inspect outputs/attention/ar_internal_study/pse_256x256x256/seed2024/run_info.json"
    ;;
  internal-all)
    internal_root="${project_root}/outputs/attention/ar_internal_study"
    for variant in \
      pse_256x256x256 \
      rqk_128x256x512 \
      rqk_512x256x128
    do
      run_one "${variant}" 2024 0 "${internal_root}"
    done
    "${python_exe}" "${project_root}/scripts/summarize_ar_internal_attention.py"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    echo "Use: smoke | all | internal-smoke | internal-all" >&2
    exit 2
    ;;
esac
