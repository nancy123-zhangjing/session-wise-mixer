#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
seed="${2:-2024}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/shared/diff_formal_common.yaml"
formal_config="${project_root}/configs/shared/diff_formal_repo_v6.yaml"
rqvae_config="${project_root}/configs/03_diffgrm_tokenizer_comparison/diff_formal_rqvae_v6.yaml"
smoke_config="${project_root}/configs/shared/diff_smoke.yaml"
profile_config="${project_root}/configs/shared/diff_profile.yaml"
embedding_dir="${project_root}/data/diff_formal_embeddings"
embedding_path="${embedding_dir}/beauty_sentence_t5_base_pca256.npy"
items_path="${embedding_dir}/beauty_item_keys.json"
manifest_path="${embedding_dir}/beauty_sentence_t5_base_pca256_manifest.json"
tokenizer_output="${project_root}/outputs/tokenizers/diff_rqvae/Beauty/seed2024"
console_dir="${project_root}/outputs/console"
run_id="diff_formal_rqvae_4x256_v6_bf16"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${embedding_dir}" "${console_dir}"

prepare_inputs() {
  "${python_exe}" "${project_root}/scripts/prepare/prepare_diff_rqvae_inputs.py" \
    --config "${common_config}" \
    --config "${rqvae_config}" \
    --output "${embedding_path}" \
    --items-output "${items_path}" \
    --manifest-output "${manifest_path}"
}

train_tokenizer() {
  local mode="${1:-formal}"
  local profile_args=()
  local log_tag="formal"
  if [[ "${mode}" == "profile" ]]; then
    profile_args=(--profile)
    log_tag="profile"
  fi
  local log_path="${console_dir}/diff_rqvae_tokenizer_${log_tag}_seed2024_$(date +%Y%m%d_%H%M%S).log"
  echo "[TOKENIZER] RQ-VAE 4x256 seed=2024 input=PCA256 mode=${mode}"
  echo "[TOKENIZER] log=${log_path}"
  "${python_exe}" "${project_root}/scripts/tokenizers/train_rqvae_tokenizer.py" \
    --config "${common_config}" \
    --config "${rqvae_config}" \
    --embedding "${embedding_path}" \
    --items "${items_path}" \
    --output-dir "${tokenizer_output}" \
    --seed 2024 \
    "${profile_args[@]}" \
    2>&1 | tee "${log_path}"
  local artifact_dir="${tokenizer_output}"
  local verify_thresholds=(--min-utilization 0.25 --min-unique-ratio 0.5)
  if [[ "${mode}" == "profile" ]]; then
    artifact_dir="${tokenizer_output}_profile"
    verify_thresholds=(--min-utilization 0.0 --min-unique-ratio 0.0)
  fi
  "${python_exe}" "${project_root}/scripts/analysis/analyze_tokenizer_geometry.py" \
    --source "${embedding_path}" \
    --reconstructed "${artifact_dir}/reconstructed_embeddings.npy" \
    --semantic-ids "${artifact_dir}/semantic_ids.npy" \
    --output "${artifact_dir}/geometry_summary.json" \
    --seed 2024
  "${python_exe}" "${project_root}/scripts/checks/verify_diff_rqvae_artifact.py" \
    --artifact-dir "${artifact_dir}" \
    --items "${items_path}" \
    --n-digit 4 \
    --codebook-size 256 \
    "${verify_thresholds[@]}"
}

require_tokenizer() {
  for path in \
    "${tokenizer_output}/semantic_ids.json" \
    "${tokenizer_output}/semantic_ids.npy" \
    "${tokenizer_output}/best.pt" \
    "${tokenizer_output}/summary.json"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing frozen RQ-VAE artifact: ${path}" >&2
      echo "Run: bash scripts/train/run_diff_rqvae_v6.sh tokenizer 2024 ${python_exe}" >&2
      exit 1
    fi
  done
}

run_one() {
  require_tokenizer
  local run_kind="$1"
  local run_seed="$2"
  local log_path="${console_dir}/diff_${run_kind}_rqvae_seed${run_seed}_$(date +%Y%m%d_%H%M%S).log"
  local extra_config=()
  local run_id_override="${run_id}"
  if [[ "${run_kind}" == "smoke" ]]; then
    extra_config=(--config "${smoke_config}")
    run_id_override="diff_smoke_rqvae_4x256"
  elif [[ "${run_kind}" == "profile" ]]; then
    extra_config=(--config "${profile_config}")
    run_id_override="diff_profile_rqvae_4x256"
  fi

  echo "[RUN] kind=${run_kind} tokenizer=RQ-VAE decoder_seed=${run_seed}"
  echo "[RUN] tokenizer_seed=2024 layout=4x256 input=PCA256"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${formal_config}" \
    --config "${rqvae_config}" \
    "${extra_config[@]}" \
    --run_id="${run_id_override}" \
    --rand_seed="${run_seed}" \
    2>&1 | tee "${log_path}"
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    "${python_exe}" scripts/checks/preflight_diff_formal.py --tokenizer rqvae
    ;;
  prepare)
    prepare_inputs
    ;;
  tokenizer-profile)
    prepare_inputs
    train_tokenizer profile
    ;;
  tokenizer)
    prepare_inputs
    train_tokenizer formal
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
      "${python_exe}" scripts/analysis/analyze_diff_cpd_attention.py \
        --seed "${run_seed}" --run-id "${run_id}"
    done
    cd "${project_root}"
    "${python_exe}" scripts/analysis/summarize_diff_formal.py \
      --run-id "${run_id}" --require-all-seeds
    ;;
  remaining-seeds)
    for run_seed in 2025 2026; do
      run_one formal_v6 "${run_seed}"
      cd "${project_root}"
      "${python_exe}" scripts/analysis/analyze_diff_cpd_attention.py \
        --seed "${run_seed}" --run-id "${run_id}"
    done
    cd "${project_root}"
    "${python_exe}" scripts/analysis/summarize_diff_formal.py \
      --run-id "${run_id}" --require-all-seeds
    ;;
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/summarize_diff_formal.py --run-id "${run_id}"
    ;;
  compare)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/summarize_diff_formal.py \
      --run-id diff_formal_pse_4x256_repo_repro_v6_bf16
    "${python_exe}" scripts/analysis/summarize_diff_formal.py \
      --run-id diff_formal_rqkmeans_4x256_v6_bf16
    "${python_exe}" scripts/analysis/summarize_diff_formal.py --run-id "${run_id}"
    "${python_exe}" scripts/analysis/compare_diff_tokenizers.py --include-rqvae
    ;;
  attention)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id "${run_id}"
    ;;
  attention-smoke)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/analyze_diff_cpd_attention.py \
      --seed "${seed}" --run-id "${run_id}" \
      --max-samples 128 --batch-size 32
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | prepare | tokenizer-profile | tokenizer | smoke | profile | formal | all-seeds | remaining-seeds | summarize | compare | attention | attention-smoke" >&2
    exit 2
    ;;
esac
