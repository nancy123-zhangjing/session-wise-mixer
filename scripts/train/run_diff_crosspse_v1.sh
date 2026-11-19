#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
seed="${2:-2024}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/shared/diff_formal_common.yaml"
formal_config="${project_root}/configs/shared/diff_formal_repo_v6.yaml"
cross_config="${project_root}/configs/04_crosspse_offline/diff_formal_crosspse_v1.yaml"
smoke_config="${project_root}/configs/shared/diff_smoke.yaml"
profile_config="${project_root}/configs/shared/diff_profile.yaml"
embedding_dir="${project_root}/data/diff_formal_embeddings"
embedding_path="${embedding_dir}/beauty_sentence_t5_base_pca256.npy"
items_path="${embedding_dir}/beauty_item_keys.json"
manifest_path="${embedding_dir}/beauty_sentence_t5_base_pca256_manifest.json"
fit_mask_path="${embedding_dir}/beauty_pse_fit_mask.npy"
fit_mask_summary="${embedding_dir}/beauty_pse_fit_mask_summary.json"
tokenizer_output="${project_root}/outputs/tokenizers/diff_crosspse/Beauty/seed2024"
reference_pse_ids="${project_root}/data/cache/AmazonReviews2014/Beauty/processed/sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8.sem_ids"
console_dir="${project_root}/outputs/console"
run_id="diff_formal_crosspse_4x256_v1_bf16"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${embedding_dir}" "${console_dir}"

prepare_inputs() {
  "${python_exe}" "${project_root}/scripts/prepare/prepare_diff_rqvae_inputs.py" \
    --config "${common_config}" \
    --config "${cross_config}" \
    --output "${embedding_path}" \
    --items-output "${items_path}" \
    --manifest-output "${manifest_path}"
  "${python_exe}" "${project_root}/scripts/prepare/prepare_diff_crosspse_fit_mask.py" \
    --config "${common_config}" \
    --config "${cross_config}" \
    --items "${items_path}" \
    --output "${fit_mask_path}" \
    --summary-output "${fit_mask_summary}"
}

train_tokenizer() {
  local mode="${1:-formal}"
  local profile_args=()
  local verify_reference_args=()
  local log_tag="formal"
  local artifact_dir="${tokenizer_output}"
  if [[ "${mode}" == "profile" ]]; then
    profile_args=(--profile --restart)
    log_tag="profile"
    artifact_dir="${tokenizer_output}_profile"
  fi
  if [[ -f "${reference_pse_ids}" ]]; then
    verify_reference_args=(--reference-pse-semantic-ids "${reference_pse_ids}")
  fi
  local log_path="${console_dir}/diff_crosspse_tokenizer_${log_tag}_seed2024_$(date +%Y%m%d_%H%M%S).log"
  echo "[TOKENIZER] Cross-PSE 4x256 seed=2024 scope=full mode=${mode}"
  echo "[TOKENIZER] fixed_base_codebooks=true input=PCA256+OPQ"
  echo "[TOKENIZER] log=${log_path}"
  "${python_exe}" "${project_root}/scripts/tokenizers/train_cross_pse_tokenizer.py" \
    --config "${common_config}" \
    --config "${cross_config}" \
    --embedding "${embedding_path}" \
    --items "${items_path}" \
    --fit-mask "${fit_mask_path}" \
    --output-dir "${tokenizer_output}" \
    --seed 2024 \
    "${profile_args[@]}" \
    2>&1 | tee "${log_path}"
  "${python_exe}" "${project_root}/scripts/checks/verify_diff_crosspse_artifact.py" \
    --artifact-dir "${artifact_dir}" \
    --items "${items_path}" \
    --n-digit 4 \
    --codebook-size 256 \
    "${verify_reference_args[@]}"
  "${python_exe}" "${project_root}/scripts/analysis/analyze_tokenizer_geometry.py" \
    --source "${artifact_dir}/opq_embeddings.npy" \
    --reconstructed "${artifact_dir}/reconstructed_opq_embeddings.npy" \
    --semantic-ids "${artifact_dir}/semantic_ids.npy" \
    --output "${artifact_dir}/geometry_summary.json" \
    --codebook-size 256 \
    --seed 2024
}

require_tokenizer() {
  for path in \
    "${tokenizer_output}/semantic_ids.json" \
    "${tokenizer_output}/semantic_ids.npy" \
    "${tokenizer_output}/best.pt" \
    "${tokenizer_output}/summary.json" \
    "${tokenizer_output}/geometry_summary.json"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing frozen Cross-PSE artifact: ${path}" >&2
      echo "Run: bash scripts/train/run_diff_crosspse_v1.sh tokenizer 2024 ${python_exe}" >&2
      exit 1
    fi
  done
}

run_one() {
  require_tokenizer
  local run_kind="$1"
  local run_seed="$2"
  local log_path="${console_dir}/diff_${run_kind}_crosspse_seed${run_seed}_$(date +%Y%m%d_%H%M%S).log"
  local extra_config=()
  local run_id_override="${run_id}"
  if [[ "${run_kind}" == "smoke" ]]; then
    extra_config=(--config "${smoke_config}")
    run_id_override="diff_smoke_crosspse_4x256_v1"
  elif [[ "${run_kind}" == "profile" ]]; then
    extra_config=(--config "${profile_config}")
    run_id_override="diff_profile_crosspse_4x256_v1"
  fi

  echo "[RUN] kind=${run_kind} tokenizer=Cross-PSE decoder_seed=${run_seed}"
  echo "[RUN] tokenizer_seed=2024 layout=4x256 input=PCA256+OPQ"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${formal_config}" \
    --config "${cross_config}" \
    "${extra_config[@]}" \
    --run_id="${run_id_override}" \
    --rand_seed="${run_seed}" \
    2>&1 | tee "${log_path}"
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    "${python_exe}" tests/test_cross_pse_module.py
    "${python_exe}" scripts/checks/preflight_diff_formal.py --tokenizer cross_pse
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
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/summarize_diff_formal.py --run-id "${run_id}"
    ;;
  compare)
    cd "${project_root}"
    "${python_exe}" scripts/analysis/compare_diff_tokenizers.py \
      --include-cross-pse \
      --crosspse-name cross_pse_v1 \
      --crosspse-dir "${project_root}/outputs/reports/${run_id}"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | prepare | tokenizer-profile | tokenizer | smoke | profile | formal | attention | attention-smoke | summarize | compare" >&2
    exit 2
    ;;
esac
