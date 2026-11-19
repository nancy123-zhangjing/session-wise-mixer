#!/usr/bin/env bash
set -euo pipefail

variant="${1:-refined}"
stage="${2:-preflight}"
seed="${3:-2024}"
python_exe="${4:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/diff_formal_common.yaml"
formal_config="${project_root}/configs/diff_formal_repo_v6.yaml"
smoke_config="${project_root}/configs/diff_smoke.yaml"
profile_config="${project_root}/configs/diff_profile.yaml"
embedding_dir="${project_root}/data/diff_formal_embeddings"
embedding_path="${embedding_dir}/beauty_sentence_t5_base_pca256.npy"
items_path="${embedding_dir}/beauty_item_keys.json"
manifest_path="${embedding_dir}/beauty_sentence_t5_base_pca256_manifest.json"
fit_mask_path="${embedding_dir}/beauty_pse_fit_mask.npy"
fit_mask_summary="${embedding_dir}/beauty_pse_fit_mask_summary.json"
reference_pse_ids="${project_root}/data/cache/AmazonReviews2014/Beauty/processed/sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8.sem_ids"
base_artifact_dir="${project_root}/outputs/tokenizers/diff_pse_base/Beauty/seed2024"
console_dir="${project_root}/outputs/console"

case "${variant}" in
  refined)
    selected_config="${project_root}/configs/diff_formal_refinedpse_v1.yaml"
    tokenizer_output="${project_root}/outputs/tokenizers/diff_refinedpse/Beauty/v1_seed2024"
    run_id="diff_formal_refinedpse_4x256_v1_bf16"
    protocol="controlled_trainable_pse_no_cross_v1"
    preflight_name="refined_pse"
    label="Refined-PSE-v1"
    expected_selection=""
    ;;
  cross_v3)
    selected_config="${project_root}/configs/diff_formal_crosspse_v3.yaml"
    tokenizer_output="${project_root}/outputs/tokenizers/diff_crosspse/Beauty/v3_seed2024"
    run_id="diff_formal_crosspse_4x256_v3_bf16"
    protocol="controlled_trainable_cross_codebook_pse_v3"
    preflight_name="cross_pse_v3"
    label="Cross-PSE-v3"
    expected_selection=""
    ;;
  refined_v2_catalog)
    selected_config="${project_root}/configs/diff_formal_refinedpse_v2_catalog.yaml"
    tokenizer_output="${project_root}/outputs/tokenizers/diff_refinedpse/Beauty/v2_catalog_seed2024"
    run_id="diff_formal_refinedpse_4x256_v2_catalog_bf16"
    protocol="controlled_trainable_pse_catalog_selection_v2"
    preflight_name="refined_pse_catalog"
    label="Refined-PSE-v2-catalog"
    expected_selection="fit_catalog_reconstruction"
    ;;
  cross_v3_catalog)
    selected_config="${project_root}/configs/diff_formal_crosspse_v3_catalog.yaml"
    tokenizer_output="${project_root}/outputs/tokenizers/diff_crosspse/Beauty/v3_catalog_seed2024"
    run_id="diff_formal_crosspse_4x256_v3_catalog_bf16"
    protocol="controlled_trainable_cross_codebook_pse_v3_catalog"
    preflight_name="cross_pse_v3_catalog"
    label="Cross-PSE-v3-catalog"
    expected_selection="fit_catalog_reconstruction"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

catalog_refined_summary="${project_root}/outputs/tokenizers/diff_refinedpse/Beauty/v2_catalog_seed2024/summary.json"
catalog_cross_summary="${project_root}/outputs/tokenizers/diff_crosspse/Beauty/v3_catalog_seed2024/summary.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${embedding_dir}" "${console_dir}"

prepare_inputs() {
  "${python_exe}" "${project_root}/scripts/prepare_diff_rqvae_inputs.py" \
    --config "${common_config}" \
    --config "${selected_config}" \
    --output "${embedding_path}" \
    --items-output "${items_path}" \
    --manifest-output "${manifest_path}"
  "${python_exe}" "${project_root}/scripts/prepare_diff_crosspse_fit_mask.py" \
    --config "${common_config}" \
    --config "${selected_config}" \
    --items "${items_path}" \
    --output "${fit_mask_path}" \
    --summary-output "${fit_mask_summary}"
}

train_tokenizer() {
  local mode="${1:-formal}"
  local profile_args=()
  local reference_args=()
  local log_tag="formal"
  local artifact_dir="${tokenizer_output}"
  if [[ "${mode}" == "profile" ]]; then
    profile_args=(--profile --restart)
    log_tag="profile"
    artifact_dir="${tokenizer_output}_profile"
  fi
  if [[ -f "${reference_pse_ids}" ]]; then
    reference_args=(--reference-pse-semantic-ids "${reference_pse_ids}")
  fi
  local log_path="${console_dir}/diff_${variant}_tokenizer_${log_tag}_seed2024_$(date +%Y%m%d_%H%M%S).log"
  echo "[TOKENIZER] ${label} 4x256 tokenizer_seed=2024 mode=${mode}"
  echo "[TOKENIZER] paired configuration=${selected_config}"
  echo "[TOKENIZER] log=${log_path}"
  "${python_exe}" "${project_root}/scripts/train_cross_pse_tokenizer.py" \
    --config "${common_config}" \
    --config "${selected_config}" \
    --embedding "${embedding_path}" \
    --items "${items_path}" \
    --fit-mask "${fit_mask_path}" \
    --base-artifact-dir "${base_artifact_dir}" \
    --output-dir "${tokenizer_output}" \
    --seed 2024 \
    "${profile_args[@]}" \
    2>&1 | tee "${log_path}"
  "${python_exe}" "${project_root}/scripts/verify_diff_crosspse_artifact.py" \
    --artifact-dir "${artifact_dir}" \
    --items "${items_path}" \
    --n-digit 4 \
    --codebook-size 256 \
    "${reference_args[@]}"
  "${python_exe}" "${project_root}/scripts/analyze_tokenizer_geometry.py" \
    --source "${artifact_dir}/opq_embeddings.npy" \
    --reconstructed "${artifact_dir}/reconstructed_opq_embeddings.npy" \
    --semantic-ids "${artifact_dir}/semantic_ids.npy" \
    --output "${artifact_dir}/geometry_summary.json" \
    --codebook-size 256 \
    --seed 2024
}

require_tokenizer() {
  local path
  for path in \
    "${tokenizer_output}/semantic_ids.json" \
    "${tokenizer_output}/semantic_ids.npy" \
    "${tokenizer_output}/best.pt" \
    "${tokenizer_output}/summary.json" \
    "${tokenizer_output}/geometry_summary.json"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing frozen ${label} artifact: ${path}" >&2
      echo "Run the tokenizer stage first." >&2
      exit 1
    fi
  done
  local readiness_args=(
    --summary "${tokenizer_output}/summary.json"
    --expected-protocol "${protocol}"
    --min-item-change 0.0001
    --strict
  )
  if [[ -n "${expected_selection}" ]]; then
    readiness_args+=(--expected-selection-protocol "${expected_selection}")
  fi
  "${python_exe}" "${project_root}/scripts/inspect_crosspse_tokenizer.py" \
    "${readiness_args[@]}"
}

run_one() {
  require_tokenizer
  local run_kind="$1"
  local run_seed="$2"
  local log_path="${console_dir}/diff_${run_kind}_${variant}_seed${run_seed}_$(date +%Y%m%d_%H%M%S).log"
  local extra_config=()
  local selected_run_id="${run_id}"
  if [[ "${run_kind}" == "smoke" ]]; then
    extra_config=(--config "${smoke_config}")
    selected_run_id="diff_smoke_${variant}_4x256"
  elif [[ "${run_kind}" == "profile" ]]; then
    extra_config=(--config "${profile_config}")
    selected_run_id="diff_profile_${variant}_4x256"
  fi
  echo "[RUN] kind=${run_kind} tokenizer=${label} decoder_seed=${run_seed}"
  echo "[RUN] frozen tokenizer seed=2024 layout=4x256 input=PCA256+OPQ"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model DIFF_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${formal_config}" \
    --config "${selected_config}" \
    "${extra_config[@]}" \
    --run_id="${selected_run_id}" \
    --rand_seed="${run_seed}" \
    2>&1 | tee "${log_path}"
}

case "${stage}" in
  preflight)
    cd "${project_root}"
    "${python_exe}" scripts/test_cross_pse_module.py
    "${python_exe}" scripts/preflight_diff_formal.py --tokenizer "${preflight_name}"
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
  inspect)
    inspect_args=(
      --summary "${tokenizer_output}/summary.json"
      --expected-protocol "${protocol}"
      --min-item-change 0.0001
    )
    if [[ -n "${expected_selection}" ]]; then
      inspect_args+=(--expected-selection-protocol "${expected_selection}")
    fi
    "${python_exe}" "${project_root}/scripts/inspect_crosspse_tokenizer.py" \
      "${inspect_args[@]}"
    ;;
  tokenizer-compare)
    cd "${project_root}"
    "${python_exe}" scripts/compare_pse_refinement_tokenizers.py \
      --refined-summary "${catalog_refined_summary}" \
      --cross-summary "${catalog_cross_summary}" \
      --output "${project_root}/outputs/reports/diff_pse_refinement_catalog_tokenizers/summary.json"
    ;;
  smoke|profile|formal)
    run_one "${stage}" "${seed}"
    ;;
  attention|attention-smoke)
    cd "${project_root}"
    attention_args=(--seed "${seed}" --run-id "${run_id}")
    if [[ "${stage}" == "attention-smoke" ]]; then
      attention_args+=(--max-samples 128 --batch-size 32)
    fi
    "${python_exe}" scripts/analyze_diff_cpd_attention.py "${attention_args[@]}"
    ;;
  summarize)
    cd "${project_root}"
    "${python_exe}" scripts/summarize_diff_formal.py --run-id "${run_id}"
    ;;
  compare)
    cd "${project_root}"
    compare_args=(--include-rqvae --include-refined-pse --include-cross-pse-v3)
    if [[ -n "${expected_selection}" ]]; then
      compare_args+=(
        --refinedpse-dir "${project_root}/outputs/reports/diff_formal_refinedpse_4x256_v2_catalog_bf16"
        --refinedpse-name refined_pse_v2_catalog
        --crosspse-v3-dir "${project_root}/outputs/reports/diff_formal_crosspse_4x256_v3_catalog_bf16"
        --crosspse-v3-name cross_pse_v3_catalog
      )
    fi
    "${python_exe}" scripts/compare_diff_tokenizers.py "${compare_args[@]}"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: preflight | prepare | tokenizer-profile | tokenizer | inspect | tokenizer-compare | smoke | profile | formal | attention-smoke | attention | summarize | compare" >&2
    exit 2
    ;;
esac
