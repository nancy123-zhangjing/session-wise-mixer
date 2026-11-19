#!/usr/bin/env bash
set -euo pipefail

seed="${1:-2024}"
stage="${2:-all}"
python_exe="${3:-python}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/shared/ar_formal_common.yaml"
rqvae_config="${project_root}/configs/01_ar_baseline_3x256/ar_formal_rqvae.yaml"
rqkmeans_config="${project_root}/configs/01_ar_baseline_3x256/ar_formal_rqkmeans.yaml"
embedding_path="${project_root}/data/formal_embeddings/beauty_sentence_t5_xl.npy"
items_path="${project_root}/data/formal_embeddings/beauty_item_keys.json"
manifest_path="${project_root}/data/formal_embeddings/beauty_sentence_t5_xl_manifest.json"
rqvae_output="${project_root}/outputs/tokenizers/rqvae/Beauty/seed${seed}"
rqkmeans_output="${project_root}/outputs/tokenizers/rqkmeans/Beauty/seed${seed}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${project_root}/outputs/console" "${project_root}/data/formal_embeddings"

prepare_embeddings() {
  "${python_exe}" "${project_root}/scripts/prepare/prepare_formal_embeddings.py" \
    --config "${common_config}" \
    --output "${embedding_path}" \
    --items-output "${items_path}" \
    --manifest-output "${manifest_path}"
}

train_rqvae() {
  "${python_exe}" "${project_root}/scripts/tokenizers/train_rqvae_tokenizer.py" \
    --config "${common_config}" \
    --config "${rqvae_config}" \
    --embedding "${embedding_path}" \
    --items "${items_path}" \
    --output-dir "${rqvae_output}" \
    --seed "${seed}"
  "${python_exe}" "${project_root}/scripts/analysis/analyze_tokenizer_geometry.py" \
    --source "${embedding_path}" \
    --reconstructed "${rqvae_output}/reconstructed_embeddings.npy" \
    --semantic-ids "${rqvae_output}/semantic_ids.npy" \
    --output "${rqvae_output}/geometry_summary.json" \
    --seed "${seed}"
}

run_ar_rqvae() {
  local log_path="${project_root}/outputs/console/ar_formal_rqvae_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model AR_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${rqvae_config}" \
    --rand_seed="${seed}" \
    --sid_fit_seed="${seed}" \
    --rqvae_seed="${seed}" \
    --precomputed_semantic_ids_path="${rqvae_output}/semantic_ids.json" \
    2>&1 | tee "${log_path}"
}

run_ar_rqkmeans() {
  local log_path="${project_root}/outputs/console/ar_formal_rqkmeans_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
  cd "${diff_root}"
  "${python_exe}" main.py \
    --model AR_GRM \
    --dataset AmazonReviews2014 \
    --config "${common_config}" \
    --config "${rqkmeans_config}" \
    --rand_seed="${seed}" \
    --sid_fit_seed="${seed}" \
    --rq_kmeans_seed="${seed}" \
    --tokenizer_artifact_dir="${rqkmeans_output}" \
    2>&1 | tee "${log_path}"
  "${python_exe}" "${project_root}/scripts/analysis/analyze_tokenizer_geometry.py" \
    --source "${embedding_path}" \
    --reconstructed "${rqkmeans_output}/reconstructed_embeddings.npy" \
    --semantic-ids "${rqkmeans_output}/semantic_ids.npy" \
    --output "${rqkmeans_output}/summary.json" \
    --seed "${seed}"
}

case "${stage}" in
  prepare) prepare_embeddings ;;
  rqvae-tokenizer) prepare_embeddings; train_rqvae ;;
  rqvae-ar) run_ar_rqvae ;;
  rqkmeans-ar) prepare_embeddings; run_ar_rqkmeans ;;
  all)
    prepare_embeddings
    train_rqvae
    run_ar_rqvae
    run_ar_rqkmeans
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    echo "Use: prepare | rqvae-tokenizer | rqvae-ar | rqkmeans-ar | all" >&2
    exit 2
    ;;
esac
