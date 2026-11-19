#!/usr/bin/env bash
set -euo pipefail

variant="${1:-all}"
seed="${2:-2024}"
python_exe="${3:-python}"
mode="${4:-formal}"

if [[ "${mode}" != "profile" && "${mode}" != "formal" ]]; then
  echo "Unknown mode: ${mode}" >&2
  echo "Use: profile | formal" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
common_config="${project_root}/configs/ar_formal_common.yaml"
embedding_path="${project_root}/data/formal_embeddings/beauty_sentence_t5_xl.npy"
items_path="${project_root}/data/formal_embeddings/beauty_item_keys.json"
manifest_path="${project_root}/data/formal_embeddings/beauty_sentence_t5_xl_manifest.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

mkdir -p "${project_root}/outputs/console" "${project_root}/data/formal_embeddings"

prepare_embeddings() {
  "${python_exe}" "${project_root}/scripts/prepare_formal_embeddings.py" \
    --config "${common_config}" \
    --output "${embedding_path}" \
    --items-output "${items_path}" \
    --manifest-output "${manifest_path}"
}

run_variant() {
  local name="$1"
  local overlay="$2"
  local artifact_dir="${3:-}"
  local codebook_sizes="${4:-}"
  local log_path="${project_root}/outputs/console/ar_${mode}_${name}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"

  echo "[RUN] mode=${mode} variant=${name} seed=${seed}"
  echo "[RUN] log=${log_path}"
  cd "${diff_root}"

  local args=(
    main.py
    --model AR_GRM
    --dataset AmazonReviews2014
    --config "${common_config}"
    --config "${overlay}"
    --rand_seed="${seed}"
    --sid_fit_seed="${seed}"
  )
  if [[ -n "${artifact_dir}" ]]; then
    mkdir -p "${artifact_dir}"
    args+=(
      --rq_kmeans_seed="${seed}"
      --tokenizer_artifact_dir="${artifact_dir}"
    )
  fi

  # A complete one-epoch run verifies the ragged output heads, strict
  # item-level evaluator, and position diagnostics before a formal run.
  # best_epoch=1 also keeps this metadata out of the formal summarizer.
  if [[ "${mode}" == "profile" ]]; then
    args+=(
      --epochs=1
      --eval_start_epoch=1
      --eval_interval=1
      --patience=1
    )
  fi

  "${python_exe}" "${args[@]}" 2>&1 | tee "${log_path}"

  if [[ -n "${artifact_dir}" ]]; then
    read -r -a codebook_size_args <<< "${codebook_sizes}"
    "${python_exe}" "${project_root}/scripts/analyze_tokenizer_geometry.py" \
      --source "${embedding_path}" \
      --reconstructed "${artifact_dir}/reconstructed_embeddings.npy" \
      --semantic-ids "${artifact_dir}/semantic_ids.npy" \
      --output "${artifact_dir}/summary.json" \
      --codebook-sizes "${codebook_size_args[@]}" \
      --seed "${seed}"
  fi
}

run_pse() {
  run_variant \
    "formal_pse_3x256" \
    "${project_root}/configs/ar_formal_pse.yaml"
}

run_rqk_128_256_512() {
  run_variant \
    "formal_rqkmeans_128x256x512" \
    "${project_root}/configs/ar_formal_rqkmeans_128_256_512.yaml" \
    "${project_root}/outputs/tokenizers/rqkmeans_128x256x512/Beauty/seed${seed}" \
    "128 256 512"
}

run_rqk_512_256_128() {
  run_variant \
    "formal_rqkmeans_512x256x128" \
    "${project_root}/configs/ar_formal_rqkmeans_512_256_128.yaml" \
    "${project_root}/outputs/tokenizers/rqkmeans_512x256x128/Beauty/seed${seed}" \
    "512 256 128"
}

prepare_embeddings
case "${variant}" in
  pse) run_pse ;;
  rqk-128-256-512) run_rqk_128_256_512 ;;
  rqk-512-256-128) run_rqk_512_256_128 ;;
  all)
    run_pse
    run_rqk_128_256_512
    run_rqk_512_256_128
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    echo "Use: pse | rqk-128-256-512 | rqk-512-256-128 | all" >&2
    exit 2
    ;;
esac
