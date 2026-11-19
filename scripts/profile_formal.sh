#!/usr/bin/env bash
set -euo pipefail

seed="${1:-2024}"
python_exe="${2:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
common_config="${project_root}/configs/ar_formal_common.yaml"
rqvae_config="${project_root}/configs/ar_formal_rqvae.yaml"
rqkmeans_config="${project_root}/configs/ar_formal_rqkmeans.yaml"
embedding_path="${project_root}/data/formal_embeddings/beauty_sentence_t5_xl.npy"
items_path="${project_root}/data/formal_embeddings/beauty_item_keys.json"
rqvae_output="${project_root}/outputs/tokenizers/rqvae/Beauty/seed${seed}"
rqkmeans_output="${project_root}/outputs/tokenizers/rqkmeans/Beauty/seed${seed}_profile"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

bash "${project_root}/scripts/run_formal_pair.sh" "${seed}" prepare "${python_exe}"

"${python_exe}" "${project_root}/scripts/train_rqvae_tokenizer.py" \
  --config "${common_config}" \
  --config "${rqvae_config}" \
  --embedding "${embedding_path}" \
  --items "${items_path}" \
  --output-dir "${rqvae_output}" \
  --seed "${seed}" \
  --profile

cd "${project_root}/external/DiffGRM"
"${python_exe}" main.py \
  --model AR_GRM \
  --dataset AmazonReviews2014 \
  --config "${common_config}" \
  --config "${rqkmeans_config}" \
  --rand_seed="${seed}" \
  --sid_fit_seed="${seed}" \
  --rq_kmeans_seed="${seed}" \
  --run_id=ar_profile_rqkmeans_3x256 \
  --tokenizer_artifact_dir="${rqkmeans_output}" \
  --epochs=1 \
  --eval_start_epoch=1 \
  --eval_interval=1 \
  --patience=1 \
  --ar_beam_search='{"pre_cut_num":[32,32,32,32],"beam_search_num":[32,32,32,32],"top_k_final":10,"self_kv_cache":true,"cross_kv_cache":true,"seed":42}'
