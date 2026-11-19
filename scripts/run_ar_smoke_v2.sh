#!/usr/bin/env bash
set -euo pipefail

python_exe="${1:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diff_root="${project_root}/external/DiffGRM"
config_path="${project_root}/configs/ar_smoke_v2_unique.yaml"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${project_root}/data/huggingface}"

if [[ ! -d "${diff_root}" ]]; then
    echo "external/DiffGRM is missing. Run bash scripts/bootstrap_upstreams.sh first." >&2
    exit 1
fi

"${python_exe}" -c \
    "import torch, accelerate, datasets, faiss, sentence_transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"${python_exe}" "${project_root}/scripts/test_collision_aware_eval.py"

cd "${diff_root}"
"${python_exe}" main.py \
    --model AR_GRM \
    --dataset AmazonReviews2014 \
    --config "${config_path}"
