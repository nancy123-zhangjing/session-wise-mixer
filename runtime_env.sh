#!/usr/bin/env bash
# Source this file from Bash; it changes only the current shell environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source runtime_env.sh (running it as a subprocess cannot activate your shell)." >&2
  exit 2
fi
_grm_runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_grm_conda_sh="/inspire/hdd/global_user/<inspire-account>/miniconda3/etc/profile.d/conda.sh"
_grm_env_prefix="/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp"
if [[ ! -f "${_grm_conda_sh}" || ! -x "${_grm_env_prefix}/bin/python" ]]; then
  echo "This helper is for the documented Inspire shared environment; paths are unavailable here." >&2
  unset _grm_runtime_root _grm_conda_sh _grm_env_prefix
  return 1
fi
source "${_grm_conda_sh}"
conda activate "${_grm_env_prefix}" || return 1
export HF_HOME="${_grm_runtime_root}/data/huggingface"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled
cd "${_grm_runtime_root}" || return 1
python -c 'import sys; print("Python:", sys.executable)'
unset _grm_runtime_root _grm_conda_sh _grm_env_prefix
