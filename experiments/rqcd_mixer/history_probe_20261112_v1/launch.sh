set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_history_probe_20261112_v1
SOURCE=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_mixer_formal_20261111_v1
SUBSET=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_landing_diagnostic_20261112_v1/results
PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
cd "$ROOT"
exec > >(tee -a probe.log) 2>&1
exec "$PY" "$ROOT/probe.py" --source "$SOURCE" --subset-root "$SUBSET" --output "$ROOT/results"
