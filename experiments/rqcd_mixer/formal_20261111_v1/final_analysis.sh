#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_mixer_formal_20261111_v1
PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
cd "$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR="$ROOT/.mplconfig"
trap 'echo $? > "$ROOT/final_analysis.exit"' EXIT
"$PY" -u audit_checkpoints.py "$ROOT" --output checkpoint_audit.json
"$PY" -m scripts.rqcd_mixer.run report --runs "$ROOT/B0" "$ROOT/B1" "$ROOT/B2" --output "$ROOT/report"
