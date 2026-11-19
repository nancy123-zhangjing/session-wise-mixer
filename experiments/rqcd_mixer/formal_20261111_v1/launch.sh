set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_mixer_formal_20261111_v1
PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
DATA=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/inputs/dataset
ARM="$1"
case "$ARM" in B0|B1|B2) ;; *) exit 2;; esac
exec > >(tee -a "$ARM.log") 2>&1
if ! test -f "${ARM}_rq/COMPLETE.json"; then
 resume=()
 if test -f "${ARM}_rq/last.pt"; then resume=(--resume); fi
 "$PY" -m scripts.rqcd_mixer.run pretrain --dataset "$DATA" --prepared prepared --arm "$ARM" --runtime runtime.locked.json --output "${ARM}_rq" "${resume[@]}"
fi
if ! test -f "$ARM/TRAINING_COMPLETE.json"; then
 resume=()
 if test -f "$ARM/last.pt"; then resume=(--resume); fi
 "$PY" -m scripts.rqcd_mixer.run train --dataset "$DATA" --prepared prepared --pretrained "${ARM}_rq" --arm "$ARM" --runtime runtime.locked.json --output "$ARM" "${resume[@]}"
fi
"$PY" -m scripts.rqcd_mixer.run test --dataset "$DATA" --prepared prepared --pretrained "${ARM}_rq" --arm "$ARM" --runtime runtime.locked.json --run "$ARM"
