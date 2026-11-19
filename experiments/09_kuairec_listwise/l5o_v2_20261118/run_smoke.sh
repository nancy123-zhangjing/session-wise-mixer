#!/bin/bash
# Protocol v2 GPU short test (1 H100). Never produces results; writes only under $V/smoke.
set -u
B=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09
V=$B/l5o_v2_20261118
C=$V/code_302f9b7
W=$B/l5o_wave_20261117_v1
P=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
SM=$V/smoke
RECEIPT=$B/l5o_sasrec_20261117_v1/run/COMPLETE.json
DATA=$B/l5o_stage0_20261117_v1/dataset
ASSETS=$B/l5o_stage0_20261117_v1/assets
cd $C
export PYTHONPATH=$C/ar_deps:$C OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $SM
nvidia-smi > $SM/nvidia_smi.txt
$P -u -m scripts.session_wise_ar.v2_probe --receipt $RECEIPT --dataset $DATA --content $ASSETS/content_features.npy \
  --wave $W --output $SM/probe > $SM/probe.log 2>&1
echo "probe $?" >> $SM/STATUS.txt
for slot in "h H0" "h H1" "tiger T0A" "tiger T1A" "sasrec SASREC" "cf CF-M0" "cf CF-M1"; do
  set -- $slot
  $P -u -m scripts.kuairec_data.l5o_v2_worker $1 $2 2024 --smoke --root $SM/root --receipt $RECEIPT \
    --dataset $DATA --assets $ASSETS --cf-runtime $C/experiments/rqcd_mixer/targets_formal_20261113_v1/runtime.locked.json \
    > $SM/worker_$1_$2.log 2>&1
  echo "$1 $2 $?" >> $SM/STATUS.txt
done
echo DONE >> $SM/STATUS.txt
