set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_mixer_benchmark_20261111_v1
PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
$PY -m scripts.rqcd_mixer.run prepare --dataset /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/inputs/dataset --content /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/assets/pse_v1/content_features.npy --output prepared
$PY -m scripts.rqcd_mixer.run benchmark --dataset /inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/inputs/dataset --prepared prepared --output benchmark
