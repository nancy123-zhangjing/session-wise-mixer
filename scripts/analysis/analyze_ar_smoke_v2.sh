#!/usr/bin/env bash
set -euo pipefail

python_exe="${1:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
processed_dir="${project_root}/data/cache/AmazonReviews2014/Beauty/processed"

sid_map="$(
    find "${processed_dir}" -maxdepth 1 -type f \
        -name 'item_id2tokens_*_4sem_csuffix_5total.npy' \
        -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
)"
if [[ -z "${sid_map}" ]]; then
    echo "Collision-aware SID mapping was not found in ${processed_dir}." >&2
    exit 1
fi

report_dir="${project_root}/outputs/reports"
figure_dir="${project_root}/outputs/figures/ar_smoke_v2"
mkdir -p "${report_dir}" "${figure_dir}"

"${python_exe}" "${project_root}/scripts/analysis/analyze_sid_collisions.py" \
    --input "${sid_map}" \
    --semantic-digits 4 \
    --codebook-size 256 \
    --require-unique-full \
    --output "${report_dir}/ar_smoke_v2_sid.json"

position_csv="$(
    find "${project_root}/outputs/position_loss_logs/AmazonReviews2014/AR_GRM" \
        -maxdepth 1 -type f -name '*_position_loss.csv' \
        -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
)"
if [[ -z "${position_csv}" ]]; then
    echo "Position-loss CSV was not found." >&2
    exit 1
fi

"${python_exe}" "${project_root}/scripts/analysis/plot_position_loss.py" \
    --input "${position_csv}" \
    --output-dir "${figure_dir}"

echo "SID report: ${report_dir}/ar_smoke_v2_sid.json"
echo "Figures: ${figure_dir}"
