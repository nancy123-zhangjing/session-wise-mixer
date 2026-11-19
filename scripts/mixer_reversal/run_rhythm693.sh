#!/bin/bash
# Post-run analysis of the H rhythm control (l5o_v2_rhythm693_20261119), CPU notebook only.
# Waits (up to 20 hours) until all six arms x seeds have their three test summaries, then runs
#   1. the pre-registered paired tests (scripts.kuairec_data.l5o_paired) per seed and node;
#   2. the read-only breakdowns (scripts.mixer_reversal.analyze v2 / traj, family h) on this root.
# Usage: run_rhythm693.sh <control root> <paired-test code dir> <analysis code dir>
set -u
N=$1; PC=$2; AC=$3
D=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1/dataset
OUT=$N/analysis
mkdir -p "$OUT/paired"
deadline=$(( $(date +%s) + 20 * 3600 ))
while :; do
  n=0
  for s in 2024 2025 2026; do for a in H0 H1; do for x in smoothed best final; do
    [ -s "$N/h/seed$s/$a/test/$x/summary.json" ] && n=$((n + 1))
  done; done; done
  echo "$(date -Is) test summaries: $n/18" > "$OUT/wait.txt"
  [ "$n" -eq 18 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then touch "$OUT/TIMEOUT"; exit 0; fi
  sleep 600
done
cd "$PC"
for s in 2024 2025 2026; do for x in smoothed best final; do
  python3 -m scripts.kuairec_data.l5o_paired --dataset "$D" --base "$N/h/seed$s/H0/test/$x/per_sample.jsonl.gz" \
    --variant "$N/h/seed$s/H1/test/$x/per_sample.jsonl.gz" --output "$OUT/paired/seed${s}_$x.json" \
    > "$OUT/paired/seed${s}_$x.log" 2>&1
  echo "$(date -Is) paired $s $x rc=$?" >> "$OUT/status.txt"
done; done
cd "$AC"
python3 -m scripts.mixer_reversal.analyze traj --root "$N" --family h --output "$OUT/traj_h.json" > "$OUT/traj_h.log" 2>&1
echo "$(date -Is) traj rc=$?" >> "$OUT/status.txt"
python3 -m scripts.mixer_reversal.analyze v2 --root "$N" --family h --output "$OUT/v2_h.json" > "$OUT/v2_h.log" 2>&1
echo "$(date -Is) v2 rc=$?" >> "$OUT/status.txt"
touch "$OUT/DONE"
