#!/bin/bash
# Post-run analysis of step 1 (oldsplit_v2code_20261119: current code on the old split), CPU notebook only.
# Waits (up to 30 hours) until all six arms x seeds have their two test summaries (validation best, R35),
# then scores them with the same functions as the accepted old-split re-scoring:
#   python3 -m scripts.mixer_reversal.analyze old --rerun-root <root>
# (exposure NDCG@5 gate against the stored per-sample values, EV, user-cluster bootstrap, validation curves).
# Usage: run_oldsplit.sh <step-1 root> <analysis code dir> [draws]
set -u
N=$1; AC=$2; DRAWS=${3:-2000}
OUT=$N/analysis
mkdir -p "$OUT"
deadline=$(( $(date +%s) + 30 * 3600 ))
while :; do
  n=0
  for s in 2024 2025 2026; do for a in H0 H1; do for x in best final; do
    [ -s "$N/h/seed$s/$a/test/$x/summary.json" ] && [ -s "$N/h/seed$s/$a/test/$x/per_sample.jsonl.gz" ] && n=$((n + 1))
  done; done; done
  echo "$(date -Is) test summaries: $n/12" > "$OUT/wait.txt"
  [ "$n" -eq 12 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then touch "$OUT/TIMEOUT"; exit 0; fi
  sleep 600
done
cd "$AC"
python3 -m scripts.mixer_reversal.analyze old --rerun-root "$N" --draws "$DRAWS" --output "$OUT/old_rerun.json" \
  > "$OUT/old_rerun.log" 2>&1
rc=$?
echo "$(date -Is) old_rerun rc=$rc" >> "$OUT/status.txt"
[ "$rc" -eq 0 ] && touch "$OUT/DONE" || touch "$OUT/FAILED"
