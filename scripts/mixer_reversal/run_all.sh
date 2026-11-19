#!/bin/bash
# Automated read-only pipeline for the Mixer-reversal checks (CPU notebook next to Inspire storage).
# Usage: run_all.sh <output root> <protocol v2 root>
# H and the old split run at once; the TIGER part waits until all six TIGER test phases are complete
# (up to 30 hours), then runs. Markers: H_DONE, ALL_DONE, TIGER_TIMEOUT; progress in status.txt.
set -u
OUT=$1
V=$2
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/out" "$OUT/logs"
step() {
  local name=$1
  shift
  if [ -s "$OUT/out/$name.json" ]; then echo "$(date -Is) $name already present" >> "$OUT/status.txt"; return 0; fi
  for attempt in 1 2; do
    if python3 -m scripts.mixer_reversal.analyze "$@" --output "$OUT/out/$name.json" > "$OUT/logs/${name}_$attempt.log" 2>&1; then
      echo "$(date -Is) $name ok" >> "$OUT/status.txt"
      return 0
    fi
    echo "$(date -Is) $name failed (attempt $attempt)" >> "$OUT/status.txt"
    sleep 60
  done
  return 1
}
echo "$(date -Is) start pid $$" >> "$OUT/status.txt"
step old old
step traj_h traj --root "$V" --family h
step v2_h v2 --root "$V" --family h
touch "$OUT/H_DONE"
deadline=$(( $(date +%s) + 30 * 3600 ))
while :; do
  n=0
  for s in 2024 2025 2026; do for a in T0A T1A; do [ -f "$V/tiger/seed$s/$a/test/complete.json" ] && n=$((n + 1)); done; done
  echo "$(date -Is) TIGER test phases complete: $n/6" > "$OUT/tiger_wait.txt"
  [ "$n" -eq 6 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then touch "$OUT/TIGER_TIMEOUT"; exit 0; fi
  sleep 600
done
step traj_tiger traj --root "$V" --family tiger
step v2_tiger v2 --root "$V" --family tiger
touch "$OUT/ALL_DONE"
echo "$(date -Is) all done" >> "$OUT/status.txt"
