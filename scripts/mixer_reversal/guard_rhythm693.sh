#!/bin/bash
# Guard for the H rhythm-control jobs (run locally, needs the inspire CLI).
# Every 10 min: read the job list. A job is resubmitted (workers resume from last.pt) only when
#   * the job list was read successfully and the job's latest attempt is in a terminal non-success
#     state (FAILED / STOPPED / CANCELLED), and
#   * env-zzl confirms that the job's seeds do not yet have all their test summaries.
# Network or CLI errors never count as job failures. At most two resubmissions per job.
# Exits after a resubmission (so the new job is checked by hand), when both jobs succeeded, or at the deadline.
set -u
cd /home/nancy/projects/experiment1
N=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_v2_rhythm693_20261119
E=experiments/09_kuairec_listwise/l5o_v2_rhythm693_20261119
declare -A SEEDS=([h-a]="2024 2025" [h-b]="2026")
end=$(( $(date +%s) + ${1:-12} * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  list=$(timeout 100 inspire --account sii-shared --json job list --workspace 分布式训练空间 2>/dev/null)
  if [ -z "$list" ]; then echo "$(date -u +%H:%M) job list unreadable; waiting"; sleep 600; continue; fi
  done_jobs=0
  for job in h-a h-b; do
    latest=""; status=""
    for suffix in -r3 -r2 ""; do
      st=$(echo "$list" | python3 -c "
import json,sys
d=json.load(sys.stdin)['data']['items']; s=[j['status'] for j in d if j['name']=='nancy-l5o-r693-$job$suffix']; print(s[0] if s else '')" 2>/dev/null)
      if [ -n "$st" ]; then latest="$suffix"; status="$st"; break; fi
    done
    echo "$(date -u +%H:%M) $job${latest} $status"
    [ "$status" = "SUCCEEDED" ] && done_jobs=$((done_jobs + 1)) && continue
    case "$status" in FAILED|STOPPED|CANCELLED) ;; *) continue ;; esac
    n=$(timeout 120 inspire --account sii-shared notebook exec env-zzl --workspace CPU资源空间 "n=0; for s in ${SEEDS[$job]}; do for a in H0 H1; do for x in smoothed best final; do [ -s $N/h/seed\$s/\$a/test/\$x/summary.json ] && n=\$((n+1)); done; done; done; echo COUNT \$n" 2>/dev/null | grep -o 'COUNT [0-9]*' | cut -d' ' -f2)
    want=$(( $(echo ${SEEDS[$job]} | wc -w) * 6 ))
    if [ -z "$n" ]; then echo "  env-zzl unreadable; not resubmitting"; continue; fi
    if [ "$n" -ge "$want" ]; then echo "  tests complete ($n/$want); no resubmission"; done_jobs=$((done_jobs + 1)); continue; fi
    case "$latest" in "") next=-r2 ;; -r2) next=-r3 ;; *) echo "  two resubmissions used; stopping"; exit 0 ;; esac
    timeout 100 inspire --account sii-shared job events nancy-l5o-r693-$job$latest --workspace 分布式训练空间 2>&1 | tail -5
    python3 -m scripts.kuairec_data.l5o_v2_submit $job --code $N/code_d27d29f --root $N --h-rhythm old693 \
      --prefix nancy-l5o-r693 --suffix $next --submit >> $E/SUBMIT_resume.jsonl 2>&1
    echo "  RESUBMITTED $job as nancy-l5o-r693-$job$next (tests $n/$want)"; tail -1 $E/SUBMIT_resume.jsonl
    exit 0
  done
  [ "$done_jobs" -eq 2 ] && { echo "BOTH_SUCCEEDED"; exit 0; }
  sleep 600
done
echo DEADLINE
