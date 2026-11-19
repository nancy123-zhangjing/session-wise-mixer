#!/bin/bash
# Pre-registered protocol v2 TIGER comparison: T1A minus T0A per seed and node (smoothed/best/final),
# predictions resolved from test/complete.json by prediction_sha256; scripts.kuairec_data.l5o_paired, code_7311328.
set -u
V=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_v2_20261118
D=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1/dataset
O=$V/analysis/tiger_paired
mkdir -p $O
cd $V/code_7311328
echo "$(date -Is) start" >> $O/status.txt
for s in 2024 2025 2026; do
  for n in smoothed best final; do
    pick() { python3 - "$V/tiger/seed$s/$1/test" "$n" <<'PY'
import json, glob, sys
b, n = sys.argv[1], sys.argv[2]
sha = json.load(open(f'{b}/complete.json'))[n]['prediction_sha256']
hits = [p for p in glob.glob(f'{b}/step_*/summary.json') if json.load(open(p))['prediction_sha256'] == sha]
assert len(hits) == 1, hits
print(hits[0].rsplit('/', 1)[0] + '/predictions.jsonl.gz')
PY
    }
    b=$(pick T0A) && v=$(pick T1A) || { echo "$(date -Is) $s $n pick failed" >> $O/status.txt; continue; }
    echo "$s $n base=$b variant=$v" >> $O/inputs.txt
    python3 -m scripts.kuairec_data.l5o_paired --dataset $D --base $b --variant $v --output $O/${n}_$s.json \
      && echo "$(date -Is) $s $n ok" >> $O/status.txt || echo "$(date -Is) $s $n FAILED" >> $O/status.txt
  done
done
echo "$(date -Is) done" >> $O/status.txt
touch $O/DONE
