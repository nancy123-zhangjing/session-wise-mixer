"""Per-day L5O training volume, test/validation start days and same-day pool coverage.

Read-only. Run 2026-11-17 on env-zzl as an inline `python3 -I -c` with this exact code
(argument: dataset directory); output saved in DAY_VOLUME.txt.
"""
import sys

import numpy as np
import pyarrow.parquet as pq

L = sys.argv[1]
DAY = 86400000000
tr = pq.read_table(f'{L}/train.parquet', columns=['target_timestamp_us', 'target_item_ids'])
ts = tr['target_timestamp_us'].combine_chunks().flatten().to_numpy(); it = tr['target_item_ids'].combine_chunks().flatten().to_numpy()
ts = ts[it > 0]; it = it[it > 0]
te = pq.read_table(f'{L}/test.parquet', columns=['target_timestamp_us', 'target_item_ids'])
tts = np.array(te['target_timestamp_us'].to_pylist()); tit = np.array(te['target_item_ids'].to_pylist())
va = pq.read_table(f'{L}/validation.parquet', columns=['target_timestamp_us'])
vts = np.array(va['target_timestamp_us'].to_pylist())
t0 = min(ts.min(), tts[tts > 0].min(), vts[vts > 0].min())
d = (ts - t0) // DAY; td = (tts[:, 0] - t0) // DAY; vd = (vts[:, 0] - t0) // DAY
print('train events per day 60..76', np.bincount(d, minlength=77)[60:77].tolist())
print('test users by start day 60..76', np.bincount(td, minlength=77)[60:77].tolist(), 'below60', int((td < 60).sum()))
print('val users by start day 60..76', np.bincount(vd, minlength=77)[60:77].tolist())
print('tercile sizes', np.bincount(np.digitize(td, [71, 72]), minlength=3).tolist())
for day in (68, 70, 71, 72, 73, 74):
    s = set(it[d == day].tolist()); t = tit[td == day].reshape(-1); t = t[t > 0]
    print('day', day, 'train distinct', len(s), 'test targets', len(t), 'in same-day train pool',
          round(float(np.mean([x in s for x in t])) if len(t) else -1, 3))
gap = (tts[:, 0] - vts[:, 0]) / 3.6e9
print('val->test gap hours quantiles', np.quantile(gap, [.1, .5, .9]).round(2).tolist())
