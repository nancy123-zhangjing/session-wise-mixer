"""Read-only: time span of L5O train and test lists, grouped by the Asia/Shanghai date of the list's first record."""
import json
import numpy as np, pyarrow.compute as pc, pyarrow.parquet as pq
D = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1/dataset'
GROUPS = {'07-05..07-12': (18448, 18456), '07-13..07-31': (18456, 18475), '08-01..08-10': (18475, 18485),
          '08-11..08-26': (18485, 18501), '08-27..08-31': (18501, 18506), '09-01..09-03': (18506, 18509),
          '09-04': (18509, 18510), '09-05': (18510, 18511)}  # days since 1970-01-01 (18448 = 2020-07-05)
out = {}
for s in ('train', 'test'):
    t = pq.read_table(f'{D}/{s}.parquet', columns=['target_timestamp_us'])
    ts = pc.list_flatten(t.column('target_timestamp_us').combine_chunks()).to_numpy().astype(np.int64).reshape(-1, 5)
    span = (ts[:, -1] - ts[:, 0]) / 6e7; d = (ts[:, 0] // 10**6 + 8 * 3600) // 86400
    out[s] = {}
    for g, (a, b) in GROUPS.items():
        m = (d >= a) & (d < b)
        if m.any():
            sp = span[m]
            out[s][g] = dict(n=int(m.sum()), le10m=float((sp <= 10).mean()), m10_60=float(((sp > 10) & (sp <= 60)).mean()),
                             h1_24=float(((sp > 60) & (sp <= 1440)).mean()), gt1d=float((sp > 1440).mean()))
print(json.dumps(out, indent=1))
