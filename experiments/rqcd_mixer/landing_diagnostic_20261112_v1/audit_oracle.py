"""CPU-only explanation of oracle mismatches; no inference/training rerun."""
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch

torch.set_num_threads(4)
source, out = map(Path, sys.argv[1:3])
meta = json.loads((source/'prepared/PREPARED.json').read_text())
content = np.load(meta['content_path'])
summary = json.loads((out/'COMPLETE.json').read_text())
unique_content = np.unique(content, axis=0)
report = {'content_path': meta['content_path'], 'catalog_count': len(content),
          'distinct_content_vectors': len(unique_content), 'arms': {}}
for arm,r in summary['results'].items():
    state = torch.load(r['checkpoint'], map_location='cpu', weights_only=False)
    z = state['catalog']['z'].numpy()
    unique_z = np.unique(z, axis=0)
    errors = r['oracle']['original_z']['errors']
    identical_content = sum(np.array_equal(content[e['target']-1],content[e['rq']-1]) for e in errors)
    identical_z = sum(np.array_equal(z[e['target']-1],z[e['rq']-1]) for e in errors)
    nonidentical = []
    for e in errors:
        i,j = e['target']-1,e['rq']-1
        if not np.array_equal(z[i],z[j]):
            nonidentical.append({**e, 'max_abs_z_difference': float(np.max(np.abs(z[i]-z[j]))),
                'squared_content_distance_fp64': float(np.sum((content[i].astype('float64')-content[j])**2)),
                'target_z_norm': float(np.linalg.norm(z[i])),
                'fp32_mm_distance_target': float(np.sum(z[i]*z[i])*2-2*(z[i]@z[i])),
                'fp32_mm_distance_selected': float(np.sum(z[i]*z[i])+np.sum(z[j]*z[j])-2*(z[i]@z[j]))})
    report['arms'][arm] = {'distinct_latent_vectors': len(unique_z), 'oracle_errors': len(errors),
        'errors_with_identical_input_content': identical_content,
        'errors_with_identical_latent_vector': identical_z, 'nonidentical_errors': nonidentical}
    del state
(out/'ORACLE_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
