"""Copy pre-training parents after a device-only source fix; preserve originals."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.session_wise_ar.runtime import source_identity
from scripts.rqcd_mixer.common import read_state, atomic_save, write_json, sha256


def main():
    current = source_identity()
    for family in ('tiger', 'diger'):
        source = ROOT/'parents'/family/'initial.pt'
        target = ROOT/'parents_devicefix'/family
        if target.exists():
            raise RuntimeError('Destination already exists; inspect before reuse')
        state = read_state(source)
        differences = {k for k in set(current) | set(state['sources'])
                       if current.get(k) != state['sources'].get(k)}
        assert differences == {'scripts/session_wise_ar/model.py'}, differences
        old_sources = state['sources']
        state['sources'] = current
        target.mkdir(parents=True)
        atomic_save(target/'initial.pt', state)
        restored = read_state(target/'initial.pt')
        for name in ('plain', 'intra'):
            for key, tensor in state['tokenizers'][name].items():
                assert torch.equal(tensor, restored['tokenizers'][name][key])
        for key, tensor in state['recommender'].items():
            assert torch.equal(tensor, restored['recommender'][key])
        write_json(target/'manifest.json', dict(family_config=state['family_config'],
                   binding=state['binding'], initial_sha256=sha256(target/'initial.pt'), sources=current))
        write_json(target/'DEVICE_FIX_PROVENANCE.json', dict(original_path=str(source),
                   original_sha256=sha256(source), old_sources=old_sources,
                   changed_files=sorted(differences), tensors_unchanged=True,
                   reason='Synchronize upstream cached device after module move; no model/settings change'))
        print(family, 'copied with identical tensors', flush=True)


if __name__ == '__main__':
    main()
