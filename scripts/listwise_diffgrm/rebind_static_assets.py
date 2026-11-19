"""Reuse only static PSE assets under a new split, with an explicit new manifest."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from .assets import dataset_binding, load_assets, sha256


def rebind(source, dataset, output):
    source, dataset, output = map(Path, (source, dataset, output))
    original = load_assets(source)
    binding = dataset_binding(dataset)
    manifest = copy.deepcopy(original.manifest)
    if manifest['fit_scope'] != 'full_static_catalog_no_behavior':
        raise ValueError('Cannot rebind behavior-fitted assets')
    for name in ('item_map.json', 'item_text.jsonl'):
        if binding['files'][name] != manifest['dataset_binding']['files'][name]:
            raise ValueError(f'Static catalog changed: {name}')
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in manifest['artifacts'].items():
        shutil.copyfile(source / name, output / name)
        if sha256(output / name) != expected['sha256']:
            raise ValueError(f'Copy verification failed: {name}')
    manifest['source_asset_manifest_sha256'] = sha256(source / 'manifest.json')
    manifest['source_dataset_binding'] = manifest['dataset_binding']
    manifest['dataset_binding'] = binding
    manifest['reuse_note'] = 'Static catalog only; no behavioral checkpoint, optimizer or trained Mixer reused'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return load_assets(output, dataset)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('source', 'dataset', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    rebind(a.source, a.dataset, a.output)
