"""Full freeze35 archive, fresh safe extraction, and disclosed CPU restoration.

pack/verify use only the standard library. Data/PSE and historical prediction
dependencies remain EXTERNAL and are hashed explicitly; the archive is NOT
self-contained. restore needs an explicitly selected existing CPU Python.
No platform API, GPU allocation, training, deletion or automatic retry exists.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile

FORMAT = 'stage09_freeze35_full_backup_v1'
MANIFEST = 'DELIVERY_MANIFEST.json'
BLOCK = 4 * 1024 * 1024
NAMES = {'A': 'H1-Freeze20-R35', 'B': 'H0-Freeze32-R35', 'C': 'H0-Freeze20-R35'}
START = {'A': (33, 26, 200), 'B': (41, 32, 360), 'C': (5, 5, 0)}
PARENTS = {'A': '1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9',
           'B': '8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e',
           'C': 'b1adf68c7c5b3f1818773c1ef34adaeac4df49be57a867c6b11bcaf38cdf9166'}
ANCESTOR = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique)


def read(path): return decode(Path(path).read_bytes())


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(encoded(value))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''): value.update(block)
    return value.hexdigest()


def no_links(path):
    path = Path(os.path.abspath(path))
    for node in [*reversed(path.parents), path]:
        if not node.exists() and not node.is_symlink(): continue
        info = node.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symlink/reparse path forbidden: ' + str(node))
    return path


def member_name(name):
    if not isinstance(name, str): raise ValueError('Archive name must be text')
    value = PurePosixPath(name)
    if (not name or value.is_absolute() or value.as_posix() != name or '\\' in name or ':' in name
            or any(ord(c) < 32 for c in name)
            or any(p in ('', '.', '..') or p.endswith(('.', ' ')) for p in value.parts)):
        raise ValueError('Unsafe member: ' + repr(name))
    for part in value.parts:
        if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', part.split('.')[0], re.I):
            raise ValueError('Windows reserved member: ' + name)
    return value


def register(name, known):
    parts = member_name(name).parts
    for length in range(1, len(parts) + 1):
        prefix = '/'.join(parts[:length]); identity = (prefix, length == len(parts))
        previous = known.get(prefix.casefold())
        if previous is not None and (previous != identity or identity[1]):
            raise ValueError('Duplicate/case/file-directory collision: ' + name)
        known[prefix.casefold()] = identity


def child(root, name): return no_links(Path(root).joinpath(*member_name(name).parts))


def file_record(path):
    path = no_links(path)
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode): raise ValueError('Not a regular file: ' + str(path))
    before = path.stat(); digest = sha(path); after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while hashing: ' + str(path))
    return dict(bytes=after.st_size, sha256=digest)


def check(path, record):
    if file_record(path) != {k: record[k] for k in ('bytes', 'sha256')}:
        raise ValueError('Size/SHA mismatch: ' + str(path))
    return Path(path)


def inventory(root):
    root = no_links(root); result, known = {}, {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs: no_links(Path(directory) / name)
        for name in sorted(files):
            path = no_links(Path(directory) / name); relative = path.relative_to(root).as_posix()
            register(relative, known); result[relative] = file_record(path)
    return dict(sorted(result.items()))


def plan():
    rows = [('R', e) for e in range(1, 6)]
    for cycle in range(10): rows += [('T', 5 + 3 * cycle), *[('R', 6 + 3 * cycle + j) for j in range(3)]]
    return rows


def node_specs(branch):
    start, epoch, t = START[branch]
    result = {'initial.pt': dict(phase=start, update=0, boundary_prepared=False, R=epoch * 693, T=t)}
    for phase in range(start, 45):
        kind, epoch = plan()[phase]
        if kind == 'T' and branch == 'C' and phase < 25: t += 40
        name = f'phase{phase:02d}_{kind}_R{epoch:02d}_T{t:03d}.pt'
        result[name] = dict(phase=phase + 1, update=0, boundary_prepared=False, R=epoch * 693, T=t)
    if branch == 'C': result['R20_pre_freeze.pt'] = dict(phase=25, update=0, boundary_prepared=False, R=13860, T=200)
    result['R35_training_boundary.pt'] = dict(phase=44, update=693, boundary_prepared=True, R=24255, T=t)
    return result


def runtime_coverage(root):
    """No tensor loading: bind every required filename/receipt and selection."""
    coverage, roles = {}, []
    for branch, name in NAMES.items():
        directory = root / 'runs' / name
        if (directory / '.running').exists(): raise ValueError('Run still has an ownership lock: ' + name)
        request, complete = read(directory / 'request.json'), read(directory / 'complete.json')
        selected, tests = read(directory / 'selection.json'), read(directory / 'test_predictions_complete.json')
        if (request['branch'] != branch or request['parent_sha256'] != PARENTS[branch]
                or complete['status'] != 'training_and_validation_complete' or complete['branch'] != branch
                or (complete['phase'], complete['r_updates'], complete['t_updates']) != (45, 24255, 360 if branch == 'B' else 200)
                or sha(directory / 'selection.json') != complete['selection_sha256']
                or tests['branch'] != branch or tests['selection_sha256'] != complete['selection_sha256']):
            raise ValueError('Incomplete/mixed branch: ' + branch)
        chosen = selected['selected']; expected_tests = {35, chosen['epoch']} if branch in ('A', 'C') else {35}
        if len(tests['results']) != len(expected_tests) or {r['epoch'] for r in tests['results']} != expected_tests:
            raise ValueError('Missing or extra planned test endpoints')
        original_coverage = read(directory / 'CHECKPOINT_COVERAGE.json')
        if (sha(directory / 'CHECKPOINT_COVERAGE.json') != complete['checkpoint_coverage_sha256']
                or original_coverage.get('status') != 'passed'):
            raise ValueError('Unaccepted checkpoint coverage')
        specs = node_specs(branch); rows = {}
        for filename, clocks in specs.items():
            path = directory / filename
            receipt = read(directory / 'checkpoint_receipts' / (filename + '.json'))
            check(path, receipt)
            if (receipt['path'] != filename or receipt['r_updates'] != clocks['R'] or receipt['t_updates'] != clocks['T']
                    or receipt['progress'] != {k: clocks[k] for k in ('phase', 'update', 'boundary_prepared')}
                    or set(receipt['fields']) != {'model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'binding'}
                    or not all(v is True for v in receipt['fields'].values())):
                raise ValueError('Incomplete/mislabelled boundary: ' + branch + '/' + filename)
            rows[filename] = dict(receipt, expected_clocks=clocks)
        if len(original_coverage['records']) != len(specs) - 1:
            raise ValueError('Original coverage omitted a required phase')
        for filename in set(specs) - {'R35_training_boundary.pt'}:
            if original_coverage['records'][filename] != {k: v for k, v in rows[filename].items() if k != 'expected_clocks'}:
                raise ValueError('Coverage does not bind actual boundary receipt')
        if sha(directory / 'last.pt') != complete['last_sha256']: raise ValueError('Rolling last changed')
        coverage[branch] = dict(name=name, immutable_count=len(rows), nodes=rows,
            rolling_last=file_record(directory / 'last.pt'), selected=chosen)
        for role, filename in [('initial', 'initial.pt'), ('R35', next(n for n in specs if n.startswith('phase44_')))]:
            roles.append(dict(branch=branch, role=role, path='goal/runs/' + name + '/' + filename, kind='recovery35'))
        if branch == 'C': roles.append(dict(branch=branch, role='R20', path='goal/runs/' + name + '/R20_pre_freeze.pt', kind='recovery35'))
        if chosen['kind'] == 'new':
            filename = PurePosixPath(chosen['checkpoint']).name
            if filename not in specs or rows[filename]['sha256'] != chosen['checkpoint_sha256']:
                raise ValueError('Selected best is outside immutable coverage')
            roles.append(dict(branch=branch, role='best', path='goal/runs/' + name + '/' + filename, kind='recovery35'))
        elif branch == 'A' and chosen['kind'] == 'reference':
            roles.append(dict(branch=branch, role='best', epoch=chosen['epoch'], source_sha256=chosen['checkpoint_sha256'], kind='historical_h1'))
        else: raise ValueError('Unsupported selected model origin')
    if sum(v['immutable_count'] for v in coverage.values()) != 63: raise ValueError('Expected all 63 immutable checkpoints')
    return coverage, roles


def execution_inventory(root, expected_sha):
    code = root / 'code'; path = code / 'FREEZE35_EXECUTION_MANIFEST.json'
    if sha(path) != expected_sha: raise ValueError('Execution manifest differs from caller-approved SHA')
    manifest = read(path)
    if (manifest.get('format') != 'stage09_h_freeze35_execution_v1'
            or manifest.get('package_mode') != 'committed' or manifest.get('gpu_execution_allowed') is not True):
        raise ValueError('Not a committed formal execution package')
    expected = dict(manifest['execution_files']); expected.update(manifest['metadata'])
    expected[path.name] = file_record(path)
    if inventory(code) != expected: raise ValueError('Frozen source inventory differs')
    return manifest


def external_dependencies(parents, references):
    result = {'dataset': copy.deepcopy(parents['dataset']), 'assets': copy.deepcopy(parents['assets']), 'historical_results': []}
    seen = set()
    for split in ('validation', 'test'):
        for rows in references[split].values():
            for row in rows.values():
                for key in ('summary', 'predictions'):
                    identity = row[key]
                    if identity in seen: continue
                    seen.add(identity)
                    result['historical_results'].append(dict(remote=row[key], local=row['local'][key],
                        bytes=row[key + '_bytes'], sha256=row[key + '_sha256']))
    return result


def mapped_path(path, mappings):
    # Exact, longest-prefix relocation without changing the recorded manifest.
    text = str(path)
    for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
        if text == old: return Path(new)
        for separator in ('/', '\\'):
            if text.startswith(old.rstrip('/\\') + separator):
                tail = text[len(old.rstrip('/\\')) + 1:].replace('\\', '/')
                return Path(new).joinpath(*PurePosixPath(tail).parts)
    return Path(text)


def verify_dependencies(dependencies, mode, *, dataset=None, assets=None, mappings=None):
    mappings = mappings or {}; checked = {}
    for group, override in [('dataset', dataset), ('assets', assets)]:
        row = dependencies[group]; root = no_links(override or mapped_path(row[mode], mappings))
        if sha(root / 'manifest.json') != row['manifest_sha256']:
            raise ValueError('External manifest differs: ' + group)
        for name, record in row['files'].items(): check(child(root, name), record)
        checked[group] = dict(path=str(root), files=len(row['files']), manifest_sha256=row['manifest_sha256'])
    for row in dependencies['historical_results']: check(mapped_path(row[mode], mappings), row)
    checked['historical_results'] = len(dependencies['historical_results'])
    return checked


def pack(root, archive, *, execution_sha, release_receipt, references_mode='remote'):
    root, archive = no_links(root), no_links(archive)
    if archive.exists() or archive.is_relative_to(root): raise FileExistsError('Archive must be NEW and outside source root')
    archive.parent.mkdir(parents=True, exist_ok=True)
    execution_inventory(root, execution_sha)
    release = read(release_receipt)
    if (release.get('status') != 'verified' or release.get('all_goal_jobs_released') is not True
            or not release.get('evidence')):
        raise ValueError('Explicit completed/released-job evidence is required before full packaging')
    coverage, roles = runtime_coverage(root)
    parents = read(root / 'code/FREEZE35_PARENTS.json'); references = read(root / 'code/FREEZE35_REFERENCE_RESULTS.json')
    dependencies = external_dependencies(parents, references)
    dependency_receipt = verify_dependencies(dependencies, references_mode)
    original = inventory(root); payload = {'goal/' + name: root / name for name in original}
    def add(name, source, expected):
        path = no_links(source)
        if sha(path) != expected: raise ValueError('Required historical state bytes differ: ' + str(path))
        payload[name] = path
    for branch in NAMES:
        row = parents['parents'][branch]
        if row['sha256'] != PARENTS[branch]: raise ValueError('Parent identity changed')
        add(f'references/parents/{branch}.pt', row[references_mode], PARENTS[branch])
    ancestor = parents['ancestors']['A']
    if ancestor['sha256'] != ANCESTOR: raise ValueError('H1 R20 ancestor identity changed')
    add('references/ancestors/H1_R20.pt', ancestor[references_mode], ANCESTOR)
    for role in roles:
        if role['kind'] == 'historical_h1':
            ref = references['validation']['A'][str(role['epoch'])]
            source = ref['checkpoint'] if references_mode == 'remote' else ref['local']['checkpoint']
            name = f'references/historical/A_R{role["epoch"]}.pt'
            add(name, source, role['source_sha256']); role['path'] = name
    payload['references/tools/backup_delivery.py'] = no_links(__file__)
    payload['references/lifecycle/release.json'] = no_links(release_receipt)
    for number, row in enumerate(release['evidence']):
        path = no_links(row['path'])
        if sha(path) != row['sha256']: raise ValueError('Release evidence changed')
        payload[f'references/lifecycle/evidence{number:03d}.json'] = path
    known = {}; records = {}
    for name, path in sorted(payload.items()): register(name, known); records[name] = file_record(path)
    preflight = [name for name in records if 'preflight' in name.lower() and name.endswith(('.pt', '.json', '.jsonl', '.log'))]
    manifest = dict(format=FORMAT, source_root=str(root), files=records, payload_files=len(records),
        payload_bytes=sum(v['bytes'] for v in records.values()), execution_manifest_sha256=execution_sha,
        checkpoint_coverage=coverage, model_restore_roles=roles, dependencies=dependencies,
        self_contained=False, software_environment_bundled=False,
        dependency_note='Dataset/PSE and historical prediction files are external; all declared SHA must pass at restore. No standalone-archive restore claim.',
        dependencies_verified_at_pack=dependency_receipt, preflight_separate_scope=dict(included=preflight,
            formal_recovery_sources=False, note='Disposable bundles are retained evidence, never selected as formal state'),
        release_receipt='references/lifecycle/release.json', release_evidence_scope='Bound external release evidence; no platform query by this tool',
        pack_tool_sha256=sha(__file__), required_immutable_checkpoints=63, rolling_last_checkpoints=3,
        actual_model_restore='pending; byte/coverage verification is not a model restore')
    data = encoded(manifest); manifest_sha = hashlib.sha256(data).hexdigest()
    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo(MANIFEST); info.size = len(data); info.mode = 0o644; info.mtime = 0
        tar.addfile(info, io.BytesIO(data))
        for name, path in sorted(payload.items()):
            check(path, records[name]); info = tarfile.TarInfo(name)
            info.size = records[name]['bytes']; info.mode = 0o644; info.mtime = 0
            with Path(path).open('rb') as stream: tar.addfile(info, stream)
    if inventory(root) != original: raise ValueError('Source root changed during packing; retain partial archive, do not accept')
    for name, path in payload.items(): check(path, records[name])
    receipt = dict(status='packed_not_model_restored', format=FORMAT, archive=str(archive), **file_record(archive),
                   manifest_sha256=manifest_sha, payload_files=len(records), immutable_checkpoints=63,
                   self_contained=False, dependencies_verified=True, model_restore_pending=True)
    write_new(archive.with_suffix(archive.suffix + '.receipt.json'), receipt)
    return receipt


def extract_bytes(archive, destination, archive_sha, manifest_sha):
    """Generic byte verifier. Public verify additionally enforces real coverage."""
    archive, destination = no_links(archive), no_links(destination)
    if destination.exists() or sha(archive) != archive_sha: raise ValueError('Archive SHA differs or destination is not fresh')
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers(); known = {}
        for member in members:
            register(member.name, known)
            if not member.isreg() or member.sparse: raise ValueError('Only nonsparse regular members are permitted')
        if len([m for m in members if m.name == MANIFEST]) != 1: raise ValueError('Exactly one manifest is required')
        meta = next(m for m in members if m.name == MANIFEST)
        if meta.size > 32 * 1024 * 1024: raise ValueError('Manifest exceeds the explicit bound')
        raw = tar.extractfile(meta).read()
        if hashlib.sha256(raw).hexdigest() != manifest_sha: raise ValueError('Manifest SHA mismatch')
        manifest = decode(raw)
        expected = manifest['files']
        if {m.name for m in members} != {MANIFEST, *expected}: raise ValueError('Archive members differ from full manifest')
        for member in members:
            if member.name != MANIFEST and member.size != expected[member.name]['bytes']:
                raise ValueError('Tar member size differs: ' + member.name)
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            path = child(destination, member.name); path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(); written = 0
            with tar.extractfile(member) as source, path.open('xb') as target:
                for block in iter(lambda: source.read(BLOCK), b''):
                    target.write(block); digest.update(block); written += len(block)
            wanted = dict(bytes=len(raw), sha256=manifest_sha) if member.name == MANIFEST else expected[member.name]
            if written != wanted['bytes'] or digest.hexdigest() != wanted['sha256']:
                raise ValueError('Member content SHA differs: ' + member.name)
    for name, row in expected.items(): check(child(destination, name), row)
    return manifest


def verify(archive, destination, *, archive_sha, manifest_sha, dataset=None, assets=None, mappings=None):
    manifest = extract_bytes(archive, destination, archive_sha, manifest_sha)
    destination = Path(destination)
    if (manifest.get('format') != FORMAT or manifest.get('required_immutable_checkpoints') != 63
            or manifest.get('self_contained') is not False): raise ValueError('Not the full formal freeze35 archive')
    execution_inventory(destination / 'goal', manifest['execution_manifest_sha256'])
    coverage, _ = runtime_coverage(destination / 'goal')
    if coverage != manifest['checkpoint_coverage']: raise ValueError('Restored coverage differs from pack proof')
    dependency_proof = verify_dependencies(manifest['dependencies'], 'local', dataset=dataset, assets=assets, mappings=mappings)
    result = dict(format=FORMAT, status='bytes_and_structural_coverage_passed', archive_sha256=archive_sha,
        manifest_sha256=manifest_sha, extracted_payloads=len(manifest['files']), all_member_and_disk_sha256_verified=True,
        immutable_checkpoints=63, rolling_last_checkpoints=3, dependencies=dependency_proof, self_contained=False,
        actual_model_restore='pending', cuda_exact_restore=False, source_inputs_modified=False)
    write_new(destination / 'BACKUP_VERIFICATION.json', result)
    return result


def memory_relocation(saved, controller, fingerprint, verify_binding, *, c20=None):
    """Metadata-only copy after strict source/config/data verification.

    The caller checks original bytes first. This never changes model/optimizer/
    scheduler/progress/RNG tensors or the source file. C's gate hashes are mapped
    to the SAME R20 state under the disclosed local binding.
    """
    original = copy.deepcopy(saved)
    proof = verify_binding(original['execution_binding'], controller.execution_binding)
    if proof['extensions']: raise ValueError('CPU restore may relocate paths, not add runtime sources')
    expected_protocol = copy.deepcopy(original['protocol'])
    expected_protocol['source_proof']['relocations'] = controller.protocol['source_proof']['relocations']
    if expected_protocol != controller.protocol: raise ValueError('Scientific recovery protocol differs beyond path relocation')
    if original['parent_binding'] != controller.parent_binding or original['parent_hashes'] != controller.parent_hashes:
        raise ValueError('Authenticated parent metadata differs')
    adapted = copy.deepcopy(original)
    adapted['execution_binding'] = copy.deepcopy(controller.execution_binding)
    adapted['inner']['binding'] = copy.deepcopy(controller.execution_binding)
    adapted['protocol'] = copy.deepcopy(controller.protocol)
    gate_proof = None
    if adapted.get('replay_gate') is not None:
        if c20 is None: raise ValueError('C gate restoration requires its archived R20 pre-freeze state')
        old_sha = fingerprint(c20['inner'])
        if (adapted['replay_gate_state_sha256'] != old_sha
                or adapted['replay_gate']['checkpoint_state_sha256'] != old_sha):
            raise ValueError('C gate does not bind archived R20 state')
        r20 = copy.deepcopy(c20['inner']); r20['binding'] = copy.deepcopy(controller.execution_binding)
        new_sha = fingerprint(r20)
        gate_proof = dict(old_r20_inner_sha256=old_sha, relocated_r20_inner_sha256=new_sha,
                          old_gate_protocol_sha256=adapted['replay_gate']['protocol_sha256'],
                          relocated_gate_protocol_sha256=controller.protocol_sha256)
        adapted['replay_gate_state_sha256'] = new_sha
        adapted['replay_gate']['checkpoint_state_sha256'] = new_sha
        adapted['replay_gate']['protocol_sha256'] = controller.protocol_sha256
    for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'progress', 'r_updates', 't_updates'):
        if fingerprint(original['inner'][key]) != fingerprint(adapted['inner'][key]):
            raise ValueError('Metadata relocation changed scientific state: ' + key)
    return adapted, dict(only_disclosed_metadata_paths_changed=True, source_proof=proof,
        binding_hash_method='frozen recovery.fingerprint SHA256 value hash',
        old_execution_binding_sha256=fingerprint(original['execution_binding']),
        relocated_execution_binding_sha256=fingerprint(adapted['execution_binding']),
        old_protocol_sha256=fingerprint(original['protocol']), relocated_protocol_sha256=fingerprint(adapted['protocol']),
        gate_mapping=gate_proof, original_archive_bytes_untouched=True)


def historical_cpu_controller(saved, ancestor, *, dataset, assets):
    """Authenticate old H1 first, then explicitly rebind an in-memory CPU copy."""
    from scripts.listwise_diffgrm.run import build_context
    from scripts.listwise_diffgrm.recovery import RecoveryController as Legacy, fingerprint
    from scripts.listwise_diffgrm.recovery35 import verify_binding
    inner = saved.get('inner', saved)
    config = copy.deepcopy(inner['binding']['config']); config.update(dataset=dataset, assets=assets)
    trainer, _, data, current = build_context(config, 'cpu', inner['precision'], True)
    sealed = saved['execution_binding'] if 'inner' in saved else current
    proof = verify_binding(inner['binding'], current)
    original = Legacy(trainer, 'freeze', ancestor['binding'], sealed, ANCESTOR)
    original.fork(ancestor, exact_rng=False)
    if 'inner' in saved:
        original.restore(saved, exact_rng=False)  # Original sealed envelope, never softened.
    elif (saved['arm'] != 'H1' or (saved['r_updates'], saved['t_updates']) != (13860, 200)
          or fingerprint(saved) != fingerprint(ancestor)):
        raise ValueError('Historical raw selection is not the authenticated R20 ancestor')
    cpu = Legacy(trainer, 'freeze', ancestor['binding'], current, ANCESTOR)
    cpu.fork(ancestor, exact_rng=False)
    if 'inner' in saved:
        adapted = copy.deepcopy(saved)
        adapted['execution_binding'] = copy.deepcopy(current)
        adapted['inner']['binding'] = copy.deepcopy(current)
        adapted['protocol'] = copy.deepcopy(cpu.protocol)
        cpu.restore(adapted, exact_rng=False)
    return cpu, data, inner, dict(original_legacy_envelope_authenticated=True,
        source_proof=proof, old_execution_binding_sha256=fingerprint(inner['binding']),
        relocated_execution_binding_sha256=fingerprint(current),
        allowed_runtime_extensions=proof['extensions'], original_archive_bytes_untouched=True,
        only_path_and_declared_runtime_extension_metadata_rebound=True)


def cpu_worker(extracted, branch, dataset, assets, output):
    """Real CPU-only model, optimizers, SID and target-free finite forward."""
    extracted = Path(extracted).resolve(); code = extracted / 'goal/code'
    if os.environ.get('FREEZE35_CPU_RESTORE_SUPERVISED') != '1':
        raise RuntimeError('Use the authenticated restore supervisor, not the private worker directly')
    verification = read(extracted / 'BACKUP_VERIFICATION.json')
    if (sha(extracted / 'BACKUP_VERIFICATION.json') != os.environ.get('FREEZE35_CPU_VERIFICATION_SHA256')
            or verification.get('status') != 'bytes_and_structural_coverage_passed'
            or sha(extracted / MANIFEST) != verification['manifest_sha256']):
        raise ValueError('CPU worker has no authenticated byte-verification receipt')
    execution_inventory(extracted / 'goal', read(extracted / MANIFEST)['execution_manifest_sha256'])
    os.environ.update(CUDA_VISIBLE_DEVICES='', DIFFGRM_ENGINE_ROOT=str(code / 'external/DiffGRM'), PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True; sys.path.insert(0, str(code))
    import torch
    from scripts.listwise_diffgrm.recovery35 import build_recovery35, verify_binding, REFERENCE_R20_PREDICTION_SHA256
    from scripts.listwise_diffgrm.recovery import fingerprint
    from scripts.listwise_diffgrm.training import rng_state, restore_rng
    from scripts.listwise_diffgrm.recovery_diagnostics import diagnostic_context
    from scripts.listwise_diffgrm.run import microbatches
    if torch.version.cuda is not None: raise RuntimeError('Explicit CPU-only Torch environment required')
    module = sys.modules['scripts.listwise_diffgrm.recovery35']
    if Path(module.__file__).resolve() != code / 'scripts/listwise_diffgrm/recovery35.py':
        raise ValueError('Runtime imported from another checkout')
    torch.set_num_threads(4); host_rng = rng_state()
    manifest = read(extracted / MANIFEST); original_files = manifest['files']
    def load(relative):
        path = child(extracted, relative); check(path, original_files[relative])
        return torch.load(path, map_location='cpu', weights_only=False)
    controller, _, data = build_recovery35(extracted / f'references/parents/{branch}.pt', branch, 'cpu',
        dataset=dataset, assets=assets, ancestor_path=extracted / 'references/ancestors/H1_R20.pt' if branch == 'A' else None,
        inspection=True)
    roles = [row for row in manifest['model_restore_roles'] if row['branch'] == branch]
    directory = 'goal/runs/' + NAMES[branch] + '/'
    initial = load(directory + 'initial.pt')
    c20 = load(directory + 'R20_pre_freeze.pt') if branch == 'C' else None
    freeze_anchor = c20['freeze_hashes'] if c20 is not None else initial['freeze_hashes']
    checked_states, restored = {}, {}
    try:
        # Every named state plus rolling last gets real field/shape/clock checks.
        for filename in [*node_specs(branch), 'last.pt']:
            relative = directory + filename; state = load(relative)
            if (state.get('format') != initial['format'] or state.get('inspection_only') is not False
                    or any(state[key] != initial[key] for key in ('protocol', 'parent_binding', 'execution_binding', 'parent_hashes'))
                    or state['inner']['arm'] != NAMES[branch] or state['inner']['binding'] != state['execution_binding']):
                raise ValueError('Checkpoint source/envelope identity differs: ' + filename)
            phase = state['inner']['progress']['phase']
            anchor = freeze_anchor if phase >= controller.freeze_phase else None
            if state['freeze_hashes'] != anchor: raise ValueError('Checkpoint changed canonical freeze anchor')
            gate = state.get('replay_gate')
            if branch == 'C' and phase >= 25:
                needs_gate = phase > 25 or state['inner']['progress']['boundary_prepared']
                if needs_gate and gate is None: raise ValueError('C checkpoint passed R20 without a gate')
                if gate is not None:
                    old_r20 = fingerprint(c20['inner'])
                    protocol_sha = hashlib.sha256(json.dumps(state['protocol'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    if (gate.get('status') != 'passed' or gate.get('branch') != 'C'
                            or gate.get('protocol_sha256') != protocol_sha
                            or gate.get('checkpoint_state_sha256') != old_r20
                            or state.get('replay_gate_state_sha256') != old_r20
                            or gate.get('reference_prediction_sha256') != REFERENCE_R20_PREDICTION_SHA256):
                        raise ValueError('C checkpoint gate/source binding differs')
            elif gate is not None: raise ValueError('Replay gate in a non-gated checkpoint')
            controller.validate_clocks(state['inner'], freeze_hashes=anchor)
            controller._validate_structures(state['inner'])
            sid = state['inner']['alignment']['hard_codes']; size = controller.trainer.config['model']['codebook_size']
            if (sid.ndim != 2 or sid.shape[1] != 4 or not sid[0].eq(-1).all()
                    or not ((sid[1:] >= 0) & (sid[1:] < size)).all()):
                raise ValueError('Checkpoint SID range/PAD layout is invalid')
            if set(state['inner']['rng']) != {'python', 'numpy', 'torch', 'cuda'}:
                raise ValueError('Incomplete RNG state')
            checked_states[filename] = dict(sha256=original_files[relative]['sha256'], fields_shapes_clocks_verified=True,
                                            sid_sha256=fingerprint(sid), sid_range_pad_verified=True, rng_fields_complete=True)
            del state
        for role in roles:
            relative = role['path']
            if relative in restored: continue
            state = load(relative)
            if role['kind'] == 'recovery35':
                active, active_data, source_inner = controller, data, state['inner']
                adapted, relocation = memory_relocation(state, active, fingerprint, verify_binding, c20=c20)
                active.restore(adapted, exact_rng=False)
            elif role['kind'] == 'historical_h1' and branch == 'A':
                active, active_data, source_inner, relocation = historical_cpu_controller(state,
                    load('references/ancestors/H1_R20.pt'), dataset=dataset, assets=assets)
            else: raise ValueError('Unknown critical restore role')
            after = active.checkpoint(); inner = after['inner']
            for key in ('model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler'):
                if fingerprint(inner[key]) != fingerprint(source_inner[key]): raise ValueError('Actual loaded state differs: ' + key)
            expected_rng = copy.deepcopy(source_inner['rng']); expected_rng['cuda'] = None
            if fingerprint(inner['rng']) != fingerprint(expected_rng): raise ValueError('CPU RNG restore differs')
            sid_before = fingerprint(active.trainer.alignment.hard_codes)
            before = fingerprint(after)
            batch = next(microbatches(active_data, [0, 1], 2))
            with diagnostic_context(active.trainer), torch.autocast('cpu', enabled=False):
                ids, mask = batch['history_item_ids'], batch['history_mask']
                if ids.shape != (2, 50) or not mask.any(1).all(): raise ValueError('Expected two real padded history rows')
                history = active.trainer.alignment.hard_codes[ids]
                encoded_history = active.trainer.model._encode(history, mask)
                logits = active.trainer.model._decode(encoded_history, torch.full((2, 4), -1, dtype=torch.long), item_index=0)
                if not torch.isfinite(encoded_history).all() or not torch.isfinite(logits).all(): raise ValueError('Nonfinite target-free forward')
            if fingerprint(active.checkpoint()) != before or fingerprint(active.trainer.alignment.hard_codes) != sid_before:
                raise ValueError('Minimal inference changed state/SID')
            restored[relative] = dict(status='cpu_relocated_restore_passed', source_sha256=original_files[relative]['sha256'],
                relocation=relocation, scientific_parameters_and_both_optimizers_identical=True,
                cpu_rng_restored=True, cuda_rng_restored=False,
                retained_original_cuda_rng_sha256=fingerprint(source_inner['rng']['cuda']),
                r_updates=inner['r_updates'], t_updates=inner['t_updates'], progress=inner['progress'],
                history_only_batch=[2, 50], decoder_logits_shape=list(logits.shape), finite_forward=True,
                sid_not_refreshed=True, updates_performed=0)
            check(child(extracted, relative), original_files[relative])
    finally:
        restore_rng(host_rng)
    write_new(output, dict(status='passed', branch=branch, cpu_only=True, cuda_exact_restore=False,
        source_environment=dict(python=sys.version, torch=torch.__version__), checked_states=checked_states,
        restored=restored, roles=roles, critical_source_bytes_rehashed_after_restore=True,
        archive_or_external_inputs_written_by_tool=False))


def restore(extracted, output, *, python, verification_sha, dataset=None, assets=None, mappings=None):
    extracted, output = no_links(extracted), no_links(output)
    verification = read(extracted / 'BACKUP_VERIFICATION.json')
    if sha(extracted / 'BACKUP_VERIFICATION.json') != verification_sha or verification['status'] != 'bytes_and_structural_coverage_passed':
        raise ValueError('Explicit authentic byte-verification receipt is required')
    if sha(extracted / MANIFEST) != verification['manifest_sha256']: raise ValueError('Manifest changed since byte verification')
    manifest = read(extracted / MANIFEST)
    execution_inventory(extracted / 'goal', manifest['execution_manifest_sha256'])
    dependency_proof = verify_dependencies(manifest['dependencies'], 'local', dataset=dataset, assets=assets, mappings=mappings)
    if output.exists() or output.is_relative_to(extracted): raise FileExistsError('Restore receipts need a new directory outside archive tree')
    output.mkdir(parents=True, exist_ok=False)
    for branch in NAMES:
        command = [python, '-B', str(Path(__file__).resolve()), '_cpu-worker', '--extracted', str(extracted),
                   '--branch', branch, '--dataset', dependency_proof['dataset']['path'],
                   '--assets', dependency_proof['assets']['path'], '--output', str(output / (branch + '.json'))]
        with (output / (branch + '.log')).open('xb') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                    FREEZE35_CPU_RESTORE_SUPERVISED='1', FREEZE35_CPU_VERIFICATION_SHA256=verification_sha), timeout=1800)
        if process.returncode: raise RuntimeError('CPU restore failed, preserve receipts/logs; no automatic retry: ' + branch)
    results = {branch: read(output / (branch + '.json')) for branch in NAMES}
    if any(row.get('status') != 'passed' for row in results.values()): raise ValueError('Missing real branch restore')
    for branch, result in results.items():
        expected_roles = {role['path'] for role in manifest['model_restore_roles'] if role['branch'] == branch}
        if (set(result['restored']) != expected_roles
                or set(result['checked_states']) != {*node_specs(branch), 'last.pt'}):
            raise ValueError('Critical model or all-state metadata coverage is incomplete')
    execution_inventory(extracted / 'goal', manifest['execution_manifest_sha256'])
    verify_dependencies(manifest['dependencies'], 'local', dataset=dataset, assets=assets, mappings=mappings)
    receipt = dict(status='real_cpu_restores_passed', format=FORMAT, branches=results, dependency_proof=dependency_proof,
        byte_verification_sha256=verification_sha, self_contained=False, cuda_exact_restore=False,
        gpu_or_training_performed=False, goal_completion_not_implied=True)
    write_new(output / 'FULL_MODEL_RESTORE.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack'); p.add_argument('--root', required=True); p.add_argument('--archive', required=True)
    p.add_argument('--execution-manifest-sha', required=True); p.add_argument('--release-receipt', required=True)
    p.add_argument('--references-mode', choices=('remote', 'local'), default='remote')
    p = sub.add_parser('verify'); p.add_argument('--archive', required=True); p.add_argument('--destination', required=True)
    p.add_argument('--archive-sha', required=True); p.add_argument('--manifest-sha', required=True)
    p = sub.add_parser('restore'); p.add_argument('--extracted', required=True); p.add_argument('--output', required=True)
    p.add_argument('--python', required=True); p.add_argument('--verification-sha', required=True)
    p = sub.add_parser('_cpu-worker'); p.add_argument('--extracted', required=True); p.add_argument('--output', required=True)
    p.add_argument('--branch', choices=tuple(NAMES), required=True)
    for name in ('verify', 'restore', '_cpu-worker'):
        sub.choices[name].add_argument('--dataset'); sub.choices[name].add_argument('--assets')
    for name in ('verify', 'restore'): sub.choices[name].add_argument('--path-map', help='Explicit JSON mapping old local roots to current roots')
    args = parser.parse_args(argv)
    if args.command == 'pack':
        result = pack(args.root, args.archive, execution_sha=args.execution_manifest_sha,
                      release_receipt=args.release_receipt, references_mode=args.references_mode)
    elif args.command == 'verify':
        result = verify(args.archive, args.destination, archive_sha=args.archive_sha, manifest_sha=args.manifest_sha,
                        dataset=args.dataset, assets=args.assets, mappings=read(args.path_map) if args.path_map else None)
    elif args.command == 'restore':
        result = restore(args.extracted, args.output, python=args.python, verification_sha=args.verification_sha,
                         dataset=args.dataset, assets=args.assets, mappings=read(args.path_map) if args.path_map else None)
    else:
        cpu_worker(args.extracted, args.branch, args.dataset, args.assets, args.output); return
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'sha256', 'manifest_sha256', 'self_contained')}, ensure_ascii=False))


if __name__ == '__main__': main()
