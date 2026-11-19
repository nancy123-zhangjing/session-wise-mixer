"""Bind a completed local archive/Git restore to the real remote CPU restore.

All checks use real files. No success receipt is written until every check
passes. Cross-platform BYTE_VERIFICATION JSON is compared semantically while
each original receipt hash is retained. This does not complete the Goal,
allocate resources, execute CUDA, or construct a local Pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]
CAMPAIGN = 'goal_20261020_long_mixer_v1'
ARMS = ('no_mixer_frozen', 'no_mixer_joint', 'mixer_frozen', 'mixer_joint')
PACKAGE_SHA = '23ca79aa50e980c8f85be4447715c665a965e5a79aabecc3e361d0f58683e300'
ARCHIVE_SHA = 'c2ad35b0c87b219e6435a970dfd62bf5677cf7e7c2924b34c53e48c6fec47af7'
MANIFEST_SHA = '14bc3502ee5a207dd64ccf29657099080e982650418d1f9016e9ca633c4b1ce8'
COMMITS = {'code': '69682698c15ba796b98e6329c950061a2f68a05b',
           'engine': '1c879252f6395d6e64462511823c3929b9c42c00'}
TREES = {'code': 'fe625851b4edaa5436a0ce328b9ab19206a47c00',
         'engine': 'f1bccd3ec4e201aa759bf595d394663e0e9af801'}
COUNTS = {'code': 1456, 'engine': 37}
BOUNDARY = 'ad7b971c7e525e9fea6fb8e362a5c49dccb2473c'
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def sha(path, *, aggregate=None):
    """Hash stable bytes; optionally update the archive digest in the same pass."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            if aggregate is not None:
                aggregate.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing: ' + str(path))
    return digest.hexdigest()


def member(root, name):
    need(isinstance(name, str) and name and '\\' not in name and ':' not in name,
         'Unsafe member name')
    relative = PurePosixPath(name)
    need(not relative.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
         'Archive-relative path required')
    path = root.joinpath(*relative.parts)
    need(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(),
         'Missing/nonregular/escaped member: ' + str(path))
    parent = path.parent
    while parent != root:
        need(parent.is_relative_to(root) and not parent.is_symlink(), 'Symlink member parent')
        parent = parent.parent
    return path


def git(repo, *args):
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(repo), *args],
                            text=True, encoding='utf-8', capture_output=True)
    need(result.returncode == 0, 'Read-only Git check failed: ' + result.stderr.strip())
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, default=ROOT / 'outputs/stage05_long/package_r1')
    parser.add_argument('--backup-root', type=Path,
                        default=ROOT / 'outputs/stage05_long/extract_r1' / CAMPAIGN)
    parser.add_argument('--git-restore-root', type=Path, default=ROOT / 'outputs/stage05_long/git_restore_r1')
    parser.add_argument('--remote-evidence', type=Path, default=EVIDENCE / 'backup_evidence_remote_r1')
    parser.add_argument('--scientific-audit', type=Path,
                        help='Default: verified backup delivery/final_audit_r1/AUDIT.json; raw bytes must match the archive')
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'FINAL_BACKUP_BINDING_AUDIT.json')
    args = parser.parse_args()
    package_dir, backup, recovered, remote, output = (
        args.package_dir.resolve(), args.backup_root.resolve(), args.git_restore_root.resolve(),
        args.remote_evidence.resolve(), args.output.resolve())
    need(backup.name == CAMPAIGN and not output.exists(), 'Require campaign backup and a new binding receipt')
    need(all(not output.is_relative_to(p) for p in (package_dir, backup, recovered, remote)),
         'Keep the new audit outside its immutable input directories')

    # Preserve raw JSON identity for PACKAGE and MANIFEST, not reserialization.
    package_path = member(package_dir, 'PACKAGE.json')
    remote_package_path = member(remote, 'delivery/final_package_r1/PACKAGE.json')
    manifest_path = member(backup, 'MANIFEST.json')
    remote_manifest_path = member(remote, 'delivery/package_extract_r1/' + CAMPAIGN + '/MANIFEST.json')
    need(package_path.read_bytes() == remote_package_path.read_bytes()
         and sha(package_path) == PACKAGE_SHA, 'Local/remote original PACKAGE bytes differ')
    need(manifest_path.read_bytes() == remote_manifest_path.read_bytes()
         and sha(manifest_path) == MANIFEST_SHA, 'Local/remote original MANIFEST bytes differ')
    package, manifest = read(package_path), read(manifest_path)
    need(package['campaign_id'] == manifest['campaign_id'] == CAMPAIGN
         and package['format'] == manifest['format'] == 'stage05_long_full_byte_backup_v1'
         and package['archive_sha256'] == ARCHIVE_SHA and package['manifest_sha256'] == MANIFEST_SHA,
         'Wrong final archive/campaign identity')
    rows = {r['member']: r for r in manifest['files']}
    need(len(rows) == len(manifest['files']) == package['files'] == 1871
         and sum(r['size_bytes'] for r in rows.values()) == package['uncompressed_bytes'] == 1701159613,
         'Final archive member inventory/byte count differs')

    # Independent final transport check, regardless of downloader status files.
    parts = package['parts']
    need(len(parts) == 3 and len({r['member'] for r in parts}) == 3
         and [r['member'] for r in parts] == ['mixer_long_backup.tar.gz.part%04d' % i for i in (1, 2, 3)]
         and sum(r['size_bytes'] for r in parts) == package['archive_size_bytes'] == 1536215400,
         'Archive part ordering/size differs')
    archive_digest, part_checks = hashlib.sha256(), []
    for row in parts:
        path = member(package_dir, row['member'])
        need(path.parent == package_dir and path.stat().st_size == row['size_bytes']
             and sha(path, aggregate=archive_digest) == row['sha256'], 'Incomplete or damaged final downloaded part')
        part_checks.append(dict(row))
    need(archive_digest.hexdigest() == ARCHIVE_SHA, 'Concatenated archive bytes differ')

    # Receipt line endings may differ; the complete JSON structure must agree.
    local_byte_path = member(backup, 'BYTE_VERIFICATION.json')
    remote_byte_path = member(remote, 'delivery/package_extract_r1/' + CAMPAIGN + '/BYTE_VERIFICATION.json')
    local_byte, remote_byte = read(local_byte_path), read(remote_byte_path)
    need(local_byte == remote_byte and canonical(local_byte) == canonical(remote_byte),
         'Local/remote BYTE_VERIFICATION JSON semantics differ')
    need(local_byte['status'] == 'FULL_BACKUP_BYTE_VERIFICATION_PASS_NOT_RESTORE'
         and local_byte['campaign_id'] == CAMPAIGN and local_byte['package_sha256'] == PACKAGE_SHA
         and local_byte['manifest_sha256'] == MANIFEST_SHA and local_byte['files'] == 1871
         and local_byte['all_member_bytes_verified'] is True
         and local_byte['all_14_locked_inputs_included'] is True
         and local_byte['model_restore_verified'] is False, 'Incomplete local full-byte verification')
    local_byte_sha, remote_byte_sha = sha(local_byte_path), sha(remote_byte_path)

    # Rehash every extracted member, including all nine real formal states and
    # fourteen inputs. No receipt alone substitutes for these actual file bytes.
    member_hashes = {}
    for name, row in rows.items():
        path = member(backup, name)
        need(path.stat().st_size == row['size_bytes'], 'Extracted member size changed: ' + name)
        digest = sha(path)
        need(digest == row['sha256'], 'Extracted member bytes changed: ' + name)
        member_hashes[name] = digest
    inputs = read(member(backup, 'delivery/INPUTS.json'))
    input_names = {'inputs/' + r['path'] for r in inputs['records']}
    need(inputs['campaign_id'] == CAMPAIGN and len(inputs['records']) == len(input_names) == 14
         and input_names == {n for n in rows if n.startswith('inputs/')}
         and manifest['all_14_locked_inputs_included'] is True, 'Fourteen locked input streams differ')
    for row in inputs['records']:
        name = 'inputs/' + row['path']
        need(rows[name]['sha256'] == row['sha256'] and rows[name]['size_bytes'] == row['size_bytes'],
             'Input lock differs from verified actual bytes')
    need(inputs['counts'] == dict(users=22363, items=12101, interactions=198502,
                                 train=105668, val=22363, test=22363), 'Beauty data identity differs')

    identity = read(member(backup, 'formal/identity.json'))
    need(identity['campaign_id'] == CAMPAIGN and identity['execution_backend'] == 'cuda'
         and identity['inputs_sha256'] == member_hashes['delivery/INPUTS.json']
         and identity['protocol_sha256'] == member_hashes['code/configs/05_crosspse_joint/long/mixer_long_seed2024.json'],
         'Formal source/input identity differs')
    for directory, expected in COMMITS.items():
        need(manifest['source_locks'][directory]['commit'] == expected
             and manifest['source_locks'][directory]['tracked_files'] == COUNTS[directory], 'Source snapshot version differs')
    need(manifest['source_locks']['engine']['shallow_boundary'] == [BOUNDARY], 'Missing engine shallow boundary')

    # Scientific receipt must be the original complete audit carried in this
    # archive, with final evaluation and real checkpoint tensor checks enabled.
    science_path = (args.scientific_audit.resolve() if args.scientific_audit
                    else member(backup, 'delivery/final_audit_r1/AUDIT.json'))
    science = read(science_path)
    need(sha(science_path) == manifest['artifact_audit_sha256'] == rows['delivery/final_audit_r1/AUDIT.json']['sha256']
         and science['status'] == 'FORMAL_ARTIFACT_AUDIT_PASS_NOT_DELIVERY'
         and science['campaign_id'] == CAMPAIGN and science['identity'] == identity
         and science['tests_checked'] is True and science['checkpoints_checked'] is True,
         'Scientific audit is partial or belongs to different artifact bytes')
    need(set(science['summaries']) == set(ARMS), 'Scientific audit omitted an arm')

    # Remote CPU proof binds the remote receipt's own raw bytes, never the
    # Windows reserialized local BYTE_VERIFICATION hash.
    remote_restore_path = member(remote, 'delivery/restore_check_cpu_r1/' + CAMPAIGN + '/RESTORE_PASS.json')
    remote_summary_path = member(remote, 'delivery/restore_check_execution_r1/REMOTE_RESTORE_SUMMARY.json')
    remote_audit_path = member(remote, 'REMOTE_RESTORE_EVIDENCE_AUDIT.json')
    cpu, remote_summary, remote_audit = read(remote_restore_path), read(remote_summary_path), read(remote_audit_path)
    need(remote_summary['status'] == 'REMOTE_FINAL_ARCHIVE_BYTE_AND_MODEL_RESTORE_PASS'
         and remote_summary['package_sha256'] == PACKAGE_SHA and remote_summary['archive_sha256'] == ARCHIVE_SHA
         and remote_summary['manifest_sha256'] == MANIFEST_SHA
         and remote_summary['byte_verification_sha256'] == remote_byte_sha
         and remote_summary['cpu_restore_receipt_sha256'] == sha(remote_restore_path)
         and remote_summary['formal_states_restored'] == 8
         and remote_summary['original_package_and_state_bytes_unchanged'] is True,
         'Remote CPU/byte summary raw-byte linkage differs')
    need(cpu['status'] == 'FORMAL_FULL_BACKUP_CPU_RESTORE_PASS' and cpu['campaign_id'] == CAMPAIGN
         and cpu['byte_verification_sha256'] == remote_byte_sha and cpu['manifest_sha256'] == MANIFEST_SHA
         and cpu['model_source'] == 'verified_backup/code' and cpu['engine_source'] == 'verified_backup/engine'
         and cpu['all_14_inputs_copied_from_verified_backup'] is True
         and cpu['original_checkpoints_unchanged'] is True and cpu['execution_backend'] == 'cpu'
         and cpu['cuda_execution_verified_here'] is False
         and cpu['restorer_sha256'] == rows['delivery/analysis_tools_v3r2/scripts/analysis/restore_mixer_long_backup.py']['sha256'],
         'Real remote CPU consumer scope/source differs')
    need(remote_audit['status'] == 'REMOTE_FINAL_ARCHIVE_AND_CPU_RESTORE_EVIDENCE_PASS_NOT_LOCAL_BACKUP'
         and remote_audit['bindings']['package_raw_sha256'] == PACKAGE_SHA
         and remote_audit['bindings']['manifest_raw_sha256'] == MANIFEST_SHA
         and remote_audit['bindings']['remote_byte_verification_raw_sha256'] == remote_byte_sha
         and remote_audit['bindings']['remote_cpu_restore_raw_sha256'] == sha(remote_restore_path),
         'Independent remote sidecar audit identity differs')
    for row in remote_audit['source_sidecar_members']:
        path = member(remote, row['member'])
        need(path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'], 'An original remote sidecar changed')
    remote_byte_log = member(remote, 'delivery/restore_check_execution_r1/byte_verification.log')
    cpu_log_path = member(remote, 'delivery/restore_check_execution_r1/cpu_model_restore.log')
    cpu_log = cpu_log_path.read_text(encoding='utf-8')
    start = cpu_log.rfind('{\n  "status": "FORMAL_FULL_BACKUP_CPU_RESTORE_PASS"')
    need(read(remote_byte_log) == remote_byte and start >= 0 and json.loads(cpu_log[start:]) == cpu
         and 'INFO:root:Device: cpu' in cpu_log, 'Remote consumer logs and actual saved receipts differ')

    states = {(arm, filename) for arm in ARMS for filename in ('best.pt', 'last.pt')}
    cpu_states = {(r['arm'], r['checkpoint']): r for r in cpu['checkpoints']}
    need(len(cpu_states) == len(cpu['checkpoints']) == 8 and set(cpu_states) == states,
         'Eight distinct actual CPU checkpoint restorations are required')
    seal_path = member(backup, 'formal/SELECTION_SEALED.json')
    seal = read(seal_path)
    need(seal['identity'] == identity and set(seal['arms']) == set(ARMS), 'Four-arm sealed selection differs')
    checkpoint_bindings = []
    for arm in ARMS:
        train_name, progress_name = 'formal/' + arm + '/TRAIN_COMPLETE.json', 'formal/' + arm + '/progress.json'
        train, progress = read(member(backup, train_name)), read(member(backup, progress_name))
        summary = science['summaries'][arm]
        need(science['source_receipt_sha256'][arm]['TRAIN_COMPLETE.json'] == member_hashes[train_name]
             and science['source_receipt_sha256'][arm]['progress.json'] == member_hashes[progress_name]
             and train['progress'] == progress and train['identity'] == identity
             and train['status'] == 'trained_not_tested' and train['arm'] == arm,
             'Scientific audit/training receipt/progress chain differs')
        need(summary['complete'] is True and summary['recommender_epochs'] == 35
             and summary['recommender_updates'] == 3640
             and summary['tokenizer_updates'] == (60 if arm.endswith('_joint') else 0)
             and summary['noisy_calls'] == (1920 if arm.endswith('_joint') else 0), 'Full literal training budget differs')
        need(seal['arms'][arm] == {k: train[k] for k in ('best_sha256', 'selected_decoder_epoch', 'last_sha256')},
             'Sealed best selection differs from actual training')
        for filename in ('best.pt', 'last.pt'):
            name = 'formal/' + arm + '/' + filename
            row, audited = cpu_states[arm, filename], summary['checkpoint_audit'][filename[:-3]]
            field = 'best_sha256' if filename == 'best.pt' else 'last_sha256'
            epoch = train['selected_decoder_epoch'] if filename == 'best.pt' else 35
            token = 0 if arm.endswith('_frozen') or epoch <= 5 else 6 * ((epoch - 6) // 3 + 1)
            need(member_hashes[name] == train[field] == row['sha256']
                 and row['decoder_epochs'] == audited['recommender_epochs'] == epoch
                 and row['decoder_updates'] == epoch * 104
                 and row['tokenizer_updates'] == audited['tokenizer_updates'] == token,
                 'Restored state/file/scientific checkpoint stage differs')
            need(all(row[k] is True for k in ('strict_model_optimizer_scheduler_restore', 'python_numpy_cpu_rng_restored',
                                             'official_SID_restored', 'nonpersistent_E0_anchor_restored'))
                 and HEX64.fullmatch(row['cuda_rng_bytes_preserved_sha256']) is not None,
                 'Full CPU optimizer/scheduler/RNG/SID/anchor restoration is missing')
            if filename == 'last.pt':
                need(audited['model_tensor_sha256'] == train['final_model_tensor_sha256']
                     and audited['SID_sha256'] == train['final_SID_sha256'], 'Final scientific tensor/SID audit differs')
            # These two named model digests deliberately use different algorithms.
            # Their common binding is the complete checkpoint file's raw SHA.
            checkpoint_bindings.append(dict(member=name, checkpoint_raw_sha256=row['sha256'],
                size_bytes=rows[name]['size_bytes'], decoder_epochs=epoch, decoder_updates=epoch * 104,
                tokenizer_updates=token, scientific_model_tensor_raw_stream_sha256=audited['model_tensor_sha256'],
                cpu_model_tensor_json_summary_digest_sha256=row['model_tensor_sha256'],
                tensor_digest_algorithms_distinct_and_not_compared=True,
                strict_cpu_model_optimizer_scheduler_rng_sid_anchor_restored=True))
        test_name = 'formal/' + arm + '/TEST_RESULT.json'
        test = read(member(backup, test_name))
        need(test['best_sha256'] == train['best_sha256'] and test['selection_sha256'] == member_hashes['formal/SELECTION_SEALED.json']
             and test['actual_dataset_split'] == test['split'] == 'test'
             and test['decoder_epoch'] == train['selected_decoder_epoch']
             and test['selected_tokenizer_updates'] == summary['best_tokenizer_updates']
             and all(test['metrics'][k] == v for k, v in summary['selected_test'].items()),
             'Scientific selected test is bound to a different checkpoint')

    formal_names = {'formal/common/initial.pt'} | {'formal/' + a + '/' + n for a, n in states}
    need(len(formal_names) == 9 and formal_names.issubset(member_hashes), 'Nine complete formal states are missing')

    # Recheck the actual recovered Git repositories as well as their receipt.
    git_path = member(recovered, 'FINAL_GIT_RESTORE_PASS.json')
    gr = read(git_path)
    need(gr['status'] == 'FINAL_DOWNLOADED_BACKUP_GIT_RESTORE_PASS_NOT_MODEL_RESTORE'
         and gr['campaign_id'] == CAMPAIGN and gr['package_sha256'] == PACKAGE_SHA
         and gr['manifest_sha256'] == MANIFEST_SHA and gr['byte_verification_sha256'] == local_byte_sha
         and gr['actual_final_downloaded_package_used'] is True and gr['verified_backup_unchanged'] is True
         and gr['all_backup_members_rehashed'] is True and gr['all_downloaded_parts_rehashed'] is True
         and gr['engine_shallow_boundary'] == BOUNDARY
         and gr['helper_sha256'] == rows['delivery/final_verification_tools/restore_git_from_final_backup.py']['sha256'],
         'Actual final downloaded-backup Git restore linkage differs')
    repositories = {}
    for name in ('code', 'engine'):
        repo, declared = recovered / name, gr['repositories'][name]
        expected_shallow = name == 'engine'
        need(declared['head'] == COMMITS[name] and declared['tree'] == TREES[name]
             and declared['tracked_files'] == COUNTS[name] and declared['clean'] is True
             and declared['all_tracked_bytes_equal_verified_backup'] is True
             and declared['shallow'] is expected_shallow, 'Git restore receipt source inventory differs')
        need(git(repo, 'rev-parse', 'HEAD') == COMMITS[name]
             and git(repo, 'rev-parse', 'HEAD^{tree}') == TREES[name]
             and git(repo, 'status', '--porcelain') == ''
             and (git(repo, 'rev-parse', '--is-shallow-repository') == 'true') is expected_shallow,
             'Actual recovered Git source differs from the successful receipt')
        raw = subprocess.check_output(['git', '--no-optional-locks', '-C', str(repo), 'ls-files', '-z'])
        names = [n for n in raw.decode('utf-8').split('\0') if n]
        need(len(names) == COUNTS[name], 'Actual restored tracked count differs')
        for relative in names:
            need(sha(member(repo, relative)) == member_hashes[name + '/' + relative], 'Recovered source byte changed')
        repositories[name] = dict(head=COMMITS[name], tree=TREES[name], tracked_files=len(names),
                                  shallow=expected_shallow, current_tracked_bytes_equal_backup=True)
    need((recovered / 'engine/.git/shallow').read_bytes() == (BOUNDARY + '\n').encode('ascii'),
         'Engine shallow boundary is missing or not the verified ASCII bytes')
    sources = {(r['repository'], r['path']): r['sha256'] for r in gr['formal_execution_sources']}
    expected_sources = {(directory, path): digest for directory, field in (('code', 'root_sources'), ('engine', 'engine_sources'))
                        for path, digest in identity[field].items()}
    need(sources == expected_sources and len(sources) == len(gr['formal_execution_sources']) == 19,
         'Nineteen formal execution-source identities differ')
    for (directory, path), digest in sources.items():
        need(member_hashes[directory + '/' + path] == digest, 'Formal execution source differs from actual backup bytes')

    # All the input proofs explicitly keep Goal/CUDA claims out of this binder.
    need(all(v.get('goal_complete') is False for v in (package, manifest, local_byte, remote_byte, cpu, remote_summary, gr, science))
         and gr['cuda_execution_verified'] is False and remote_summary['cuda_execution_verified_here'] is False,
         'An input proof unexpectedly claimed Goal completion or CUDA execution')
    result = dict(status='FINAL_FULL_BACKUP_RECEIPTS_BOUND_NOT_GOAL_COMPLETION', all_checks_passed=True,
        campaign_id=CAMPAIGN, utc=datetime.now(timezone.utc).isoformat(), binder_sha256=sha(Path(__file__)),
        paths=dict(package_dir=str(package_dir), verified_backup_root=str(backup), git_restore_root=str(recovered),
                   remote_evidence=str(remote), scientific_audit=str(science_path)),
        package_raw_sha256=PACKAGE_SHA, archive_sha256=ARCHIVE_SHA, manifest_raw_sha256=MANIFEST_SHA,
        original_local_remote_package_and_manifest_bytes_identical=True,
        archive_parts=part_checks, archive_size_bytes=package['archive_size_bytes'],
        all_local_archive_parts_and_concatenation_rehashed=True,
        byte_verification_json_semantics_identical=True,
        byte_verification_canonical_json_sha256=hashlib.sha256(canonical(local_byte)).hexdigest(),
        local_byte_verification_raw_sha256=local_byte_sha, remote_byte_verification_raw_sha256=remote_byte_sha,
        byte_verification_raw_receipt_bytes_identical=local_byte_path.read_bytes() == remote_byte_path.read_bytes(),
        local_git_restore_raw_sha256=sha(git_path), remote_cpu_restore_raw_sha256=sha(remote_restore_path),
        remote_summary_raw_sha256=sha(remote_summary_path), independent_remote_evidence_audit_sha256=sha(remote_audit_path),
        scientific_audit_raw_sha256=sha(science_path), all_manifest_members_rehashed=True,
        verified_manifest_member_count=len(rows), verified_uncompressed_bytes=package['uncompressed_bytes'],
        locked_input_count=14, locked_input_bytes=sum(rows[n]['size_bytes'] for n in input_names),
        formal_complete_state_count=9, formal_complete_state_bytes=sum(rows[n]['size_bytes'] for n in formal_names),
        formal_complete_state_hashes={n: member_hashes[n] for n in sorted(formal_names)},
        eight_real_cpu_restores_bound_to_actual_local_states_and_scientific_audit=True,
        common_initial_reconstruction_enforced_by_locked_cpu_restorer=True,
        checkpoint_bindings=checkpoint_bindings, model_tensor_digest_algorithms_distinct_and_not_mixed=True,
        recovered_git_repositories=repositories, actual_recovered_git_tracked_count=sum(COUNTS.values()),
        formal_execution_source_hashes_checked=19, local_full_backup_verified=True,
        local_final_downloaded_package_git_recovery_verified=True,
        remote_cpu_consumer_restoration_bound=True, local_pipeline_model_restore_executed_here=False,
        cpu_restore_execution_backend='cpu', cuda_execution_verified_here=False, goal_complete=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(dict(status=result['status'], all_checks_passed=True, output=str(output),
                         output_sha256=sha(output), local_byte_verification_raw_sha256=local_byte_sha,
                         remote_byte_verification_raw_sha256=remote_byte_sha,
                         byte_verification_json_semantics_identical=True, goal_complete=False),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
