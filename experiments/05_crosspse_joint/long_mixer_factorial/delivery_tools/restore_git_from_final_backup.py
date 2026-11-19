"""Restore Git from a final downloaded and byte-verified backup, never live assets.

Pure standard library plus Git. Does not construct a model, allocate resources,
execute CUDA, alter the verified backup, or mark the experimental Goal complete.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
CAMPAIGN = 'goal_20261020_long_mixer_v1'
EXPECTED = {
    'code.bundle': 'b3cb871bd303d6fe45279d9929c7b1bd2693569a3f6574f4af27830e8fc0d397',
    'code_r2.bundle': '81c1f50eef5802c665ad5b170bba9e8576c40409d826ef6211b524249df8d319',
    'engine.bundle': '69faf55e0188bcb6ac1cbe619ada855428fee0c07bdb4487e57b760a35b47d2c',
}
COMMITS = {'code': '69682698c15ba796b98e6329c950061a2f68a05b',
           'engine': '1c879252f6395d6e64462511823c3929b9c42c00'}
TREES = {'code': 'fe625851b4edaa5436a0ce328b9ab19206a47c00',
         'engine': 'f1bccd3ec4e201aa759bf595d394663e0e9af801'}
BOUNDARY = 'ad7b971c7e525e9fea6fb8e362a5c49dccb2473c'


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def member(root, name):
    relative = Path(name)
    path = root / relative
    need(not relative.is_absolute() and '..' not in relative.parts
         and path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(),
         'Unsafe or missing backup member: ' + str(name))
    parent = path.parent
    while parent != root:
        need(parent.is_relative_to(root) and not parent.is_symlink(), 'Symlink backup member parent')
        parent = parent.parent
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, required=True,
                        help='Final locally downloaded PACKAGE.json and listed parts')
    parser.add_argument('--backup-root', type=Path, required=True,
                        help='Extraction already verified with the locked byte verifier')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Fresh independent directory, outside the extracted backup')
    args = parser.parse_args()
    package, backup, output = args.package_dir.resolve(), args.backup_root.resolve(), args.output_dir.resolve()
    need(backup.name == CAMPAIGN and not output.exists()
         and not output.is_relative_to(backup) and not backup.is_relative_to(output),
         'Require verified campaign backup and a new independent Git output directory')
    verified = read(backup / 'BYTE_VERIFICATION.json')
    need(verified['status'] == 'FULL_BACKUP_BYTE_VERIFICATION_PASS_NOT_RESTORE'
         and verified['campaign_id'] == CAMPAIGN and verified['all_member_bytes_verified'] is True
         and verified['all_14_locked_inputs_included'] is True, 'Full final byte-verification receipt required')
    need(sha(package / 'PACKAGE.json') == verified['package_sha256'], 'Downloaded package receipt differs from byte verification')
    receipt = read(package / 'PACKAGE.json')
    need(receipt['campaign_id'] == CAMPAIGN and receipt['manifest_sha256'] == verified['manifest_sha256'],
         'Final package/manifest identity differs')
    # Recheck the actual downloaded part bytes. Full concatenated archive/member
    # verification was performed by the separately locked byte verifier.
    for row in receipt['parts']:
        path = member(package, row['member'])
        need(path.parent == package and path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'],
             'A downloaded part changed since verification')
    need(sha(backup / 'MANIFEST.json') == verified['manifest_sha256'], 'Verified manifest changed')
    manifest = read(backup / 'MANIFEST.json')
    need(manifest['campaign_id'] == CAMPAIGN, 'Wrong internal campaign')
    members = {row['member']: row for row in manifest['files']}
    need(len(members) == len(manifest['files']), 'Duplicate manifest member')
    for name, row in members.items():
        path = member(backup, name)
        need(path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'], 'Verified member changed: ' + name)
    for name, expected in EXPECTED.items():
        need(sha(member(backup, 'prep/' + name)) == expected, 'Wrong source bundle bytes: ' + name)
    for directory, commit in COMMITS.items():
        need(manifest['source_locks'][directory]['commit'] == commit, 'Manifest source commit differs')
    need(manifest['source_locks']['engine']['shallow_boundary'] == [BOUNDARY], 'Shallow boundary was not retained')

    output.mkdir(parents=True, exist_ok=False)
    commands = []
    def git(repo, *args):
        result = subprocess.run(['git', '-C', str(repo), *args], text=True, encoding='utf-8', capture_output=True)
        commands.append(dict(cwd=str(repo), argv=['git', '-C', str(repo), *args], exit_code=result.returncode,
                             stdout=result.stdout.strip(), stderr=result.stderr.strip()))
        need(result.returncode == 0, 'Git command failed: ' + json.dumps(commands[-1], ensure_ascii=False))
        return result.stdout.strip()
    code, engine = output / 'code', output / 'engine'
    for repo in (code, engine):
        git(output, 'init', str(repo))
        for key, value in (('core.autocrlf', 'false'), ('core.eol', 'lf'), ('core.longpaths', 'true')):
            git(repo, 'config', key, value)
    ref = 'refs/heads/codex/stage05-mixer-long'
    for name, commit in (('code.bundle', 'fd04bfd6d9ab8009fdc4fb1d613a4db3adacb98f'),
                         ('code_r2.bundle', COMMITS['code'])):
        bundle = str(backup / 'prep' / name)
        git(code, 'bundle', 'verify', bundle)
        git(code, 'fetch', '--no-tags', bundle, ref + ':' + ref)
        need(git(code, 'rev-parse', ref) == commit, 'Source bundle ref differs')
    git(code, 'checkout', '--detach', COMMITS['code'])
    # The engine bundle intentionally has shallow history. Write ASCII bytes
    # before fetch; PowerShell default text encoding must not introduce a BOM.
    (engine / '.git/shallow').write_bytes((BOUNDARY + '\n').encode('ascii'))
    bundle = str(backup / 'prep/engine.bundle')
    git(engine, 'bundle', 'verify', bundle)
    git(engine, 'fetch', '--no-tags', bundle, 'HEAD:refs/heads/codex/stage05-engine-restore')
    git(engine, 'checkout', '--detach', COMMITS['engine'])
    repositories = {}
    for repo, expected_count in ((code, 1456), (engine, 37)):
        git(repo, 'fsck', '--full')
        need(git(repo, 'rev-parse', 'HEAD') == COMMITS[repo.name]
             and git(repo, 'rev-parse', 'HEAD^{tree}') == TREES[repo.name]
             and git(repo, 'status', '--porcelain') == '', 'Restored source checkout differs')
        names = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0')
        names = [name for name in names if name]
        need(len(names) == expected_count, 'Restored tracked file inventory differs')
        for name in names:
            original = member(backup, repo.name + '/' + name)
            need((repo / name).is_file() and sha(repo / name) == sha(original),
                 'Restored tracked path missing or byte-different: ' + name)
        repositories[repo.name] = dict(head=COMMITS[repo.name], tree=TREES[repo.name], tracked_files=len(names),
            all_tracked_bytes_equal_verified_backup=True, clean=True,
            shallow=git(repo, 'rev-parse', '--is-shallow-repository') == 'true')
    need(not repositories['code']['shallow'] and repositories['engine']['shallow'], 'Unexpected Git history depth')
    identity = read(backup / 'formal/identity.json')
    source_checks = []
    for repo, field in ((code, 'root_sources'), (engine, 'engine_sources')):
        for name, expected in identity[field].items():
            need(sha(repo / name) == expected, 'Recovered execution source differs from formal identity')
            source_checks.append(dict(repository=repo.name, path=name, sha256=expected))
    need(len(source_checks) == 19, 'Formal execution source inventory differs')
    # Confirm that the read-only source operation did not change the backup.
    for name, row in members.items():
        need(sha(member(backup, name)) == row['sha256'], 'Git reconstruction changed a backup member')
    result = dict(status='FINAL_DOWNLOADED_BACKUP_GIT_RESTORE_PASS_NOT_MODEL_RESTORE', campaign_id=CAMPAIGN,
        utc=datetime.now(timezone.utc).isoformat(), package_sha256=verified['package_sha256'],
        manifest_sha256=verified['manifest_sha256'], byte_verification_sha256=sha(backup / 'BYTE_VERIFICATION.json'),
        helper_sha256=sha(Path(__file__)), bundle_hashes=EXPECTED, engine_shallow_boundary=BOUNDARY,
        verified_backup_unchanged=True, all_backup_members_rehashed=True, all_downloaded_parts_rehashed=True,
        repositories=repositories, formal_execution_sources=source_checks, commands=commands,
        actual_final_downloaded_package_used=True, model_optimizer_scheduler_rng_sid_restore_verified=False,
        cuda_execution_verified=False, goal_complete=False)
    with (output / 'FINAL_GIT_RESTORE_PASS.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
