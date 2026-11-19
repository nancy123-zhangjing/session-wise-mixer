"""Read-only Stage09 warmup precheck: no training import, GPU or platform call."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def owned(relative):
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            'Unsafe manifest-relative path')
    path = ROOT / relative
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Path escapes Goal root')
    require(not path.is_symlink(), 'Symlink rejected: ' + relative)
    return path


def git(repo, *args):
    env = os.environ.copy()
    env.update(GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0',
               GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    command = ['git', '-c', 'safe.directory=' + str(repo),
               '-c', 'core.hooksPath=' + os.devnull, '-c', 'core.fsmonitor=false',
               '-c', 'submodule.recurse=false', '-C', str(repo), *args]
    result = subprocess.run(command, env=env, text=True, capture_output=True,
                            timeout=60, check=False)
    require(result.returncode == 0, 'Read-only Git check failed: ' + str(repo))
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['J', 'H'], required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.manifest_sha256):
        parser.error('--manifest-sha256 must be 64 lowercase hex characters')
    manifest_path = owned('delivery/FORMAL_WARMUP_BINDING.json')
    require(digest(manifest_path) == args.manifest_sha256, 'Binding manifest SHA differs')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['root'] == str(ROOT), 'Different Goal root')
    output = owned('runs/warmup_' + args.architecture)
    require(not output.exists(), 'Warmup output exists; inspect prior job and resume explicitly')
    repositories = {}
    for relative, expected in manifest['frozen_repositories'].items():
        repo = owned(relative)
        require((repo / '.git').is_dir() and not (repo / '.git').is_symlink(),
                'Expected standalone Git repository: ' + relative)
        require(Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() == repo.resolve(),
                'Unexpected worktree root')
        head = git(repo, 'rev-parse', 'HEAD')
        require(head == expected, 'Frozen HEAD differs: ' + relative)
        require(not git(repo, 'status', '--porcelain=v1', '--untracked-files=all'),
                'Frozen Git tree is dirty: ' + relative)
        repositories[relative] = head
    actual_sources = set()
    for directory in ('code/scripts/listwise_diffgrm', 'code/external/DiffGRM/genrec',
                      'code/external/DiffGRM/genrec/models/DIFF_GRM'):
        actual_sources.update(p.relative_to(ROOT).as_posix() for p in owned(directory).glob('*.py'))
    require(actual_sources == set(manifest['source_binding_paths']),
            'Runtime Python file set differs from accepted binding')
    checked_bytes = 0
    for relative, expected in manifest['file_sha256'].items():
        path = owned(relative)
        require(path.is_file(), 'Bound file missing: ' + relative)
        require(digest(path) == expected, 'Bound file SHA differs: ' + relative)
        checked_bytes += path.stat().st_size
    config = json.loads(owned('config/efficiency_r2.json').read_text(encoding='utf-8'))
    require(config == manifest['runtime_config'], 'Runtime config differs')
    require(config['seed'] == 2024 and config['training']['microbatch_size'] == 256
            and config['training']['effective_batch_size'] == 1024
            and config['training']['warmup_epochs'] == 5
            and config['evaluation']['batch_size'] == 32, 'Accepted settings differ')
    engine = os.environ.get('DIFFGRM_ENGINE_ROOT')
    require(not engine or Path(engine).resolve() == owned('code/external/DiffGRM').resolve(),
            'Environment overrides the frozen DiffGRM engine')
    print(json.dumps(dict(status='passed', architecture=args.architecture,
        checked_at_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
        output_absent=str(output), repositories=repositories,
        bound_files=len(manifest['file_sha256']), checked_bytes=checked_bytes,
        manifest_sha256=args.manifest_sha256, large_dataset_files_read=False,
        training_imported=False, gpu_called=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
