"""Execute the explicitly authorized bounded short acceptance, never formal jobs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'manifest-sha', 'preparer-sha', 'job-sha', 'self-sha'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    spec = importlib.util.spec_from_file_location('bounded_job_helpers', root / 'delivery/job.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for path, expected in ((root / 'delivery/job.py', a.job_sha), (__file__, a.self_sha),
                           (root / 'delivery/prepare.py', a.preparer_sha)):
        if mod.sha256(path) != expected:
            raise ValueError('Short-acceptance deployment changed: ' + str(path))
    code = root / 'code'
    env = mod.environment()
    env['DIFFGRM_ENGINE_ROOT'] = str(code / 'external/DiffGRM')
    # The platform's 0.45-hour cap also includes this bounded verification.
    subprocess.run([a.python, '-B', str(root / 'delivery/prepare.py'), 'verify', '--code-root', str(code),
                    '--manifest-sha', a.manifest_sha, '--references', 'remote'],
                   check=True, cwd=code, env=env, timeout=90)
    inputs = json.loads((code / 'FREEZE35_PARENTS.json').read_text(encoding='utf-8'))
    command = [a.python, '-B', '-u', '-m', 'scripts.listwise_diffgrm.recovery35_preflight',
               '--dataset', inputs['dataset']['remote'], '--assets', inputs['assets']['remote'],
               '--ancestor', inputs['ancestors']['A']['remote'], '--output', str(root / 'preflight_r1'),
               '--max-seconds', '1500', '--profile-updates', '3', '--validation-samples', '128']
    for branch in 'ABC':
        command += ['--parent-' + branch.lower(), inputs['parents'][branch]['remote']]
    os.chdir(code)
    os.execve(a.python, command, env)


if __name__ == '__main__':
    main()
