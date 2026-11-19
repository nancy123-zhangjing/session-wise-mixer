"""Run commands on the CPU notebook over the SDK Jupyter transport.

Used because the cached SSH/rtunnel bridge was broken after the notebook restart
(TunnelError). Read-only usage in this analysis.
"""
import base64
import os
import shlex
import sys

from inspire import InspireClient

NOTEBOOK = 'nancy-s09-e-prep-1101-r1'
WORKSPACE = 'CPU资源空间'
PYTHON = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
os.environ.update(NO_PROXY='.sii.edu.cn', HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='')


def run(command, timeout=120):
    with InspireClient(account='sii-shared') as client:
        client.login()
        r = client.notebooks.exec(NOTEBOOK, command=command, workspace=WORKSPACE,
                                  timeout=timeout, transport='jupyter')
    if r.returncode != 0 or not r.completed:
        raise RuntimeError(f'remote failed rc={r.returncode} completed={r.completed}: {r.output[-2000:]}')
    return r.output


def python(source, timeout=120):
    encoded = base64.b64encode(source.encode()).decode()
    code = f"import base64; exec(base64.b64decode({encoded!r}))"
    return run(PYTHON + ' -B -c ' + shlex.quote(code), timeout)


if __name__ == '__main__':
    print(run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120))
