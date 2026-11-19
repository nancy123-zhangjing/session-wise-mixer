"""Archive small final report and audit receipts only; never trains or decodes."""
import base64
import datetime
import hashlib
import json
import poll_progress as p


def remote(script):
    encoded = base64.b64encode(script.encode()).decode()
    command = p.PYTHON + ' -c "import base64;exec(base64.b64decode(\'' + encoded + '\'))" # '
    result = p.call(['notebook', 'exec', 'env-zzl', '--workspace', 'CPU资源空间', command, '--timeout', '30'])
    assert result['data']['returncode'] == 0, result
    return next(json.loads(line) for line in result['data']['output'].splitlines() if line.startswith('{'))


def main():
    read = lambda name: json.loads((p.ROOT / name).read_text(encoding='utf-8'))
    evidence = read('FINAL_ACCEPTANCE_ARTIFACTS.json')
    accepted = read('ACCEPTED_RESULT.json')
    assert evidence['complete'] and accepted['status'] == 'passed'
    assert accepted['updates']['R_added'] == 6237 and accepted['updates']['T_added'] == 0
    assert accepted['full_checkpoint_restore_verified']
    assert read('FINAL_LOCAL_SOURCE_CHECK.json')['status'] == 'passed'
    platform = p.call(['job', 'status', 'nancy-s09-r26freeze-1106-v1', '--workspace', '分布式训练空间'])
    assert platform['data']['status'] == 'SUCCEEDED'
    p.write(p.ROOT / 'PLATFORM_LATEST.json', platform)
    report = (p.ROOT / 'FINAL_REPORT.md').read_bytes()
    report_sha = hashlib.sha256(report).hexdigest()
    script = f'''
import base64, hashlib, json
from pathlib import Path
root = Path({p.REMOTE!r})
audit = root / 'acceptance/r26_freeze'
result = audit / 'RESULT.json'
manifest = audit / 'ARTIFACT_MANIFEST.json'
assert hashlib.sha256(result.read_bytes()).hexdigest() == {evidence['result_sha256']!r}
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == {evidence['manifest_sha256']!r}
target = root / 'FINAL_REPORT.md'
target.write_bytes(base64.b64decode({base64.b64encode(report).decode()!r}))
assert hashlib.sha256(target.read_bytes()).hexdigest() == {report_sha!r}
print(json.dumps(dict(report_sha256={report_sha!r}, report_bytes=target.stat().st_size, independent_audit_unchanged=True)))
'''
    receipt = remote(script)
    closure = dict(status='complete', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   arm='r26_freeze', platform_status='SUCCEEDED', gpu_released=True,
                   shared_cpu='env-zzl retained', remote_root=p.REMOTE,
                   training_updates=accepted['updates'], best_epoch=accepted['best_epoch'],
                   independent_acceptance='passed', restore_verified=True,
                   remote_artifact_count=evidence['remote_files_size_verified'],
                   result_sha256=evidence['result_sha256'], manifest_sha256=evidence['manifest_sha256'],
                   report=receipt, large_artifacts='retained remotely; not downloaded or deleted')
    p.write(p.ROOT / 'FINAL_CLOSURE.json', closure)
    content = (p.ROOT / 'FINAL_CLOSURE.json').read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    confirmation = remote(f'''
import base64, hashlib, json
from pathlib import Path
target = Path({p.REMOTE!r}) / 'FINAL_CLOSURE.json'
target.write_bytes(base64.b64decode({base64.b64encode(content).decode()!r}))
assert hashlib.sha256(target.read_bytes()).hexdigest() == {sha!r}
print(json.dumps(dict(status='passed', closure_sha256={sha!r}, bytes=target.stat().st_size)))
''')
    p.write(p.ROOT / 'FINAL_DELIVERY_READBACK.json', confirmation)
    (p.ROOT / 'STATUS.md').write_text('# 已完成\n\n训练、验证、测试、独立验收和恢复检查全部通过；GPU 已释放。最终报告和关闭收据已同步远端并 SHA256 回读一致。共享 CPU env-zzl 保留。详见 FINAL_REPORT.md、FINAL_CLOSURE.json 和 FINAL_DELIVERY_READBACK.json。\n', encoding='utf-8')
    print(json.dumps(confirmation))


if __name__ == '__main__':
    main()
