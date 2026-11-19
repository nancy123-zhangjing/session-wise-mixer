"""Finalize only after the GPU worker and independent CPU audit have passed."""
import base64
import hashlib
import json
import shlex

import control as c


METRIC = 'list/ndcg@5'


def delta(ours, original):
    return (ours / original - 1) * 100


def describe(value):
    return '数值接近' if abs(value) <= 1 else ('高于原 H1' if value > 0 else '低于原 H1')


def write_remote(files):
    payload = {name: base64.b64encode(data).decode() for name, data in files.items()}
    source = ('import base64, hashlib, json\nfrom pathlib import Path\n'
              f'files={payload!r}\nreceipts={{}}\n'
              'for name, encoded in files.items():\n'
              '    data = base64.b64decode(encoded)\n'
              '    path = Path(name)\n'
              '    temp = path.with_suffix(".tmp")\n'
              '    temp.write_bytes(data)\n'
              '    temp.replace(path)\n'
              '    assert path.read_bytes() == data\n'
              '    receipts[name] = hashlib.sha256(data).hexdigest()\n'
              'print(json.dumps(receipts))\n')
    command = f'cd {c.REMOTE} && {c.PYTHON} -B -c {shlex.quote(source)}'
    readback = json.loads(c.remote(command, 'DELIVERY_READBACK.json'))
    if readback != {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}:
        raise ValueError('Final report and closure readback mismatch')
    for name, data in files.items():
        (c.HERE / name).write_bytes(data)


def main():
    job = c.cli(['job', 'status', c.JOB, '--workspace', c.WORKSPACE], 'PLATFORM_LATEST.json')
    instances = c.cli(['job', 'instances', c.JOB, '--workspace', c.WORKSPACE], 'TERMINAL_INSTANCES.json')
    workers = instances.get('items', [])
    if job['status'] != 'SUCCEEDED' or len(workers) != 1 or workers[0]['status'] != 'instance_success':
        raise ValueError('Terminal GPU release is not confirmed')
    report_data = json.loads(c.remote(f'cd {c.REMOTE} && cat ACCEPTED_RESULT.json', 'ACCEPTANCE_READBACK.json'))
    if report_data['status'] != 'passed':
        raise ValueError('Independent audit has not passed')
    manifest_bytes = base64.b64decode(c.remote(f'cd {c.REMOTE} && base64 -w0 ARTIFACT_MANIFEST.json'))
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != report_data['manifest_sha256'] or len(json.loads(manifest_bytes)) != report_data['artifact_count']:
        raise ValueError('Artifact manifest readback mismatch')
    (c.HERE / 'ARTIFACT_MANIFEST.json').write_bytes(manifest_bytes)
    c.write('ACCEPTED_RESULT.json', report_data)
    validation = report_data['validation']
    best = report_data['best_epoch']
    test = report_data['best_test']
    original_best = validation['20']['original'][METRIC]
    val_best = validation[str(best)]['ours'][METRIC]
    val_diff = delta(val_best, original_best)
    test_diff = delta(test['ours'][METRIC], test['original'][METRIC])
    lines = ['# H1 去中心编号标记对照', '',
             '从原 H1 R5/T0 分叉，关闭中心编号标记并保留码本身份，沿原 35R/400T 时钟训练到 R20/T200。',
             '', '| 完整验证节点 | 原 H1 NDCG@5 | 去编号标记 NDCG@5 | 相对变化 |',
             '|---|---:|---:|---:|']
    for epoch, row in validation.items():
        old, new = row['original'][METRIC], row['ours'][METRIC]
        lines.append(f'| R{epoch} | {old:.9f} | {new:.9f} | {delta(new, old):+.2f}% |')
    lines += ['', f'新组按验证选择 R{best}，验证 NDCG@5 为 {val_best:.9f}，相对原 H1 best '
              f'（R20，{original_best:.9f}）变化 {val_diff:+.2f}%，{describe(val_diff)}。', '',
              '| 最佳点测试 | 原 H1 | 去编号标记 | 相对变化 |', '|---|---:|---:|---:|']
    for key in (METRIC, 'list/hr@5', 'list/recall@5'):
        old, new = test['original'][key], test['ours'][key]
        lines.append(f'| {key.removeprefix("list/")} | {old:.9f} | {new:.9f} | {delta(new, old):+.2f}% |')
    lines += ['', f'最佳点测试 NDCG@5 {describe(test_diff)}。±1% 仅作单 seed 的数值描述。', '',
              '新增 10,395 次推荐器更新和 200 次 Tokenizer 更新。完整验证节点及父状态均通过严格恢复，'
              '五个新验证节点和最佳点测试均通过逐样本独立复算。', '',
              f'单 H100 作业 {c.JOB} 成功结束并释放 GPU。完整 checkpoint 与预测留在 {c.REMOTE}，'
              f'SHA256 清单覆盖 {report_data["artifact_count"]} 个文件。共享 CPU 入口 env-zzl 保留。', '']
    report = '\n'.join(lines).encode('utf-8')
    closure = dict(status='complete', job=c.JOB, platform_status='SUCCEEDED', gpu_released=True,
                   shared_cpu_retained='env-zzl', root=c.REMOTE, artifact_manifest_sha256=manifest_sha,
                   report_sha256=hashlib.sha256(report).hexdigest(), acceptance=report_data['status'])
    write_remote({'FINAL_REPORT.md': report,
                  'FINAL_CLOSURE.json': (json.dumps(closure, ensure_ascii=False, indent=2) + '\n').encode()})
    print(json.dumps(closure, ensure_ascii=False))


if __name__ == '__main__':
    main()
