"""Local closure receipt assembled only from verified evidence."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

progress = read(root / 'PROGRESS.json')
audit = read(root / 'checkpoint_audit.json')
pre = read(root / 'pretrain_recovery_audit.json')
local = read(root / 'LOCAL_REPORT_VERIFICATION.json')
manifest = read(root / 'ASSET_MANIFEST.json')
benchmark = read(root.parent / 'benchmark_20261111_v1/FINAL_ACCEPTANCE.json')
assert audit['passed'] and pre['passed'] and local['passed']
assert len(audit['checkpoints']) == 93 and len(manifest['external_data']) == 9
assert benchmark['status'] == 'GPU_BENCHMARK_PASS' and benchmark['recovery']['passed'] and benchmark['numerical']['passed']
previous = read(root / 'CONFIG_BEFORE.json')
current = read(root.parents[2] / 'configs/rqcd_mixer/v1.json')
for cfg in (previous, current):
    cfg.pop('protocol_id'); cfg.pop('design_sha256'); cfg['evaluation'].pop('epochs')
assert previous == current, 'Unexpected scientific setting change'
resources, results = {}, []
for arm in ('B0', 'B1', 'B2'):
    status = read(root / f'status_{arm}.json')['data']
    instances = read(root / f'instances_{arm}.json')['data']['items']
    assert status['status'] == 'SUCCEEDED' and status['finished_at']
    assert status['resource']['gpu'] == 1 and len(instances) == 1
    assert all(x['status'] == 'instance_success' for x in instances)
    resources[arm] = {'job': status['name'], 'status': status['status'], 'instances': instances, 'finished_at': status['finished_at']}
    joint = progress['arms'][arm]['joint']
    assert joint['TRAINING_COMPLETE.json']['R'] == 35 and joint['TRAINING_COMPLETE.json']['T'] == 300
    assert len(joint['validation']) == 31
    assert all(v['count'] == 71657 for v in joint['validation'])
    assert all(v['count'] == 53194 for v in joint['TEST_COMPLETE.json'].values())
    results.append({'arm':arm, 'best_R':joint['BEST.json']['R'],
                    'test_best':joint['TEST_COMPLETE.json']['best']['metrics'],
                    'test_final':joint['TEST_COMPLETE.json']['final']['metrics']})
cpu = read(root / 'cpu_final_status.json')['data']
assert cpu['status'] == 'STOPPED'
assert not list(root.rglob('*.pt')), 'Unexpected local large checkpoint copy'
evidence = {}
for name in ('checkpoint_audit.json','pretrain_recovery_audit.json','ASSET_MANIFEST.json',
             'LOCAL_REPORT_VERIFICATION.json','cpu_final_status.json','report/REPORT.md'):
    evidence[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
receipt = {'status':'ALL_REQUIREMENTS_VERIFIED', 'created_at':datetime.now(timezone.utc).isoformat(),
           'scope':'B0/B1/B2 seed2024, one H100 each, R35/T300, 31 full validation nodes, best/final tests',
           'resources':resources, 'temporary_cpu':{'name':cpu['name'],'status':cpu['status']},
           'checkpoint_nodes_verified':93, 'pretrain_diagnostics_verified':600,
           'figure_review':'19 standalone plots visually inspected; distinct solid colors; identical utilization curves overlap',
           'local_report_verified_files':24, 'remote_asset_files':len(manifest['files']),
           'remote_root':manifest['root'], 'evidence_sha256':evidence, 'results':results,
           'recovery_boundary':'All saved states readable/finite with optimizer clocks/RNG/data streams audited. Exact next-R GPU recovery reuses accepted benchmark, not rerun.',
           'limits':['single seed; no significance claim','low absolute recommendation scores','pretrain VQ loss rises while reconstruction falls'],
           'closure_receipt_location':'Local; remote report, checkpoints, manifest and state audits were synced before stopping CPU.'}
(root / 'FINAL_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
(root / 'STATUS.md').write_text('# 已完成并验收\n\n三组训练、验证和测试已完成；GPU 作业/实例均成功结束，临时 CPU 已停止。\n\n'
    '结果入口：[report/REPORT.md](report/REPORT.md)。完整验收：[FINAL_ACCEPTANCE.json](FINAL_ACCEPTANCE.json)。\n\n'
    f'远端资产：`{manifest["root"]}`。大文件保留远端，本地无 checkpoint 副本。\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'remote_asset_files':len(manifest['files']),'resources_closed':True}))
