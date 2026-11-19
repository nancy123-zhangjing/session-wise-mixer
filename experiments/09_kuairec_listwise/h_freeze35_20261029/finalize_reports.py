"""Bind completed evidence and assemble delivery documentation; no model/network work."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
RUNS = ROOT / 'outputs/stage09_restore/h_freeze35_20261029_final_r1_restored/goal/runs'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    now = datetime.now(timezone.utc).isoformat()
    evidence = [
        'GOAL_OBJECTIVE.md', 'PROTOCOL.md', 'EXECUTION_PACKAGE.json',
        'DETERMINISTIC_PACKAGE_AUDIT.json', 'INPUT_AUDIT.json', 'INPUT_AUDIT_COVERAGE.json',
        'H100_ACCEPTANCE_AUDIT.json', 'C_R20_REPLAY_FREEZE_EVIDENCE.json',
        'FINAL_PACK_RECEIPT.json', 'FINAL_TRANSFER_VERIFICATION.json', 'FINAL_BACKUP_VERIFICATION.json',
        'final_cpu_restore_r1/FULL_MODEL_RESTORE.json', 'final_cpu_restore_r1/INDEPENDENT_REVIEW.json',
        'final_cpu_restore_r1/CHECKPOINT_COVERAGE.csv',
        'final_analysis/A/analysis_report.json', 'final_analysis/B/analysis_report.json',
        'final_analysis/C/analysis_report.json', 'final_comparison_r1/analysis_report.json',
        'final_analysis/A/SUPPLEMENTAL_BEST_COMPARISON_BINDING.json',
        'early_a_cpu_r2/analysis/COMPLETE.json', 'early_ab_comparison_20261029/COMPLETE.json',
        'final_independent_review_r1/INDEPENDENT_REVIEW.json',
        'final_independent_review_r1/MAIN_REPORT_REVIEW.json',
        'final_figures_r1/PLOT_PROVENANCE.json', 'final_figures_r1/VISUAL_REVIEW.json',
        'GPU_LEDGER.json', 'final_lifecycle_r1/RELEASE.json', 'final_lifecycle_r1/REMOTE_UPLOAD_VERIFIED.json',
        'cold_popularity_audit_20261030/PROTOCOL.md', 'cold_popularity_audit_20261030/C_EXTRA_MODELS.json',
        'cold_popularity_audit_20261030/c_supplement_r1/COMPLETE.json',
        'cold_popularity_audit_20261030/VERIFY_C_SUPPLEMENT.json',
        'cold_popularity_audit_20261030/final_report_review_r1/REVIEW.json',
        'cold_popularity_audit_20261030/c_supplement_tables_r1/EXPORT_RECEIPT.json',
        'cold_popularity_audit_20261030/FINAL_REPORT.md',
        'cold_popularity_audit_20261030/REPORT_C_SUPPLEMENT.md',
        'cold_popularity_audit_20261030/STATUS.json',
    ]
    for n in ['DETERMINISTIC_PACKAGE_AUDIT.json','H100_ACCEPTANCE_AUDIT.json',
              'final_cpu_restore_r1/INDEPENDENT_REVIEW.json',
              'final_analysis/A/analysis_report.json','final_analysis/B/analysis_report.json',
              'final_analysis/C/analysis_report.json','final_comparison_r1/analysis_report.json',
              'final_independent_review_r1/INDEPENDENT_REVIEW.json',
              'final_independent_review_r1/MAIN_REPORT_REVIEW.json',
              'cold_popularity_audit_20261030/VERIFY_C_SUPPLEMENT.json',
              'cold_popularity_audit_20261030/final_report_review_r1/REVIEW.json']:
        assert read(TASK/n)['status'] == 'passed', n
    assert sha(TASK/'GOAL_OBJECTIVE.md') == '74f3a37ae3a57e9602d63b62501efb46782bfc72da42c02c9e4dd3143048cab7'
    restore = read(TASK/'final_cpu_restore_r1/FULL_MODEL_RESTORE.json')
    assert restore['status'] == 'real_cpu_restores_passed'
    coverage = read(TASK/'final_cpu_restore_r1/INDEPENDENT_REVIEW.json')
    assert (coverage['immutable_states'], coverage['rolling_last_states'], coverage['all_checked_states'],
            coverage['critical_role_count'], coverage['unique_actual_critical_restores']) == (63,3,66,10,9)
    assert not coverage['anomalies']
    backup = read(TASK/'FINAL_BACKUP_VERIFICATION.json')
    assert backup['extracted_payloads'] == 3405 and backup['all_member_and_disk_sha256_verified']
    assert backup['status'] == 'bytes_and_structural_coverage_passed'
    ledger = read(TASK/'GPU_LEDGER.json')
    assert ledger['cumulative_gpu_seconds'] == 78518 and ledger['all_goal_jobs_released']
    assert ledger['budget_gpu_seconds'] == 108000
    assert not ledger['price_estimate']['settlement_verified']
    assert read(TASK/'final_lifecycle_r1/RELEASE.json')['all_goal_jobs_released']
    cold = read(TASK/'cold_popularity_audit_20261030/STATUS.json')
    assert cold['all_requested_diagnostics_complete']
    assert not cold['pending_C_fixed_R32'] and not cold['pending_C_selected_best']
    notification = read(TASK/'NOTIFICATIONS.json')
    sent = {x['event'] for x in notification['sent']}
    assert set(notification['required_h1_events']) <= sent
    assert {'C_test_independently_audited','cold_popularity_complete_C_report'} <= sent
    source_pins = {
        'backup_delivery.py':'f99e2293ac7dc528b53151ecd0dddbb582095ef564846021fa9af12ef2df811d',
        'download_archive.py':'1217d52db527bdb2d803047d74c56ccebf4f56530ed4cdc93b6440aa3ea8eec1',
        'analyze_results.py':'47522df77e2eab06e598454ed331e6a939609bf0f63ddb9087d78f7f19aa9677',
        'analyze_results_compat.py':'c5f6c819ea3889c8365bcee44336164b7617f9651745f5452c9061111ef8a7b4',
        'cold_popularity_audit_20261030/analyze.py':'3270debadddbd528a5b9b423a5b6ccae89246ab1351ac02d27279eca34ea7267',
        'cold_popularity_audit_20261030/verify.py':'b51c6d3acf597aa481a2b1b90bd371af011d77465bdd302782f57319413cd33b',
    }
    for n, h in source_pins.items():
        assert sha(TASK/n) == h, n
        evidence.append(n)
    terminal = {}
    for b, name, t in [('A','H1-Freeze20-R35',200),('B','H0-Freeze32-R35',360),('C','H0-Freeze20-R35',200)]:
        r = RUNS/name
        d = read(r/'complete.json')
        assert d['status'] == 'training_and_validation_complete'
        assert d['r_updates'] == 24255 and d['t_updates'] == t
        assert not [x for x in r.rglob('*') if x.is_file() and ('failed' in x.name.lower() or 'failure' in x.name.lower())]
        terminal[b] = {'run':name,'r_updates':d['r_updates'],'t_updates':d['t_updates'],
            'complete_sha256':sha(r/'complete.json'), 'selection_sha256':sha(r/'selection.json'),
            'test_predictions_complete_sha256':sha(r/'test_predictions_complete.json'),
            'milestones':{x.name:sha(x) for x in sorted((r/'milestones').glob('*.json'))},
            'failure_markers':[]}
    assert read(RUNS/'H0-Freeze20-R35/replay_gate.json')['status'] == 'passed'
    pairs = read(TASK/'final_comparison_r1/analysis_report.json')
    assert pairs['selected_epochs'] == {'A':32,'B':35,'C':23}
    assert pairs['inputs_unchanged'] and len(pairs['comparisons']) == 6
    independent = read(TASK/'final_independent_review_r1/INDEPENDENT_REVIEW.json')
    assert independent['inputs_unchanged'] and independent['c_models_fully_recomputed'] == 13
    assert len(independent['comparisons']) == 6 and not independent['torch_imported']
    cpu_file = TASK/'final_lifecycle_r1/CPU_RELEASE.json'
    cpu = read(cpu_file) if cpu_file.exists() else {'status':'unconfirmed_connection_reset','gpu_count':0}
    if cpu_file.exists(): evidence.append('final_lifecycle_r1/CPU_RELEASE.json')
    draft = (TASK/'FINAL_MAIN_REPORT_DRAFT.md').read_text(encoding='utf-8')
    intro = '2026-10-30。主实验与新增冷启动/热门偏置统计已完成并通过独立复核。完整运行备份和关键状态真实恢复均已通过。本文为最终科学报告；后处理备份及实际 Git 提交由 [交付索引](DELIVERY_INDEX.md) 和 `FINAL_DELIVERY_CLOSURE.json` 分别绑定，不以本文替代这些收据。'
    draft = draft.replace(draft.split('\n')[2], intro, 1)
    draft = draft.replace('上述归档先于最终分析、恢复收据和本报告生成；这些后处理交付件不能声称已包含在该tar中。最终交付归档/索引、Git提交及Goal验收由主任务随后整合，此草稿不代为确认。',
        '上述运行归档先于最终分析、恢复收据和本报告生成；这些后处理文件作为另一份交付归档和 Git 版本保存，不能声称已包含在运行 tar 中。见 [交付索引](DELIVERY_INDEX.md) 与 [最终验收记录](FINAL_DELIVERY_AUDIT.md)。')
    draft = draft.replace('该草稿汇总结果，不覆盖那些历史记录。','本报告汇总结果，原阶段通知和历史记录保留。')
    (TASK/'REPORT.md').write_text(draft,encoding='utf-8')
    evidence.append('REPORT.md')
    acceptance = {'format':'freeze35_final_scientific_acceptance_v1','status':'scientific_and_artifact_acceptance_passed',
        'checked_utc':now,'original_scientific_goal_requirements_verified':True,
        'cold_diagnostic_complete_and_delivered':True,'all_goal_gpu_allocations_released':True,
        'gpu_seconds':78518,'gpu_hours':78518/3600,'settlement_verified':False,
        'cpu_file_entry':cpu,'terminal_runs':terminal,'common_best':{'A':32,'C':23},
        'checkpoint_coverage':{'immutable':63,'last':3,'fields_checked':66,'critical_roles':10,'actual_unique_cpu_restores':9},
        'archive':{'bytes':6846443520,'sha256':backup['archive_sha256'],'payloads':3405,'self_contained':False},
        'source_pins_unchanged':source_pins,
        'evidence':{n:{'sha256':sha(TASK/n),'bytes':(TASK/n).stat().st_size} for n in evidence},
        'remaining_administrative_closure':['postprocessing_archive_and_git_bundle_fresh_restore','task_only_git_commit','automation_stop_after_final_delivery'],
        'goal_complete':False,
        'boundary':'Receipt binds completed scientific, byte, restore, diagnostic and GPU evidence; Git/postprocessing/automation closure is a later separate receipt. CPU status is explicit; no new inference or training performed by this script.'}
    write(TASK/'FINAL_ACCEPTANCE.json',acceptance)
    index={'format':'freeze35_final_delivery_index_v1','created_utc':now,
        'main_report':'REPORT.md','cold_report':'cold_popularity_audit_20261030/FINAL_REPORT.md',
        'acceptance':{'path':'FINAL_ACCEPTANCE.json','sha256':sha(TASK/'FINAL_ACCEPTANCE.json')},
        'runtime_archive':{'local_path':str(ROOT/'outputs/stage09_restore/h_freeze35_20261029_final_r1.tar'),
            'remote_path':'/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_final_r1.tar',
            'bytes':6846443520,'sha256':backup['archive_sha256'],'manifest_sha256':backup['manifest_sha256'],
            'fresh_extraction':str(RUNS.parent.parent),'self_contained':False,'dependencies':backup['dependencies']},
        'execution_extension_commit':'e4ca4b32920c95ef7190ad475d18e6f31fa3d276',
        'postprocessing_receipt':'FINAL_DELIVERY_CLOSURE.json',
        'notes':['Runtime tar does not include later reports and restore receipts.','The later closure receipt binds the content commit, separate postprocessing archive and Git bundle; it is not included in the archive it hashes.','External data/PSE/history and software runtime must be preserved.'],
        'evidence':acceptance['evidence']}
    write(TASK/'DELIVERY_INDEX.json',index)
    (TASK/'DELIVERY_INDEX.md').write_text('''# 第09阶段冻结续训交付索引

- [主实验完整报告](REPORT.md)：固定R35、共同网格best、自身起点、真实训练/验证曲线。
- [冷启动与热门偏置完整报告](cold_popularity_audit_20261030/FINAL_REPORT.md)：五模型前后段与逐目标贡献、频次和集中度，包含公平C R32与C best R23。
- [最终科学验收](FINAL_ACCEPTANCE.json)、[验收说明](FINAL_DELIVERY_AUDIT.md)、[机器可读索引](DELIVERY_INDEX.json)。
- [66状态覆盖表](final_cpu_restore_r1/CHECKPOINT_COVERAGE.csv)、[9文件实际恢复](final_cpu_restore_r1/FULL_MODEL_RESTORE.json)、[恢复独立复审](final_cpu_restore_r1/REVIEW.md)。
- [C与A−C第二实现统计复核](final_independent_review_r1/REPORT.md)、[五模型诊断复核](cold_popularity_audit_20261030/VERIFY_C_SUPPLEMENT.json)、[诊断报告审核](cold_popularity_audit_20261030/final_report_review_r1/REVIEW.md)。

运行备份在 `outputs/stage09_restore/h_freeze35_20261029_final_r1.tar`，6,846,443,520字节，SHA256 `9006d8d835ed735bd79bfe382ae362ebc6266ba9c1522bf9a20e6b0dcb399f16`。新鲜解压目录为同级 `h_freeze35_20261029_final_r1_restored/`，3,405份payload逐SHA与关键恢复通过。它需要已有数据/PSE及历史预测，不是自包含包；具体外部依赖及绑定见JSON索引和原归档内DELIVERY_MANIFEST。

恢复用法保留在 [BACKUP_USAGE.md](BACKUP_USAGE.md)，使用原CPU环境及冻结版本；不要从当前开发HEAD替换原训练源码。科学扩展提交是 `e4ca4b32920c95ef7190ad475d18e6f31fa3d276`，最终文档/统计Git提交是另外的交付版本。

最终分析、图表、恢复收据和中文报告生成于运行包之后，另存后处理归档及Git bundle。实际路径、文件SHA、内容提交、独立解压/克隆校验与收尾提交在 `FINAL_DELIVERY_CLOSURE.json` 中记录；该闭环收据本身另行提交，不能自称包含在它所哈希的归档中。

本轮四项已授权GPU分配共21.810556 GPU小时，全部释放；费用仅按历史单价估算约21.81点，结算未核实。原0 GPU CPU文件入口的关闭状态单列于生命周期证据，不混入GPU耗时。
''',encoding='utf-8')
    (TASK/'FINAL_DELIVERY_AUDIT.md').write_text('''# 第09阶段三组冻结续训：最终验收记录

本文件替换进度版的展示入口；原进度报告原字节保存在 `stage_reports/`。实际检查及SHA绑定见 [FINAL_ACCEPTANCE.json](FINAL_ACCEPTANCE.json)。科学与运行产物验收已通过，后处理归档/Git/跟踪收尾以 `FINAL_DELIVERY_CLOSURE.json` 为准。

| 验收项 | 结论与证据 |
|---|---|
| 既定三组、预算、代码与输入 | 仅原A/B/C及短测；锁定执行包、历史源和5份扩展不变，数据/PSE逐SHA。H100真实下一步、冻结隔离和恢复连续性短验收通过 |
| C重放与R20冻结 | 原5个完整验证关口及20阶段轨迹通过，R20独立保存；预测一致不冒称恢复缺失旧权重的bitwise状态 |
| 训练与验证完成 | 三组真实complete/milestones及所有要求节点齐全；R35均24,255 R，A/C T200、B T360；无失败标记 |
| 固定终点与best测试 | A/B/C固定35及A32/C23必要best全部逐样本复算；C13份完整预测全部复核；旧B32有效结果复用 |
| 配对区间与选模 | 原2000次seed2024用户聚类bootstrap；A/C原共同网格best32/23，B仅父32对固定35；第二实现覆盖C自身比较与A−C共6对，旧A/B独立证据复用 |
| 提前通知 | H1 R29/R32/R35完整验证、R35训练和测试核验均按NOTIFICATIONS已报告；C测试核验和五模型完整诊断报告亦已登记，不把阶段报告当Goal结项 |
| 冷启动/热门诊断 | 固定旧时间切点、三覆盖类+训练目标频次、逐目标命中/DCG、热门与集中度、C同R32及best23全部完成；全数组/区间复算和每模型1027原始聚合抽查、全量计数，7表573行及两报告108行数值表独立审核 |
| 完整运行包 | 6,846,443,520字节整包、manifest与3,405payload SHA通过；独立安全解压及全部外部依赖SHA通过；包非自包含 |
| checkpoint覆盖 | 63不可覆盖+3last全部字段、形状、有限性、计数及来源检查，66行覆盖表齐全 |
| 实际恢复 | 10关键角色去重9文件完成模型、双优化器、scheduler、CPU RNG加载和无目标有限前向，零更新；源码/输入不变；CPU不冒充CUDA精确恢复 |
| 三组曲线 | 真实新增R-loss/学习率及完整validation曲线与CSV、PNG/PDF、来源SHA齐全；PNG视觉核验通过 |
| GPU与费用 | 四次已知分配78,518GPU秒=21.810556小时，全部释放，30小时内；控制器重复终止不多算新分配；账单结算未核实 |
| 报告与交接 | REPORT、五模型FINAL_REPORT、DELIVERY_INDEX、EXECUTION、PROJECT_HANDOFF与INSPIRE稳定资产索引更新；单seed/选模/冻结点/不同T边界明确 |
| 最后闭环 | 后处理归档、Git bundle、独立恢复及任务限定提交由FINAL_DELIVERY_CLOSURE记录；原0 GPU CPU入口状态单列，未确认时不写已关闭；所有要求通过后才停止09-h1并完成Goal |

没有为诊断训练或模型推理，没有增加GPU/seed/beam/Decay/评估协议，没有由提前看到的H1测试修改H0。所有未包含在先前运行tar的后处理产物单独打包；不以文件存在、平台成功或单个PASS替代完整验收。
''',encoding='utf-8')
    print(json.dumps({'status':acceptance['status'],'acceptance_sha256':sha(TASK/'FINAL_ACCEPTANCE.json'),
                      'main_report_sha256':sha(TASK/'REPORT.md'),'evidence_files':len(evidence),'cpu_status':cpu['status']}))

if __name__ == '__main__':
    main()
