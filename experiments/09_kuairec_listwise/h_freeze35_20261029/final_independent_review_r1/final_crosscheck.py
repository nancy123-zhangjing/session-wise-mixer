"""Final small-evidence crosscheck only; no model, prediction or archive replay."""
from pathlib import Path
import csv
import difflib
import hashlib
import json
from datetime import datetime, timezone

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
ROOT = REPO / 'outputs/stage09_restore/h_freeze35_20261029_final_r1_restored'
OUT = Path(__file__).resolve().parent
started = datetime.now(timezone.utc).isoformat()
checks, issues, hashes = [], [], {}

def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    if not condition:
        issues.append(name)

def digest(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise RuntimeError('Refusing large-file hash in final crosscheck: ' + str(path))
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes.setdefault(str(path), value)
    return value

def read(path):
    path = Path(path)
    digest(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def receipt(name):
    return read(TASK / name)

acceptance = receipt('FINAL_ACCEPTANCE.json')
index = receipt('DELIVERY_INDEX.json')
for name in ['GOAL_OBJECTIVE.md', 'FINAL_DELIVERY_AUDIT.md', 'DELIVERY_INDEX.md',
             'REPORT.md', 'FINAL_MAIN_REPORT_DRAFT.md', 'BACKUP_USAGE.md']:
    digest(TASK / name)
evidence = []
for name, expected in acceptance['evidence'].items():
    path = TASK / name
    actual = {'sha256': digest(path), 'bytes': path.stat().st_size}
    check('acceptance evidence bytes/SHA: ' + name, actual == expected)
    evidence.append({'path': name, **actual, 'matches_acceptance_and_index': actual == expected == index['evidence'].get(name)})
check('exactly 44 acceptance evidence records', len(evidence) == 44)
check('index evidence identical to acceptance', index['evidence'] == acceptance['evidence'])
check('index acceptance SHA binding', digest(TASK / index['acceptance']['path']) == index['acceptance']['sha256'])
check('science acceptance explicitly not Goal complete', acceptance['goal_complete'] is False)
check('CPU file entry uncertainty explicit and zero GPU', acceptance['cpu_file_entry'] == {'status': 'unconfirmed_connection_reset', 'gpu_count': 0})
for path, expected in acceptance['source_pins_unchanged'].items():
    check('frozen/helper source pin: ' + path, digest(TASK / path) == expected)

main_review = receipt('final_independent_review_r1/MAIN_REPORT_REVIEW.json')
draft = TASK / 'FINAL_MAIN_REPORT_DRAFT.md'
check('reviewed draft SHA and passed review', main_review['status'] == 'passed' and digest(draft) == main_review['manuscript_sha256'] and not main_review['unresolved_findings'])
old = draft.read_text(encoding='utf-8-sig').splitlines()
new = (TASK / 'REPORT.md').read_text(encoding='utf-8-sig').splitlines()
changes = []
for kind, a, b, c, d in difflib.SequenceMatcher(None, old, new).get_opcodes():
    if kind != 'equal':
        changes.append({'kind': kind, 'draft_lines': [a + 1, b], 'final_lines': [c + 1, d], 'before': old[a:b], 'after': new[c:d]})
check('only three approved delivery-framing substitutions', len(changes) == 3 and all(x['kind'] == 'replace' and len(x['before']) == len(x['after']) == 1 for x in changes) and [x['draft_lines'][0] for x in changes] == [3, 163, 172])

manifest = read(ROOT / 'DELIVERY_MANIFEST.json')
backup = receipt('FINAL_BACKUP_VERIFICATION.json')
transfer = receipt('FINAL_TRANSFER_VERIFICATION.json')
restore = receipt('final_cpu_restore_r1/FULL_MODEL_RESTORE.json')
restore_review = receipt('final_cpu_restore_r1/INDEPENDENT_REVIEW.json')
check('3405 manifest payload records', len(manifest['files']) == manifest['payload_files'] == backup['extracted_payloads'] == acceptance['archive']['payloads'] == 3405)
check('runtime archive SHA/size cross-binding', index['runtime_archive']['sha256'] == acceptance['archive']['sha256'] == transfer['sha256'] == backup['archive_sha256'] == restore_review['archive_sha256_recomputed'] and index['runtime_archive']['bytes'] == acceptance['archive']['bytes'] == transfer['bytes'] == 6846443520)
check('fresh manifest SHA bound', digest(ROOT / 'DELIVERY_MANIFEST.json') == backup['manifest_sha256'] == index['runtime_archive']['manifest_sha256'] == restore_review['manifest_sha256'])
check('byte-stage receipt followed by actual restore', backup['status'] == 'bytes_and_structural_coverage_passed' and restore['status'] == 'real_cpu_restores_passed' and restore['byte_verification_sha256'] == digest(TASK / 'FINAL_BACKUP_VERIFICATION.json'))
check('independent actual restore and byte review passed', restore_review['status'] == 'passed' and not restore_review['anomalies'] and restore_review['all_3405_payloads_SHA_recomputed_after_restore'] and restore_review['source_evidence_unchanged_before_after_review'])
check('external dependency counts and non-self-contained boundary', index['runtime_archive']['dependencies'] == backup['dependencies'] and backup['dependencies']['dataset']['files'] == 8 and backup['dependencies']['assets']['files'] == 12 and backup['dependencies']['historical_results'] == 26 and manifest['self_contained'] is False and manifest['software_environment_bundled'] is False and acceptance['archive']['self_contained'] is False)
check('actual restore CPU-only and no training', restore['cuda_exact_restore'] is False and restore['gpu_or_training_performed'] is False and restore['goal_completion_not_implied'] is True)

coverage_path = TASK / 'final_cpu_restore_r1/CHECKPOINT_COVERAGE.csv'
rows = list(csv.DictReader(coverage_path.open(encoding='utf-8-sig', newline='')))
check('CSV 66 state rows, 63 immutable and 3 last', len(rows) == 66 and sum(x['immutable'] == 'True' for x in rows) == 63 and sum(x['node'] == 'last.pt' for x in rows) == 3)
check('66/9/10 exact acceptance coverage', acceptance['checkpoint_coverage'] == {'immutable': 63, 'last': 3, 'fields_checked': 66, 'critical_roles': 10, 'actual_unique_cpu_restores': 9})
check('ten required roles resolve to nine files', len(manifest['model_restore_roles']) == 10 and len({x['path'] for x in manifest['model_restore_roles']}) == 9)
check('CSV actual restore flags identify nine distinct files', sum(x['actually_critical_restored'] == 'True' for x in rows) == 9)
for row in rows:
    branch, name = row['branch'], row['node']
    record = restore['branches'][branch]['checked_states'][name]
    file_record = manifest['files'][row['archive_relative_path']]
    check('state metadata cross-binding: ' + branch + '/' + name,
          row['sha256'] == row['independent_current_sha256'] == record['sha256'] == file_record['sha256'] and int(row['bytes']) == file_record['bytes'] and int(row['R_epoch']) * 693 == int(row['R_updates']) and all(row[x] == 'True' for x in ['worker_fields_shapes_clocks_verified', 'worker_original_LR_schedule_checked', 'worker_rng_fields_complete', 'worker_sid_range_pad_verified']) and row['updates_performed_by_restore'] == ('0' if row['actually_critical_restored'] == 'True' else ''))
for branch, expected_count in [('A', 15), ('B', 7), ('C', 44)]:
    actual = restore['branches'][branch]
    expected_paths = {x['path'] for x in manifest['model_restore_roles'] if x['branch'] == branch}
    check('branch exact checked/actual role coverage: ' + branch, len(actual['checked_states']) == expected_count and set(actual['restored']) == expected_paths and actual['archive_or_external_inputs_written_by_tool'] is False and actual['critical_source_bytes_rehashed_after_restore'])
    for path, item in actual['restored'].items():
        check('actual CPU restore flags: ' + path, item['status'] == 'cpu_relocated_restore_passed' and item['cpu_rng_restored'] and item['cuda_rng_restored'] is False and item['scientific_parameters_and_both_optimizers_identical'] and item['sid_not_refreshed'] and item['finite_forward'] and item['updates_performed'] == 0 and item['source_sha256'] == manifest['files'][path]['sha256'] and item['relocation']['only_disclosed_metadata_paths_changed'] and item['relocation']['original_archive_bytes_untouched'])

terminal = {}
for branch, expected in acceptance['terminal_runs'].items():
    directory = ROOT / 'goal/runs' / expected['run']
    complete, selection, tests = [read(directory / f) for f in ['complete.json', 'selection.json', 'test_predictions_complete.json']]
    for name, key in [('complete.json', 'complete_sha256'), ('selection.json', 'selection_sha256'), ('test_predictions_complete.json', 'test_predictions_complete_sha256')]:
        check('terminal SHA: ' + branch + '/' + name, digest(directory / name) == expected[key])
    check('terminal counters and phase: ' + branch, complete['r_updates'] == expected['r_updates'] == 24255 and complete['t_updates'] == expected['t_updates'] == {'A': 200, 'B': 360, 'C': 200}[branch] and complete['phase'] == 45 and complete['status'] == 'training_and_validation_complete')
    check('terminal checkpoint coverage digest: ' + branch, digest(directory / 'checkpoint_coverage.json') == complete['checkpoint_coverage_sha256'])
    check('selection-only tests and exact planned epochs: ' + branch, tests['selection_sha256'] == complete['selection_sha256'] == expected['selection_sha256'] and [x['epoch'] for x in tests['results']] == {'A': [35, 32], 'B': [35], 'C': [35, 23]}[branch])
    if branch in ['A', 'C']:
        winner = min(selection['candidates'].values(), key=lambda x: (-x['ndcg5'], x['epoch']))['epoch']
        check('common validation grid and earliest tie: ' + branch, set(selection['candidates']) == {'20','23','26','29','32','35'} and winner == selection['selected']['epoch'] == acceptance['common_best'][branch])
    else:
        check('B fixed endpoint not common-grid selection', selection['candidates'] == {} and selection['fixed_endpoint'] == 35 and selection['selected']['epoch'] == 35)
    milestones = {}
    for name, expected_sha in expected['milestones'].items():
        data = read(directory / 'milestones' / name)
        check('milestone SHA: ' + branch + '/' + name, digest(directory / 'milestones' / name) == expected_sha)
        milestones[name] = data
    failures = [str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and ('fail' in p.name.lower() or 'error' in p.name.lower())]
    check('no run failure/error file markers: ' + branch, failures == expected['failure_markers'] == [])
    terminal[branch] = {'run': expected['run'], 'r_updates': complete['r_updates'], 't_updates': complete['t_updates'], 'selected_epoch': selection['selected']['epoch'], 'test_epochs': [x['epoch'] for x in tests['results']], 'milestones': milestones, 'failure_markers': failures}

statistics = {}
all_models = {}
for name in ['final_analysis/A/analysis_report.json', 'final_analysis/B/analysis_report.json', 'final_analysis/C/analysis_report.json', 'final_comparison_r1/analysis_report.json']:
    data = receipt(name)
    check('accepted original bootstrap: ' + name, data['status'] == 'passed' and data['bootstrap'] == {'repeats': 2000, 'seed': 2024, 'confidence': 0.95, 'unit': 'user cluster'} and data['inputs_unchanged'])
    for label, model in data['models'].items():
        audit = read(Path(model['audit_path']))
        check('model per-sample audit binding: ' + name + '/' + label, digest(Path(model['audit_path'])) == model['audit_sha256'] and audit['passed'] and audit['predictions_sha256'] == model['predictions_sha256'] and audit['summary_sha256'] == model['summary_sha256'])
        all_models[label] = model
    for label, comp in data['comparisons'].items():
        check('paired comparison binding: ' + name + '/' + label, digest(Path(comp['path'])) == comp['sha256'])
    statistics[name] = {'model_count': len(data['models']), 'models': list(data['models']), 'comparisons': list(data['comparisons']), 'status': data['status']}
check('C 13 full result audits', len(statistics['final_analysis/C/analysis_report.json']['models']) == 13)
check('required validation model coverage', set(all_models) == {f'A_R{x}_validation' for x in [20,23,26,29,32,35]} | {f'A_R{x}_test' for x in [20,26,32,35]} | {f'B_R{x}_{s}' for x in [32,35] for s in ['validation','test']} | {f'C_R{x}_validation' for x in [5,8,11,14,17,20,23,26,29,32,35]} | {f'C_R{x}_test' for x in [23,35]})
second = receipt('final_independent_review_r1/INDEPENDENT_REVIEW.json')
check('second implementation C+AC scope and no overclaim', second['status'] == 'passed' and second['c_models_fully_recomputed'] == 13 and len(second['comparisons']) == 6 and second['existing_A_B_branch_audits_repeated'] is False and second['final_A_B_pair_independently_recomputed'] is False)
check('five C replay gates and training trajectory accepted', [x['epoch'] for x in main_review['replay_gates']] == [8,11,14,17,20] and all(x['top_list_agreement'] == 1 and x['ndcg5_absolute_delta'] == 0 for x in main_review['replay_gates']) and '20 trajectory receipts agree' in main_review['checks'])
h100 = receipt('H100_ACCEPTANCE_AUDIT.json')
check('separate H100 next-step/isolation/state-side-effect tests', h100['status'] == 'passed' and all(x['status'] == 'passed' and x['exact_disk_next_update_replay'] and x['frozen_module_optimizer_hard_sid_isolation'] and x['FP32_decode_same_output_and_no_training_state_side_effects'] and x['ends_at_exact_input_state'] for x in h100['branches'].values()))

notifications = receipt('NOTIFICATIONS.json')
notification_review = []
for event in notifications['required_h1_events']:
    found = [x for x in notifications['sent'] if x['event'] == event and x.get('branch') == 'A']
    check('exactly one required H1 notification: ' + event, len(found) == 1)
    item = found[0]
    check('H1 notification evidence SHA: ' + event, digest(TASK / item['evidence']) == item['evidence_sha256'])
    notification_review.append({k: item[k] for k in ['event','reported_utc','evidence','evidence_sha256']})
check('H1 test notification before C freeze20 acceptance', datetime.fromisoformat(next(x['reported_utc'] for x in notifications['sent'] if x['event'] == 'test_independently_audited')) < datetime.fromisoformat(next(x['reported_utc'] for x in notifications['sent'] if x['event'] == 'C_replay_R20_freeze_accepted')))

cold_status = receipt('cold_popularity_audit_20261030/STATUS.json')
cold_review = receipt('cold_popularity_audit_20261030/final_report_review_r1/REVIEW.json')
export = receipt('cold_popularity_audit_20261030/c_supplement_tables_r1/EXPORT_RECEIPT.json')
check('all initial and C diagnostic work complete and delivered', cold_status['status'] == 'all_requested_diagnostics_verified_and_delivered' and cold_status['initial_verified'] and cold_status['initial_report_delivered'] and not cold_status['pending_C_fixed_R32'] and not cold_status['pending_C_selected_best'] and cold_status['all_requested_diagnostics_complete'] and cold_status['no_additional_gpu'] and cold_status['original_goal_completion_not_implied'])
check('C diagnostic exactly fixed32 and common best23', cold_status['c_models'] == ['C_freeze_R32','C_best_R23'])
check('five-model report independent review passed', cold_review['status'] == 'passed' and not cold_review['blocking_findings'] and cold_review['all_108_numeric_rows_match_source_rounding'] and cold_review['all_csv_cells_match_locked_JSON'])
check('seven exported tables 573 rows', export['status'] == 'passed' and len(export['files']) == 7 and sum(export['table_rows'].values()) == cold_review['csv_total_rows'] == 573)
for name, expected in export['files'].items():
    path = TASK / 'cold_popularity_audit_20261030/c_supplement_tables_r1' / name
    check('cold table SHA and row count: ' + name, digest(path) == expected and len(list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))) == export['table_rows'][Path(name).stem])
stat_scope = cold_review['source_statistical_verification']
check('cold raw-row second-implementation sampling boundary', stat_scope['models'] == 5 and stat_scope['array_rows_per_model'] == stat_scope['raw_rows_counted_per_model'] == 71657 and stat_scope['raw_rows_sampled_per_model'] == 1027)

release = receipt('final_lifecycle_r1/RELEASE.json')
ledger = receipt('GPU_LEDGER.json')
seconds = sum(x['gpu_seconds'] for x in release['allocations'])
check('four known GPU allocations released within total30h', release['status'] == 'verified' and release['all_goal_jobs_released'] and release['known_allocation_count'] == 4 and seconds == release['total_gpu_seconds'] == acceptance['gpu_seconds'] == ledger['cumulative_gpu_seconds'] == 78518 and seconds <= ledger['budget_gpu_hours'] * 3600 == 108000 and release['includes_preflight_gpu_seconds'] == 219)
check('allocation accounting not platform billing', acceptance['settlement_verified'] is False and release['settlement_verified'] is False and ledger['settlement_verified'] is False and ledger['cumulative_is_final'])
for path in [REPO/'docs/guides/PROJECT_HANDOFF.md', REPO/'experiments/09_kuairec_listwise/REPORT.md']:
    digest(path)
handoff = (REPO/'docs/guides/PROJECT_HANDOFF.md').read_text(encoding='utf-8-sig')
check('current handoff links science and explicit closure boundary', all(t in handoff[:2500] for t in ['最终科学交付','0.31%','21.810556','FINAL_DELIVERY_CLOSURE','不要在闭环收据产生前替Goal宣告完成']))

requirements = [
    ('I/III', '原历史源码、完整起点、数据/PSE绑定及恢复扩展', ['INPUT_AUDIT.json','DETERMINISTIC_PACKAGE_AUDIT.json','EXECUTION_PACKAGE.json','H100_ACCEPTANCE_AUDIT.json'], 'passed', 'CPU前查与H100真实下一步验收分开；最终使用冻结代码。'),
    ('II/III', 'A/B/C各自起点、R/T阶段、原35R学习率、冻结隔离', ['final_independent_review_r1/MAIN_REPORT_REVIEW.json','final_cpu_restore_r1/INDEPENDENT_REVIEW.json'], 'passed', '新增R=6237/2079/20790，T=0/0/200；末态R24255/T200,360,200；无阶段压缩。'),
    ('III/C', 'C重放5关口、20段轨迹与独立R20冻结分界', ['C_R20_REPLAY_FREEZE_EVIDENCE.json','final_independent_review_r1/MAIN_REPORT_REVIEW.json','final_cpu_restore_r1/CHECKPOINT_COVERAGE.csv'], 'passed', 'R8/11/14/17/20全部通过原阈值；不声称缺失旧R20权重bitwise重现。'),
    ('IV', '授权并行预算、短测隔离及全部GPU释放', ['EXECUTION.md','GPU_LEDGER.json','final_lifecycle_r1/RELEASE.json'], 'passed', '3个唯一正式Job与219秒短测；累计78518秒，<=30小时；费用估算非结算。'),
    ('V', '全部完整验证、固定35、共同网格best与B父32', ['final_analysis/A/analysis_report.json','final_analysis/B/analysis_report.json','final_analysis/C/analysis_report.json'], 'passed', 'A共同best32/C23；B35是固定终点，不冒充共同网格选模。'),
    ('VI', '计划内5次新测试、旧测试复用与逐样本审计', ['final_analysis/A/analysis_report.json','final_analysis/B/analysis_report.json','final_analysis/C/analysis_report.json'], 'passed', 'A35/32、B35、C35/23；53,194样本/4,153用户；旧A23/26验证审计明确复用。'),
    ('VI', '用户配对bootstrap与主要/补充比较', ['final_comparison_r1/analysis_report.json','final_analysis/A/SUPPLEMENTAL_BEST_COMPARISON_BINDING.json','final_independent_review_r1/INDEPENDENT_REVIEW.json'], 'passed', '原2000/2024/95%；第二实现覆盖C13份与6项C/AC区间，A/B已验历史复核不重复冒称。'),
    ('VII', 'H1三次全验证、训练完成、测试审计分开及时通知', ['NOTIFICATIONS.json','H1_STAGE_REPORT.md','H1_TEST_REPORT.md'], 'passed', '五个必需通知唯一且证据SHA匹配；H1测试报告先于C冻结20完成，无等待C。'),
    ('VIII', '全节点完整保存及66状态覆盖', ['final_cpu_restore_r1/CHECKPOINT_COVERAGE.csv','final_cpu_restore_r1/INDEPENDENT_REVIEW.json'], 'passed', '63不可覆盖+3last；A15/B7/C44；字段、时钟、LR、SID、来源SHA检查不是66次模型实例恢复。'),
    ('VIII', '关键起点、C20、终态、best实际恢复', ['final_cpu_restore_r1/FULL_MODEL_RESTORE.json','final_cpu_restore_r1/INDEPENDENT_REVIEW.json'], 'passed', '10角色/9文件；CPU RNG真实恢复、CUDA RNG仅保留；零更新；不是CUDA精确续训或完整评估。'),
    ('VIII', '完整运行包下载、全成员SHA及独立解压', ['FINAL_TRANSFER_VERIFICATION.json','FINAL_BACKUP_VERIFICATION.json','final_cpu_restore_r1/INDEPENDENT_REVIEW.json'], 'passed', '6,846,443,520字节/3405payload；已有独立全SHA复核；8数据/12PSE/26历史预测依赖，非自包含。'),
    ('IX', '曲线、中文报告、路径索引与交接', ['REPORT.md','DELIVERY_INDEX.json','DELIVERY_INDEX.md','final_figures_r1/PLOT_PROVENANCE.json','final_figures_r1/VISUAL_REVIEW.json',str(REPO/'docs/guides/PROJECT_HANDOFF.md')], 'passed', '最终稿科学内容与已通过234项复审的草稿完全相同；仅三处交付说明替换。'),
    ('新增CPU诊断', '初版三模型及C公平R32、共同best23补充', ['cold_popularity_audit_20261030/STATUS.json','cold_popularity_audit_20261030/VERIFY_C_SUPPLEMENT.json','cold_popularity_audit_20261030/final_report_review_r1/REVIEW.json'], 'passed', '完整五模型；原时间切点、三类覆盖、训练目标频次分层、逐目标贡献及推荐集中；无训练/推理/新GPU。'),
    ('科学解释', '0.31%描述性分解、warm mid、head与集中度、session判断', ['cold_popularity_audit_20261030/FINAL_REPORT.md','cold_popularity_audit_20261030/REPORT_C_SUPPLEMENT.md','cold_popularity_audit_20261030/final_report_review_r1/REVIEW.json'], 'passed', '全数组/区间复算+每模型1027条原始推荐抽查+全量行计数；不是每条原始推荐第二实现复核；不作因果或session切分有效承诺。'),
    ('IX/行政', '后处理归档、Git bundle新鲜恢复、仅本任务Git提交', ['FINAL_DELIVERY_CLOSURE.json'], 'pending_administrative_closure', '不在运行tar内；须由根任务后续真实收据绑定，本文不验收。'),
    ('行政', '0GPU CPU文件入口最终关闭确认', ['FINAL_ACCEPTANCE.json:cpu_file_entry'], 'pending_administrative_closure', '当前ECONNRESET导致关闭状态未确认，不将其算入GPU，也不宣称已关闭。'),
    ('最终Goal', '交付后停止跟踪并最终Goal签收', ['FINAL_DELIVERY_CLOSURE.json'], 'pending_administrative_closure', '待前述收尾后执行；本复核Goal complete=false。'),
]
requirements = [dict(zip(['section','requirement','evidence','status','boundary'], x)) for x in requirements]
for name, expected in list(hashes.items()):
    check('source unchanged through crosscheck: ' + name, digest(Path(name)) == expected)
result = {
    'format': 'freeze35_final_document_crosscheck_v1',
    'status': 'scientific_crosscheck_passed_administrative_closure_pending' if not issues else 'findings_require_review',
    'started_at_utc': started,
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'reviewer': 'independent child connection_local_audit',
    'scope': 'Small evidence hashes, receipt cross-binding, manifest/CSV/role coverage, final-vs-reviewed-draft text comparison and original Goal/diagnostic requirement mapping. No large-payload rehash, tensor deserialization, model, inference, network, Git or original-file edits.',
    'evidence_count': len(evidence), 'evidence': evidence,
    'document_sha256': {n: digest(TASK/n) for n in ['FINAL_ACCEPTANCE.json','DELIVERY_INDEX.json','DELIVERY_INDEX.md','FINAL_DELIVERY_AUDIT.md','REPORT.md','GOAL_OBJECTIVE.md']},
    'reviewed_draft_sha256': main_review['manuscript_sha256'],
    'approved_framing_changes': changes,
    'scientific_body_unchanged': len(changes) == 3 and not issues,
    'terminal_runs': terminal,
    'checkpoint_coverage': acceptance['checkpoint_coverage'],
    'runtime_payload_count': len(manifest['files']),
    'archive_SHA_recomputed_in_this_crosscheck': False,
    'prior_independent_full_SHA_review_bound': digest(TASK/'final_cpu_restore_r1/INDEPENDENT_REVIEW.json'),
    'statistics': statistics,
    'h1_notification_records': notification_review,
    'cold_raw_verification_scope': stat_scope,
    'requirements': requirements,
    'uncompleted_scientific_items': [] if not issues else issues,
    'administrative_closure_pending': [x for x in requirements if x['status'] == 'pending_administrative_closure'],
    'historical_pending_receipts': {
        'test_predictions_complete_and_milestones': 'Generated before independent metric audit; pending/false flags remain immutable. Later SHA-bound branch per-sample audits close the scientific check.',
        'FINAL_BACKUP_VERIFICATION.actual_model_restore': 'Byte-stage pending is historical and explicitly followed by FULL_MODEL_RESTORE and INDEPENDENT_REVIEW.',
        'INPUT_AUDIT_COVERAGE_and_early_checklist': 'Pre-execution/pre-download snapshots; not the final state.'
    },
    'boundaries': {
        'goal_complete': False,
        'cpu_entry_closed_confirmed': False,
        'platform_settlement_verified': False,
        'all_66_states_actually_reloaded_as_models': False,
        'actual_unique_cpu_restores': 9,
        'cuda_exact_restore': False,
        'runtime_archive_self_contained': False,
        'postprocessing_archive_included_in_runtime_tar': False,
        'all_raw_cold_recommendations_second_implementation_rechecked': False,
        'single_seed_or_selection_uncertainty_resolved': False,
        'parent_may_close_Goal_using_this_receipt_alone': False,
    },
    'source_SHA_before_after': hashes,
    'original_inputs_unchanged': not any(x.startswith('source unchanged') for x in issues),
    'checks_count': len(checks), 'checks': checks, 'findings': issues,
    'script_sha256': digest(Path(__file__)),
}
(OUT/'FINAL_CROSSCHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
lines = ['# 最终科学交付独立交叉核验', '',
         '结论：科学与运行产物核验通过；行政收尾仍待根任务完成，整个 Goal 尚不能由本记录宣告完成。' if not issues else '结论：发现待复查项：' + '; '.join(issues), '',
         f'核验时间：{result["completed_at_utc"]}。本次仅读取小型证据/清单并写本目录复核结果，没有网络、模型执行、大文件重新哈希或修改受审文件。', '',
         'FINAL_ACCEPTANCE 的44项证据大小和SHA全部吻合；DELIVERY_INDEX的44项清单完全相同，且正确绑定验收JSON。最终REPORT与通过234项检查的草稿相比仅3处交付/历史通知措辞变化，所有科学数值、表格、图与解释内容不变。', '',
         'A/B/C末态均R24255，T分别200/360/200；固定35与A32/C23必要测试齐备。3份complete、3份selection、3份测试生成收据、10个milestones均逐SHA核对，没有run目录failure/error文件标记。B的selected35字段只是固定终点，不能改写成共同网格best。', '',
         '66状态=63不可覆盖+3last，A/B/C分别15/7/44；10个关键角色对应9个真实CPU恢复文件。CSV逐行与manifest及真实恢复字段绑定一致。3,405运行payload及6,846,443,520字节归档采用已独立全SHA复核的绑定收据，本次未重复大文件哈希。原8份数据、12份PSE、26份历史结果依赖及软件环境须另外保留，运行包非自包含。', '',
         '科学要求映射：', '', '| 要求 | 状态 | 核验范围/边界 |', '|---|---|---|']
for item in requirements:
    lines.append('| '+item['requirement']+' | '+('通过' if item['status']=='passed' else '行政收尾待完成')+' | '+item['boundary']+' |')
lines += ['',
          '旧收据中的`validation_pending`、`independent_audit_pending`和字节验收`actual_model_restore=pending`均是当时的事件边界，已经由后续验证、逐样本审计、FULL_MODEL_RESTORE及独立恢复复核分别补齐；不应篡改旧收据，也不应据此误判仍有科学项未完成。', '',
          '必须保留的解释边界：CPU恢复不等同CUDA精确续训；66状态检查不等同66次模型实例恢复；新增统计第二实现覆盖C的13份结果及6项C/AC配对，A/B历史审计按原证据复用。冷诊断第二实现覆盖全数组及bootstrap、每模型1027条原始推荐抽查和71,657条全量计数，不是每条原始推荐的第二实现重算。0.31%是固定粗分层的描述性组成贡献，warm mid内部变化、训练head比例和top10集中必须分别解释；session留出没有在本轮验证有效。', '',
          '资源确认：4项GPU分配含219秒短测，共78,518秒=21.810555556小时，均有释放证据，未超累计30小时；不是平台结算确认。原0 GPU文件入口由于ECONNRESET尚未确认关闭，必须作为单独的行政未决事项保留。', '',
          '根任务后续仍须完成：仅本任务Git提交、后处理交付归档与Git bundle独立恢复、0 GPU入口关闭状态确认、最终交付/闭环收据及停止跟踪。最终报告/索引对后处理归档的交付措辞不能替代尚待生成的实际闭环收据。所有这些完成前，`goal_complete=false`。', '',
          f'机器可读证据：[FINAL_CROSSCHECK.json](FINAL_CROSSCHECK.json)，SHA256 `{hashlib.sha256((OUT/"FINAL_CROSSCHECK.json").read_bytes()).hexdigest()}`。', '']
(OUT/'FINAL_CROSSCHECK.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'status': result['status'], 'checks': len(checks), 'findings': issues, 'json_sha256': hashlib.sha256((OUT/'FINAL_CROSSCHECK.json').read_bytes()).hexdigest(), 'md_sha256': hashlib.sha256((OUT/'FINAL_CROSSCHECK.md').read_bytes()).hexdigest()}, ensure_ascii=False))
