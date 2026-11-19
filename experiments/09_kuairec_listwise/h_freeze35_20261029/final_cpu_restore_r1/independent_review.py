"""Offline receipt/byte audit; stdlib only, no deserialization or model execution.

This separate reviewer never imports the backup implementation or Torch. It
independently derives the expected nodes and reads all frozen payload bytes.
Existing evidence is read-only; the three report outputs must not exist.
"""
import csv
import hashlib
import io
import json
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TASK = REPO / 'experiments/09_kuairec_listwise/h_freeze35_20261029'
OUT = Path(__file__).resolve().parent
ROOT = REPO / 'outputs/stage09_restore/h_freeze35_20261029_final_r1_restored'
EXPECTED_MANIFEST = 'b4650e450f9d367e9ae5aa8780b393c2c7a79d8f9730c1aeafb2618a559cac11'
EXPECTED_VERIFICATION = '1d531247dd4cf57a0a422ff76091a579139045c29332da90e55065058d95550b'
EXPECTED_HELPER = 'f99e2293ac7dc528b53151ecd0dddbb582095ef564846021fa9af12ef2df811d'
NAMES = {'A': 'H1-Freeze20-R35', 'B': 'H0-Freeze32-R35', 'C': 'H0-Freeze20-R35'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def record(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}

def file_check(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Not a regular source file: ' + str(path))
    require(not (getattr(path.stat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT), 'Reparse source: ' + str(path))
    require(path.stat().st_size == expected['bytes'], 'Size mismatch: ' + str(path))
    actual = digest(path)
    require(actual == expected['sha256'], 'SHA mismatch: ' + str(path))
    return actual

def expected_nodes(branch):
    # Derive from 35 R epochs and the ten four-phase T/R/R/R cycles.
    start_epoch, start_phase, frozen_t = {'A': (26, 33, 200), 'B': (32, 41, 360), 'C': (5, 5, 200)}[branch]
    def clocks(epoch, t, phase, update=0, boundary=False):
        return {'R': epoch * 693, 'T': t, 'phase': phase, 'update': update, 'boundary_prepared': boundary}
    nodes = {'initial.pt': clocks(start_epoch, 0 if branch == 'C' else frozen_t, start_phase)}
    for cycle in range(10):
        first = 5 + 4 * cycle
        t = min((cycle + 1) * 40, 200) if branch == 'C' else frozen_t
        for offset in range(4):
            phase = first + offset
            if phase < start_phase:
                continue
            epoch = 5 + 3 * cycle + offset
            kind = 'T' if offset == 0 else 'R'
            name = f'phase{phase:02d}_{kind}_R{epoch:02d}_T{t:03d}.pt'
            nodes[name] = clocks(epoch, t, phase + 1)
    if branch == 'C':
        nodes['R20_pre_freeze.pt'] = clocks(20, 200, 25)
    nodes['R35_training_boundary.pt'] = clocks(35, frozen_t, 44, 693, True)
    nodes['last.pt'] = clocks(35, frozen_t, 45)
    return nodes

def progress(clocks):
    return {k: clocks[k] for k in ('phase', 'update', 'boundary_prepared')}

def main():
    outputs = [OUT / n for n in ('INDEPENDENT_REVIEW.json', 'CHECKPOINT_COVERAGE.csv', 'REVIEW.md')]
    require(not any(p.exists() for p in outputs), 'Preserve existing review; outputs already exist')
    started = datetime.now(timezone.utc).isoformat()
    evidence_paths = [ROOT / 'DELIVERY_MANIFEST.json', ROOT / 'BACKUP_VERIFICATION.json',
                      TASK / 'FINAL_PACK_RECEIPT.json', TASK / 'FINAL_TRANSFER_VERIFICATION.json',
                      TASK / 'FINAL_BACKUP_VERIFICATION.json', OUT / 'FULL_MODEL_RESTORE.json',
                      TASK / 'backup_delivery.py', ROOT / 'references/tools/backup_delivery.py',
                      *[OUT / (b + ext) for b in NAMES for ext in ('.json', '.log')]]
    evidence_before = {str(p): record(p) for p in evidence_paths}
    manifest = read(ROOT / 'DELIVERY_MANIFEST.json')
    verification = read(ROOT / 'BACKUP_VERIFICATION.json')
    pack, transfer, backup = [read(TASK / n) for n in ('FINAL_PACK_RECEIPT.json', 'FINAL_TRANSFER_VERIFICATION.json', 'FINAL_BACKUP_VERIFICATION.json')]
    full = read(OUT / 'FULL_MODEL_RESTORE.json')
    require(digest(ROOT / 'DELIVERY_MANIFEST.json') == EXPECTED_MANIFEST, 'Wrong final manifest')
    require(digest(ROOT / 'BACKUP_VERIFICATION.json') == EXPECTED_VERIFICATION, 'Wrong final byte-verification receipt')
    require(digest(TASK / 'FINAL_BACKUP_VERIFICATION.json') == EXPECTED_VERIFICATION, 'Copied backup receipt differs')
    require(verification == backup and verification['status'] == 'bytes_and_structural_coverage_passed', 'Byte verification mismatch')
    require(verification['all_member_and_disk_sha256_verified'] is True and verification['source_inputs_modified'] is False, 'Byte verification scope')
    require(pack['status'] == 'packed_not_model_restored' and transfer['status'] == 'BYTES_VERIFIED_ONLY', 'Transfer/pack stages')
    require(pack['sha256'] == transfer['sha256'] == verification['archive_sha256'], 'Archive identity chain')
    require(pack['manifest_sha256'] == transfer['manifest_sha256'] == verification['manifest_sha256'] == EXPECTED_MANIFEST, 'Manifest chain')
    require(transfer['block_sha256_verified'] is True and transfer['full_archive_and_embedded_manifest_sha256_verified'] is True, 'Transfer SHA scope')
    require(transfer['bytes'] == pack['bytes'] == 6846443520, 'Archive size identity')
    archive_sha = file_check(Path(transfer['local']), {'bytes': pack['bytes'], 'sha256': pack['sha256']})
    require(full['status'] == 'real_cpu_restores_passed' and full['byte_verification_sha256'] == EXPECTED_VERIFICATION, 'Restore chain')
    require(full['self_contained'] is False and full['cuda_exact_restore'] is False and full['gpu_or_training_performed'] is False and full['goal_completion_not_implied'] is True, 'Restore scope must remain bounded')
    require(full['dependency_proof'] == verification['dependencies'], 'Restore dependency proof differs')
    require(set(full['branches']) == set(NAMES), 'Final branch set')
    require(manifest['pack_tool_sha256'] == EXPECTED_HELPER, 'Manifest helper version')
    require(digest(TASK / 'backup_delivery.py') == digest(ROOT / 'references/tools/backup_delivery.py') == EXPECTED_HELPER, 'Actual reviewed helper bytes')
    files = manifest['files']
    require(len(files) == manifest['payload_files'] == verification['extracted_payloads'] == 3405, 'Payload count')
    require(sum(v['bytes'] for v in files.values()) == manifest['payload_bytes'] == 6841589719, 'Payload bytes')
    disk_paths = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    require(disk_paths == set(files) | {'DELIVERY_MANIFEST.json', 'BACKUP_VERIFICATION.json'}, 'Unexpected or missing extracted files')
    folded = [p.casefold() for p in files]
    require(len(set(folded)) == len(folded), 'Case-colliding manifest paths')
    payload_hashes = {}
    for relative, expected in files.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and '\\' not in relative and ':' not in relative, 'Unsafe payload path')
        payload_hashes[relative] = file_check(ROOT / relative, expected)
    print('All 3405 payload SHA checks passed', flush=True)
    execution = read(ROOT / 'goal/code/FREEZE35_EXECUTION_MANIFEST.json')
    require(payload_hashes['goal/code/FREEZE35_EXECUTION_MANIFEST.json'] == manifest['execution_manifest_sha256'], 'Execution manifest identity')
    require(len(execution['execution_files']) == execution['execution_file_count'] == 3034, 'Execution source count')
    for section in ('execution_files', 'metadata'):
        for name, item in execution[section].items():
            require(files['goal/code/' + name] == item, 'Execution inventory record differs: ' + name)
    dependency_records = []
    for name in ('dataset', 'assets'):
        dep = manifest['dependencies'][name]
        directory = Path(full['dependency_proof'][name]['path'])
        require(directory == Path(dep['local']), 'Unexpected external path relocation')
        require(digest(directory / 'manifest.json') == dep['manifest_sha256'], 'External manifest digest')
        dependency_records.append(record(directory / 'manifest.json'))
        for relative, item in dep['files'].items():
            file_check(directory / relative, item)
            dependency_records.append({'path': str(directory / relative), **item})
    for item in manifest['dependencies']['historical_results']:
        file_check(Path(item['local']), item)
        dependency_records.append({'path': item['local'], 'bytes': item['bytes'], 'sha256': item['sha256']})
    require(len(manifest['dependencies']['historical_results']) == 26, 'Historical dependency count')
    rows, branch_results, roles_expected = [], {}, []
    gate = read(ROOT / 'goal/runs/H0-Freeze20-R35/replay_gate.json')
    require(gate['status'] == 'passed' and gate['branch'] == 'C' and set(gate['checks']) == {'8', '11', '14', '17', '20'}, 'Archived C gates incomplete')
    for b, name in NAMES.items():
        d = full['branches'][b]
        require(d == read(OUT / (b + '.json')), 'Supervisor/worker mismatch: ' + b)
        require(d['branch'] == b and d['status'] == 'passed' and d['cpu_only'] is True and d['cuda_exact_restore'] is False, 'Worker status/scope')
        require(d['critical_source_bytes_rehashed_after_restore'] is True and d['archive_or_external_inputs_written_by_tool'] is False, 'Worker source preservation')
        require(d['source_environment']['torch'] == '2.6.0+cpu', 'Worker not expected CPU environment')
        expected = expected_nodes(b)
        coverage = manifest['checkpoint_coverage'][b]
        require(coverage['name'] == name and set(coverage['nodes']) == set(expected) - {'last.pt'}, 'Independent node schedule mismatch')
        require(len(expected) == {'A': 15, 'B': 7, 'C': 44}[b] and len(coverage['nodes']) == coverage['immutable_count'], 'Node count')
        require(set(d['checked_states']) == set(expected), 'Actual checked-state set mismatch')
        run_dir = ROOT / 'goal/runs' / name
        completion = read(run_dir / 'complete.json')
        original_coverage = read(run_dir / 'CHECKPOINT_COVERAGE.json')
        require(completion['status'] == 'training_and_validation_complete' and completion['branch'] == b, 'Completion record')
        require(completion['phase'] == 45 and completion['r_updates'] == 24255 and completion['t_updates'] == (360 if b == 'B' else 200), 'Completion clocks')
        require(completion['checkpoint_coverage_sha256'] == payload_hashes[f'goal/runs/{name}/CHECKPOINT_COVERAGE.json'], 'Coverage digest chain')
        require(original_coverage['status'] == 'passed' and set(original_coverage['records']) == set(expected) - {'last.pt', 'R35_training_boundary.pt'}, 'Original coverage set')
        selection = read(run_dir / 'selection.json')
        require(completion['selection_sha256'] == payload_hashes[f'goal/runs/{name}/selection.json'], 'Selection digest chain')
        require(selection['selected'] == coverage['selected'], 'Selected checkpoint mismatch')
        require(selection['selected']['epoch'] == {'A': 32, 'B': 35, 'C': 23}[b], 'Unexpected selected epoch')
        if b != 'B':
            require(set(selection['candidates']) == {'20', '23', '26', '29', '32', '35'}, 'Common candidate grid mismatch')
            winner = sorted(selection['candidates'].values(), key=lambda item: (-item['ndcg5'], item['epoch']))[0]
            require(winner == selection['selected'], 'Common validation selection/tie rule')
        else:
            require(selection['fixed_endpoint'] == 35 and selection['candidates'] == {}, 'B must remain fixed endpoint')
        prefix = f'goal/runs/{name}/'
        best_name = Path(selection['selected']['checkpoint']).name
        require(best_name in expected and selection['selected']['checkpoint_sha256'] == files[prefix + best_name]['sha256'], 'Selected state outside authenticated coverage')
        roles = [{'branch': b, 'kind': 'recovery35', 'path': prefix + 'initial.pt', 'role': 'initial'},
                 {'branch': b, 'kind': 'recovery35', 'path': prefix + ('phase44_R_R35_T360.pt' if b == 'B' else 'phase44_R_R35_T200.pt'), 'role': 'R35'}]
        if b == 'C':
            roles.append({'branch': b, 'kind': 'recovery35', 'path': prefix + 'R20_pre_freeze.pt', 'role': 'R20'})
        roles.append({'branch': b, 'kind': 'recovery35', 'path': prefix + best_name, 'role': 'best'})
        roles_expected.extend(roles)
        require(d['roles'] == roles and set(d['restored']) == {r['path'] for r in roles}, 'Actual critical-role coverage')
        for filename, clocks in expected.items():
            relative = prefix + filename
            checked = d['checked_states'][filename]
            require(checked['sha256'] == payload_hashes[relative], 'Checked-state source digest')
            require(all(checked[k] is True for k in ('fields_shapes_clocks_verified', 'sid_range_pad_verified', 'rng_fields_complete')), 'State validation flags')
            require(bool(re.fullmatch('[0-9a-f]{64}', checked['sid_sha256'])), 'SID digest absent')
            if filename != 'last.pt':
                item = coverage['nodes'][filename]
                receipt = read(run_dir / 'checkpoint_receipts' / (filename + '.json'))
                require(item == dict(receipt, expected_clocks=clocks), 'Receipt/independent clock mismatch: ' + filename)
                require(receipt['r_updates'] == clocks['R'] and receipt['t_updates'] == clocks['T'] and receipt['progress'] == progress(clocks), 'Receipt counter mismatch')
                require(receipt['path'] == filename and receipt['sha256'] == checked['sha256'] and receipt['bytes'] == files[relative]['bytes'], 'Receipt byte identity')
                require(set(receipt['fields']) == {'model', 'alignment', 'r_optimizer', 't_optimizer', 'scheduler', 'rng', 'binding'} and all(v is True for v in receipt['fields'].values()), 'Missing checkpoint field')
                if filename != 'R35_training_boundary.pt':
                    require(original_coverage['records'][filename] == receipt, 'Original checkpoint record mismatch')
            else:
                endpoint = prefix + ('phase44_R_R35_T360.pt' if b == 'B' else 'phase44_R_R35_T200.pt')
                require(coverage['rolling_last'] == files[relative] and files[relative] == files[endpoint] and completion['last_sha256'] == checked['sha256'], 'Rolling last is not byte-identical final endpoint')
            anchor_name = 'R20_pre_freeze.pt' if b == 'C' else 'initial.pt'
            frozen = clocks['phase'] >= (41 if b == 'B' else 25)
            if frozen:
                require(checked['sid_sha256'] == d['checked_states'][anchor_name]['sid_sha256'], 'Frozen SID digest drift')
            critical = relative in d['restored']
            if critical:
                result = d['restored'][relative]
                require(result['status'] == 'cpu_relocated_restore_passed' and result['source_sha256'] == checked['sha256'], 'Critical restore/source mismatch')
                require(result['r_updates'] == clocks['R'] and result['t_updates'] == clocks['T'] and result['progress'] == progress(clocks), 'Loaded critical clocks')
                require(result['scientific_parameters_and_both_optimizers_identical'] is True and result['cpu_rng_restored'] is True and result['cuda_rng_restored'] is False, 'Loaded tensors or RNG boundary')
                require(result['history_only_batch'] == [2, 50] and result['decoder_logits_shape'] == [2, 4, 256] and result['finite_forward'] is True, 'Actual finite history forward')
                require(result['sid_not_refreshed'] is True and result['updates_performed'] == 0, 'Unexpected SID/training update')
                require(bool(re.fullmatch('[0-9a-f]{64}', result['retained_original_cuda_rng_sha256'])), 'CUDA RNG provenance missing')
                relocation = result['relocation']
                require(relocation['only_disclosed_metadata_paths_changed'] is True and relocation['original_archive_bytes_untouched'] is True, 'Relocation scope')
                proof = relocation['source_proof']
                require(proof['extensions'] == {} and set(proof['relocations']) == {'dataset', 'assets'}, 'Undisclosed path/source change')
                for k in ('dataset', 'assets'):
                    require(Path(proof['relocations'][k]['execution']) == Path(full['dependency_proof'][k]['path']) and proof['relocations'][k]['parent'] == manifest['dependencies'][k]['remote'], 'Path relocation proof mismatch')
                gate_mapping = relocation['gate_mapping']
                if b == 'C' and clocks['phase'] > 25:
                    require(gate_mapping['old_r20_inner_sha256'] == gate['checkpoint_state_sha256'] and gate_mapping['old_gate_protocol_sha256'] == gate['protocol_sha256'], 'C gate relocation source mismatch')
                    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in gate_mapping.values()), 'C gate mapped digests')
                else:
                    require(gate_mapping is None, 'Gate mapping outside post-R20 C')
            rows.append({'branch': b, 'run': name, 'node': filename, 'immutable': filename != 'last.pt', 'archive_relative_path': relative,
                         'bytes': files[relative]['bytes'], 'sha256': checked['sha256'], 'independent_current_sha256': payload_hashes[relative],
                         'R_epoch': clocks['R'] // 693, 'R_updates': clocks['R'], 'T_updates': clocks['T'],
                         'phase': clocks['phase'], 'phase_update': clocks['update'], 'boundary_prepared': clocks['boundary_prepared'],
                         'worker_fields_shapes_clocks_verified': True, 'worker_original_LR_schedule_checked': True,
                         'worker_rng_fields_complete': True, 'worker_sid_range_pad_verified': True, 'sid_sha256': checked['sid_sha256'],
                         'frozen_sid_digest_matches_anchor': frozen, 'critical_roles': '|'.join(r['role'] for r in roles if r['path'] == relative),
                         'actually_critical_restored': critical, 'actual_cpu_rng_restored': critical,
                         'actual_cuda_rng_restored': False, 'finite_forward_verified': critical,
                         'updates_performed_by_restore': 0 if critical else '', 'review_method': 'byte-and-receipt-audit;no-model-rerun'})
        branch_results[b] = {'immutable_nodes': len(expected) - 1, 'rolling_last': 1, 'checked_states': len(expected),
                             'actual_critical_restores': len(d['restored']), 'roles': roles, 'selected_epoch': selection['selected']['epoch'],
                             'expected_node_set_matches_manifest_and_worker': True, 'clocks_independently_derived_and_matched': True,
                             'all_current_state_SHA_match_pre_restore_manifest_and_worker': True, 'all_frozen_SID_digests_match_anchor': True}
    require(roles_expected == manifest['model_restore_roles'], 'Manifest role list mismatch')
    require(len(rows) == 66 and sum(r['immutable'] for r in rows) == manifest['required_immutable_checkpoints'] == verification['immutable_checkpoints'] == 63, 'Final exact coverage')
    require(manifest['rolling_last_checkpoints'] == verification['rolling_last_checkpoints'] == 3, 'Rolling count')
    require(sum(r['actually_critical_restored'] for r in rows) == 9 and len(roles_expected) == 10, 'Critical dedup count')
    for p in evidence_paths:
        require(record(p) == evidence_before[str(p)], 'Original evidence changed during review')
    for row in rows:
        require(digest(ROOT / row['archive_relative_path']) == row['sha256'], 'Checkpoint changed during review')
    require('torch' not in sys.modules, 'Reviewer unexpectedly imported Torch')
    csv_buffer = io.StringIO(newline='')
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
    summary = {'status': 'passed', 'review_scope': 'independent manifest, receipts, schedule/role coverage and current raw-byte verification; no model rerun',
               'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'reviewer': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__), 'stdlib_only': True,
                            'torch_imported': False, 'checkpoint_deserialization_performed': False, 'model_or_training_or_network_performed': False},
               'source_evidence': list(evidence_before.values()), 'source_evidence_unchanged_before_after_review': True,
               'archive_sha256_recomputed': archive_sha, 'manifest_sha256': EXPECTED_MANIFEST, 'verification_sha256': EXPECTED_VERIFICATION,
               'all_3405_payloads_SHA_recomputed_after_restore': True, 'payload_bytes': manifest['payload_bytes'],
               'execution_files_bound_and_unchanged': 3034, 'execution_metadata_files': len(execution['metadata']),
               'external_dependency_records_rehashed': dependency_records, 'branches': branch_results,
               'immutable_states': 63, 'rolling_last_states': 3, 'all_checked_states': 66,
               'critical_role_count': 10, 'unique_actual_critical_restores': 9, 'all_66_source_SHA_rechecked_at_review_end': True,
               'tensor_validation_basis': 'Actual final worker receipts plus SHA-bound implementation review; independent reviewer did not deserialize or rerun tensors.',
               'reviewed_validation_guards': {
                   'fields_shapes_finite': 'recovery35.py:211-239 validates model/alignment keys, tensor shape/dtype/finite and optimizer parameter/moment mapping/finite.',
                   'R_T_and_LR': 'recovery35.py:148-209 and training.py:121-134 validate cursor-derived R/T, Adam steps, original full-35R scheduler and original T LR, DIGER original total400 and frozen alignment/T optimizer fingerprints.',
                   'all_states': 'backup_delivery.py:485-517 checks every required node+last, source/envelope identity, canonical freeze anchor, C replay gate, SID Nx4/PAD=-1/range and full RNG keys.',
                   'critical_actual_restore': 'backup_delivery.py:519-555 performs strict state load, compares model/alignment/both optimizers/scheduler fingerprints, CPU RNG, history-only B2xL50 finite forward and full-state/SID unchanged, then rehashes source.',
                   'relocation': 'backup_delivery.py:369-411 permits only authenticated dataset/PSE path metadata; scientific tensor/counter/RNG fingerprints unchanged; C gate binds original archived R20 before mapping.',
                   'source_runtime': 'backup_delivery.py:446-476 requires CPU-only Torch and exact extracted recovery35 module path, and uses archived parent/ancestor states.',
                   'supervisor': 'backup_delivery.py:564-597 authenticates verification/manifest, supervises three workers, compares exact roles/state sets, rechecks execution sources and dependencies before final receipt.'},
               'boundaries': {'cuda_exact_restore': False, 'cuda_rng_actually_restored': False, 'cuda_rng_preserved_by_digest': True,
                              'cpu_python_numpy_torch_rng_actually_restored': True, 'all_66_deserialized_by_original_worker': True,
                              'all_66_independently_deserialized_by_reviewer': False, 'all_66_actual_model_reload_claim': False,
                              'finite_forward_is_full_recommendation_evaluation': False, 'self_contained_archive': False,
                              'software_environment_bundled': False, 'metric_or_bootstrap_acceptance': False,
                              'GPU_release_or_platform_settlement_acceptance': False, 'Goal_complete_implied': False},
               'coverage_csv': {'path': str(outputs[1]), 'rows': 66, 'sha256': hashlib.sha256(csv_bytes).hexdigest()},
               'anomalies': []}
    md = '''# 最终 checkpoint 与 CPU 恢复独立复核

结论：**通过本项验收**。逐一核对最终 manifest、实际 A/B/C 恢复明细及原始边界收据，并独立重算归档、3,405 个解压文件和外部依赖 SHA；全部与已认证清单一致。原收据、归档和源码未修改。独立复核使用标准库读取 JSON/原始字节，没有反序列化 checkpoint、重建模型、执行推理或训练。

| 组 | 不可覆盖状态 | rolling last | 实际结构检查 | 关键角色 | 实际关键文件恢复 |
|---|---:|---:|---:|---:|---:|
| A | 14 | 1 | 15 | initial / R35 / best R32 | 3 |
| B | 6 | 1 | 7 | initial / R35 / best（固定 R35） | 2 |
| C | 43 | 1 | 44 | initial / R35 / R20 / best R23 | 4 |
| 合计 | 63 | 3 | 66 | 10 个角色 | 9 个唯一文件 |

应有节点由独立日程算式生成，再与 manifest、逐 checkpoint 收据及 worker 的 `checked_states` 比对，集合完全一致。35R 终点累计 24,255 次 R 更新，A/C 固定 200T，B 固定 360T；C 重放从 R5/T0 到 R20/T200。R35 未完成验证时保存的 training boundary（phase44/update693/prepared=true）与完整终点（phase45/update0）分别保留；三个 last 均与各自完整 R35 文件字节相同。C 的 R20 前冻结状态单独覆盖。

原真实恢复 worker 已对 66 个状态读取并校验字段、模型/alignment 形状/类型/有限性、优化器映射及 moment 形状/有限性、R/T Adam 时钟、原 35R LR 日程及 T LR、DIGER 时钟、冻结指纹、SID 范围/PAD、RNG 字段。独立复核核对这些实际明细并审查 SHA 绑定的校验实现；没有声称独立再次检验张量内容。逐节点 SHA、R/T 计数、阶段与关键恢复覆盖见 `CHECKPOINT_COVERAGE.csv`。

9 个关键文件确有模型、alignment、双优化器及 scheduler 的实际加载后指纹比较，CPU Python/NumPy/Torch RNG 恢复，真实 B2×L50 历史的有限前向（logits 2×4×256）。全部零训练更新、SID 不刷新、前向前后完整状态一致。每个关键来源文件在原恢复后已复哈希；本复核再次核全部 66 个状态，并在复核结束前再核一次，均匹配恢复前 manifest。B 的两个角色共享同一 R35 文件，去重并无遗漏。

路径重定位仅发生在内存中的已认证 dataset/PSE 绑定元数据；保留旧/新绑定指纹及 C 的 R20 gate 映射，不能据此改变科学参数或磁盘 checkpoint。C 的后冻结关键状态均正确绑定原 R20 replay gate。

范围：CPU 恢复没有实际恢复 CUDA RNG，只保留原 CUDA RNG 指纹；不能称 CUDA 精确续训验收。归档依赖外部 dataset、PSE 和 26 份历史结果，环境未打包，不能称单包自包含。这里只验收 checkpoint 覆盖和真实 CPU 恢复，不替代指标/bootstrap、冷启动补充、资源释放结算或整个 Goal 验收。原 `BACKUP_VERIFICATION` 的 model restore pending 是历史阶段边界，后续完成由本次 `FULL_MODEL_RESTORE` 及其明细补充证明，原收据保持不变。

最终归档 SHA256：`9006d8d835ed735bd79bfe382ae362ebc6266ba9c1522bf9a20e6b0dcb399f16`。
最终 manifest SHA256：`b4650e450f9d367e9ae5aa8780b393c2c7a79d8f9730c1aeafb2618a559cac11`。
认证字节验证收据 SHA256：`1d531247dd4cf57a0a422ff76091a579139045c29332da90e55065058d95550b`。

独立复核详情、原证据 SHA、实现审核位置与范围见 `INDEPENDENT_REVIEW.json`；`independent_review.py` 是新建离线复核工具，不属于冻结执行源码。
'''
    with outputs[1].open('xb') as f:
        f.write(csv_bytes)
    with outputs[2].open('x', encoding='utf-8', newline='\n') as f:
        f.write(md)
    summary['review_markdown_sha256'] = digest(outputs[2])
    with outputs[0].open('x', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'status': 'passed', 'checked_states': 66, 'critical_roles': 10, 'actual_critical_restores': 9,
                      'outputs': [record(p) for p in outputs]}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
