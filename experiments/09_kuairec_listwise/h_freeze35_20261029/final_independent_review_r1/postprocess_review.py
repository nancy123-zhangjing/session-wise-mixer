"""Independent read-only postprocess archive/Git review; never loads models."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
PACK = REPO / 'outputs/stage09_restore/h_freeze35_20261029_postprocess_563f8db2dcc2'
OUT = Path(__file__).resolve().parent
TASK_REL = TASK.relative_to(REPO).as_posix()
EXPECTED_COMMIT = '563f8db2dcc2814c23439d2a7546256948272300'
EXPECTED_RECEIPT_SHA = '2f9546f40b50e4ab80752751804e3dda0cec5963b435fe5bf97c1cfc4857f61c'
started = datetime.now(timezone.utc).isoformat()
checks, findings, member_rows = [], [], []

def check(label, ok):
    checks.append({'check': label, 'passed': bool(ok)})
    if not ok:
        findings.append(label)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

receipt_path = PACK / 'VERIFICATION.json'
manifest_path = PACK / 'MANIFEST.json'
receipt = load(receipt_path)
manifest = load(manifest_path)
extract = Path(receipt['extraction'])
clone = Path(receipt['git_restore'])
protected = [receipt_path, manifest_path, PACK/'delivery.tar', PACK/'delivery.bundle', clone/'.git/config']
before = {str(p): sha(p) for p in protected}
check('received verification SHA', before[str(receipt_path)] == EXPECTED_RECEIPT_SHA)
check('passed receipt and requested content commit', receipt['status'] == 'postprocess_archive_and_git_restore_passed' and receipt['content_commit'] == manifest['content_commit'] == EXPECTED_COMMIT)
for kind in ['archive', 'git_bundle']:
    item = receipt[kind]
    path = Path(item['path'])
    check(kind + ' independently recomputed SHA and size', before[str(path)] == item['sha256'] and path.stat().st_size == item['bytes'])
check('manifest independently recomputed SHA', before[str(manifest_path)] == receipt['manifest']['sha256'])
check('411 payload count', len(manifest['members']) == manifest['file_count'] == receipt['verified_files'] == 411)

tar_rows = {}
with tarfile.open(PACK/'delivery.tar', 'r:') as archive:
    members = archive.getmembers()
    names = [x.name for x in members]
    check('tar 411 payload plus exactly one manifest', len(members) == 412 and len(set(names)) == 412 and set(names) == set(manifest['members']) | {'POSTPROCESS_MANIFEST.json'})
    check('all tar members are safe ordinary files', all(x.isfile() and not PurePosixPath(x.name).is_absolute() and '..' not in PurePosixPath(x.name).parts for x in members))
    for member in members:
        raw = archive.extractfile(member).read()
        member_sha = sha_bytes(raw)
        disk = extract / member.name
        if member.name == 'POSTPROCESS_MANIFEST.json':
            check('embedded and extracted manifest bytes identical', member_sha == sha(disk) == before[str(manifest_path)])
            continue
        expected = manifest['members'][member.name]
        disk_sha = sha(disk)
        ok = member.size == len(raw) == disk.stat().st_size == expected['bytes'] and member_sha == disk_sha == expected['sha256']
        check('raw tar and fresh disk member: ' + member.name, ok)
        normalized_sha = sha_bytes(raw.replace(b'\r\n', b'\n'))
        check('raw versus Git canonical declaration: ' + member.name,
              expected['git_blob_sha256'] == (normalized_sha if expected['git_normalized_crlf'] else member_sha))
        tar_rows[member.name] = {'path': member.name, 'bytes': len(raw), 'tar_raw_sha256': member_sha, 'extracted_raw_sha256': disk_sha, 'expected_raw_sha256': expected['sha256'], 'expected_git_blob_sha256': expected['git_blob_sha256'], 'git_normalized_crlf': expected['git_normalized_crlf']}
check('no unmanifested extracted files', {p.relative_to(extract).as_posix() for p in extract.rglob('*') if p.is_file()} == set(manifest['members']) | {'POSTPROCESS_MANIFEST.json'})

env = dict(os.environ)
env['GIT_OPTIONAL_LOCKS'] = '0'
env['GIT_TERMINAL_PROMPT'] = '0'
def git(*args, input_bytes=None):
    return subprocess.run(['git','-c','core.longpaths=true','-C',str(clone),*args], input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)

head = git('rev-parse','HEAD')
check('fresh clone HEAD is requested content commit', head.returncode == 0 and head.stdout.decode().strip() == EXPECTED_COMMIT)
status = git('status','--porcelain=v1')
check('restored Git checkout clean', status.returncode == 0 and not status.stdout.strip())
fsck = git('fsck','--full','--strict')
check('independent restored-clone fsck strict passed', fsck.returncode == 0)
bundle_verify = git('bundle','verify',str(PACK/'delivery.bundle'))
check('independent bundle verify passed', bundle_verify.returncode == 0)
paths = list(manifest['members'])
batch = git('cat-file','--batch', input_bytes=''.join(EXPECTED_COMMIT+':'+p+'\n' for p in paths).encode('utf-8'))
check('read committed canonical blobs with one local batch', batch.returncode == 0)
cursor = 0
for path in paths:
    end = batch.stdout.index(b'\n', cursor)
    header = batch.stdout[cursor:end].decode('ascii').split()
    if len(header) != 3 or header[1] != 'blob':
        raise RuntimeError('Expected blob in git cat-file batch: ' + path)
    length = int(header[2])
    raw = batch.stdout[end+1:end+1+length]
    cursor = end + 1 + length + 1
    blob_sha = sha_bytes(raw)
    clone_disk_sha = sha(clone/path)
    expected = manifest['members'][path]
    check('canonical Git blob: ' + path, len(raw) == length and blob_sha == expected['git_blob_sha256'])
    check('Git worktree has canonical or original CRLF bytes: ' + path, clone_disk_sha in [blob_sha, expected['sha256']])
    tar_rows[path].update({'git_blob_sha256': blob_sha, 'git_blob_bytes': length, 'git_checkout_sha256': clone_disk_sha})
check('Git batch consumed exactly', cursor == len(batch.stdout))
check('only INSPIRE.md normalizes CRLF', [p for p,v in manifest['members'].items() if v['git_normalized_crlf']] == receipt['git_crlf_normalized_files'] == ['INSPIRE.md'])

archive_task = extract / TASK_REL
acceptance = load(archive_task / 'FINAL_ACCEPTANCE.json')
index = load(archive_task / 'DELIVERY_INDEX.json')
science_rows = []
for path, expected in acceptance['evidence'].items():
    archived_path = TASK_REL + '/' + path
    record = tar_rows[archived_path]
    check('original scientific receipt payload binding: ' + path,
          record['bytes'] == expected['bytes'] and record['tar_raw_sha256'] == record['extracted_raw_sha256'] == expected['sha256'] and index['evidence'][path] == expected)
    science_rows.append({'path': path, 'bytes': record['bytes'], 'sha256': record['extracted_raw_sha256'], 'git_blob_sha256': record['git_blob_sha256']})
check('exactly 44 original science evidence records', len(science_rows) == 44)
check('archive acceptance JSON SHA correctly indexed', sha(archive_task/'FINAL_ACCEPTANCE.json') == index['acceptance']['sha256'])
report_pins = {
    'REPORT.md': 'd9e7c04edcd9f8cd482146bee051e8af389607a8963e0af6f7976d9edc51f4c2',
    'cold_popularity_audit_20261030/FINAL_REPORT.md': 'acb366d436260caa459845a6156fc62d1944d83410dcdcfa14e85894bb697225',
    'cold_popularity_audit_20261030/REPORT_C_SUPPLEMENT.md': '2ba7954d97a68924ce4c5a9acd889dbbac0ed61935836855e8b554f7a4bfabb0',
    'final_independent_review_r1/FINAL_CROSSCHECK.json': 'b5283763a326957da87a07a8f0a4e7544f3c81be5fad7743933cee41084daaab',
}
for path, pinned in report_pins.items():
    check('previously reviewed report pinned in fresh extraction: ' + path, sha(archive_task/path) == tar_rows[TASK_REL+'/'+path]['git_blob_sha256'] == pinned)
failed = REPO/'outputs/stage09_restore/h_freeze35_20261029_postprocess_e1e7e7832b4a'
check('failed first attempt retained without success receipt', failed.is_dir() and not (failed/'VERIFICATION.json').exists())
check('closure receipt excluded from archived payload', not any(p.endswith('/FINAL_DELIVERY_CLOSURE.json') for p in manifest['members']))
for path, expected in before.items():
    check('protected package/config bytes unchanged: ' + path, sha(Path(path)) == expected)

review = {
    'format': 'freeze35_postprocess_independent_review_v1',
    'status': 'passed' if not findings else 'findings_require_review',
    'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'content_commit': EXPECTED_COMMIT,
    'verification_receipt_sha256': EXPECTED_RECEIPT_SHA,
    'archive': receipt['archive'], 'git_bundle': receipt['git_bundle'], 'manifest': receipt['manifest'],
    'archive_bundle_manifest_SHA_independently_recomputed': True,
    'raw_payloads_independently_hashed_in_tar_and_fresh_extraction': len(tar_rows),
    'tar_total_members_including_manifest': 412,
    'all_canonical_git_blobs_independently_hashed': len(tar_rows),
    'all_clone_worktree_files_independently_hashed': len(tar_rows),
    'clone_HEAD': head.stdout.decode().strip(),
    'independent_git_fsck': {'returncode': fsck.returncode, 'stdout': fsck.stdout.decode(errors='replace'), 'stderr': fsck.stderr.decode(errors='replace')},
    'independent_bundle_verify': {'returncode': bundle_verify.returncode, 'stdout': bundle_verify.stdout.decode(errors='replace'), 'stderr': bundle_verify.stderr.decode(errors='replace')},
    'git_per_command_options': ['core.longpaths=true'],
    'git_optional_locks': '0',
    'git_config_modified': False,
    'normalized_files': ['INSPIRE.md'],
    'normalization_note': 'Tar/extraction preserve original raw bytes. Canonical Git blob content is verified separately; INSPIRE.md alone differs through CRLF-to-LF normalization. Both hashes are recorded, never treated as equal.',
    'scientific_evidence': science_rows,
    'report_pins': report_pins,
    'members': list(tar_rows.values()),
    'failed_first_attempt': {'path': str(failed), 'retained': failed.is_dir(), 'successful_verification_receipt_exists': (failed/'VERIFICATION.json').exists(), 'not_used_for_acceptance': True},
    'snapshot_scope': 'Only sealed content-commit/manifest/tar/extraction bytes are reviewed. Later legitimate workspace NOTIFICATIONS additions and final closure metadata are outside this sealed snapshot, not scientific drift.',
    'protected_hashes_before_after': before,
    'remaining_goal_work': ['zero-GPU CPU file-entry stop confirmation remains unconfirmed after ECONNRESET', 'final closure receipt and follow-up commit outside the archive', 'stop tracking only after all Goal closure requirements'],
    'boundaries': {'goal_complete': False, 'cpu_entry_closed_confirmed': False, 'network_access': False, 'model_or_Torch_execution': False, 'runtime_6_85GB_archive_rehashed_or_repackaged': False, 'existing_package_or_source_edited': False, 'closure_receipt_included_in_its_own_archive': False},
    'checks_count': len(checks), 'checks': checks, 'findings': findings,
    'script_sha256': sha(Path(__file__)),
}
(OUT/'POSTPROCESS_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
json_sha = sha(OUT/'POSTPROCESS_REVIEW.json')
text = f'''# 后处理备份与 Git 恢复独立审核

结论：{'通过' if not findings else '发现待复查项'}。本次独立读取约38.7 MB后处理包及Git bundle，未调用网络、Torch或模型，也未重读/打包6.85 GB运行模型归档。

- 内容提交与新clone HEAD均为 `{EXPECTED_COMMIT}`；新clone工作区干净，独立执行`git fsck --full --strict`与`git bundle verify`均通过。
- tar、bundle、manifest的SHA分别重新计算，与原VERIFICATION一致。tar包含411份payload和1份嵌入清单，共412个普通安全成员。
- 411份tar内原始文件与411份新解压文件的大小/SHA逐一吻合；嵌入清单与独立manifest相同，新解压目录无额外文件。
- 独立读取411份提交中的canonical Git blob并计算SHA，同时读取新clone文件。仅`INSPIRE.md`存在原始CRLF与Git LF规范化差异；raw和canonical SHA分别核对，未混为一谈。
- 原科学验收44项证据在新解压中全部维持原大小/SHA，DELIVERY_INDEX绑定一致。主报告、五模型冷诊断报告、C补充报告与FINAL_CROSSCHECK均精确匹配前次审核SHA。

本次审查的原始验证收据SHA：`{EXPECTED_RECEIPT_SHA}`。

| 对象 | 字节数 | SHA256 |
|---|---:|---|
| delivery.tar | {receipt['archive']['bytes']} | `{receipt['archive']['sha256']}` |
| delivery.bundle | {receipt['git_bundle']['bytes']} | `{receipt['git_bundle']['sha256']}` |
| MANIFEST.json | {manifest_path.stat().st_size} | `{receipt['manifest']['sha256']}` |

首次`e1e7e7832b4a`失败目录仍保留且没有成功VERIFICATION，本次不把它纳入验收。第二次仅通过每条Git命令的`-c core.longpaths=true`处理长路径；本复核未修改全局/仓库配置，新clone配置前后SHA未变。

这是已封存内容提交时点的验收。根任务随后追加的NOTIFICATIONS属于包外合法收尾元数据，不应拿当前工作区新通知去否定已封存原字节。最终闭环收据、本独立审核结果及其后续提交也在包外保存，不能自称包含于它们所哈希的旧包。

Goal仍未完成：原0 GPU CPU入口因连接重置尚未确认关闭；最终闭环收据、后续提交及停止跟踪仍由根任务完成。本次不宣称CPU已关闭、不将CPU入口算为GPU，也不以科学/备份通过代替整个Goal结项。

机器可读审核：[POSTPROCESS_REVIEW.json](POSTPROCESS_REVIEW.json)，SHA256 `{json_sha}`。共{len(checks)}项检查，未解决问题{len(findings)}项。
'''
(OUT/'POSTPROCESS_REVIEW.md').write_text(text,encoding='utf-8')
print(json.dumps({'status':review['status'],'checks':len(checks),'findings':findings,'json_sha256':json_sha,'md_sha256':sha(OUT/'POSTPROCESS_REVIEW.md')},ensure_ascii=False))
