# 阶段05长期四组：完整备份与恢复指南

本实验固定为 `goal_20261020_long_mixer_v1`。完整备份应同时包含共同初态、四臂 best/last、选择及测试收据、14件锁定输入、训练源码快照、引擎快照、Git bundles、原始记录及分析工具。旧的轻量 smoke/进度包不能替代完整备份。

远程最终归档已经通过全部字节核验，并从该归档的独立副本实际恢复四臂共八份best/last模型、优化器、scheduler、RNG及SID。证据在 `backup_evidence_remote_r1/`；其中 `REMOTE_RESTORE_EVIDENCE_AUDIT.json` 逐一绑定归档、清单、CPU执行日志和原始训练收据。本地完整包已下载，最终本地字节检查与Git恢复已从该包实际通过，见[同包绑定验收](FINAL_BACKUP_BINDING_AUDIT.json)。历史 `GIT_RESTORE_RECIPE_PROBE.json` 仅是早期源码探针，不能替代最终包验收。

本轮归档大小1,536,215,400字节，SHA256为`c2ad35b0c87b219e6435a970dfd62bf5677cf7e7c2924b34c53e48c6fec47af7`。清单包含1871条文件记录，归档另含MANIFEST.json本身。下载目录为`outputs/stage05_long/package_r1/`，校验解压目录为`outputs/stage05_long/extract_r1/goal_20261020_long_mixer_v1/`，Git恢复输出为`outputs/stage05_long/git_restore_r1/`；后两者只有真实成功收据存在时才能称为通过。

## 1. 先校验完整分卷包

保留 `PACKAGE.json` 和其中列出的全部分卷，按记录的文件名与顺序读取。使用交付时锁定哈希的 `pack_mixer_long_delivery.py`，例如：

```text
python -B <TOOLS>/scripts/analysis/pack_mixer_long_delivery.py --mode verify --package-dir <PACKAGE_DIRECTORY> --output <NEW_EXTRACT_PARENT>/goal_20261020_long_mixer_v1
```

`<TOOLS>` 是交付的独立分析工具目录；以最终分析锁及文件哈希为准。它不属于训练提交 `6968269`。输出目录必须全新，目录末级保留本 campaign 名称。校验器核对每个分卷、拼接压缩流、内部 MANIFEST、所有成员字节及14件输入；成功生成 `BYTE_VERIFICATION.json`。若失败，保留失败证据并定位损坏，不用已有部分解压目录冒充成功目录。

解压后的 `code/`、`engine/` 是训练源码快照；不包含 `.git/`。需要 Git 仓库时，在另一个新目录按第2节重建，保持已校验备份不变。Windows 路径宜短，Python需支持长路径。

## 2. 从真实 bundles 重建两个 Git 仓库

完整包中必须实际查到并核对以下文件。`prep/source_release_fd04bfd.tar.gz` 另存旧版源码打包副本，不能单独把根仓库恢复到最终训练版本；仍需要 `code_r2.bundle`。

| 完整包相对路径 | 已核对归档内 SHA256 | 恢复作用 |
| --- | --- | --- |
| `prep/code.bundle` | `b3cb871bd303d6fe45279d9929c7b1bd2693569a3f6574f4af27830e8fc0d397` | 根仓库基础 bundle，ref 指向 `fd04bfd6d9ab8009fdc4fb1d613a4db3adacb98f` |
| `prep/code_r2.bundle` | `81c1f50eef5802c665ad5b170bba9e8576c40409d826ef6211b524249df8d319` | 在基础对象已存在后，将同一 ref 更新到最终训练提交 `69682698c15ba796b98e6329c950061a2f68a05b` |
| `prep/engine.bundle` | `69faf55e0188bcb6ac1cbe619ada855428fee0c07bdb4487e57b760a35b47d2c` | 引擎 HEAD `1c879252f6395d6e64462511823c3929b9c42c00`，须同时恢复浅历史边界 |

引擎是浅仓库，边界为 `ad7b971c7e525e9fea6fb8e362a5c49dccb2473c`。bundle 没有自动携带 `.git/shallow`，因此要在 fetch **之前**写入该文件，采用 ASCII、无 BOM、行尾 LF。不能把普通 `git clone engine.bundle` 当作已验证的完整恢复步骤，也不能把这个 bundle 称作完整历史归档。

以下 Python 配方可在 Windows 或 Linux 执行。只改 `BACKUP` 和 `RECOVERY` 两个绝对路径；`RECOVERY` 必须不存在，不使用当前工作仓库、不覆盖运行目录。它会核对 bundles、按顺序 fetch、校验对象库、检查每个跟踪路径存在，并按备份中正式 identity 核对19份执行源码。Python和Git需已安装。

```python
from pathlib import Path
import hashlib, json, subprocess

BACKUP = Path(r"/absolute/verified_backup/goal_20261020_long_mixer_v1").resolve()
RECOVERY = Path(r"/absolute/new_private/source_restore").resolve()
assert BACKUP.is_dir() and not RECOVERY.exists()
expected = {
    "code.bundle": "b3cb871bd303d6fe45279d9929c7b1bd2693569a3f6574f4af27830e8fc0d397",
    "code_r2.bundle": "81c1f50eef5802c665ad5b170bba9e8576c40409d826ef6211b524249df8d319",
    "engine.bundle": "69faf55e0188bcb6ac1cbe619ada855428fee0c07bdb4487e57b760a35b47d2c",
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()

verified = json.loads((BACKUP / "BYTE_VERIFICATION.json").read_text())
assert verified["status"] == "FULL_BACKUP_BYTE_VERIFICATION_PASS_NOT_RESTORE"
assert sha(BACKUP / "MANIFEST.json") == verified["manifest_sha256"]
for name, value in expected.items():
    assert sha(BACKUP / "prep" / name) == value, name

RECOVERY.mkdir(parents=True, exist_ok=False)
code, engine = RECOVERY / "code", RECOVERY / "engine"
for repo in (code, engine):
    subprocess.check_call(["git", "init", str(repo)])
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "core.eol", "lf")
    git(repo, "config", "core.longpaths", "true")

ref = "refs/heads/codex/stage05-mixer-long"
for name, commit in (
    ("code.bundle", "fd04bfd6d9ab8009fdc4fb1d613a4db3adacb98f"),
    ("code_r2.bundle", "69682698c15ba796b98e6329c950061a2f68a05b"),
):
    bundle = str(BACKUP / "prep" / name)
    git(code, "bundle", "verify", bundle)
    git(code, "fetch", "--no-tags", bundle, ref + ":" + ref)
    assert git(code, "rev-parse", ref) == commit
git(code, "checkout", "--detach", "69682698c15ba796b98e6329c950061a2f68a05b")

boundary = "ad7b971c7e525e9fea6fb8e362a5c49dccb2473c"
(engine / ".git/shallow").write_bytes((boundary + "\n").encode("ascii"))
bundle = str(BACKUP / "prep/engine.bundle")
git(engine, "bundle", "verify", bundle)
git(engine, "fetch", "--no-tags", bundle,
    "HEAD:refs/heads/codex/stage05-engine-restore")
git(engine, "checkout", "--detach", "1c879252f6395d6e64462511823c3929b9c42c00")
assert git(engine, "rev-parse", "--is-shallow-repository") == "true"

for repo, count in ((code, 1456), (engine, 37)):
    git(repo, "fsck", "--full")
    assert git(repo, "status", "--porcelain") == ""
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"]
    ).decode().split("\0")
    names = [name for name in names if name]
    assert len(names) == count and all((repo / name).is_file() for name in names)
identity = json.loads((BACKUP / "formal/identity.json").read_text())
for repo, field in ((code, "root_sources"), (engine, "engine_sources")):
    for name, value in identity[field].items():
        assert sha(repo / name) == value, (repo, name)
print("Git source restore passed; model-state restore remains separate.")
```

Windows 实测必须设置 `core.longpaths=true`：首次未设置时，Git checkout 退出码仍为0，但15个历史深层CSV没有落盘。探针没有仅凭退出码判断成功，原失败目录予以保留。设置后全新目录的1456个根仓库文件、37个引擎文件全部存在，两仓库状态干净，`git fsck --full`通过，19份执行源码的字节哈希与正式identity一致。根仓库 tree 为 `fe625851b4edaa5436a0ce328b9ab19206a47c00`，引擎 tree 为 `f1bccd3ec4e201aa759bf595d394663e0e9af801`。

## 3. 从完整备份重建并恢复真实模型

使用具有完整 Pipeline 依赖的既有 `grm-exp` CPU 环境；当前本地 `continuous_env` 只有Torch/Numpy等必要数值组件，不能据此宣称完整Pipeline恢复成功。无需安装或升级共享环境。既有远端解释器为：

```text
/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
```

先在CPU上校验同一完整备份副本，再运行交付时锁定的恢复器。`--backup-root` 指向已校验解压根目录；`--scratch-root` 是全新的私有目录，末级也必须为campaign名。保持原备份及训练目录只读，不把scratch放在备份根内。

```text
<GRM_PYTHON> -B <TOOLS>/scripts/analysis/restore_mixer_long_backup.py --backup-root <VERIFIED_BACKUP>/goal_20261020_long_mixer_v1 --scratch-root <NEW_PRIVATE_SCRATCH_PARENT>/goal_20261020_long_mixer_v1
```

调用前设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`。恢复器会重新核对所有manifest成员字节，只从备份自身的 `code/`、`engine/` 和14件输入重建Pipeline，核对完整共同初态，然后恢复四臂各自的 `last.pt` 与 `best.pt`。检查包括：

- strict模型状态、完整Adam参数组/动量/step、推荐器scheduler及累计计数；Frozen没有Tokenizer优化器。
- Python、NumPy、CPU Torch RNG恢复；CUDA RNG是非空uint8状态字节并保留哈希。
- 正式SID、重新构造的非persistent E0 anchor、原始checkpoint未被改写。

成功输出scratch内的 `RESTORE_PASS.json`；将该收据与最终本地 `BYTE_VERIFICATION.json`、MANIFEST及PACKAGE一起保留。CPU消费者恢复不执行CUDA，也不等同于CUDA下一阶段数值重放；同后端重放证据须另引用本轮真实H100验收收据。Git恢复、字节校验、CPU模型恢复和GPU资源释放分别核验，不相互替代。

本轮上述CPU恢复已经在远端私有scratch实际执行通过，不需要为下载再申请GPU。最终本地验收比较两地PACKAGE、MANIFEST和完整归档原始字节哈希，并比较两份BYTE_VERIFICATION的解析JSON语义；Windows与Linux收据行尾可能不同，分别记录原始SHA。CPU恢复绑定远端BYTE收据，Git恢复绑定本地BYTE收据。不同审计器的`model_tensor_sha256`计算方法不同，不能跨算法直接比较；以同一checkpoint原始文件SHA连接两套内部检查。

后续若需要实际恢复训练，使用已恢复完整状态和同一冻结协议，在完整阶段边界续接；`last.pt`包含已提交阶段及累计计数，不从best重复训练已完成预算。任何新GPU运行须遵守当时明确授权和本Goal剩余预算。正式测试若已有 `TEST_STARTED.json` 却没有结果，先检查失败原因，不直接重跑。
