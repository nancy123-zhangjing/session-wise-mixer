# 后处理备份与 Git 恢复独立审核

结论：通过。本次独立读取约38.7 MB后处理包及Git bundle，未调用网络、Torch或模型，也未重读/打包6.85 GB运行模型归档。

- 内容提交与新clone HEAD均为 `563f8db2dcc2814c23439d2a7546256948272300`；新clone工作区干净，独立执行`git fsck --full --strict`与`git bundle verify`均通过。
- tar、bundle、manifest的SHA分别重新计算，与原VERIFICATION一致。tar包含411份payload和1份嵌入清单，共412个普通安全成员。
- 411份tar内原始文件与411份新解压文件的大小/SHA逐一吻合；嵌入清单与独立manifest相同，新解压目录无额外文件。
- 独立读取411份提交中的canonical Git blob并计算SHA，同时读取新clone文件。仅`INSPIRE.md`存在原始CRLF与Git LF规范化差异；raw和canonical SHA分别核对，未混为一谈。
- 原科学验收44项证据在新解压中全部维持原大小/SHA，DELIVERY_INDEX绑定一致。主报告、五模型冷诊断报告、C补充报告与FINAL_CROSSCHECK均精确匹配前次审核SHA。

本次审查的原始验证收据SHA：`2f9546f40b50e4ab80752751804e3dda0cec5963b435fe5bf97c1cfc4857f61c`。

| 对象 | 字节数 | SHA256 |
|---|---:|---|
| delivery.tar | 38666240 | `24948ec4d228da5d13ac5a02b1be751e958681e3ff749724154cf1e2fe9ea426` |
| delivery.bundle | 38649187 | `42f23cce3beea81b46ea5464331d56c79bcd8c7bf1d6ebdec22547cc2203ba0b` |
| MANIFEST.json | 141692 | `f0159a0586660d666f9a6347eeda18fd2a4c2e62d211f95f0505733496e5db0a` |

首次`e1e7e7832b4a`失败目录仍保留且没有成功VERIFICATION，本次不把它纳入验收。第二次仅通过每条Git命令的`-c core.longpaths=true`处理长路径；本复核未修改全局/仓库配置，新clone配置前后SHA未变。

这是已封存内容提交时点的验收。根任务随后追加的NOTIFICATIONS属于包外合法收尾元数据，不应拿当前工作区新通知去否定已封存原字节。最终闭环收据、本独立审核结果及其后续提交也在包外保存，不能自称包含于它们所哈希的旧包。

Goal仍未完成：原0 GPU CPU入口因连接重置尚未确认关闭；最终闭环收据、后续提交及停止跟踪仍由根任务完成。本次不宣称CPU已关闭、不将CPU入口算为GPU，也不以科学/备份通过代替整个Goal结项。

机器可读审核：[POSTPROCESS_REVIEW.json](POSTPROCESS_REVIEW.json)，SHA256 `19a8d2fd7f5b41ea65c8623db693cd357d8998c383de524adaff9449540d633f`。共1718项检查，未解决问题0项。
