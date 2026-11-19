# 本地修改验收（2026-11-06）

状态：代码与四组配置已完成；没有提交实验。

- 9项CPU回归测试全部通过，使用微型合成模型，不训练KuaiRec。
- 验证四组完整R/T预算、R26最后已用系数、线性首尾系数、起点有效码本连续、模型/优化器/RNG继承。
- 验证T只提交一次增量、历史修正不回撤、冻结组Tokenizer和T优化器全程不变。
- 验证断点恢复后下一步结果一致、跨组/篡改门控被拒绝、评估缓存绑定、起点纳入选模与同分取早。
- 修复检查中发现的新评估目录未自动创建问题，最终完整9项测试通过。
- 8个新增Python文件语法检查通过，四组入口默认预览通过，无训练副作用。
- 495个继承源码文件逐文件SHA一致。原实验目录没有修改。

## 未验范围

启智连接刷新报getaddrinfo ENOTFOUND <institution-host>，未能完成远端父文件核对。
PARENTS.lock.json未生成，不存在“父文件已通过”或“已启动”的结论。
真实父checkpoint的SHA、门控元数据和验证记录需由lock_parents.py核对；
真实CUDA起点/SID连续性在运行入口第一步更新前检查。未使用GPU。

## 证据

CPU_TESTS.log、LOCAL_VERIFICATION.json、CONFIG_CHECK.json、INHERITED_SOURCE.json。
SOURCE_MANIFEST.json封存本次v1，后续改科学设置请建立v2，不修改正在运行的v1。
