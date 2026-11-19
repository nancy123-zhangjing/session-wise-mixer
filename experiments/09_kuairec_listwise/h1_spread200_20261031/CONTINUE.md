# Stage09 D实验：当前继续位置

2026-11-01最终收尾（本条优先于下方历史记录）：网络恢复后已上传最终科学补充包及manifest、receipt，远端三份SHA256与本地完全一致，见delivery/SCIENCE_DELIVERY_REMOTE_VERIFIED.json。CPU r2已停止，r1/r2均重新Live查询为STOPPED，r2生命周期终止于2026-11-01 12:22:20；原远端科学产物保留。所有GPU此前已释放，本地相关工作进程查询为0。本轮没有重跑训练、推理、统计或恢复旧监控。

goal_completion_audit.py已实际exit0：8项全部通过，重新核268科学输入文件、r3源码证据、备份/冷恢复/资源证据。FINAL_DELIVERY_CLOSURE.json及FINAL_GOAL_ACCEPTANCE.md已创建并读回；本轮将Goal标记complete，最终状态以Goal工具为准。累计GPU上界14.37527778小时，实际结算仍未核实。主比较D比冻结A下降38.27%，负结果按协议完成，不追加实验。以下网络阻塞、缺少最终收据或待同步描述均为历史。

18:31第三连续Goal轮阻塞审计（最新）：现有CPU status仍API read ECONNRESET，现有cached connection仍TunnelError。PLATFORM_CLOSURE_BLOCKED_AUDIT_TURN3.json记录当前两查询、前两轮证据和三连续条件。科学/本地交付已完成，当前没有其他独立必要工作能推进；缺项仍为远端补充包同步/SHA、现有临时CPU停止确认和最后8项Goal关闭审计，两个最终通过文件尚不存在。本轮将请求Goal blocked（最终实际状态以Goal工具返回为准），不是完成或用户请求暂停。所有科学产物/已封存包/接续操作已保全。用户恢复接入并继续后，阻塞审计必须重新从新一轮计数，直接完成已授权收尾，不重复训练/推理/统计，不新建资源，不恢复旧监控。

18:29后第二个Goal continuation增量（最新）：当前平台status再次ECONNRESET，cached connection再次TunnelError；同一网络条件已连续2个Goal turns。PLATFORM_CLOSURE_RECHECK_TURN2.json保存两条当前查询及计数；旧PLATFORM_CLOSURE_PENDING保留首次观察，不覆盖。此turn只把EVIDENCE_PLAN更新为真实已通过的科学/备份/GPU范围和最后CPU/远端补充包缺项，没有重复统计/推理。所有本地科学与交付准备均已完成，无其他独立必要工作；用户网络恢复请求仍待回复。Goal保持active，本turn未满足3连续门槛；若下一turn相同条件仍存在、没有可推进动作，应按blocked审计设Goal blocked，而非无限重复状态。网络恢复则立即完成此前已授权收尾，不需重新请求批准。

18:26:35增量（最高优先级）：科学、模型备份/独立解压/冷恢复、统计、报告与视觉验收均已实际完成，全部5个GPU均已释放。Goal未完成仅因平台收尾连通性：启智API连续read ECONNRESET，cached CPU connection test也TunnelError。最后一次CPU r2成功查询是RUNNING，当前状态未知，未确认stop；不能报为当前正常运行或已关闭。已向用户异步请求恢复可访问qz.sii.edu.cn的网络或提供可用接入方式，等待回复。未修改全局网络配置、未新建/重启任何GPU/CPU。此网络条件首次出现在本Goal turn，目前连续计数1；不得现在标blocked，后续三连续Goal turns确实无独立进展才按规则blocked。

本地最终工具r3已冻结80文件：acceptance_tools_r3.tar.gz SHA85a70bed94d07a49b1ea1ba751eefa9ca5fba25a75a90a16496a981ed7f3b8ea（143225B），TOOLS_MANIFEST_R3 SHAd73bf4c5507ea2aeb9ee13ea4cbcc8d7bd9208c314f3a8786ae4f015ebb5f181。含最终report_d、review_report、pack_science_delivery、goal_completion_audit、GPU/传输工具；旧r1/r2保留，不再修改这80份源/收据，否则须新修订包。final_report_r2/VISUAL_REVIEW_ACCEPTED.json实际通过四PNG人工查看与最终字节一致、13生成输出SHA及数值表绑定，报告SHA65510226fb3e362dc9c8b048e7369b86d991e23cff55738848fc1d22f6fc0353。

最终科学补充包已本地封存并逐成员核验：delivery/science_delivery_r1.tar.gz（1229351B），SHAe5bc4b586f098b13a603e1d4b389fef04a994009608a76b2d6277c30a1a2f401，manifest SHA398d4d4dad40da47c40414160b2e5c7b652497c81c961e842815287c8d5dd1f4，64文件。SCIENCE_DELIVERY_RECEIPT_R1和LOCAL_VERIFIED已通过。此包尚未远端上传；模型完整主归档原本已在远端保存，本地也已SHA/新解压通过，不会因当前联网失败丢科学结果。CURRENT_SCIENTIFIC_BINDINGS_CHECK实际重新核268个科学输入SHA全部一致（非重算统计）。RESEARCH_HANDOFF_PENDING_PLATFORM.md给出结论/证据/缺项和继续指令；PLATFORM_CLOSURE_PENDING.json记录当前外部条件。禁止重新训练、推理、统计或重建已有包。

恢复后唯一剩余动作：查现有CPU r2；上传上述1.23MB补充包+manifest+receipt到ROOT/delivery，remote sha256sum核本地相等并保存delivery/SCIENCE_DELIVERY_REMOTE_VERIFIED.json（status passed_remote_sha_matches_local、archive_sha256同上）；stop r2并核两CPU STOPPED，原CLI success/data结构存closure_cpu_r1/各name_status.json，保存生命周期/停止证据；确认本轮local work无活跃进程，LOCAL_WORK_PROCESSES_CLOSED.json matching_owned_process_count=0；运行goal_completion_audit.py。它会逐8项核对并重验科学输入/包/源码/资源，只有通过才创建FINAL_DELIVERY_CLOSURE.json与FINAL_GOAL_ACCEPTANCE.md；然后update_goal complete。现阶段不应有这两个最终通过文件。需要改最终审计代码时保留r3并新修订，不能伪造状态跳过CPU或远端备份门禁。

17:50左右增量（最新）：最终科学主分析final_analysis/ANALYSIS.json实际passed：10验证精确SHA复用、两测试新预测独立复算通过，全20990步/24255R/200T和共同网格R23选择审核、三项2000用户聚类配对bootstrap完成。主测试A_R35 NDCG5 .008956341761612958、D_R35 .005528385500732038，差-.00342795626088093（-38.274%），95%区间[-.0038839484876958276,-.002975842599966817]；best A_R32 .009430792222931613 vs D_R23 .008074436410928235，差-.0013563558120033908，区间[-.001854786115176182,-.0008936153203737087]。负结果不追加实验。

final_temporal/TEMPORAL.json实际passed，既定38644/33013切片均D低于A，early delta-.00485674136、late-.00343647770，区间均低0；late-early绝对负差缩小+.00142026366，联合用户CI[.00066383221,.00224456497]，不能误写时间泛化改善（相对差early-38.98%、late-42.48%）。five训练分组和描述性分解闭合、H0历史参照均通过。final_process/PROCESS.json实际passed：31诊断/11快照/20新子集+10提取+1旧R5；全catalog映射/slotusage/位移/Mixer独立校验通过。末块20T改变60.84%商品SID，最后三R未恢复T前诊断；不同SID9513→9643、碰撞14.03%→12.41%、Mixer范数比末1.3337。后期R适应不足只是猜想，不能认定one-epoch或过拟合。

报告初版final_report已生成且四PNG实际查看无截断。随后仅修改report_d.py补全聚合@10/20/50/覆盖/slotusage/best表及事实-猜想、时间负差表述，未重算任何统计或推理；新权威报告在final_report_r2/REPORT.md，源REPORT_BINDING已经生成。其四PNG与实际查看的初版四图逐SHA完全相同，仍需保存正式VISUAL_REVIEW_ACCEPTED与最终交付清单。旧final_report及旧r2工具包保留，不能覆盖。

freeze_acceptance_tools.py刚新增--include以容纳后补GPU/传输/最终检查工具；尚未冻结r3，最终必须生成新r3并含实际report_d.py、audit_gpu_closure.py、transfer_archive_chunks.py及本轮最终验收源码，不能用r2当最终。科学/冷恢复/运维/资源GPU部分均已通过，尚缺：视觉/最终交付SHA与科学补充包备份、必要最后目标逐条验收、临时CPU r2 stop+核验及本轮跟踪设施退出。当前所有native科学进程已exit0（13864/48607/18064），无训练/推理需继续。Goal active。

17:43增量（最新）：COLD_RESTORE.json真实执行passed，7个关键状态（initial、last、首T、首R、跨半块T40、best R23、最终R35）实际CPU模型/Tokenizer/双优化器/scheduler/计数/progress恢复相等，paired T RNG CPU字段也相等。全41源checkpoint实际指纹与收据/计数/progress匹配，所有相同T计数的R间隔alignment/T optimizer/paired T RNG保持不变。恢复从新解压封存代码与engine实际导入，source未改；CUDA RNG原字节保留但CPU不应用，CUDA下一步恢复证据仍以前期短测为准。

ASYNC_DELIVERY_AUDIT.json真实执行passed_async_operational_delivery_audit：17文件运维包/门禁/成功指针及train/test退出0，11个恢复后只读任务正确cp+完整state前后相等、源预测提取一致、首30恢复前缀收据均通过。此审计不代替统计/完整更新/资源审计。

现已启动analyze_d.py，run=fresh_r1/runs/H1-Spread200-R35、output=本目录final_analysis；查本轮native session直到退出，不重复启动或删除已产生输出。它审核20990更新/40新阶段/T样本随机日程/全部验证与测试/共同网格及2000用户配对bootstrap；10份已接受验证只按SHA复用，两份测试实际独立复算。之后temporal_d、process_d、report_d尚未执行，四图人工核验/最终工具交付SHA/临时CPU r2关闭/逐条Goal验收仍待，Goal active。

17:41:30增量（覆盖下方extract运行中描述）：native session77297已exit0，独立新目录实际解压通过3464文件逐大小与SHA。fresh根LOCAL_EXTRACTION_VERIFIED.json status=passed、整包/manifest源SHA匹配。C盘余12303712256B。现在开始真实cold_restore.py，从该新目录封存代码构建CPU模型并恢复关键checkpoint/双优化器/scheduler，再审核41份边界；先查本轮工具返回native handle，不重复运行。后续完整科学/运维/报告/CPU收尾仍待，Goal active。

后续最高优先级：55块分块传输全部完成且真实逐块校验。delivery/chunks_r1/ASSEMBLED_ACCEPTED.json已实际通过，组装后本地delivery/stage09_h1_spread200_r1.tar.gz大小3643737772B、SHAca7a72833c2eeadfdc51ccfe0e8472cacd2ca7ed7d94e5d01e80d4fa5ecb485c与远端封存一致。工具SHAcdef9f106ad128b0aba9291056b3885826d4c85c2b38f379ecf8ae9139fe1b7d；旧不完整513MB文件保留.partial_r1，7个64MiB完整块经SHA验证复用，尾部未用。

已启动真实archive_delivery.py extract到全新outputs/stage09_spread200_20261031_fresh_r1；native session77297在运行，必须先poll同一handle或检查LOCAL_EXTRACTION_VERIFIED.json和进程终态，不可重复extract/删除部分解压目录。目标3464文件逐大小/SHA，整包SHA已核；尚未通过真正全目录验收时不得启动后续最终冷恢复/分析。解压后按RECOVERY.md依次cold_restore、audit_async_delivery、analyze_d、temporal_d、process_d、report_d与四图实际视觉检查；这些最终实际执行均尚未开始。所有GPU已释放且预算审计通过，CPU r2因交付仍RUNNING，最后还需停并核验。Goal active。

17:38左右增量（最高优先级）：两份新测试全部完成，各53194条。唯一续跑作业SUCCEEDED，17:04:22 Pod删除、17:04:28终态控制器收尾；FINAL_ASYNC_STATUS/INSTANCES/EVENTS及closure_platform_r1全部五作业status/instances/events为权威。FINAL_GPU_RESOURCE_AUDIT.json由audit_gpu_closure.py实际执行通过：所有GPU终态且释放、实际分配区间无重叠；创建至Pod删除加每作业60秒余量累计51751秒=14.37527778GPUh，短测834秒<1800；实际调度至Pod删除合计14.24194444h，仅费用代理不是结算，按两组live标价1点/实例h估算上界14.3753点，actual_settlement_verified=false。初始短测9:56的重复控制器清理不代表仍占GPU，1:41:50早已删除Pod。19文件gpu_closure_evidence_r1.tar.gz SHA56d405a4fc3c4e1d2320256404a4edb1eacf9e2deb46cf3c10ef33db3082e502已上传ROOT，FINAL_GPU_RESOURCE_AUDIT也已上传。原GPU_LEDGER是运行期旧快照，以FINAL_GPU_RESOURCE_AUDIT覆盖。

科学成功指针ROOT/FORMAL_COMPLETION.json：formal_async_r2 result SHAde3fa195bc41fdc1821056ba1f6a9198a4752caf1c60c9b53011a20f7cf58089、resume_gate SHA140bf9d7051140998e18e5770699e6217fe1509473f8da3f00bc6b7e47d5d2b7。result两stage退出0、no retry/no stop。原生产测试NDCG5：固定R35 .005528385496907751、best R23 .00807443640048501，尚未独立复算，不能作为最终科学验收。test_predictions_complete记录仅epochs35/23；固定35 pred SHA d363d56787dd1cdba1ce558af69ec51a46dbc17ad4f94e76e303d91e4dac4691、best23 pred74ba7b98d274352b7d2d0e4c965cc1b3d30fcb71cc95c9f8362cdb33fbfed7fe。测试只需最终analyze_d统一复算，不再新增推理。

远端封存已完成：ROOT/delivery/ARCHIVE_RECEIPT.json status sealed，3464文件/source_bytes4123253353；stage09_h1_spread200_r1.tar.gz 3643737772B，SHAca7a72833c2eeadfdc51ccfe0e8472cacd2ca7ed7d94e5d01e80d4fa5ecb485c，manifest SHA4e95e908586093433dc09ac0c7cd4b13e84ecd81ddb1e3aade2db37e0f216e32。本地delivery已下载receipt+manifest，空间验收PRE_DOWNLOAD_SPACE_CHECK通过。seal客户端120秒超时但原PID1604662真实继续并完成，不能重复seal或覆盖包。新归档包括r2工具包和GPU证据；传输helper位于远端delivery内，不修改封存科学内容。

首次整包SCP发生CPU入口运行周期中断：lifecycle旧周期01:25:40—17:21:20，新周期17:22:40，非我们主动start；内存采样接近100%，事件未直接确认OOM，只能说资源压力相关猜想。已停止明确身份的本地旧下载进程树root7212/scp65160，非实验/GPU。原前缀保留delivery/stage09_h1_spread200_r1.tar.gz.partial_r1（513146880B，SHAbd169bec84bc5b80ffbfa98c540832e63e1a96aa07785f4eb947ecf9a0d3f0ba），前7个64MiB块469762048B已按源SHA复用，剩余尾部未用。CPU_TRANSFER_INTERRUPTION_LIFECYCLE/METRICS在本地delivery保存。CPU r2已恢复RUNNING，仍需最终stop核验，原r1已停。

改用transfer_archive_chunks.py（新增离线传输工具，未入旧r2工具包，最终需新工具清单）。远端ROOT/delivery/transfer_chunks_r1已有55块与MANIFEST.json，分块时重新核整包SHA相同；manifest SHA677dd41dce901238675b1466134526abec81fc8ad22b8b4b766bfbcff693ead7。已用同一CPU入口同时最多2块SCP完成全部55块并逐块实际SHA验证，内存保持约0.35/4GB；本地delivery/chunks_r1/{PREFIX_REUSE_ACCEPTED,DOWNLOAD_SHA_RECORDS}.json证明。当前已启动assemble重新组装原文件名并核整包SHA，先查本轮工具返回的native session是否已完成；不可重复启动或覆盖现有archive。全部块均已验证，组装成功以chunks_r1/ASSEMBLED_ACCEPTED.json为准。

下一步：组装成功后按RECOVERY.md从本地delivery原文件名extract到全新outputs/stage09_spread200_20261031_fresh_r1（之前未建），逐文件SHA；再cold_restore、audit_async_delivery、analyze_d、temporal_d、process_d、report_d。它们尚未运行最终数据；十份验证统计按现有ACCEPTED SHA复用，两测试独立复算、全20990步/41cp/31诊断/时间/2000bootstrap全要求保留。实际报告四图需查看，工具若改保留r2另建r3；新audit_gpu_closure与transfer_archive_chunks等最终工具也要纳入。最终CPU关闭/本轮跟踪设施退出/交付完整SHA和逐项目标审计全待，Goal active。

16:43增量（最新）：固定R35测试已完成53194条，R23/T120测试已实际处理9024/53194条。平台最新查询RUNNING，进程stage=test/heartbeat2.88秒/stop_reason=null；同一唯一D、单H100，最近GPU32%，low窗口约2618秒（43.6min），guard_action=null。累计GPU创建至观察保守上界13.95818h，非实际结算。R35/T200训练计数不增加，允许测试仍仅[23,35]，不据测试追加实验。

本地test_R35_completion_r1/{complete.json,summary.json,METADATA_CHECKED.json}仅接受完整53194/FP32/beam32/H/seed2024/24255R/200T及与预选R35验证相同的model/execution绑定，summary SHA5d20c1032d98b2ea41b0e6ebde2a324099e7a5e24fb03a1756f97ebc39e046c4；声明pred SHA d363d56787dd1cdba1ce558af69ec51a46dbc17ad4f94e76e303d91e4dac4691尚未本地独立复算。两份测试预测统一留最终analyze_d从新解压全包独立审计，不重复提前统计。22次等待快照保存BOUNDED_OBSERVATIONS_FIXED_R35_TEST_COMPLETION.json。下一步继续只观察R23测试；成功后必须查平台终态/实例释放/成功指针，再按RECOVERY.md的archive_delivery_async_r2.py封存并下载新解压。

新增freeze_acceptance_tools.py --revision r2已实际生成69文件离线工具/静态协议/十份已接受验证receipt+audit+record+summary的版本包，旧r1保留。TOOLS_MANIFEST_R2 SHAeab16fffa01d87dd2b291b5b94bcc6fbdb444420660fbcd1094090d7f2882bac，acceptance_tools_r2.tar.gz SHA512af99b5337baf53d6f0210f21a6c64953d905614a27ba22958974fc096a04f（117484B），TOOLS_BUNDLE_R2 SHA77a34e6a806a0c4c0dadb557babd00faf3ee2b4f59d0295393de2b60f279d16c。三份已上传同root并真实sha256sum与本地核一致，见TOOLS_R2_REMOTE_SHA_CHECKED.json。包状态prepared_source_bundle_not_final_acceptance，真实最终审计未运行；若后续修工具，保留r2另建r3，最终清单必须绑定实际运行源码。运行中的模型/科学code/runtime/launcher未改。

已保存GPU_RATE_BASIS_20261031_1620.json：live job quota单H100(1,20,200)标价1点券/实例h，仅费用估算依据，不是实际结算。C盘最近可用20811313152B；归档前仍需核最终archive/source大小与本地余量。所有最终科学分析/新解压全SHA/实际冷恢复/报告视觉检查/全部GPU与临时CPU收尾仍未完成，Goal active。

16:17增量（最新）：十份新完整验证全部独立通过，已锁共同网格best=R23/T120，固定R35/T200测试正在运行（6016/53194条）。唯一GPU作业仍需单独实时查询确认，最新进程heartbeat2.79秒、stage=test、stop_reason=null，11/11恢复后只读任务有完成记录。最近GPU14%、连续低窗口约1053秒，尚无guard_action；仅必要串行测试，不加无关负载。累计创建至观察GPU保守上界13.51802h，非账单。

R35实际CPU独立复算：71657条/4733用户、0不匹配、maxerr6.653994599759017e-8；独立NDCG5 .006243133522239885，生产 .006243133519718186，HR5 .029082992589698146、Recall5 .005967316611099249，eval2837.3318秒。saved_validation_R35包含record/summary/pred、四份只读收据、SOURCE_CHECK/audit/ACCEPTED/READONLY_BINDING_ACCEPTED及selection/training_complete/SELECTION_ACCEPTED。audit SHA c8e0af8c48b35fa6532aba4ceb479b08ff901e542c30a8afed27c358d364765c；summary1a5a4a1f782c4a86425ebc0349f0ee22424a767829e664dba4b6744084e486fc，pred f553f77b3887bf3429cf5552a6931646c1848420d3571dcb4bcdc842dd7ba57d。checkpoint SHA与已备份最终d84d0b7bf3533dcfaefedb5b23564c99752b33f44f44253584c76d24f7d37050一致，完整状态38118afe9cc52438f88f2026e2e709097b03fd149fd0939368be6d8051d02e5a前后不变；8192诊断由同份完整验证提取。三份验收收据已上传root/recovery_async_r2/R35_*。

新增check_locked_selection.py已实际核验十份已接受验证的源SHA及六候选顺序/生产score/平局较早规则，选择R23、允许新测试epochs只[23,35]。SELECTION_ACCEPTED.json已上传root/recovery_async_r2；selection SHA01a7de08c5c9735c550eae81e345addd80f6432d4b11346c6df6a575c46451ec。独立FP64复算不能替代生产score重选。13次等待快照保存BOUNDED_OBSERVATIONS_R35_VALIDATION_COMPLETION.json。没有最终测试结果，不能据验证结论代替固定R35主比较。

当前单次只读观察入口已新增observe_async_once_v2.py（旧v1保留），同root共享Python -B root/observe_async_once_v2.py --root root；除原状态外读取selected_epoch、selection_candidate_epochs、tests和test_progress。仅运行一次就退出，无daemon，不改code/runtime/launcher或17文件运维包。functions store observeAsync已切v2、observeAsyncV1保留旧命令。后续验证缓存十份均复用，禁止重算。下一步只观察既定R35测试完成及随后R23测试；全部测试成功终态后按RECOVERY.md做归档/新目录SHA/真实冷恢复/运维及科学最终审计/报告/全部GPU和CPU收尾。最终工具清单包含最新版process_d/cold_restore及新增check_readonly_validation_binding/check_locked_selection/observerv2。Goal active。

16:01增量（最新）：R32完整验证已实际CPU独立验收通过，71657条/4733用户、0不匹配、maxerr4.876876169568334e-8；独立NDCG5 .008441040868489801，生产选模分数 .008441040864534052，HR5 .03973094045243312、Recall5 .008172265218414423。saved_validation_R32包含7份源文件/元数据、SOURCE_CHECK、audit、ACCEPTED及READONLY_BINDING_ACCEPTED。audit SHA ae384c85991704156a714ceedec506034a2e45e84abadf235523fc34b528f4d9；summary e636e8fb72ff9af0da2c449b5b909323c8d6ca0d9e987ba60591b75ede3b2c73；pred a6ee946962134df342677c8f6acd0e290006583921b87af617da40bf4ee1c3c3。真实远端phase40_R_R32_T180.pt SHA5922b2421179dee5746e69905dd0d7102e27f006de120aed1a92d4049a65ca18匹配，完整状态fingerprint e09ebc77ca507114ae7d13c158e84869c0115f2ad9d0f0568755642fe59735aa前后与收据一致，8192子集确实提取同份完整验证。三份指标/只读验收已上传root/recovery_async_r2/R32_*。

现在R8至R32九份新完整验证全部独立通过，最终analyze_d按SHA复用，不重复复算。R32低于R23，不提前锁best，仍等R35完整验证。23次有界等待快照保存BOUNDED_OBSERVATIONS_R32_VALIDATION_COMPLETION.json。16:00:59实际R35验证43008/71657，恢复后10/11只读任务有完成记录；平台查询仍RUNNING，heartbeat4.18秒、stop_reason=null，R35/T200训练终点不变。仅剩一个验证worker时最近GPU28%、低窗口约60秒，尚无guard_action；不能当作3h回收，也不靠额外无关计算提高负载。创建至观察累计保守GPU上界13.24944h，非账单。

下一步等待同一作业的R35完整验证，完成后按saved_validation_R32流程独立验收（phase44_R_R35_T200.pt，SHA已知d84d...）。既有launcher随后锁共同网格并自动执行固定35和best最多两份测试；不可提前测试或改参数。全部最终新解压/冷恢复/科学分析/报告/资源关闭要求仍待完成，Goal active。运行源码与运维包未改。

15:38增量（最新，已验证等待）：nancy-s09-d-async-1031-r2平台仍RUNNING，单H100节点qb-prod-gpu461。8次有界单次观察保存BOUNDED_OBSERVATIONS_LAST_TWO_VALIDATIONS_1537.json；末次15:37:41 heartbeat0.01秒，R32完整验证45024/71657、R35完整验证13024/71657，9/11恢复后只读任务完成，stop_reason=null、低利用率窗口清空、最近GPU77%。R35/T200训练终点不变、暂无新的完整验证分数/selection/test。创建至观察保守累计12.86122GPUh，非实际结算。仍只等待这两份新完整验证，完成后下载并独立验收，不重算已接受八份验证；之后已绑定launcher自行锁共同网格best与最多两份新测试。

远端本轮root当前du -sb为4026038236字节，尚非最终封存大小（后续验证/测试预测仍写入）；之前本地C可用21.52GB，正式包下载前须再核最终archive_bytes+source_bytes+余量。不得清理旧实验。现有阶段checkpoint和旧证据保留。最终新解压/全SHA/真实冷恢复、完整科学分析/报告/全部资源关闭仍待完成，Goal active。此轮无新模型计算、无重提、无旧监控/飞书写入。

15:29增量（最新）：R35完整验证已实际开始2016/71657，R32验证35008/71657；平台RUNNING、heartbeat3.01秒、9/11只读任务完成。OBSERVATION_R35_VALIDATION_STARTED.json为单次快照。训练终点R35/T200不变，选模/测试尚未开始。最后GPU样本38%，low_since刚起、guard_action=null，不将瞬时样本视为持续3h低利用率或因此重提。继续观察这两份新验证；完成后沿用saved_validation_R29流程验收。

补充离线工具状态：process_d的catalog完整状态绑定已按冻结源码的真实顺序处理（catalog在complete_phase之前、phase checkpoint在其后），显式仅重建T边界4个progress字段后比完整指纹；初始catalog无变换，其余模型/优化器/RNG等全部保持原checkpoint字节对应值。直接比较两个cursor的完整fingerprint会误报，因此不能用早期15:27工具副本。当前最新版语法通过但实际最终验收未运行；GPU源码/独立runtime/launcher未改。最终工具清单须包含此版，全部科学/备份恢复/报告/资源验收仍待完成。

15:27增量（最新）：R33诊断8192/8192完成，9/11恢复后只读任务已发布完成记录；R32验证32000/71657，R35验证待worker。新鲜heartbeat1.45秒、stop_reason=null，R35/T200训练终点不变。3次有界观察保存BOUNDED_OBSERVATIONS_R33_DIAGNOSTIC.json。下一步只观察既有R32/R35完整验证，完成后同saved_validation_R29流程做新预测独立验收；最终选择/最多两份测试仍由封存launcher自动执行，不能提前选模。

离线工具刚补覆盖缺项，尚未运行最终数据：process_d将保存行汇总收据的2e-12严格绑定与独立FP64对FP32的原core2e-6容差分开；另外独立核验catalog正确checkpoint、norm/Mixer修正/碰撞/桶/槽entropy与usage/逐槽SID变化/相对位移。cold_restore逐41份真实源state指纹/双计数/progress与receipt绑定；关键恢复核paired T RNG的CPU字段与显式排除CUDA部分。语法通过不等于验收通过。没有改运行中的code/runtime/launcher。最终验收工具包须包含最新版process_d/cold_restore及新增check_readonly_validation_binding，旧TOOLS_MANIFEST_R1不可代替。科学/备份恢复/报告/资源收尾全要求仍保留，Goal active。

15:20增量（覆盖下方旧状态）：R29完整验证实际CPU独立复算通过，71657条/4733用户、0不匹配、maxerr4.809881e-8，独立NDCG5 .007069971977536885；生产选模分数 .007069971973108773。saved_validation_R29包含record/summary/预测、SOURCE_CHECK、audit、ACCEPTED及四份只读元数据与READONLY_BINDING_ACCEPTED。audit SHA bc4fff2ec8190151bb7c19f62582dd847c0f0d55829827ebda9fa5233cd0af1a，summary51b804a76f6c7da496cbf58e713f069ce4ae77086d9056de5acb66ad87ed161b，pred f9307f0210eecd51a28ae2978bf7f7b0c2a74c31ad0fa0d89822969426d1a11c。远端实际phase36 checkpoint SHA9c6d54db18ee5a4975e5d27af39f2336ad45cd2f9e07b1e6970a2935de4dc606匹配；完整状态前后与checkpoint收据指纹d37f8e69d3e4e6a95b712b960501d4a2ba59431be8223f0c5cf96f2dc064b5e8一致，8192诊断提取同份验证。指标/只读收据三份已上传root/recovery_async_r2/R29_*。新增本地通用check_readonly_validation_binding.py，未改任何运行中的GPU源码。R8至R29共八份完整验证独立通过，最终analyze_d按SHA复用，不重算。R29低于R23，不提前锁best。

15:17平台RUNNING、新鲜心跳2.26秒，真实R35/T200端点不变；R32验证20000/71657，T200诊断5120/8192，R33诊断和R35验证待队列。7/11恢复后只读任务完成记录。最近GPU79%，无保护停止；创建至观察保守GPU累计12.52618h，非账单。C盘当前约21.52GB可用，正式包下载前仍需对照实际压缩/解压大小再核空间。下一步继续既有只读队列，R32/R35完成后同样独立验收；全部验证齐全后由已绑定launcher锁选模并进行最多两份新测试。科学终态、全包新目录SHA/真实冷恢复、最终分析/报告及资源收尾均未完成，Goal active。

15:04增量（最新）：唯一D训练更新已结束R35/T200，累计R24255；新增R20790/T200=20990条，41份阶段checkpoint收据已存在。最终phase44与last.pt实际SHA均d84d0b7bf3533dcfaefedb5b23564c99752b33f44f44253584c76d24f7d37050，最终progress phase45/update0，状态指纹38118afe9cc52438f88f2026e2e709097b03fd149fd0939368be6d8051d02e5a。最终checkpoint/收据/完整日志已先行备份R35_training_endpoint_r1，BACKUP_ACCEPTED.json实际哈希/计数/端点元数据通过，日志SHA25506f3cfa4d752738c29ba0132ddd785ffaf5c18521dec17d8cf5dbd1ca35c6，原13014行前缀字节完整保留。此备份不代替最终全包新目录逐文件SHA及真实模型/双优化器/scheduler冷恢复；收据上传root/recovery_async_r2/R35_ENDPOINT_LOCAL_BACKUP_ACCEPTED.json。
现在训练进程stage仍为train、持锁等只读任务是预期，R/T日志不再增长不是故障，不能重启或追加训练。当前平台RUNNING、心跳新鲜；R29完整验证60000/71657、R32验证4000/71657，恢复后6/11只读任务完成记录已出现，T200/R33诊断及R35验证仍排队。所有验证/诊断齐全后才锁selection并自动执行固定35与best最多两次测试，当前没有最终测试结果。创建至观察GPU保守上界约12.30125h（非账单）。17个等待快照保存BOUNDED_OBSERVATIONS_R35_TRAINING_ENDPOINT.json。下一步只观察既有验证队列，R29完成后按saved_validation_R26同流程做独立CPU验收；不得重算已验七份验证或误将训练终点标成Goal完成。

14:42增量（最新）：R26完整验证已独立CPU验收通过，71657条/4733用户，0不匹配、max逐条误差4.809881e-8，独立NDCG5=.008352733347021675；生产选模分数保持.008352733337596533。saved_validation_R26/{ACCEPTED.json,audit.json,SOURCE_CHECK.json,record.json,summary.json,per_sample.jsonl.gz,READONLY_BINDING_ACCEPTED.json及checkpoint/readonly/evaluation/diagnostic收据}均保存；audit SHA0db4b93fb3b8d3341ed5c02d55aeb7fc3ef65ce8a55969a25ecaafdaad05a0b8，summary48fd000999487b52f46ce13c7af4854173e9229f75d48952445466b2e6c702a5，pred78d770f51d1680d8c324caae6d931e8b174a1d12921898050a4de01260780e58。正确R26/T140 checkpoint真实SHA匹配，完整状态前后不变且与阶段receipt一致；8192诊断确实提取同份完整验证。指标/只读验收已上传root/recovery_async_r2/R26_METRIC_ACCEPTED.json、R26_INDEPENDENT_AUDIT.json、R26_READONLY_ACCEPTED.json。最终analyze_d按SHA复用七份已验验证（不重算原六份）。R26低于R23，不提前锁best，R29/32/35与R35主测试仍待完成。
最新真实R34/update353（R23222/T200），R29验证42016/71657，T180诊断3072/8192，其他既定任务排队；暂无其他新完整分数。创建至观察GPU保守累计约11.94038h（非账单）。R26完成等待7快照保存BOUNDED_OBSERVATIONS_R26_COMPLETION.json。下一步继续同一作业，待R35训练完成、R29等新完整验证依次成功后按相同流程验收；测试仅由已绑定launcher在全部训练/验证选模锁定后执行。全部最终恢复/交付/资源要求未完成。

14:29增量（最新）：最后20T已完成，真实R33/update294（R22470/T200）；R26完整验证65024/71657、R29验证31008/71657，R32验证已排队，均无完整新分数。ALL_T200_ACCEPTED.json已实际CPU核验200次真实T更新的全部batch/10块/计数/原400T噪声公式与温度/1e-5与1e-4 T LR/每次T原35R余弦R LR；204800次出现的有序流与原前200T严格相同，stream SHA52471ad1b359f8c7463f2c8bd2d9892038ef0e53f6cf8ec4e3151892b932c9d0。快照ALL_T200_SNAPSHOT.json SHA9c301e7459c4c9409dba5f860117b49cadeb73990de1e6726a8932b43b5eb857，审计audit_all_T200.py，收据已上传root/recovery_async_r2/ALL_T200_ACCEPTED.json。T200验收不等于R35或Goal完成。
最终离线cold_restore.py新增41个源checkpoint的paired_t_rng指纹检查，要求每个T计数不变的R间隔中alignment/T optimizer/paired_t_rng均不变；仅语法检查，实际最终冷恢复未做。没有改GPU运行源码/runtime/launcher。12个有界等待快照保存在BOUNDED_OBSERVATIONS_20261031_T200.json。下一步继续看唯一作业及R26完整验证，成功后下载新R26预测/summary/record，先source_check_validation.py --epoch 26 --directory experiments/09_kuairec_listwise/h1_spread200_20261031/saved_validation_R26，再audit_completed_validation.py --epoch 26做CPU指标独立复算；不重复已验六份旧D验证，不提前测试。

14:12增量（最新）：R31已结束，真实进入R32/update38（R21521/T180）；R26完整验证52000/71657，R29验证19008/71657，仍无完整新分数。3个恢复后只读诊断完成，T180/R30诊断等worker，科学节点全保留。平台RUNNING、心跳新鲜、最近GPU99%、无保护停止。有界只读等待9快照保存BOUNDED_OBSERVATIONS_20261031_1412.json；个别query延迟只等原handle，没有重启/重复提交。下一步继续观察同一作业：R32完成保存T180状态用于验证，随后最后20T到200，再R33—35；完整新验证成功后CPU独立审计，不提前测试。最终所有科学/交付/恢复/资源要求仍待完成。

14:00增量（最新）：R30已结束，真实进入R31/update30（R20820/T180）；R26完整验证43008/71657，R29验证9024/71657，尚无完整新分数。新增phase38/R30诊断任务与phase37/T180排队，读取不可变checkpoint，不删科学节点。3个恢复后诊断完成记录在；唯一平台作业仍RUNNING、心跳新鲜、最后GPU99%，创建至观察保守累计约11.24343h（非账单）。8轮有界只读观察已完成，快照见BOUNDED_OBSERVATIONS_20261031_1359.json，没有启动后台daemon或额外模型计算。下一步继续只观察此作业，等待R26完整验证完成后做新预测CPU独立审计；全程20990条新增更新/41个完整边界checkpoint/全部最终交付资源要求仍未完成。

13:49增量（最新）：R29已结束R20097/T160，T161—180已完成，真实进入R30/update27（R20124/T180）。R29完整验证phase36实际处理1024/71657，R26验证33024/71657，均尚无完整新分数；当前两只读worker执行R26/R29完整验证，T180诊断phase37排队，所有节点保留，读取不可变checkpoint。R29_T180_HANDOFF_OBSERVED.json记录任务绑定及400T噪声时钟10/20步观察，不代替最终全部20步配对审计/真实冷恢复。R29验证任务cp SHA9c6d54db18ee5a4975e5d27af39f2336ad45cd2f9e07b1e6970a2935de4dc606；T180任务cp SHA8a2e0274e8fed219d8989f8d13fe5b1af5bad8624ddb87afdd11fe762afd9567。平台RUNNING、新鲜心跳、最后GPU采样99%；当时创建至观察保守累计约11.04354h，账单未核。下一步仍观察此唯一作业；任何完整验证成功后下载新预测并按既有流程CPU独立审计，不提前测试。

13:37增量（最新）：R28已结束，实际进入R29/update47（R19451/T160）；R26完整验证20000/71657，仍无完整新分数。phase34/R27诊断8192全完成，done显示前后状态一致；正确checkpoint指纹及独立指标统一留最终audit_async_delivery/process_d核验，勿把完成记录等同全部科学验收。唯一作业平台RUNNING，新鲜心跳，最近GPU95%，未触发保护停止；创建至观察保守累计约10.85247GPUh，账单未核。继续等待同一作业，不追加分支、GPU或提前测试。

13:30增量：同一唯一D平台RUNNING、新鲜心跳推进到R28/update343（R19054/T160），R26完整验证14016/71657，R27诊断4096/8192；仍无新完整分数。报告模板已补真实中断/同断点恢复和只读调度说明，并新增必需--operational-audit ASYNC_DELIVERY_AUDIT.json，最终该审计11任务全部通过后才可生成；RECOVERY.md命令已同步。只做模板语法检查，未生成最终报告，没有改运行中科学代码/runtime/launcher。下一步继续等新完整验证与诊断完成，不启动其他任务。

13:25增量（当前最新）：R27已结束，真实进入R28/update37（R18748/T160），phase34诊断任务已产生。R26完整验证10016/71657，尚无完整分数。phase33/T160诊断8192全部完成，T160_readonly_acceptance_r1/ACCEPTED.json已实际核对正确checkpoint完整状态恢复与前后不变，指纹04e887be80ce7d5a572636ee6a7974a37ce266997303abc197ea41c1f113cd37；实际预测SHA7e2b78582c77966c27a9320d7a12ad047d1fe901d0c0679e885caff889ea12b4匹配，指标复算仍待最终process_d。收据已上传root/recovery_async_r2/T160_READONLY_ACCEPTED.json。当前累计创建至观察保守GPU上界约10.65244h，账单未核；平台RUNNING、心跳新鲜、最近GPU采样99%。CURRENT_ASYNC_OBSERVATION.json保存该时点观察，不是实时状态。下一步继续只观察本作业，等待R26完整验证成功后做独立CPU审计，不提前选模测试。新增root/observe_async_once.py为单次只读紧凑观察脚本，无daemon或新模型计算，执行共享Python -B root/observe_async_once.py --root root，经CPU Notebook exec调用，并另查job status；不改科学/runtime绑定。

13:14增量（覆盖下方旧进度）：R26已完整结束R18018/T140，phase32完整验证实际处理1024/71657，尚无完整验证分数；随后T141—160已执行，真实进入R27/update120（R18138/T160）。新增20T严格样本顺序/每batch SHA/原400T噪声公式与温度/学习率/计数已由audit_resumed_T160.py实际CPU验收，见RESUMED_T160_ACCEPTED.json（已上传root/recovery_async_r2/T160_ACCEPTED.json）。证据包T160_boundary_evidence.tar.gz SHA f25fa1708b56f2041327cf0b1fa67f66b58279fc008efa458bc484d640b9854e，已安全解压T160_boundary_local_r1。R26/T140 checkpoint实际SHA4ae7fa773a72aca4b2e9e61d8defb2403579cd424c8233a83993df5ee2196e0f；T160 checkpoint实际SHA db8ae2f74831545444e033c86446452ef9e6f7765ca8751308a785e597725710。第二只读任务phase33诊断实际处理1024/8192；三角色同时计算最近真实GPU采样98/99/99%，不以短窗口保证未来3小时。继续只观察同一作业：phase32完整验证成功后下载新R26预测/summary/record/checkpoint receipt，复用现有source_check_validation和audit_completed_validation流程做独立CPU审计；不提前选模/测试。phase33/后续只读任务状态完整指纹最终audit_async_delivery统一核验。

13:01增量：唯一D同断点恢复仍RUNNING且新鲜心跳推进，R25已结束R17325/T140，真实进入R26/update26（phase32、R17351/T140），阶段checkpoint收据观察见R25_BOUNDARY_OBSERVED.json。首只读任务已验收，其余任务待执行。下一个完整验证为R26，尚未产生新完整验证分数。下一步观察R26结束后的phase32验证任务及随后T160样本/400T噪声时钟延续；只读验证成功后下载新预测做独立CPU验收，不提前测试。

12:58增量：首个只读R24诊断8192条全部完成，first_async_boundary_R24/ACCEPTED.json真实核对done与任务/阶段checkpoint绑定；state_before=state_after=checkpoint_receipt.state_sha256=a71adaad5deb8d8d6478ab70785cb8fc78f83a5f3707d2f2666a6ad788a3792c。远端实际checkpoint与预测SHA也经sha256sum匹配（SOURCE_CHECK.json）；指标独立复算仍留最终process_d统一执行，不重复做。收据已上传root/recovery_async_r2/FIRST_READONLY_TASK_ACCEPTED.json。最新真实观察R25/update520、R17152/T140，平台RUNNING；首任务已通过，接下来观察R25结束、R26训练/首份恢复后完整验证及持续负载。其余10任务与最终交付尚未完成。

12:51增量：R24已完整结束，阶段checkpoint及首个只读任务元数据绑定通过（first_async_boundary_R24/OBSERVED.json，实际冷恢复仍待最终归档后做）。真实继续到R25/update128、R16760/T140；首个只读R24诊断已实际推进2048/8192，未完成。平台RUNNING及新鲜心跳再次核对。下一步观察phase30.done.json并检查其state_before/after是否等于checkpoint_receipt.state_sha256=a71adaad5deb8d8d6478ab70785cb8fc78f83a5f3707d2f2666a6ad788a3792c；勿提前声明完成。新增最终audit_async_delivery.py（语法已检，尚未实际最终验收），RECOVERY.md已补新归档入口及操作审计命令。

## 最新覆盖状态：2026-10-31 12:46，正式同断点续跑已开始

本段覆盖下方12:35和11:40状态。唯一正式恢复作业nancy-s09-d-async-1031-r2，平台RUNNING，12:43:17调度到单H100节点，启动新鲜心跳已核对。科学源码未变，独立运维包ASYNC_DEPLOYMENT.jsonSHA b9f829d07aded961e8508c129fb5af318e71c9bb78a2fc0c5d5ab626ac8b8b09；runtime async_resume.py SHA7c35eaebb242fe06291707b4c71a038bc79b4c453193dd526ddbb1857bf2d0f6；launcher job_async_r2.py SHA4a728d4574d94ab9fba2b60d7ae9c7e56c93bbc14897a6b042308f9780bf9917；gate SHA4a1520ca21f76bcbfa0ac1e6926abef9ec9681a1d2f59619b543dfa9644fb3f7。旧12:35候选runtime SHA已过时。

- CPU_ASYNC_QUEUE.json：五个真实CPU子进程、同时最多两个；已完任务不重复启动，伪造只读状态收据拒绝，失败停止且不重试，停止信号阻止新worker。CPU_ASYNC_GUARD.json：150min低利用率保存退出、40%重置窗口、300s利用率不可观测停止、累计24h及截止保存预留检查通过。CPU_ASYNC_STRUCTURE.json仍仅结构检查，不能替代GPU实测；原并行GPU等价短测已真实通过。
- 原checkpoint/log哈希再次核对；旧正式作业实时CANCELLED、所有短测已释放后，旧.running移至recovery_async_r2/ORIGINAL_RUNNING_LOCK.json，未删除。收据STALE_LOCK_ARCHIVED.json本地SHA可复核。未生成管理员批准，也未假称已申请成功；该恢复依据是真实所需工作并行的资源适配修复。
- ASYNC_FIRST30_ACCEPTED.json已实际CPU验收：原13014行SHA348e7a914323ae4fa7e587d0578406c35357ab67c3c005e212a7c7eb65186b1e完整保留；新R24/update401—430样本SHA、时钟及逐loss与隔离CUDA oracle一致；已真实达到R16369/T140。快照async_first30_training.jsonl SHAcd6ceeebd56074ee7bdee3d6550acddcea850238334c0f60319d56384f882eed。实际训练继续推进，此处是已验的固定前缀，不是实时最新计数。
- 新尝试日志/心跳/利用率/最终result路径root/jobs/formal_async_r2；任务请求/只读完成收据/子日志在其tasks/。旧root/jobs/formal不覆盖。只读worker成功后写原科学输出diagnostics/和validation_records/。新作业最长13.8h，内部49620秒、提前120秒保存、停止后60秒硬终止；持续GPU<40%达150min保存退出，未观测GPU利用率达300秒也保存退出；无自动重试。每30秒真实nvidia-smi采样，心跳每5秒。监控仍需平台实时状态与心跳新鲜度交叉判断。
- 预算已耗（此新作业前）保守35813秒=9.9480556GPUh；新作业最大49680秒加180秒释放总预留85673秒<86400。剩余任务保守串行总和加25%估43715.22秒，低于新上限。当前作业正在消耗剩余预算，不能继续声称余14.05h不变；GPU_LEDGER含本次新作业，最终需以终态分配/释放证据结算。
- 成功后新launcher写root/FORMAL_COMPLETION.json，attempt=formal_async_r2；最终封存调用root/archive_delivery_async_r2.py（已支持该独立尝试），不能误用只支持旧路径的root/archive_delivery.py。下载包后的本地archive_delivery.py也支持新指针。
- 下一步只观察此作业，核对R24结束后首个只读诊断、R26完整验证以及持续真实利用率，完成新验证独立审计；不得启动第二GPU或重放旧模型。最终四全验证/最多两新测试/过程分析/全日志与41状态审计/新目录冷恢复/报告/全部资源收尾仍未完成，Goal保持active。

## 最新覆盖状态：2026-10-31 12:35

以下旧段落中的“没有新GPU提交”“只能等待管理员”“已耗9.803333h”均已过时。正式D仍停在原R24/update400、R16339/T140，未重启。已在Goal授权的短测预算内完成两个隔离恢复检查，未改冻结科学源码、正式checkpoint或日志。

- 恢复性能短测nancy-s09-d-prof-1031-r1已SUCCEEDED并释放；下一R两次CUDA重放及profile状态精确一致，未证明换旧运行组本身能改善负载。
- 同卡并行短测nancy-s09-d-concur-1031-r1已SUCCEEDED，12:34:13删除pod，最后终态事件12:34:22；单H100，正式更新0。30次R更新逐loss及完整状态与串行一致；256条FP32完整预测行一致。三角色同时计算40个采样，平均84.55%、39个>=40%，最大采样显存14501MiB。短时结果不保证未来三小时持续达标。
- 本地证据包concurrency_preflight_evidence_r1.tar.gz SHA029400a10cd15460a3a6aafc1480867191bdfa6df239e5a0f65a63934f4316b8；已安全解压到concurrency_evidence_local_r1，报告SHA ca0784a1091f946901a80f9378e0d5ebd5e1a3676a31177e7ff3cac9b8c20002；实际GPU采样均值和计数本地复算通过。平台终态/实例/删除事件见CONCURRENCY_PLATFORM_STATUS/INSTANCES/RELEASE_EVENTS.json。
- GPU_LEDGER当前保守总已耗35813秒=9.9480556GPUh，余14.0519444GPUh；累计短测663秒<1800秒。计时采用创建至最后终态事件保守上界，不是账单结算。
- 已新增未部署候选async_resume.py，独立于冻结code，SHA765f37398f4928460fc5d805cafcbd441b01ac7e14b7c715981eab1c917fc7be。CPU_ASYNC_STRUCTURE.json只证明原step/prepare_boundary/complete_phase调用AST一致、45阶段/20990新增更新/10验证/20子集诊断覆盖，剩余11个只读任务含R26/29/32/35四全验证；不是运行通过。设计见ASYNC_RECOVERY_DESIGN.md。
- 下一步：完善只读任务队列和停止/心跳/预算门禁的运行检查，独立绑定新runtime及launcher；保持原实验绑定真实。新方案是同一D训练+必需只读验证并行，不是管理员豁免，无需伪造审批。旧job_resume_r2/resume_guard及RESUME_PLAN_R2是串行+管理员条件草稿，预算已过时，不可直接执行。
- 正式恢复前先确认全部短测释放，归档平台已终态原正式作业的过期.running锁，真实CPU恢复证据及原last/log SHA再核对；从原R24/update401开始，不能用短测后的状态。新尝试独立jobs/formal_async_r2，旧jobs/formal保留。当前无正式GPU运行，CPU入口r2仍需保留。
- Goal未完成：四份全验证、最终训练/测试、剩余过程诊断、独立最终分析、备份冷恢复、报告及资源收尾仍待完成。此前ETA已失效。

截至2026-10-31 11:40左右，Goal active但正式GPU已被平台自动回收，未完成最终验收。恢复checkpoint和日志已保全、真实CPU恢复通过；等待管理员认可的运行条件，不能按下方过时运行记录直接重提。用户已回复“申请”，材料见RECLAIM_EXCEPTION_REQUEST.md，尚无批准或已提交证据。本文替代此前累积的早期状态；历史保存在CONTINUE_HISTORY_20261031_0456.md及EXECUTION.md。不要修改封存训练代码或恢复旧监控。

## 当前权威状态

| 项目 | 已核事实 | 证据边界 |
|---|---|---|
| 唯一正式作业 | nancy-s09-d-1031-r1，平台CANCELLED、instance_failed；11:28:19 StopJob因管理员利用率回收，11:28:27删除pod，末次终态事件11:29:08 | 真实终态，不是观察超时；无新GPU提交，未获管理员运行条件前不能靠反复重启规避回收规则 |
| 当前训练 | 精确保存在R24第400步，R16339/T140；原始日志13014新增更新与last.pt完全对齐；真实CPU模型、双优化器及scheduler恢复通过 | cancellation_r1/RESTORE_READY.json，cp SHA bd28ba74e82a776a38e567d91c742dfccd1ca4591881151635a9da7efa5267fe；下一步R24第401步，无需重放，最终CUDA续跑仍未执行 |
| R8完整验证 | 71657条/4733用户，R5544/T20；独立逐样本/冷暖分组/全部指标复算通过，0不一致 | first_validation_R08/audit.json和ACCEPTED.json；不是终点/Goal通过 |
| R11完整验证 | R7623/T40，71657条/4733用户，全部指标/身份/标签/冷暖独立复算通过，0不一致 | saved_validation_R11/ACCEPTED.json及audit.json；独立NDCG5=.0057987429367 |
| R14完整验证 | R9702/T60，71657条/4733用户，全部指标/身份/标签/冷暖独立复算通过，0不一致 | saved_validation_R14/ACCEPTED.json及audit.json；独立NDCG5=.0082684433139 |
| R17完整验证 | R11781/T80，71657条/4733用户，全部指标/身份/标签/冷暖独立复算通过，0不一致 | saved_validation_R17/ACCEPTED.json及audit.json；独立NDCG5=.0078388068617 |
| R20完整验证 | R13860/T100，71657条/4733用户；独立全部指标/身份/标签/冷暖分组复算通过，0不一致；独立NDCG@5=.0077258419895 | saved_validation_R20/ACCEPTED.json及audit.json为真实通过证据；生产选模分数仍锁定.0077258419761，不能按独立FP64数值事后重选；首个共同候选完成，不提前定best |
| R23完整验证 | R15939/T120，71657条/4733用户；独立全部指标/身份/标签/冷暖分组复算通过，0不一致；独立NDCG5=.0087913676671 | saved_validation_R23/ACCEPTED.json及audit.json；生产选模分数.0087913676554，2570.99秒；只是第二个共同候选，不能提前定最终best |
| 已见指标 | 前五完整节点R8/11/14/17/20独立复算均通过；生产指标和独立聚合仅浮点微差 | 选模仍严格用生产summary，R20是首个共同候选；未据此改参或提前测试 |
| 启动门禁 | CPU九项、真实父状态、204800 T输入、30 R顺序；单H100短测通过 | 原R5锁定128条Top5一致、前20T数值配对、CUDA下一T/R精确恢复与隔离通过 |
| 真实正式前段 | 713次更新（20T+693R）逐batch SHA/样本量/双时钟/35R LR/T LR/400T噪声幅度温度全通过 | EARLY_LOG_ACCEPTANCE.json为prefix_only，不能代替全程审计 |
| GPU预算 | 累计24h，最多1卡；已完成尝试保守上界9.803333h，剩余14.196667h；拟续跑上限14h，另保留180秒释放 | GPU_LEDGER已按终态更新，原版存cancellation_r1/GPU_LEDGER_BEFORE.json；没有新增GPU预算，实际结算待核 |
| 吞吐估计 | 第一完整R/诊断估未加余量16.7483h，加25%约20.9354h | FORMAL_EARLY_BUDGET_CHECK.json只是早期预测，不改变预算或科学要求 |
| CPU入口 | nancy-s09-d-prep-1031-r2，1CPU/4GiB；r1已STOPPED | r2约01:25起23h自动停止，最终需主动关闭并核验；旧Notebook不动 |

## 固定科学协议与路径

- 唯一D：原H1 R5/T0 initial.pt → 10×(20T+3R) → R35/T200，最终R24255/T200，新增20990步，保留45个原阶段号。主要对照已有A R35/T200。
- 原首200T有序batch完全配对，原5×40T拆为10×20T。独立T RNG跨插入R保持，只在原40T边界重置；原400T噪声时钟、1e-5/1e-4 T LR和35R余弦R日程保持。
- 全验证R8/11/14/17/20/23/26/29/32/35；best仅R20/23/26/29/32/35，验证NDCG@5、平局较早；最多固定35和best两次测试。负结果也完成，不加分支。
- 远端root：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1
- 正式输出：root/runs/H1-Spread200-R35；日志root/jobs/formal/train.log、status.json、result.json。
- 共享Python：/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python，GPU torch2.6.0+cu118。
- 原父本地：outputs/stage09_restore/goal_20261027_r1/runs/H1/initial.pt，SHA 991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1。
- 本地数据outputs/kuairec_listwise_v1/run1；资产outputs/stage09_restore/goal_20261027_r1/assets/pse_v1；CPU环境.runtime/stage08_restore_env/Scripts/python.exe（torch2.6.0+cpu）。
- 代码manifest SHA c875e203a0c07ecc109b0ee6e89502cdcc31ab4887fab0d4506f4173e30b3f07；代码tar SHA 5bea789f3cc20d4058b20ba1cb0818bb02b4646f6b0e8c61736b130242769080。其他绑定见DEPLOYMENT/PARENT_BINDING/EXECUTION_PACKAGE。

## 远端工具操作

使用C:/Users/nancy/AppData/Roaming/uv/tools/inspire-skill/Scripts/python.exe -X utf8 本目录/inspire_compat.py --json --account sii-shared ...。只适配实时workspace枚举迁移，其他操作仍走真实CLI。网络/账号缓存权限需require_escalated，用户Goal已授权本协议内操作。

- job status/instances/events核对平台；CPU Notebook exec读上述共享日志。受控exec命令末尾加 ` # end` 吸收Windows SSH额外回车，保持JSON模式避免连接URL/token出现在输出。
- 不把CLI观察失败当作作业停止。先查已有作业，禁止重复提交。
- 单次最长sleep/wait不超过60秒；正常等待可以分段sleep，周期性查同一真实活跃进程。不要为了制造进展增加训练或旧统计。

## 后续验收：工具已准备，不等于验收完成

1. 等唯一作业按原协议完成。期望complete.json（R24255/T200）、selection、test_predictions_complete和jobs/formal/result，逐项检查EVIDENCE_PLAN.md。
   R20完整验证后可做半程更新日志/吞吐预算复核（新增10495步=10395R+100T）；若下载时已含下一T块，明确审核前10495步并保留原日志，不删改日志。仍不提前选最终best或运行测试。
2. 确认GPU终态/实例释放，记录实际时间及预算。不要为补足基线加实验。正式23.9h硬上限不扩。
3. 将最新archive_delivery.py上传远端root外层（不能放封存code/scripts），仅在成功终态seal。按实际体积检查C盘；早期空余26.3GB，正式父文件79.1MB。下载整包/receipt，核整包SHA，extract到全新目录再核每文件SHA；不清理旧工作区。
4. cold_restore.py从独立解压源码真实CPU加载关键checkpoint、双优化器/scheduler，审核全部41边界字段/时钟/R间Tokenizer隔离。明确CPU inspection不应用CUDA RNG，不更新、不推理，不声称CPU/CUDA轨迹等价。
5. analyze_d.py对完整新D产物审计：逐更新batch/日程/噪声、十全验证、共同网格选模、最多两测试、独立新预测指标及2000用户配对bootstrap。R8只在源码/数据/预测/summary SHA完全相同时复用早期已接受审计；R11及其他节点仍需复算。
   选模复核严格使用训练时锁定的生产summary分数，逐项核对六候选/selected记录及测试checkpoint SHA；独立FP64复算用于一致性检查，不能因微小聚合差异事后重选。cold_restore.py已显式固定新解压engine根并检查实际model模块导入位置；archive_delivery.py只标记资源释放需另核验，不自行宣称已释放。最终工具包应包含这些最新版。
6. temporal_d.py复用原冷热门initial_r1/shared_arrays和旧H0结果，仅新增A35/D35切片。切点1598745441188500，前38644后33013，五类训练定义不变，前后优势差使用同一全用户抽样；分解精确闭合但不作因果。
7. process_d.py审核31个8192状态：20份新子集、10份新完整验证提取、1份旧R5复用。20个T前后观察引用11份独立码本快照；R间隔离真实通过后才允许T前引用上一T后快照。独立核SID/碰撞/各槽计数/位移。
8. report_d.py生成中文报告与四类PNG/SVG/PDF，实际查看图表；根据实际固定终点/best、SID与碰撞、时间段与分组结果补写定量研究解释。分清事实与猜想，单seed/事后设计局限；本实验未做MEDA重置，不能据此认定one-epoch overfitting。
9. 保存实际执行的最新版工具、全部分析与报告、SHA绑定、外部依赖与恢复命令，形成最终交付。远端acceptance_tools_r1.tar.gz为早期准备副本，SHA ea38e2b1ed75055f135d494b16461208d41a1402c877fe4627a8335acdb09e16，不能冒充最新终验工具。
10. 主动停止并核验本轮CPU r2（r1已停），核验全部新GPU释放及无本轮跟踪设施遗留，保留远端科学产物。只有科学、归档恢复、资源收尾均有证据才complete Goal。

## 约束与故障恢复注意

- jobs/formal目录exist_ok=False。如果真正续跑，先查job终态/预算，保存旧attempt，增加明确attempt日志支持，不删旧目录。
- 滚动last之后可能有已记录更新；追加日志重放可产生重复。保留各attempt原始日志，核对重放并建立有来源规范日志后再验收。当前没有故障，不为假设修改封存代码。
- 不能在正式code/scripts加Python文件，否则执行绑定变化。分析/工具仅放本地本实验目录或远端root独立位置。
- 所有旧实验/归档/无关未跟踪文件/旧监控/旧Notebook保护。不git add .。不新增seed/H0训练/其他T预算/MEDA重置，不写飞书、不发消息。

## 09:07续接补充
半程日志实际独立核验已完成，不要重复。见midpoint_R20/ACCEPTED.json：10495新增更新=10395R+100T，逐batch SHA/计数/LR/400T噪声全通过，R13860/T100；原日志SHA 4f357991019247e76851580172cf53f789f44ad492e9b87813c9822cf86f2252。仅prefix，最终完整审计仍需执行。09:04:37平台仍RUNNING，R20验证13024/71657（449.25秒），正式elapsed26455.91。没有故障或重启，不要重复提交。待R20正式完整收据后核对来源，继续其余五周期。
预算半程复核已真实执行：MIDPOINT_BUDGET_CHECK.json，正式预测20.4664h含后续25%余量、90秒保留，低于23.9h；只是估计，不是结算证据，不改变预算或科学工作。无需重复此CPU统计。

## 恢复操作入口已补齐
RECOVERY.md列明归档外部依赖、最终封存/下载/独立解压/实际冷恢复/CPU分析顺序和预定输出路径；这些命令尚未对正式完整产物执行，不能算恢复通过。最终需补实际环境版本清单，按实际最终路径更新说明并保存最新工具和报告SHA。
已实际采集ENVIRONMENT_LOCAL.json与ENVIRONMENT_SHARED_INTERPRETER.json，RECOVERY.md已更新这些真实依赖版本。collector在远端root独立位置，无模型/CUDA调用，不改变科学代码。09:11 R20验证25024/71657、870.72秒，平台仍RUNNING。最终补充交付需包含这两份环境清单和collector。

## 最近运行观察（R20仍未结束）
同一正式pid201/status running，elapsed27732.082秒；R20完整验证最近50016/71657条、1746.794秒。此前连续观察43008→45024→48000→50016，属于实际活跃进程的verified wait，无故障或重启。R20完整record/summary收据仍待发布。没有新增科学任务，也无需重复半程CPU审计或环境采集。

## T120边界实测
T120后8192固定诊断实际完成（289.285647秒），spread_boundary epoch20/kindT/R13860/T120已发布；同一pid201/status running，elapsed28903.086秒。此为第六块T边界，后续仍需R21-R35及剩余完整验证/限定测试。没有新增分支或改变配置。

## R21边界实测
R21训练与首R后8192固定诊断实际完成（285.314311秒）；spread_boundary epoch21/kindR/R14553/T120已发布，同一pid201/status running，elapsed30019.058秒。下一个完整验证节点R23；已有R20指标不触发调整或追加训练。

## R22边界实测
同一pid201正式进程发布spread_boundary epoch22/kindR/R15246/T120，elapsed30869.770秒，status running；本轮无故障或重启。下一阶段R23训练，随后第六个完整验证R23。完整D验收、最大两份新测试、交付冷恢复与资源收尾仍未完成。

## 最新：R23完整验证已启动
同一正式pid201/status running，elapsed31835.582秒；R23训练已结束，完整validation进度3008/71657、108.282秒。第六个完整验证，尚未发布结果收据；等待完成后按source_check_validation.py核record/summary SHA及元数据，再继续R26/29/32/35。最大两次新测试仍待全部训练与验证后按锁定网格选择；无改参/重启/加跑。

## R20独立复算已完成；最终复用而非重算
saved_validation_R20新增真实per_sample.jsonl.gz、audit.json、ACCEPTED.json。audit_completed_validation.py --epoch20实际CPU执行：71657样本/4733用户，源身份/标签/冷暖分组/全部逐条和汇总指标，mismatches0，maxerr4.809881e-8；独立NDCG5 .0077258419894639455。audit SHA ea14af8e017ffd2195b41e1d24d68b3b717effe8fcccf3c3129c8603f73243bc。不要重复这份新D已完成统计。
analyze_d.py缓存入口已推广为R8专用目录或saved_validation_Rxx/ACCEPTED.json，核epoch/预测/summary/core/数据/audit SHA、真实样本和用户数后才复用；测试仍强制新审计。真实R8/R20收据绑定核验及语法检查通过。最终工具包要包含这个最新版analyze_d.py与audit_completed_validation.py。R11/14/17及后续节点尚待独立复算。最近R23验证31008/71657，elapsed32831.426秒，进程running。封存训练代码未改。

## 最新验收边界：前五完整验证全部独立通过
R11/R14/R17刚由audit_completed_validation.py实际CPU复算通过，各71657条/4733用户、0不一致，源身份/标签/冷暖分组/逐条与汇总全部指标均核验。独立NDCG5分别.005798742936672065、.008268443313921644、.007838806861669293；audit SHA分别76e92ec2a6f1abeeef8cea361b2efca6287d60f3cfd33f678c0eecac8b9b8093、eae3c3bc28242f4e36a81ad24b24afa6e35737b7d2d1fb0eac553d8b435342dd、859325c38ea769005c4de58ec265063b23c96d411e5e2f67c6f7075f7cef52cd。
因此R8/11/14/17/20均已独立验收，不要重算。最终analyze_d精确SHA绑定后复用这些ACCEPTED/audit；仍需读新解压预测和checkpoint逐SHA绑定。R23及后续验证尚待完整收据和独立复算，最大两新测试仍待最终。本文早期提到R11/14/17未复算或仅R8可复用的句子已过时，以此段及顶部权威表为准。最终工具包包含audit_completed_validation.py及最新版analyze_d.py。

## 最新：R23完整独立验收通过，已见T140更新
R23正式record/summary/预测已下载saved_validation_R23，source_check_validation.py及audit_completed_validation.py实际CPU执行通过：71657条/4733用户、mismatches0、源身份/标签/冷暖分组/全部指标核验，maxerr4.809881e-8，独立NDCG5 .008791367667086984。生产NDCG5 .008791367655382213、HR5 .04074968251531602、Recall5 .00833693859520849；完整验证2570.986447秒。
summary SHA 1bbb3bc3b9b6668a67214f0a56d7735fda610a0d70d33e91c02d3794f8936665，pred SHA 7daa2d8917aadd5c6a209cb888ad5ea92bab52b7c8777a0abc7b0047b716d258，audit SHA 3cd18aa776c4a0cc6bdedd8e9c2b6c3a7baee1baced0a519d08ba97f8f9d8952；cp phase28_R_R23_T120.pt SHA 6459469f6e757eb67f318a61e9c36b3c6c74f9f57eee66c8412a4c81274397d1。
前六完整验证R8/11/14/17/20/23全部独立通过，最终analyze_d精确SHA约束后复用，不重算。仍需完整新解压源文件与checkpoint绑定、全程日志审计/共同网格选择/测试/过程/时间/备份恢复/资源收尾。已见phase29 update20/20至R15939/T140，同一pid201 status running、elapsed34337.726秒；当前T后固定诊断等待完成。下一个完整验证R26。没有改参、提前测试、重启或追加分支。

## T140边界实测
T140后固定8192诊断实际完成（305.284394秒），spread_boundary epoch23/kindT/R15939/T140已发布，同一pid201/status running，elapsed34668.010秒。接下来R24训练，下一完整验证R26。前六验证独立审计已通过，不重复复算。

## 最高优先级：正式作业被平台回收，当前不得按旧RUNNING记录续提
2026-10-31 11:28:19 StopJob：管理员GPU<40%持续3h规则；11:28:27 CANCELLED/pod删除，最后终态事件11:29:08。回收前90min GPU采样平均21.83%，范围17.1%-27.1%。证据cancellation_r1/PLATFORM_STATUS.json、EVENTS/INSTANCES/POLICY/METRICS。原jobs/formal/result.json不存在，status.json停在running只是旧进程文件，不能当存活证据。
恢复点last.pt已本地下载并与远端SHA一致：bd28ba74e82a776a38e567d91c742dfccd1ca4591881151635a9da7efa5267fe。inspect_cancelled_restore.py真实CPU构建/恢复，模型、alignment、双优化器、scheduler、双计数和progress完全一致；完整CUDA/T RNG字节保留，CPU不应用CUDA。进度phase30/update400、R16339/T140；日志13014行，前10495行按已验SHA复用，新增尾2519行独立batch/LR/噪声/计数核验通过，日志精确对齐checkpoint，不需重放。RESTORE_READY.json为证据，不是最终独立解压冷恢复或CUDA续跑通过。
GPU已停止，无新GPU提交。预算已用保守上界9.803333h，剩余14.196667h，拟续跑最多14h+180s释放仍在24h内。用户回答“申请”，已准备RECLAIM_EXCEPTION_REQUEST.md；尚未代发、未获批。不能通过反复重启、假负载或改变锁定科学协议规避管理员规则。等待管理员认可的运行条件；继续完成可做的恢复准备。
续跑还需独立运维launcher支持新attempt目录和剩余预算门禁，保留jobs/formal原日志与原stale lock，不能覆盖/删除原证据；封存科学code不改。原job.py jobs/formal exist_ok=False、总forecast gate>14h，不能原样启动。archive_delivery原seal读取jobs/formal/result，成功续跑后应使用新attempt成功指针并绑定SHA，不能把旧未成功attempt改写成功。此为待实施内容。
Windows STATUS/status同名冲突已发现：SCP的进程status.json覆盖了最初大写同名本地文件，已重新查询保存真实平台终态为PLATFORM_STATUS.json。原远端文件未改，进程文件SHA仍正确；后续所有证据文件必须用完整不同名称。

## 独立续跑运维入口已准备（尚未上传/提交/获批）
新增job_resume_r2.py、resume_guard.py、RESUME_PLAN_R2.json，封存科学code不改。原job.py保留原SHA；新launcher只允许formal_resume_r2独立attempt目录、强制--resume、14h上限和累计24h预算、CPU恢复证据/原日志SHA对齐、原作业CANCELLED及管理员批准来源SHA门禁。旧.running需确认终态后归档保留，当前尚未移动。计划从R24第401步续跑，剩余7916R+60T，最多450360新评估样本；按实测更新速度、0.04秒/评估样本规划假设、25%后续余量及120s恢复短测预留，剩余预测9.7030h<14h。没有新GPU。
12项纯CPU临时夹具测试通过，CPU_RESUME_GUARD_TESTS.json；包括无/未批准许可、超预算、超单次上限、checkpoint或日志变化、父作业非终态、恢复点错位、旧锁保留等拒绝。临时批准仅存在于测试夹具并已清理，不是真实许可；没有生成POLICY_APPROVAL.json。新launcher成功后才写根FORMAL_COMPLETION.json指针，绑定新attempt result及resume_gate SHA；archive_delivery.py已支持该指针并保留旧jobs/formal失败现场，不能把原attempt改成成功。源码语法检查通过，真实CUDA续跑/完成指针/正式归档仍未执行。最终验收工具包需包含这些最新版。
CLI resources policy为只读，无豁免申请命令。已尝试页面入口：IAB打开qz.sii.edu.cn超时，恢复查tab不存在；现有Chrome无已连接浏览器扩展，native绑定报告窗口最小化，未操作登录/认证、未填表、未提交申请，也未发外部消息。当前只能确认申请材料已准备，不能声称平台没有申请入口或已申请。若用户提供管理员批准或可用申请页面，再按实际情况继续。

## 用户追问后的责任复核（RECLAIM_INCIDENT_REVIEW.md）
关键更正：启动前EXECUTION.md第26行已经记录GPU<40%持续3h回收，因此不是中断后才首次知道规则；我没有落实利用率门禁和连续窗口检查，是运行管理遗漏。中断后一度据旧status.json继续报running，亦为监控判断失误。后续声称存活必须当前平台状态与heartbeat新鲜度，不用旧文件替代。
按用户要求仅读运维指标：本次08:28-11:28三小时181个一分钟点，均<40%，平均22.5746%、最高38%；最近Freeze A 10月29日16-19点181点平均24.6878%、最高42%，最长连续<40%为104点，19:03正常完成。数据保存cancellation_r1/CURRENT_GPU_3H_DIAG.json与PREVIOUS_FREEZE_A_GPU_3H_DIAG.json。采样不证明平台内部连续瞬时判定，但平台StopJob事件明确因利用率规则回收。旧A组cuda12.8/NGC25.02，本次cuda13.2/pytorch25.06；差异确实存在，尚不能证明因果。更早4分支2卡每卡2进程、worker汇总约59.16%，不能替代最近单卡A对照。仍需核合法资源适配/管理员许可，不能盲目反复重启。用户问题是当前Goal的追问，不是取消实验。

## 最新：在原授权短测额度内启动独立恢复性能诊断，正式训练仍停止
为继续排查用户指出的资源问题，已完成CPU输入核验（R24后续401-405五个有效batch，原micro256×4，输入张量指纹已固定；无模型执行）。新独立短测nancy-s09-d-prof-1031-r1已创建，资源为原Freeze使用的cuda12.8 H100组、1/20/200、NGC25.02、priority4、shm32；0.35h平台硬上限、1200s内进程上限、无重试/通知。该短测从现有R24u400读取，隔离执行少量下一R更新、两次精确重放及profiler；不写正式checkpoint/log，不把产物接入正式续跑，不是靠重提长任务绕过利用率规则。
原短测142秒+新平台全部上限1260秒+释放预留180秒=1582秒<累计0.5GPUh，保守总预算承诺10.203333h<24。当前GPU_LEDGER为planned，需按新Job实际status/events/result更新并在结束后重新结算剩余预算。旧RESUME_PLAN_R2与14h续跑预算暂时失效，必须计入新短测后重新生成、调整上限；不能直接复用。原正式作业CANCELLED，仍R24u400/R16339/T140，源cp和log SHA再次核对未改变。
新工具recovery_performance_probe.py SHA55a74650ac0e47730c82a42cc7c0ad2874bbfcbde125508b3e488bfa49044a77；recovery_perf_job.py SHA76777ccf825eb861cde9022b9cc14ff57a209faa7a35f9cd6be90fabb6d6d55a；CPU_PERF_INPUT_PROFILE.json SHA6ad69441e56ad8482375483655c2b24800247edfafc1f7fafcc5edf6a152a1f5。全部位于远端root外层，封存code不改。结果应在root/recovery_perf_r1/RECOVERY_PROFILE.json、CUDA_PROFILE_TRACE.json；日志jobs/recovery_perf_r1/probe.log，status含heartbeat_unix，不以旧running文件当存活依据。先查同一Job平台状态，不能重复提交。
自动审批曾因引用最早discussion-only范围拦截监督脚本上传；重新读取当前active Goal明确实施/短测/恢复授权后，按原样同一命令补充授权证据重审获放行。没有绕过拒绝，新Job创建也经正常审批并成功。RECOVERY_PERF_SUBMISSION.json为创建收据。
GUI检查方面：新Chrome窗口2235422已打开为新标签页，但电脑操作工具因无法可靠识别Windows浏览器URL而强制停止；不得通过替代UI手段绕过此限制。没有登录/填表/申请提交、没有管理员批准，材料仍仅草稿。

## 最新：恢复性能短测成功并释放，存在可继续验证的合法并行调度方案
nancy-s09-d-prof-1031-r1已SUCCEEDED、instance_success、Pod删除12:10:53，末次事件12:11:00；created12:07:34→末次事件206秒保守计（Scheduled12:08:38→末次事件142秒）。recovery_perf_r1/RECOVERY_PROFILE.json SHA d8857aecd3c59fdb0c3c290c239a0cd60419842a4bcfc60d36c288a7affef8b8，与RESULT.json绑定核验通过。平台事件及实例也已本地保存。当前没有本轮GPU运行；CPU r2仍RUNNING且有继续诊断用途，尚未关闭。
真实CUDA下一R两次重放全状态指纹一致、profiler前后下一R状态一致、正式checkpoint和journal源SHA未变。旧cuda12.8组 torch2.6.0+cu118/H100 80GB HBM3，driver570.124.06、700W、1980MHz/P0。数据准备中位.3432057秒，预先准备输入后的R计算中位.8564100秒；一次R更新有38483 cudaLaunchKernel调用，CUDA事件41164，kernel时长求和.2912648秒；profiler单步wall1.338321秒，kernel/wall比.2176不是nvidia-smi利用率。换回旧组没有显示大幅吞吐改善；只能定位输入准备和大量小kernel/host发起开销，不能把全部差异归因组/镜像。
GPU_LEDGER已按206秒保守结算：全流程已用9.860555556GPUh、剩余14.139444444h；累计短测348秒（142+206），结算仍未核实。旧RESUME_PLAN_R2的35292秒预算已过时，任何正式续跑必须重生成。新短测实际前后源文件未变，正式仍R24u400/R16339/T140，6个完整验证独立通过。
下一步可做真实有用工作，不应立即宣称必须管理员批准而blocked：评估同一H100上“唯一D训练进程 + 固定checkpoint的必要验证/诊断worker”并行。只执行协议本来要求的工作、不加模型分支/seed/评估节点/重复旧结果；用独立snapshot隔离SID/RNG，所有R/T顺序、样本、优化器和400T时钟不改，验证仍FP32/beam32/batch32、选模仍全部训练后共同网格。该调度可能利用CPU/kernel空隙，提高真实GPU负载，属于性能适配，不是假负载或反复重提规避回收。
需先在剩余短测额度内做一次数值等价及真实负载检查（可计划平台<=0.25h=900秒；348+900+180释放=1428<1800，仍在0.5h与总24h内；后续正式上限按实际用量重算，可保守13.8h）。尚未创建并行短测或正式新Job。建议隔离baseline连续R401-430 vs并行同30步全状态指纹；FP32固定subset预测baseline/并行Top5一致；read-only eval两个worker各1024条可使真实工作重叠约30-40秒，采nvidia-smi实际负载，不能用profiler kernel/wall冒充利用率。所有产物preflight_only，不能接入正式状态。若不合适再寻管理员运行条件，不能靠周期重启或无关计算。
正式实现应为独立受SHA绑定的运行driver，不修改封存模型/损失/解码源码；显式记录运行调度改变及科学组件原绑定。训练阶段边界完整checkpoint与原一致，评估worker从不可变阶段checkpoint读取，绝不能读会变化的last.pt。full validation/cycle-end subset与T后/firstR后诊断可按固定snapshot排队并行，最终等所有任务成功才共同网格选模/最多两测试。最终all41状态、source/metrics/预算/资源仍需真实验收。当前job_resume_r2.py是旧串行且要求管理员approval的草案，不能直接拿来当并行driver。
