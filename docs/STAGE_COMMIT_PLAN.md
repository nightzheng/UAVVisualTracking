# 首次阶段性 Git 提交计划

初次准备：2026-10-14；授权更新：2026-10-16。**用户已确认 MIT 版权主体为 `nightzheng`，并批准按复核后的 86 文件清单暂存及首次本地提交。** 本文件记录提交前的授权与执行计划；最终 Commit SHA 和实际文件数以 Git 历史为准。不推送、不合并、不创建 Tag 或 Release。

## Git 现状与建议

仓库已有初始提交 `1cb28b4`，仅跟踪 README/.gitignore。Stage 0、Stage 1 和本周工程尚未提交，不需要重置、重写初始历史或清理文件。建议一次提交完整的 v0.1.0 科研工程，便于保持包、配置、测试、来源和结论彼此一致。

建议 commit message：

```text
feat: establish reproducible UAV visual tracking research baseline
```

若用户更希望逻辑拆分，可在批准后按“包与基础契约”“评价/标注/受控实验执行器与配置”“报告/许可/公开摘要”分组。拆分仅选择文件与必要内容，不删除或重构现有成果；不同分组可能有引用依赖，不把中间提交宣传为完整可运行版本。本轮不执行任何拆分操作。

## 拟提交与保留本地

| 拟提交 | 原因/边界 |
|---|---|
| README.md、.gitignore、pyproject.toml | 正式项目首页、打包与路径隔离；保持现有版本/依赖配置 |
| LICENSE、THIRD_PARTY_NOTICES.md | `nightzheng` 有权授权的原创代码采用 MIT；官方 CC BY-NC/其他依赖/Pexels 素材独立，不受本 MIT 覆盖 |
| src/uav_visual_tracking/** | 自有数据/轨迹/可视化/评价/已知变换契约和固定模型适配；不包含官方核心 |
| configs/** | 可公开配置、评价协议、冻结基线；本地私有配置例外 |
| scripts/**、tests/** | 现有执行器、标注工具源码、工程测试；不自动启动实验 |
| docs 中的报告/方案/迁移映射/完整性摘要 | 能力与未通过条件、来源可追溯；原始本机机器记录例外 |
| experiments/summaries/** | 已有实验的无敏感数字与来源 SHA，非完整轨迹/素材 |
| experiments/templates/**、experiments/README.md | 480 行全空白模板及使用边界，非人工 GT |

全量候选文件名见 [STAGE_COMMIT_FILES_v0.1.0.txt](STAGE_COMMIT_FILES_v0.1.0.txt)。2026-10-16 复核仍为原 86 文件，路径无增减。与此前准备快照相比，仅六个既有候选更新：LICENSE、README、THIRD_PARTY_NOTICES、本计划、准备报告及机器可读准备摘要。按清单精确暂存后再次审核 staged diff、文件大小、内容及工作区与暂存树一致性。

| 应忽略/不提交 | 存储策略 |
|---|---|
| 视频、权重、原始大数据 | /data、/weights、checkpoints 与模型/视频后缀规则，实际文件保留 |
| 原始帧、人工标签、轨迹、运行视频、资源采样 | experiments/runs、annotations、stage1、weekly 子路径 |
| 本机完整审计/运行记录 | docs/migration/audit_run_20261012、docs/v0/STAGE0_RUN_RECORD.json、STAGE1_EVIDENCE_RECORD.json；公开摘要另存 |
| 环境、缓存、构建、IDE、日志和私有配置/密钥 | .local、Conda/venv、build/dist/egg-info、工具缓存、本地配置及凭据规则 |

不忽略整个 experiments/configs/docs，不全局忽略 JSON/CSV/NPZ/PNG。修复根 data 路径规则后，src/data 五个模块已进入候选范围。目前没有已跟踪的大型数据；如果以后发现，先报告并考虑只移除跟踪，不能自动删除本地文件。

## 能力、验证与未完成目标

本次主要新增的是可安装科研包、固定 CoTracker3 离线适配、统一查询/轨迹/可见性/时间身份输出、可视化、人工复核门槛及稀疏 GT 评价接口、少量局部裁剪先导执行器、V1 已知二维变换契约、配置与来源/阶段文档。

已有真实 GPU 证据来自 Stage 0/1 与本周先导，不能把本次提交准备称为新模型复现。本轮通过 37 项现有测试、离线 wheel 构建/隔离安装、19 个模块导入、实际 CLI help、八配置解析、历史 23 manifest/95 产物指纹、忽略/链接/敏感模式与体积检查及 git diff --check，详见 [准备报告](RELEASE_READINESS_v0.1.0.md)。

V0 GT 0/480，真实材料身份保持、真实精度/恢复和独立近景任务未通过；V1 真实相机运动估计与条件空间恢复未完成；在线模型未验证。不得在 commit 描述中写“稳定跟踪/高精度巡检/完整自运动恢复/飞控已实现”。

## 提交条件与停止点

1. 已获用户确认：`Copyright (c) 2026 nightzheng`；完整 MIT 正文及第三方边界保留，相关文档同步。已在独立目录重新构建并验证带确认版权的本地 wheel。
2. 已获用户批准：原 86 文件首次阶段本地提交，提交信息使用上文指定文本。提交前再次验证 37 项测试、打包/导入/CLI、忽略/链接/隐私/体积与 Git 一致性。
3. 按清单执行 git add，核对暂存路径恰为这 86 文件、staged diff 无新增问题且与工作区内容一致；然后执行本地 git commit。
4. 核对提交信息、父提交、实际文件数、Commit SHA 与干净工作区，报告结果后停止。提交失败则如实报告，不重置、不清理已有成果。

本轮停止点是已授权的本地提交及结果报告；不执行推送、合并、Tag 或 Release，不修改结果、标签、模型或追加实验。研究目标尚未完成的状态保持原样。
