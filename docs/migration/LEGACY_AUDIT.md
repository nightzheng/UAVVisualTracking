# 旧成果只读审计与迁移方案

审计日期：2026-10-12（Asia/Shanghai）。状态：**阶段 A 技术审计保留；2026-10-13 裁决已修正，阶段 B–D 已恢复。**

本报告第 1–7 节的实测数据主要对应 2026-10-12 阶段 A；当时未运行 CoTracker 神经网络推理。2026-10-13 新推理与迁移见 V0_PRELIMINARY_RESULTS.md 和 MIGRATION_RECORD.md，历史目录继续只读。此前暂停依据是用户需求中的明确要求：“若存在多个冲突版本、许可证风险或不明确的数据来源，先暂停报告并等裁决与执行状态。”

## 1. 范围与证据

- 新工程根目录为当前 UAVVisualTracking 仓库。初始提交 `1cb28b4d68d8d18ec5fc17fdeb891b6b46404cc6`，仅有 README.md 和 .gitignore；开始时工作区干净。
- 旧成果根目录为用户提供的 Cotracker 目录，检查了 `co-tracker/`、`CoTracker3_代码与成果_20261007/` 和 `video/`。`video/` 当前为空；实际视频在两个成果的 `data/` 目录。
- 读取官方仓库配置、HEAD、许可证、setup.py、hubconf.py、demo.py、online_demo.py、predictor.py；重点阅读 camera_motion_compensation 的 run.py、target_tracking.py、point_selector.py、motion_compensation.py、affine_compensation.py、visualization.py 及测试；核查 presentation_demo 推理/缓存/渲染流程、打包入口、清单、环境记录和历史结论。
- 核对全部 146 项打包清单的大小与 SHA256，抽查 11 个视频文件（含重复副本）的首、中、尾帧，读取 28 个 tracks.npz 的数组结构与数值有限性；**未对全部视频逐帧解码，也未人工确认全部预测的身份正确性**。
- [公开审计摘要](../../experiments/summaries/LEGACY_AUDIT_SUMMARY.json)保留源记录指纹及完整性结果；含本机路径的原始机器证据 `audit_run_20261012/LEGACY_AUDIT_EVIDENCE.json` 仍保留本地，记录文件哈希、环境、逐项清单结果、视频元数据、缓存结构和完整测试日志，不随 Git 发布。[审计入口](audit_legacy.py)接受显式目录参数，不加载模型权重。
- 未提供综述 v1.0、阶段计划 v0.5 的文件路径，未假定其内容；未发现可用于本轮定位/可见性评测的 GT 标注。

## 2. 来源、版本、版权和迁移门槛

| 项目 | 核查结果 | 判断 |
| --- | --- | --- |
| 官方代码 | remote 指向 facebookresearch/co-tracker；HEAD 为 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`；`git diff HEAD --name-only`为空 | 固定此提交作为外部依赖候选；不复制全库 |
| 本地官方核心 | `cotracker/`有 31 个 Python 文件；无已跟踪文件的本地差异 | 官方核心未见本地改动；新增实验在未跟踪目录 |
| 两份成果 | 53 个同路径 Python 文件逐字节一致；未发现同路径代码冲突；原目录另外含 9 个展示代码/测试文件 | 打包副本不是一个冲突算法版本；缺少的展示文件不影响核心版本判断 |
| 打包证据 | 本轮 146/146 清单校验通过 | 证明打包文件与清单一致，不能独立证明模型曾成功运行 |
| 官方许可 | 本地 LICENSE.md 与[固定提交的官方许可证](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md)均声明 CC BY-NC 4.0 | 保留署名、原许可证及非商业限制；不能改为 MIT |
| 离线权重 | 本地 scaled_offline.pth 为 101890938 字节，SHA256 为 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` | 与[官方模型文件页](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth)公开 SHA256 一致，内容身份已核实；实际下载过程未记录 |
| 在线权重 | 在旧成果目录和现用 Torch Hub checkpoints 中未找到 scaled_online.pth | 历史在线缓存存在；当前不能据此宣称在线模型可重推理 |
| 小组独立代码 | 各成员独立编写，各自拥有原创贡献；2026-10-13 用户确认所提供实验代码为本人独立编写，不包含其他成员代码 | 用户本人有权处置的模块允许迁移、修改、整合；来源哈希见 SOURCE_MAP.json。其他成员仍需内部科研复用同意，不能推定 MIT 公开分发；作者未知文件仅暂停自身 |
| 数据 | **来源平台已由用户确认：Pexels；单条素材的原始 URL、作者及下载记录仍待补全。** | 列入 DATA_REGISTER.json；允许符合平台许可/条款的已下载视频本地非商业推理。原素材不提交 Git、不独立打包分发，不批量抓取或构建公开 ML 数据集 |

原目录的未跟踪项还包括演示临时产物、egg-info、缓存和输出；全部保留，不清理，不重置。当前没有发现必须合并的冲突版本。阶段 A 的代码授权与平台来源门槛已由 2026-10-13 用户裁决解除，单条素材登记不足不再阻塞本地演示。

## 3. 本轮环境实测

在已有 Conda `cotracker` 环境中运行只读审计；未新建环境、安装包或修改用户环境。

| 组件 | 实测版本/状态 |
| --- | --- |
| Python | 3.10.21（Anaconda，Windows AMD64） |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA（PyTorch 编译版本） | 12.8；torch.cuda.is_available() 为 True |
| GPU / 驱动 / 显存 | NVIDIA GeForce RTX 5070 Ti / 610.62 / 16303 MiB |
| NumPy / OpenCV | 2.2.6 / opencv-python 5.0.0.93 |
| matplotlib / imageio | 3.10.9 / 2.37.4 |
| imageio-ffmpeg / tqdm | 0.6.0 / 4.70.1 |

这是**环境导入、GPU 探测、视频抽样解码和旧算法单元测试通过**的组合，尚未证明模型 CUDA 运算、峰值显存、完整推理或 CPU 推理通过。系统默认 Python 为 3.13.13，部分工具运行提示无法定位基础解释器；Conda 列表命令失败。现阶段用明确的已有 `cotracker` 解释器，不将默认环境视为可复现环境。setup.py 自身没有声明完整推理依赖，打包 requirements.txt 也未包含全部 PyTorch/视频依赖，阶段 B 应补足经过实测的最小安装说明。

## 4. 实际功能与证据强度

| 功能 | 旧实现实际支持 | 本轮证据与不足 |
| --- | --- | --- |
| 视频输入 | OpenCV 读取、按 start_sec 裁剪、帧数限制、最大宽度缩放、BGR→RGB；保留原始/处理尺寸 | 11 个文件共 33 个首中尾采样帧全部解码成功；没有完整视频解码验证 |
| 模型加载 | 官方 Predictor；显式本地 checkpoint；自动 cuda/cpu/mps 选择 | 实测 torch 与 GPU 可用、离线权重哈希正确；本轮未加载模型 |
| 初始化 | 人工点、规则网格、人工目标+背景网格（hybrid）；查询帧、ROI 排除、坐标检查 | 已审阅相关实现；不是自动语义目标检测 |
| 离线推理 | run.py 用公开 Predictor 且隐式添加 support grid；target_tracking.infer 用 `_compute_sparse_tracks` 禁用 support grid | **两种实验配置不同**；未来必须显式记录，不能混合后称为同一基线 |
| 在线推理 | CoTrackerOnlinePredictor，window_len=16，step=8，16 帧重叠窗口和真实短尾 | mock 调度测试通过；没有真实在线权重实测；不是逐帧严格因果输出 |
| 结果 | NPZ 轨迹/布尔可见性、JSON 选点/配置/耗时；含部分模型身份信息 | 28 个缓存可无 pickle 读取，pred_tracks 均为有限值；有限值不代表正确跟踪 |
| 可视化 | 轨迹尾线、可见实心/不可见空心、视频和轨迹图、在线离线对照 | 本轮内存绘图测试通过；未导出新模型视频 |
| 性能 | 计时前后 CUDA 同步，纯推理时间、有效 FPS | 不含模型加载、完整解码、可视化和写盘；无完善峰值显存/端到端/生成时刻记录 |
| 评价 | 预测可见率、首次不可见、在线离线轨迹差异、背景残差 | 属无 GT 诊断；没有统一 GT 定位误差、可见性评价与真实遮挡/恢复事件标注接口 |
| 补偿 | 背景共同平移、中位数/MAD；partial-affine + RANSAC；首次缺口后停止累计补偿 | 数学测试通过；只能作为 V1 候选实验，尚无相机 GT 或米制状态验证 |

官方 Predictor 使用内部插值尺寸与 `align_corners=True` 的 `(W-1)/(H-1)` 坐标缩放。迁移不能把它与外部视频缩放规则混为一谈。返回的可见性为阈值化布尔量；在线底层的 visibility/confidence 合成后被阈值化，当前适配层没有独立返回置信度，不应生成虚构 confidence。

旧 read_clip 主要使用 `frame_index/fps` 的名义时间和 CFR 裁剪假设，没有逐帧 PTS 或采集时钟。视频抽样 PTS 可以读取，但不证明所有文件全段均为 CFR。正式输出应记录时间来源；VFR 应保留实际解码 PTS 或明确拒绝不满足条件的数据。

target_tracking.predict_online 已把整段视频读入 CPU 内存；每次仅上传窗口。输出为最后一次调用返回的累计轨迹，没有保存每次窗口结束时已可用的输出快照。因此其历史导出可受后续窗口修订影响，不能当作过去控制时刻已经可获得的在线结果。离线结果使用整段上下文；backward_tracking=False 并不等于没有未来帧。

## 5. 数据清单与历史结果边界

下表是原工作目录视频元数据；打包中的 5 个副本与这些同名文件哈希一致。城市别名 grid_demo_city_pov 与 Urban Explorer 文件也完全相同，所以 11 个文件仅对应 5 个独立视频内容。

| 视频 | 尺寸 | 名义 FPS | 帧数 | 本轮定位 |
| --- | --- | --- | --- | --- |
| cyclist.mp4 | 3840×2160 | 29.97003 | 214 | 普通动态目标候选，平台已确认，单条登记待补全 |
| woman_walking.mp4 | 3840×2160 | 23.97602 | 186 | 普通目标/相机运动候选，平台已确认，单条登记待补全 |
| wind_turbine.mp4 | 1920×1080 | 29.97003 | 794 | 风机候选；真实转速、未锁死状态与部分可见场景适配性尚未验收 |
| Pexels - Urban Explorer Walking Through Busy City Street.mp4 | 1920×1080 | 23.97602 | 204 | 城市背景/网格候选，平台已确认，单条登记待补全 |
| Pexels - Urban Scooter Commute Through Bustling City Streets.mp4 | 1920×1080 | 60 | 2843 | 城市背景候选，平台已确认，单条登记待补全 |

旧结论明确指出多个历史 wind_turbine 结果共用文件名但尺寸/FPS 不同。当前同名视频不能替代旧缓存当年的输入。具有源视频哈希的 presentation_demo/cache 可优先追溯；其他旧结果必须逐一核对 source identity。`_smoke` 中有合成/模拟验证，不能进入真实模型成绩表。

历史风机四点缓存为 240×4×2，包含 online 和 offline；城市缓存为 120×225×2；woman_walking_grid 为 144×226×2。这些只能证明已有数组存在。旧报告记录过叶片身份歧义、恢复跳点和可见性误判，本轮未重新逐帧确认其数量；无 GT 时不能声称定位精度、真实恢复率或相机补偿收益。

## 6. 本轮执行的测试

19/19 通过，0 failure、0 error，unittest 记录时间 0.077 秒：

- 原 `test_motion_compensation` 的 6 项纯数组测试：非零查询帧、平移/异常点、可见性条件、背景不足、统计口径、ROI 排除。
- 原 `test_affine_compensation` 的 10 项纯数组测试：变换方向/累计、逆变换、平移旋转尺度、RANSAC、退化/不可逆、缺口传播与共同帧统计。
- 原 `test_target_tracking` 的 3 项选择测试：诊断指标口径、mock 在线窗口/短尾调度、内存中的可见/不可见绘图。

**这不是 CoTracker3 神经网络复现成功，也不是 V0 验收通过。** 未执行模型权重加载、真实离线/在线推理、GPU 峰值显存、CPU 推理、视频导出、GT 指标、新工程安装。此为阶段 A 当时的执行边界；现已通过新推理推进阶段 B/C，见新结果报告。

对旧目录所有非 .git、非 __pycache__ 文件进行前后 SHA256 比较，共 9241 个文件，全部一致。审计使用 `-B` 禁止创建 bytecode；所有新增证据只写入新工程。

## 7. 四类成果与迁移映射

用户原文提出分四类但未给类别名。本报告采用以下审计分类，按 2026-10-13 裁决区分授权：

| 类别 | 内容 | 决策 |
| --- | --- | --- |
| 可直接依赖 | 固定版本官方 CoTracker3 及已核实离线权重 | 外部依赖与独立权重管理；保留 CC BY-NC 4.0 |
| 已确认模块可适配 | 小组视频读取、查询点、推理调度、结果保存、基础绘图及测试 | 本人贡献已确认并适配；其他作者仅在内部科研同意后复用，公开许可另核 |
| 仅作研究参考 | 历史缓存、Translation/partial-affine 原型、展示与差异分析 | 不作为新基线成绩；V1 原型留后续单独验证 |
| 不迁移 | .git、环境、egg-info、缓存、PPT/训练脚本、模拟成绩、过期绝对路径配置 | 避免复制第三方全库、硬编码和误报成绩 |

| 旧位置（相对 co-tracker） | 拟议新位置 | 必要适配 |
| --- | --- | --- |
| 官方 cotracker/predictor.py、models/ | 外部固定提交依赖；tracking/cotracker3.py 只放适配器 | 公共入口优先；若保留私有稀疏接口，固定版本并记录支持网格策略 |
| experiments/camera_motion_compensation/run.py::read_clip | data/video.py | 解码时间源、裁剪起始帧、尺寸/坐标映射、错误检查 |
| point_selector.py 与 run.py::validate_points | data/queries.py | 人工/网格标签、查询时间、坐标域、点 ID、重复点映射 |
| target_tracking.py::infer/predict_online/online_windows | tracking/cotracker3.py | checkpoint 必填、模式/窗口、每次输出可用时刻、禁止冒充严格因果 |
| target_tracking.py::save_tracks/load_tracks | data/results.py | NPZ 数组+JSON manifest+CSV 逐帧导出；输入/权重/配置哈希 |
| visualization.py、target_tracking.py 绘图函数 | visualization/tracks.py | 只提取基础轨迹绘图；适配统一结果，保留不可见点含义 |
| target_tracking.py::visibility_statistics 等 | evaluation/diagnostics.py | 无 GT 诊断；另建显式 GT 对齐和评价接口 |
| 旧纯计算/绘图测试 | tests/ | 授权后适配；另增解析、坐标、结果读写和 GT 指标测试 |
| motion_compensation.py、affine_compensation.py | docs/v1/V1_EGOMOTION_PLAN.md 的候选来源 | 阶段内仅方案，不迁移成正式通过的 V1 算法 |

## 8. 推荐方案与裁决与执行状态

推荐以原工作目录作为迁移引用基准，以校验通过的打包副本作为来源证据；两者共同代码完全一致。依赖固定官方提交，不复制官方源码。保留两种支持网格策略的差别，并各自命名配置。

### 2026-10-13 裁决与当前执行边界

- 用户本人原创且有权处置的代码允许迁移、修改、整合。用户随后确认：“我们都是各自写代码我不包含别人的代码”，适用于本次询问的 `experiments/camera_motion_compensation/` 推理/视频/查询/结果/绘图模块。
- 其他成员原创贡献归对应作者所有，保留作者身份，取得内部科研复用同意后可迁移；组内合作不自动授权 MIT 公开分发。不能确认作者的具体文件独立标记待确认，不暂停其他模块。
- CoTracker3 官方代码/权重仍为 CC BY-NC 4.0，固定提交、外部依赖、独立权重管理，不复制官方核心。
- **来源平台已由用户确认：Pexels；单条素材的原始 URL、作者及下载记录仍待补全。** [数据清单](../v0/DATA_REGISTER.json)记录 SHA256、尺寸、FPS、作者/URL空缺和候选页面。已尝试搜索优先风机/人物素材，未把无法精确匹配的候选当成确定来源。
- 已下载视频只用于符合 Pexels 许可/条款的本地非商业推理、可视化和无 GT 诊断，不提交 Git、不作独立素材包分发、不暗示人物代言。不自动抓取或扩展公开数据集；公开/大规模 ML 使用另行核查平台条款和必要授权。
- 原始小组目录只读。阶段 B–D 的实际迁移与真实离线模型运行分别见 [迁移记录](MIGRATION_RECORD.md)、[新结果报告](../v0/V0_PRELIMINARY_RESULTS.md)。在线权重尚缺，不作为本轮必达条件。
