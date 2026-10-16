# UAVVisualTracking

**Visual Perception, State Estimation and Active Tracking of Partially Visible Dynamic Targets**

A research framework for reproducible visual tracking and conditional state estimation of partially visible dynamic targets, with slowly rotating wind-turbine blades as the primary application. It is not a complete autonomous UAV inspection system.

本项目研究低速旋转风机叶片的无人机视觉跟踪与巡检：在目标仅部分进入视场时保持局部观测，区分材料点身份、可见区域和模型推断，并逐步建立相机运动、几何尺度、状态估计及主动视觉的验证条件。CoTracker3 是当前候选科研基线，不是最终算法选型。

## 研究目标与当前状态

部分可见条件下，持续输出位置不等于保持同一材料点身份。移动相机与目标运动共同影响图像轨迹；单目像素位置也不自动提供深度或米制尺度。本项目按 V0 局部持续跟踪 → V1 运动补偿与有条件空间恢复 → V2 预测和主动视觉闭环推进。V3 仅为待审查的物理验证扩展。

工程版本为 **v0.1.0**。2026-10-16 用户已确认 MIT 署名并批准首次本地阶段提交；本轮按核验后的文件清单提交。

| 层级 | 已完成与证据 | 尚未通过或未开展 |
|---|---|---|
| 工程 | 通用轨迹接口、固定离线模型、视频/NPZ/CSV/manifest 输出、GT 评价入口；37 项单元测试通过 | 测试通过不证明真实定位精度 |
| V0 复现 | Stage 0 五次真实 GPU 推理；Stage 1 十次受控调用（含两次暖机） | 人工复核 GT 为 **0/480**；不能报告定位准确率或身份恢复率 |
| 局部视场先导 | 四种视场 × support grid 开/关，八组保存的真实离线结果 | 同一个视频的裁剪代理；不是四组独立真实近景采集，材料点可靠性未验证 |
| V1 准备 | 输入/输出及假设设计；三类已知二维变换的数学契约验证 | 没有真实背景运动估计、独立相机运动验证或米制状态恢复 |
| 在线与后续 | 离线/在线权限分开记录 | 在线权重/模型未验证；未训练或微调，未开展 SAM 2/DA3、完整 V1、V2 飞控或实机 |

预测可见率、有限坐标、轨迹平滑和视频演示均属于诊断，不能代替 GT 精度。当前局部样例存在跳变及身份关联风险，不能宣称局部巡检通过。

## 环境与安装

本机已验证 Windows + Conda 环境：Python **3.10.21**，PyTorch **2.11.0+cu128**，torchvision **0.26.0+cu128**，PyTorch CUDA **12.8**，NumPy **2.2.6**，opencv-python **5.0.0.93**；GPU 为 RTX 5070 Ti，驱动 610.62。Matplotlib 3.10.9 用于部分证据绘图脚本。其他平台及依赖组合未完成同等 GPU 验证。

建议复用已验证环境；需新环境时可按以下流程准备。下面给出安装指令，本次提交准备没有重新下载/安装 GPU 依赖，也没有执行新环境的完整 GPU 复现。

```powershell
conda create -n uavvisualtracking python=3.10 pip
conda activate uavvisualtracking
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.2.6 opencv-python==5.0.0.93 matplotlib==3.10.9 setuptools==78.1.0
python -m pip install --no-deps --no-build-isolation -e .
uav-track --help
```

在仓库根目录安装；Git 及兼容的 NVIDIA 驱动需预先可用。[官方 CUDA 12.8 wheel 索引](https://download.pytorch.org/whl/cu128/)提供相应 torch/torchvision 包。项目 `pyproject.toml` 只声明 NumPy/OpenCV 基础依赖，不能期待它自动安装 GPU 环境。已有环境中不要为此升级依赖。Conda 环境名是示例，不依赖某位成员的本机路径。

### 固定官方源码和离线权重

适配器要求一个包含 Git 元数据的固定官方 checkout；会核验 HEAD 和模型已跟踪文件。已有只读 checkout 可直接使用。需要新 checkout 时：

```powershell
$upstreamDir = "<COTRACKER_SOURCE_DIR>"
git clone https://github.com/facebookresearch/co-tracker.git $upstreamDir
git -C $upstreamDir checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
```

将 `<COTRACKER_SOURCE_DIR>` 替换为新的本地目录，不对旧小组目录执行安装或 checkout。当前适配器从该源码路径加载官方模块，不需要把上游源码复制进本项目。可选 `cotracker` extra 固定同一 VCS commit，但不能替代运行时 checkout/版本检查；上游 setup.py 本身没有安装完整运行依赖。

从 [facebook/cotracker3 官方离线权重页面](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth)自行准备 `scaled_offline.pth`，放在 Git 外或 `weights/`。预期大小 **101890938 bytes**，SHA256：

```text
2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834
```

```powershell
$checkpoint = "<SCALED_OFFLINE_PTH>"
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash.ToLowerInvariant() -ne "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834") { throw "Checkpoint SHA256 mismatch" }
```

程序不会自动下载权重，也不会在缺 CUDA 时静默切换 CPU。CPU 可显式选择，但实际 CPU 模型推理尚未验证。

## Quick Start

使用你有权使用的本地视频，无需仓库外缺失的 Pexels 素材。以下示例采用 **24 帧、3×3 无语义查询网格**，与正式风机四点基线分开；网格点不自动代表叶片或其他目标。

1. 按上文准备环境、固定 checkout 与校验后的权重。
2. 复制通用配置为本地私有配置。其 `queries.strategy=grid`、`grid_size=3` 已与真实代码一致；如需材料点，将其改为 manual 并填写首帧处理图中可辨认位置及唯一 point_ids。

```powershell
Copy-Item configs/tracking/official_offline_grid.json configs/tracking/demo.local.json
```

3. 替换三个输入占位路径，并使用一个尚不存在的输出目录：

```powershell
uav-track --config configs/tracking/demo.local.json --video "<LOCAL_VIDEO_MP4>" --checkpoint "<SCALED_OFFLINE_PTH>" --cotracker-source "<COTRACKER_SOURCE_DIR>" --output experiments/runs/my_new_demo --device cuda
```

等价入口为 `python -B -m uav_visual_tracking.cli`，参数相同。程序拒绝覆盖已有结果。`demo.local.json`、数据与运行目录被忽略；人工点必须对应实际视频、查询帧和处理尺寸，不能把已有风机四点直接用于任意视频。

4. 检查 `tracks.npz`、`tracks.csv`、`manifest.json`、`metrics.json`、`tracked.mp4` 和 `preview.png`。manifest 保存源视频/配置/权重哈希、版本、处理尺寸、时间源、时序权限和资源信息。PTS 是视频播放时间，不是传感器采集墙钟。
5. 运行工程测试：

```powershell
python -B -m unittest discover -s tests -v
```

测试包含算术/合成坐标夹具与短视频编解码检查，不是风机 GT。人工标注规范与复核门槛见 [V0_GT_ANNOTATION_SPEC.md](docs/v0/V0_GT_ANNOTATION_SPEC.md)；已有 `prepare_stage1_annotations.py` 可对匹配的运行准备原帧标注包，`evaluate_stage1.py` / `compare_stage1_gt.py` 只评价有明确人工来源及复核状态的条目。

## 工程结构

```text
src/uav_visual_tracking/  # tracking、data、visualization、evaluation、geometry、common 和 CLI
configs/                 # 跟踪配置、评价协议、冻结基线及局部先导条件
scripts/                 # 已有推理、标注准备、评价、证据整理与完整性检查
tests/                   # unittest 工程契约与数学检查
docs/                    # 迁移、阶段报告、V0/V1 方案和提交准备
experiments/             # 可公开摘要/空白模板；运行和数据产物按子路径忽略
```

`geometry` 目前仅实现已知二维变换的坐标契约，不是完整 V1。原始视频、帧、权重、人工标注产物、资源采样和大体积逐帧结果保留本地；不忽略整个 experiments/configs/docs，也不全局忽略 JSON/CSV/NPZ/PNG。

## 实验与可复现性

正式基线 **V0-offline-wind120-20261013-r1** 见 [冻结快照](configs/evaluation/v0_frozen_20261013.json)和[能力状态](docs/v0/V0_BASELINE_STATUS.md)。它固定 120 帧、960×540、四个旧查询、官方 commit/权重、6 iterations、window_len=60、seed=0、无反向跟踪。必须自行获得与记录 SHA 匹配的素材才能重现同一风机输入。

- support on 使用官方公共查询接口增加 6×6 辅助点；off 使用固定版本 sparse 接口，不添加辅助点。两种配置保留，未据预测可见率选优。
- 当前是完整片段离线推理，可以访问未来帧。backward_tracking=false 不等于因果在线；离线 FPS 不是机载端到端实时控制帧率。
- GT 仍为 **0/480**。无 GT 仅报告预测可见性、跳变、越界、返回候选与资源诊断；这些不是定位误差、真实遮挡或身份恢复率。
- 本周局部裁剪保持同一源视频/查询意图和提示成本，并记录 ROI/坐标映射；目标尺度、宽高比与背景仍有共同变化，不能称独立近景数据或单因素结论。

公开数值与来源指纹在 [实验摘要目录](experiments/summaries/README.md)，完整轨迹/视频及含本机路径的原始机器记录不随 Git 发布。摘要复用已有真实运行，不将旧缓存或本次软件测试冒充新模型实验。

阅读：[Stage 0](docs/STAGE0_REPORT.md) · [Stage 1](docs/STAGE1_REPORT.md) · [本周报告](docs/WEEKLY_PROGRESS_20261012_1018.md) · [局部先导](docs/v0/V0_PARTIAL_VISIBILITY_PLAN.md) · [GT 进度](docs/v0/V0_GT_PROGRESS.md)。

## Roadmap

| 阶段 | 已完成 | 后续研究 |
|---|---|---|
| V0 | 离线复现、工程契约、标注工具、固定局部先导 | 人工 GT、独立局部数据、真实定位/材料身份/恢复质量 |
| V1 | 二维变换数学验证与[实验设计](docs/v1/V1_EGOMOTION_PLAN.md) | 背景二维估计、独立相机参考、几何/尺度条件与退化识别 |
| V2 | 尚未开展 | 短时预测、时间有效性、主动视觉闭环及受控验证 |
| V3 | 仅候选扩展 | 在前层证据与平台条件明确后单独审查 |

## License、Third-party 与数据

`nightzheng` 拥有相应授权权利的原创代码采用 [MIT](LICENSE)，版权主体已由用户于 2026-10-16 确认为 `nightzheng`。本项目 MIT **不覆盖** CoTracker3 官方 CC BY-NC 4.0 源码及模型权重、其他第三方组件或 Pexels 素材，也不代表整个系统获得商业使用授权。

CoTracker3 官方 [固定版本许可证](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md)及[模型许可](https://huggingface.co/facebook/cotracker3)为 **CC BY-NC 4.0**。本项目仅在该许可范围内作为科研基线使用。迁移来源与其他组件许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [SOURCE_MAP.json](docs/migration/SOURCE_MAP.json)。

Pexels 素材不随仓库发布，不受本项目 MIT 授权。平台来源已由用户确认，单条 URL/作者/下载记录仍待补全，见[数据登记](docs/v0/DATA_REGISTER.json)。使用须核查 [Pexels 许可](https://www.pexels.com/license/)和[服务条款](https://www.pexels.com/terms-of-service/)；不批量抓取、公开素材包或暗示人物代言。公开数据集与更大规模 ML 用途另行核查必要授权。

首次本地阶段提交已获用户批准，核验范围与结果见 [v0.1.0 准备报告](docs/RELEASE_READINESS_v0.1.0.md)及[阶段提交计划](docs/STAGE_COMMIT_PLAN.md)。授权仅限暂存和本地提交；不推送、不合并、不创建 Tag 或 Release。
