# 迁移记录（阶段 B–D）

日期：2026-10-13。阶段 A 技术审计与原始机器证据保留，仅按用户裁决修正归属/数据使用边界。本轮已建立可安装项目并完成真实 GPU 离线模型冒烟。

## 归属与许可裁决

小组成员各自独立编写。本次询问的 `experiments/camera_motion_compensation/` 视频、查询点、推理、轨迹保存和基础绘图模块，用户确认本人独立贡献不包含他人代码；允许迁移、修改、整合，适用项目自有代码 MIT。来源 SHA256 与新路径见 [SOURCE_MAP.json](SOURCE_MAP.json)。

其他成员贡献归对应作者，内部科研复用须取得同意并保留作者，不推定 MIT 公开分发；作者未知文件独立暂缓。本次没有混入其他成员代码。CoTracker3 官方核心/权重继续 CC BY-NC 4.0，未复制整个源码、Git 历史或环境。

视频来源平台已由用户确认：Pexels；单条原始 URL、作者和下载记录仍待补全。已下载文件仅做许可/条款范围内本地非商业推理，无批量抓取、训练、公开数据集或素材包；原素材不提交 Git。

## 实际迁移映射

| 旧路径（相对 co-tracker） | 新模块 | 复用及适配 |
| --- | --- | --- |
| experiments/camera_motion_compensation/run.py::read_clip | data/video.py | 保留 OpenCV裁剪、宽度/纵横比、RGB转换；增加逐帧PTS、实际解码索引、时间源、CFR观察及截断标记 |
| point_selector.py / run.py 输入检查 | data/queries.py | 适配人工点、有效坐标、查询帧、重复点检查；增加点ID与初始化标签；新网格明确为5%–95%均匀采样，不冒充官方网格策略 |
| target_tracking.py::infer 与 run.py 推理分支 | tracking/cotracker3.py | 保留 official Predictor、local checkpoint、eval/inference_mode、CUDA同步与支持网格差异；固定官方提交/权重hash |
| target_tracking.py::save_tracks/load_tracks | data/results.py | 适配 NPZ 保存流程；新增 JSON身份、CSV逐帧结果、原图/处理图坐标、生成时刻与时序权限 |
| visualization.py::draw_point/draw_history | visualization/drawing.py | **原函数逐字提取**；保留不可见空心/淡色轨迹；新video.py接通用结果并完整解码核验输出 |
| target_tracking.py::visibility_statistics | evaluation/metrics.py | 适配无GT可见率/首次不可见；新增预测状态/出入画事件与显式GT键对齐、EPE及可见性评价 |
| 官方 cotracker/ | 外部固定依赖 | commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`；核验HEAD/model tracked文件；只开发适配器 |

源码新路径均在 `src/uav_visual_tracking/`。tracking/base.py提供后端独立契约，未创建SLAM/control等无需求空架构。

## 未迁移内容

- .git、环境、egg-info、bytecode、原视频/权重不进入正式源码。
- PPT串接、筛片、人工模拟 `_smoke`、旧缓存/成绩不作为新推理输入或新模型成绩。
- Translation/partial-affine仅作V1候选参考，不作为正式通过的V1算法。
- GUI选点暂以JSON人工查询替代；语义定位未实现。
- 窗口在线接口当前明确拒绝，缺在线权重；不重命名离线冒充因果。
- 旧绝对路径配置不复制；公开配置用占位路径，运行时显式传入本地位置。

## 工程验证

19/19新单元测试通过，覆盖数据解析/PTS、坐标转换、非零查询帧、结果读写、预测事件、GT手算/键对齐/NaN覆盖率和禁止覆盖。合成夹具不作为模型成绩。wheel已成功构建并隔离安装到 `.local/installed`，没有更新Conda包；以下五次真实推理从已安装包运行。

官方apple24帧/9点、人物120帧/3点、风机120帧/4点supports开/关、supports开重复均成功。每次输出新NPZ/CSV/JSON/MP4/PNG，五个视频共504帧完整解码，CSV共2016条点帧。重复运行最大坐标差0、可见性相同；仅说明当前条件重复性。详见 [V0_PRELIMINARY_RESULTS.md](../v0/V0_PRELIMINARY_RESULTS.md)。

模型/用户源码/视频52项运行前后hash不变；完整旧目录完整性检查见结果证据。未提交、推送、合并、训练或飞行。大型产物在Git忽略的 `experiments/runs/stage0_20261013/`，摘要/配置/来源hash在docs/configs。
