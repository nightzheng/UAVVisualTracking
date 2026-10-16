# V0 最小基线规范 v0.1

日期：2026-10-13。目标是局部物理点持续跟踪的可复现输入/输出和失效观察闭环；推理成功不等于稳定跟踪验收通过。实现通用Tracker契约和固定CoTracker3离线适配器，支持网格两配置明确区分。

## 输入与配置

入口为 `uav-track --config <JSON>`，显式覆盖 `--video --checkpoint --cotracker-source --output --device`。device为cuda/cpu，无自动fallback；错误版本/权重、缺文件、重复/越界点或已有输出目录报错，失败产物标记error.json。

- clip：start_sec、duration、max_frames、max_width。请求起始帧按nominal FPS计算，实际解码索引须符合请求；最大宽度缩小不放大、保持纵横比。
- queries：manual/grid、frame（截取后片段索引）、points、point_ids、coordinate_space=processing/original。人工点与自动网格分别标记；没有语义目标定位能力。
- 当前风机4点/人物3点沿用旧960×540处理图人工坐标，仅属开发演示初始化。
- grid_size为边长，N=grid_size²；新网格5%–95%边界内均匀采样，标签automatic_uniform_grid_no_semantics，不等同官方get_points_on_a_grid或辅助support grid。
- 模型：固定commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` 与官方scaled_offline权重hash，offline=True/v2=False/window_len=60/6 iterations/backward_tracking=False；RGB float32 0..255，模型内部H×W=384×512。
- support_grid=true调用官方公开forward（隐式6×6辅助网格），false调用固定版本 `_compute_sparse_tracks(...add_support_grid=False)`；私有入口需固定版本，不能将两者混为同配置。

## 坐标与时间

像素中心左上原点，x向右、y向下，浮点px；尺寸为[width,height]，视频数组T,H,W,3。查询点须在[0,W−1]×[0,H−1]内；模型越界/不可见推断保留，不代表真实物理位置。

外部OpenCV resize映射 `p_dst=(p_src+0.5)*size_dst/size_src-0.5`；逆映射原图用相反尺寸。缩小后极边缘点若映出有效像素中心区间会拒绝，不静默修正。内部官方align_corners=True用(W−1)/(H−1)变换，交由模型处理，不能与外部转换混用。

frame_index为源视频索引，clip_frame_index为截取后索引。优先使用实际解码POS_MSEC且有限/递增；否则标time_basis=nominal_fps_assumption并用index/fps。**这是播放时间，不是传感器采集墙钟。**当前实测片段观察到CFR；VFR保留PTS，但CFR视频导出器拒绝冒充保持VFR。视频起点定位仍依赖FPS假设，VFR精确时间裁剪留后续完善。

generated_at_utc是预测转CPU后的UTC墙钟；available_after_frame_index为已读源末帧；video_context_gap_seconds=末PTS−当前PTS，仅为上下文跨度，不是实时硬件延迟。

## 时序与可见性

| 模式 | 状态 | 未来权限 |
| --- | --- | --- |
| 完整片段离线 | 实现并真实验证 | 访问完整截取片段，最终一次输出；causal=false |
| 重叠窗口在线 | 旧实现已审计；新适配器未实现，权重缺失 | 后续必须逐窗口记录可用末帧与修订历史 |
| 固定滞后输出 | 未实现 | 不以配置命名冒充 |
| 逐帧严格因果 | 未实现 | backward_tracking=false不代表因果 |

可见性为官方阈值化bool预测，不是GT/独立confidence。不可见期间位置可能为模型推断，结果不生成confidence。查询前inactive；CSV坐标/可见性留空，NPZ保留原始数组供追溯；评价只纳入active。

## 输出与评价

- tracks.npz：schema_version、T×N×2 tracks、T×N bool visibility、frame_indices、timestamps、point_ids、queries和initialization，无pickle。
- manifest.json：Video ID/hash、配置hash、尺寸/时间源、模型提交/权重hash、支持网格、环境、生成时刻/权限、资源/性能、产物hash、status。
- tracks.csv：源/片段帧索引、PTS、Point ID、初始化/active、原图/处理图xy、预测可见性、生成时刻/末帧/上下文跨度、配置/模型身份。
- metrics.json、tracked.mp4、preview.png；视频逐帧解码核验帧数/FPS，保留失败点、不平滑轨迹。

NPZ保留dtype并直接用于评价，JSON保存复杂身份，CSV便于逐条审查。无GT只算预测可见率、首次预测不可见、预测状态返回/出入画事件与非有限值，不能作为真实遮挡/身份恢复/定位精度。

GT NPZ需tracks/visibility/valid（mask均bool）、frame_indices、point_ids、video_sha256、coordinate_space='processing'、resolution=[W,H]。同视频hash和尺寸，按源帧与Point ID对齐；缺标注用valid=false。不将预测当GT。对GT可见/active/annotated点帧计算EPE（包含预测不可见坐标），另报非有限预测与覆盖率；对可见性报TP/FP/FN/TN、precision/recall/F1，零分母/无样本为null。

## 协议与资源边界

现有素材仅是开发演示，**没有冻结测试集，也无真实GT**。后续新增局部叶片/标注数据后预先冻结输入、查询、参数和阈值；不能据测试集反复选点、调阈值。本轮不报告真实恢复率、三维/米制状态、真实转速。

推理计时为CUDA同步的输入tensor/upload+模型，不含加载/解码/输出下载/渲染。端到端另计decode/input hash/source检查/load/inference/export及视频完整解码，不含最后manifest写入/hash。cold与warm运行分列，不据单次耗时推断支持网格优劣或实时性能。

显存为torch.cuda allocator max_memory_allocated/reserved bytes，**不是驱动实测驻留显存**。长片段计数超过本机物理显存，驻留量/原因未验证，保留原始计数。CPU可显式选择，本轮未实测CPU模型推理可行性。
