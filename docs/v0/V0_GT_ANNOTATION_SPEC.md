# V0 人工 GT 标注规范（Stage 1）

2026-10-13。当前是真实视频的**空白标注准备包**，人工已复核 GT 为 0。不得把初始化查询、模型预测、自动插值或软件单元测试样例作为风机 GT。

## 视频与可编辑文件

视频 SHA256：`9dc01e3f3ebd7400ec71685e7660488b8aa630888dd7caec4921c734049556eb`。原图 1920×1080，处理图 960×540，29.97002997 FPS；原始帧 0–119，视频时间 0–3.970633 秒。时间来自解码 PTS，不是传感器采集时钟，也不自动证明覆盖了某片叶片完整 360° 旋转。

标注包：`experiments/annotations/stage1_wind_120/`，包含 120 张无预测原图、带帧哈希的 `annotations.blank.json`、480 行 `annotations.blank.csv`、可双击打开的 `index.html`。当前本机预览地址 `http://127.0.0.1:8766/`；服务仅绑定本机。网页支持原帧播放、逐帧切换、代表帧/加密帧跳转、原像素放大、人工点选、草稿保存、逐条复核和 JSON/CSV 下载/JSON 恢复。浏览器存储不是交付文件，退出前下载 JSON。正式评价以 JSON 为准；CSV 是可编辑对照表，不能省略点定义与复核信息后直接当正式 GT 导入。

## 点定义与叶片身份

| ID | 初始意图 | 人工确认要求 |
|---|---|---|
| p000 | 轮毂/轴心参考，查询 (493,277) | 是几何中心参考，不应称为叶片材料点；人工定义具体中心判据 |
| p001 / A | 初始上方叶片，查询 (591,75) | 确认红色末端区域中具体材料位置；名称不代表恰为叶尖 |
| p002 / B | 初始右下叶片，查询 (626,470) | 同上；不能随叶片交换方位重新编号 |
| p003 / C | 初始左方叶片，查询 (278,296) | 同上；身份沿原视频连续观察，无法判定时标不确定 |

旧查询多处于近叶尖的均匀红色区域，缺乏可唯一复核的内部纹理。初始“在叶片上”不等于可持续辨认的物理点。标注人须填写材料点定义，复核人确认定义及其与旧查询的对应关系。若无法确认旧查询对应的材料点，该点不得进入 GT 评价。若选择更明确的色带交界、划痕或其他内部外观点，应建立新点 ID 和查询配置，然后在同一新查询上重跑网格开/关，不能用新叶尖 GT 偷换旧查询的身份。

下一批点应包含三片叶片各自的叶尖/独特内部表面标记，以及塔身可辨认固定标记或稳定背景参照。轴心可作几何参考，但不能替代非旋转材料参照。现有光滑表面没有可可靠辨认的内部标记时，保留待补数据状态，不能强行选择。

## 标注字段与状态

顶层：`schema_version=1`、`annotation_kind=human_visual`、视频 SHA/ID、原图与处理图尺寸、坐标空间、时间基准、帧表/图像哈希、点定义、人工案例复核列表。每条记录按 `(frame_index, point_id)` 唯一定位：

| 字段 | 含义 |
|---|---|
| frame_index / timestamp_seconds | 原始帧号 / 该帧解码视频时间 |
| point_id | 固定材料点或明确定义的参考 ID |
| x / y | 960×540 图中像素中心坐标，可为小数；左上为 (0,0)，有效范围 x∈[0,959]、y∈[0,539] |
| state | visible / occluded / out_of_frame / not_visible_unknown / uncertain / pending |
| occluded / out_of_frame | 与状态相符的布尔值；原因或身份不明时为 null |
| uncertainty | none / identity_ambiguous / low_precision / pending |
| source | human_visual 或 unannotated；不接受 model_prediction |
| annotator / reviewer | 实际标注人与复核人身份/代号 |
| review_status / reviewed_at_utc | pending / draft / reviewed，以及实际复核时间 |
| note | 原始外观线索、不确定原因、遮挡证据 |

不可见状态的 x/y 必须为 null，不推测不可见位置。身份不明与“确实不可见”不同：uncertain 条目排除定位和可见性指标；无法说明不可见原因但确知身份和不可见性时可用 not_visible_unknown。点定义未确认、未复核、非人工来源、不确定条目全部排除。声明复核者和时间只是可追溯的人工声明，软件不能替代实际复核。

## 标注顺序与复核

1. 独立查看原视频/原图，先定义 A/B/C 和具体材料位置，完成初始查询对应确认。首次定位不显示预测。
2. 先标 0,10,20,…,110,119，覆盖完整 120 帧时间范围，不凭三片周期重复认定材料身份恢复。
3. 重点加密 28、36、52、54、55、68 的前后 ±3 帧；另补 56–63、73–80 及 83、89、91、105、119 前后。前一组来自预测状态变化，后一组来自实际跳变/返回候选。这只指导检查时段，不提供 GT 位置。
4. 条件允许再补全 120 帧。对未知材料点不插值，不用运动轨迹或模型预测补标签。
5. 第二次按时间顺序看原帧复核身份、状态、位置，再逐条签名。优先另一位复核人；若同一人第二遍复核，记录此限制，不能虚构独立复核。
6. 修改已复核点坐标/状态撤销该条复核；修改点定义撤销该 ID 的全部复核。下载带版本名的 JSON；保留上一版、哈希、复核说明。
7. 冻结同一份人工 GT 后再展示预测叠加，形成独立 `case_reviews`。身份切换/恢复必须有具体图像证据、关联的物理点 ID、结果配置 SHA 和人工复核记录。不得回改 GT 去追随预测。

## 评价与统计分母

协议在 `configs/evaluation/stage1_protocol.json`。键为视频 SHA、原始帧号、物理点 ID、PTS 和处理分辨率；稀疏标注不插值，未标条目 `valid=False`。对齐导出可复用 Stage 0 的 NPZ 接口。

定位 EPE 使用全部已复核、无不确定性、GT 可见且查询已生效的条目，不因预测不可见而删除。有限预测计算均值/中位数/P90；非有限预测单独计数并在位置正确率中算失败。阈值 ≤2/5/10/20 处理像素，正确率分母为上述全部 GT 可见条目。可见性 P/R/F1 分母来自全部可见性已确认的复核条目，报告 TP/FP/FN/TN；零分母为 null。

失跟定义：GT 可见时预测不可见、非有限，或 EPE>20 像素。只报告首个**已标帧**的失跟观察时间与此前已标正确帧。稀疏跨帧遮挡/恢复只给区间，不认定精确起始时间和全时段恢复率。几何回到阈值内不等于材料身份恢复。

例：在项目根目录设置 `PYTHONPATH` 为本项目 `src`，使用现有 Python 环境执行：

```powershell
python -B scripts/evaluate_stage1.py --run experiments/runs/stage1_20261013_controlled/measured_02_on --gt experiments/annotations/stage1_wind_120/wind_gt_human.json --raw-frames experiments/annotations/stage1_wind_120/raw --output experiments/stage1/evaluation_on_reviewed_v1
python -B scripts/evaluate_stage1.py --run experiments/runs/stage1_20261013_controlled/measured_03_off --gt experiments/annotations/stage1_wind_120/wind_gt_human.json --raw-frames experiments/annotations/stage1_wind_120/raw --output experiments/stage1/evaluation_off_reviewed_v1
python -B scripts/compare_stage1_gt.py --on experiments/runs/stage1_20261013_controlled/measured_02_on --off experiments/runs/stage1_20261013_controlled/measured_03_off --gt experiments/annotations/stage1_wind_120/wind_gt_human.json --output experiments/stage1/gt_comparison_reviewed_v1.json
```

输出目录必须新建，不能覆盖既有运行。`evaluate_stage1.py` 保存评价 JSON、逐点帧 CSV、对齐且经复核门槛筛选的 GT NPZ，以及仅有已复核 GT 帧的绿 GT / 红预测叠加图。当前空白文件评价覆盖 0/480、各真实 GT 指标 null，符合不得冒充真实 GT 的边界。
