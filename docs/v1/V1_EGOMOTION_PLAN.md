# V1 自运动补偿实验设计与接口

更新于 2026-10-13。**未实现从背景估计运动的完整 V1，未验证真实三维自运动恢复或米制目标运动。** 本周只实现已知二维坐标变换契约验证。原 Stage 0 方案完整保存在 [history/V1_EGOMOTION_PLAN_stage0_20261013.md](history/V1_EGOMOTION_PLAN_stage0_20261013.md)。下列初步框架保留；本周详细实验与准入见后文。

## 输入、假设和接口

读取V0源帧/PTS/Point ID、处理坐标、预测可见性、query及生成时刻，保留未补偿轨迹。背景ROI/身份须与动态目标分离；相邻帧需有限/可见、空间分布充足的静态背景对应。对应可来自CoTracker或独立背景特征匹配，记录来源与质量。

可选姿态/IMU/VIO需要时间戳、相机内参/畸变、外参及同步误差。姿态辅助分离旋转光流/参考方向，VIO提供相机相对运动；只有姿态/单目像素轨迹不能得到米制目标速度，还需深度、平移尺度和不确定性。

## 基线次序和输出

| 候选 | 假设 | 输出 |
| --- | --- | --- |
| 背景共同平移 | 小运动、近似共同二维移动 | robust平移、背景有效数/内点/残差、状态 |
| partial-affine + RANSAC | 平移+图像旋转+统一尺度近似 | 2×3变换/齐次H、空间覆盖、残差、不可逆标记 |
| 单应性 | 近似平面或纯旋转等条件 | H、条件数/空间覆盖、误差/失败状态 |
| 已知姿态 | 标定/同步有效，有旋转信息 | 旋转参考坐标/残差；平移与深度仍待建模 |

旧Translation/partial-affine仅作独立候选参考，固定参数比较，不把旧无GT背景残差当新精度证据。单应性并非通用补偿：强视差、非平面景深、大面积动态前景、背景不足时可能失效，内点多不等于模型正确。

H_t将t−1映至t，G_t=H_t @ G_(t−1)，query锚点G_q=I，p_ref,t=inv(G_t) @ p_t。输出与V0帧/点键对应：raw_xy、compensated_xy、reference_frame、transform、valid/failure_reason、背景统计、生成时刻/可用末帧。首次缺口后累计变换无效，单步恢复不自动填补缺口；重新锚定必须新reference ID。**参考坐标是像素参考系，不是米制世界坐标。**

## 数据、评价与失效

- 现有人物/城市可做无GT先导，Pexels单条来源仍待登记。当前全转子风机素材不能代替真实无人机近距离局部叶片数据。
- 数据需分相机静止/平移/旋转、目标静止/运动、弱纹理/遮挡/背景不足、显著视差。开发集与冻结测试集分开，预先固定ROI/查询、参数/失效规则。
- 有相机GT：标定同步后评旋转/相对平移方向；有深度尺度再评米制平移。有目标GT则评参考系重投影/定位误差、失效识别与恢复延迟。
- 无相机GT：只报告held-out静态背景残差、空间覆盖、失败/累计不可用比例与人工观察，不用拟合背景残差下降证明目标真实运动更准。估计背景与诊断背景分离。
- 失效标记覆盖背景数不足、空间集中、残差异常、退化/不可逆、视差、多动态运动、可见性误判和同步偏差。开发集定阈值，冻结测试不反复调参。

未来depth_per_point/uncertainty与独立尺度标定接入，评价单目相对运动和尺度可观测性。暂不实现SLAM/VIO、DA3融合、角速度估计或主动飞控。

## 本周设计依据与坐标方向

已只读核对用户确认的综述 v1.1 §4.3、§6.4、§7.3，文件 SHA `b7260d05466d30235787589bbe6594b0a14b4dbfecd5afa0d10387694c9dbfcb`。二维图像补偿、独立相机运动参考和真实三维恢复分别验证，不能相互替代。

统一列向量 `[x,y,1]^T`，左上原点、x 右、y 下。I_t 是当前帧图像坐标，I_r 是声明参考帧 r 的图像坐标，单位均为像素。V0 裁剪坐标先通过 ROI 平移映到共同源处理图 960×540，原始 1920×1080 坐标另保留；不把官方内部 512×384 坐标直接交给补偿接口。

命名固定为 `previous_to_current`、`reference_to_current`、`current_to_reference`。单步 H 将 t−1 映到 t，累计 G_r→t=H_t−1→t @ G_r→t−1；目标补偿为 inv(G_r→t) @ p_raw,t 后除齐次第三分量。若直接拟合参考帧 r→t，标 `direct_reference_matching`，不可把它误作单步再次连乘。首次累计缺口后不自动恢复；直接参考匹配或新锚点必须另记成功证据/新 reference ID。

OpenCV image warp 默认接收 source→destination 前向矩阵，内部逆采样；本周未使用 WARP_INVERSE_MAP。目标坐标补偿则用矩阵逆。[OpenCV 图像变换文档](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)支持此方向约定。图像插值误差与坐标求逆误差分别统计。

将来独立位姿另声明 T_W_C（camera→world）、内参/畸变、外参、尺度和同步误差。纯旋转且标定有效时，H_r→t=K_t R_Ct←Cr inv(K_r) 可作对照；相机平移涉及深度，不能一般地用同一个 H 恢复全部目标真实运动。裁剪/resize 后 K 必须转换到同一图像坐标。

## 本周输入与输出契约

| 输入 | 必须携带 |
|---|---|
| 帧与时间 | video SHA、源 frame_index、PTS/采集时间类型、分辨率、ROI/resize 矩阵、时钟域/同步误差、结果生成及可用末帧 |
| 背景对应 | bg_feature_id、prev/current xy、观测有效掩膜、来源与静态判据、ROI 和空间覆盖；估计/held-out 诊断 ID 分开 |
| 局部目标点 | point_id/叶片 ID、raw xy、query 来源/帧、active、预测可见性、有限性、人工身份确认/不确定状态；保留原轨迹 |
| 可选独立相机参考 | reference_id、K/畸变、T_W_C 或相对旋转/平移、单位/尺度、协方差、来源和测量时间；未知为 null |
| 配置 | 三类模型之一、参考帧、拟合/诊断划分、固定参数、失败准入和未来信息权限 |

背景可以来自独立特征匹配或点跟踪，但须排除叶片动态点，声明静态依据。来自同一 RGB 模型的观测可能相关，不能当作独立传感器。估计背景与 held-out 诊断背景按 ID 和空间分层预先划分；不拿全部拟合点残差当独立验证。

输出按 `(video_id, source_frame_index, point_id, reference_id)` 对齐：raw_xy、compensated_xy_px、前向/逆矩阵、model_kind、transform_valid、point_numerical_valid、observation_valid、failure_reason、背景数/内点/覆盖/诊断残差、来源、生成/可用时间和 future_access。矩阵有效、预测可见和材料身份可信分别保留；未知协方差为 null。

observation_valid 至少需 active、输入可见/质量掩膜、有限坐标、变换有效和可用身份状态。数学投影有效不能让隐藏或身份不明的点变成可靠材料测量；保留数值补偿用于诊断时须标明有效性来源。输出不附未经标定的 m、m/s 或真实角速度。

局部算法只用声明输入的背景，不偷偷使用裁剪外全景轮毂/背景。全景辅助若作为额外输入，单列其信息成本。

## 三类估计基线的几何与失败条件

以下是待实现的图像运动估计器，不是本周已验证的算法。数学最少对应数不等于真实准入数，实际需更多空间分布良好的静态对应。

| 基线 | 自由度与假设 | 拟合/评价计划 | 主要退化 |
|---|---|---|---|
| 平移 | 2 DOF；小运动下共同背景位移 | 鲁棒中位或固定鲁棒规则；held-out 位移残差/覆盖率 | 旋转、尺度、深度相关视差 |
| partial-affine | 4 DOF；统一尺度+平面旋转+平移，无 shear | 拟用 estimateAffinePartial2D + 固定 RANSAC；至少两不同对应为数学最低条件 | 强透视、不同深度运动、集中点、误静态标签 |
| 单应性 | 8 DOF；静态近似平面或纯旋转等条件 | 拟用 findHomography + robust 规则；至少四个非退化对应，held-out 重投影/空间分布 | 非平面平移视差、多运动、共线、近不可逆/投影地平线 |

接口依据：[OpenCV 几何估计文档](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。嵌套模型比较需同输入及剔除规则；不因增加自由度或删除困难点使拟合残差下降，就宣称解耦成功。

失败字段至少含 insufficient_background、poor_spatial_coverage、dynamic_foreground_dominant、parallax_model_mismatch、ill_conditioned_transform、projection_horizon、invalid_timestamp_sync、accumulated_gap、identity_unknown。实际数量/残差阈值依据开发数据的真值精度校准后冻结，目前不凭通用文献数字宣布通过。

## 已知二维变换的数学验证

本周实际生成 `experiments/weekly/20261013_v1_known_transforms/`：同一原帧 0 的静态图像，translation、partial_affine、homography 各 6 帧。每帧保存前向/逆矩阵、插值策略、有效图像 mask、20 个程序定义校验点和源图哈希，共 **360 个数学校验点帧**；没有新增人工风机 GT，也没有三组独立真实相机序列。

`geometry/transforms.py` 仅提供已知 H 投影/求逆及累计缺口传播，没有从背景图像求 H。以独立 OpenCV perspectiveTransform 生成当前校验点，再映回参考点，最大求逆误差为平移 **1.27e-13 px**、partial-affine **3.42e-13 px**、单应 **3.42e-13 px**。这些是数值误差，不是运动估计精度。

测试覆盖图像整数平移方向、partial-affine/H 求逆、H 非零尺度等价、NaN/有效掩膜、投影地平线、奇异/病态矩阵、非交换累计顺序和缺口后不得自动恢复。条件数 1e10/齐次分母 1e-10 是当前数值保护，不是现实估计质量阈值；条件数依赖坐标归一化，真实拟合准入另校准。

图像 bilinear warp、边界黑色，mask nearest warp；mask 只表示程序采样是否来自原图，不是叶片可见性。未执行噪声/离群点背景拟合比较，不以零噪声 oracle 求逆证明估计器通过。

复现：src 加入 PYTHONPATH 后，`python -B scripts/prepare_v1_known_transforms.py --image experiments/annotations/stage1_wind_120/raw/000000.png --output <new-directory>`。输出目录必须新建。

## 独立真实运动数据与阶段评价

本周只规划，不操作实体设备。优先静止相机/静态背景对照，用刚性固定和独立姿态测量确认近静止；再用标定相机加位移台/角度编码器或外部追踪采集受控平移/旋转，登记同步、机械回差与参考精度。随后增加多深度背景验证视差失配，再增加动态样件与独立目标角度/身份参考。真实移动相机局部叶片需独立标定/同步/许可齐备；不启动 V2 或无人机实机。

可补充核查 [EuRoC MAV 官方数据页](https://projects.asl.ethz.ch/datasets/euroc-mav/)及 [TUM RGB-D 数据页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)用于接口/静态运动对照，不替代旋转叶片 GT，不把多传感成绩改写为纯 RGB。本周未下载数据或大规模实验。EuRoC 官方也列出参考测量精度/时间同步的限制，使用前须逐序列核查。

评价分三层：**M** 已知二维变换数学契约（本周通过）；**E** 从背景估计二维运动（未实现/未验证）；**R** 独立真实相机/目标运动解耦（数据未准备）。M 通过不能宣称 E/R 通过。

E 固定背景、目标与匹配权限，报告 held-out 重投影均值/中位/P90、有效空间覆盖、失败识别/误接受比例和时间成本。已知合成 H 可评模型误差，但真实材料轨迹仍需人工 GT。真实相机参考存在时另评旋转/相对平移方向，有独立深度尺度再评米制平移；无 GT 仅报告诊断残差/可用性。

R 分组声明纯 RGB、额外相机状态与几何/尺度锚点的信息成本；背景不足、强视差、高动态占比、不可逆、不同步与身份不明须显式失败或保留相对/多解状态。滤波不创造观测，平滑补偿轨迹不能代替真实状态。审阅后再决定小规模 E 实验，本轮停止，不进入完整 V1 或融合算法。
