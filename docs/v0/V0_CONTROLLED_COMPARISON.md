# V0 support grid 受控比较（Stage 1）

2026-10-13。完成的是**真实离线运行、暖机后的速度比较、独立资源采样与评价协议对齐检查**。人工 GT 未提供；真实定位误差和身份指标待评价，不选择最佳配置。

## 固定条件与运行证据

真实运行目录：`experiments/runs/stage1_20261013_controlled/`。`protocol.json` 记录 10 次运行的 manifest/tracks 哈希，`gpu_samples.jsonl` 保存逐条采样及进程/适配器实例名称。不是 Stage 0 或旧小组预测缓存。

固定视频 SHA `9dc01e3f3ebd7400ec71685e7660488b8aa630888dd7caec4921c734049556eb`，原始帧 0–119，处理图 960×540，查询帧 0，四个查询与 Stage 0 相同。官方版本 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`，权重 SHA `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`。offline=True、window_len=60、6 iterations、seed=0、无反向跟踪。网格开启使用官方公共查询接口添加 6×6 支持点，关闭使用固定版本 sparse 接口不添加支持点；官方模型源码和架构未改。

单一已加载模型、同一解码输入、同一进程。先各暖机一次，随后 A B B A / B A A B（A=网格开），每种配置 4 次计时。没有调整分辨率、帧数、精度或 allocator；不调用 empty_cache。两种配置均启用相同独立采样器。计时含输入张量构造/上传和模型，CUDA 同步；不含加载、解码、CPU 结果下载与文件导出。离线每次使用完整片段未来信息，表中的吞吐不是在线因果系统帧率。

## 暖机后速度结果

| 配置 | 有效计时次数 | 中位时间 / 120 帧 | 最小–最大时间 | 由中位时间换算吞吐 |
|---|---:|---:|---:|---:|
| support on | 4 | 3.739815 s | 3.728980–3.915927 s | 32.087 FPS |
| support off | 4 | 3.680110 s | 3.645892–3.692412 s | 32.608 FPS |

两次暖机为 9.316343 s（on）与 3.957144 s（off），全部排除。on 的中位耗时约高 1.62%；仅一次机器会话、每种四次，不给显著性结论，也不推广为一般性能优势。Stage 0 的首轮冷启动 9.985 s 与后续 4.312 s 不再用作配置速度结论。同配置重复的坐标最大差为 0，可见性完全相同。

逐次计时图保留本地 `experiments/stage1/evidence_v2/controlled_latency.png`，公开数值见 [Stage 1 摘要](../../experiments/summaries/STAGE1_SUMMARY.json)。关闭网格预测可见率更高，但此数值不代表准确率。

## GT 比较状态与可复用入口

`experiments/stage1/gt_comparison_pending.json` 已校验两种结果的帧、PTS、点 ID、查询数组、视频与模型身份及除 support_grid 外的配置完全相同；两种结果使用同一个空白人工标注 JSON SHA。当前正式 GT 覆盖 **0/480 点帧**，EPE 均值/中位数/P90、2/5/10/20 px 正确率、可见性 P/R/F1、真实失跟时间、确认身份切换与恢复率均不能报告。

每种评价文件均标记 `awaiting_human_annotation_and_review`。人工 GT 后复用 `scripts/evaluate_stage1.py` 与 `scripts/compare_stage1_gt.py`（命令见标注规范），保存逐帧误差、指标和绿 GT/红预测图，并为每种配置独立复核错误事件。不得补零误差、以查询点自一致结果替代 GT，或由稀疏标注推断整段恢复率。

## 独立 GPU 内存采样

显卡 RTX 5070 Ti，驱动 610.62，设备报告物理容量 16303 MiB（15.921 GiB）。采样器以真实推理进程 PID 筛选 Windows GPU Process Memory，另读 GPU Adapter Memory 和 nvidia-smi 设备 FB 用量，共 **168 条，0 错误**。设置的循环额外等待为 0.2 秒，实际采样起始间隔中位为 **0.299 秒**；查询耗时中位 0.085 秒，最大 0.846 秒。每次有效推理有 12–13 条过程标签采样。采样窗口内读数并非同一瞬间，可能遗漏瞬时峰值。

| 指标 | 观测 | 含义及边界 |
|---|---:|---|
| PyTorch peak allocated | on 17.584660 GiB；off 17.584659 GiB | 当前计时轮张量分配器峰值；不是专用物理显存驻留证明 |
| PyTorch peak reserved | 两种 20.582031 GiB | 同一暖进程缓存分配器管理的空间，跨轮缓存保留 |
| Windows 进程 DedicatedUsage 采样峰值 | 每个计时轮 12.970154 GiB | Windows 独立专用内存用量报告，非逐分配物理驻留追踪 |
| Windows 进程 SharedUsage 采样峰值 | 每个计时轮 7.955078 GiB | Windows 独立共享内存用量报告；保留缓存/上下文环境，不能全归属于当前活跃张量 |
| Windows 进程 TotalCommitted 采样峰值 | 每个计时轮 20.925232 GiB | 进程提交量，与实际专用驻留含义不同 |
| nvidia-smi 设备 FB used | 全过程 2394–15335 MiB | 包括桌面及其他进程；全过程最大值不是此进程独占峰值 |

内存过程曲线保留本地 `experiments/stage1/evidence_v2/gpu_memory_samples.png`，公开数值见上述摘要。每个计时轮相同的 OS 峰值与同进程长期缓存、格式化计数器刷新有关，不能用这些相同读数得出两种配置独立驻留峰值严格相等。

[PyTorch 2.11 官方文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)区分张量占用与缓存分配器保留空间；[Microsoft WDDM residency 文档](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/driver-residency-in-wddm-2-0)描述驻留预算和内存管理；[Microsoft 已知计数器问题](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value)提醒某些 GPU 进程内存报告可能不准确。[NVIDIA 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)说明 WDDM 的显存由 Windows KMD 管理，nvidia-smi 的进程级 GPU memory usage 在 WDDM 下不可用。

因此，超过物理容量的 allocator 数字与同时采到共享用量相容，不能称“实际占用了超过 16GB 的专用显存”。本轮有独立专用/共享/设备级用量证据，**尚未取得逐分配 residency 或 page-fault/迁移 trace**，不证明每个活跃 tensor 的存储位置，也不证明共享内存就是速度差的原因。

## 优化建议的证据门槛

暂不变更正式基线。若以后评估缩短片段、减小图像、混合精度或其他内存方案，先独立列为候选配置：预计可降低输入/特征/注意力缓存成本，但可能改变小标记可辨性、长时身份关联、数值误差及未来上下文。必须在冻结 GT 上同查询对比，并另做独立进程/缓存策略一致的资源实验；不能为了使数字小于 16GB 静默改变现有配置。
