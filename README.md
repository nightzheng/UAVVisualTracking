# UAVVisualTracking

UAVVisualTracking 是旋转风机叶片局部视觉巡检的科研工作台。它在 Gazebo 中采集单目 RGB，将视觉观测与已登记的叶片材料区域关联，并使用反应式或恒速预测控制固定站位或本体–云台联合运动。系统记录分段拍摄回执，再用独立几何评价照片质量与表面覆盖。

| 功能 | 用途 |
|---|---|
| 转子曲线编辑 | 编辑连续角速度关键点、预览解析积分相位、导入和导出 JSON |
| 实验调度 | 运行冻结配置或开发配置，查看进度和日志，取消运行 |
| 结果复核 | 对照策略、浏览巡检照片与覆盖结果、下载实测轨迹和 1080p Demo |

## 运行条件

工作台需要 Python 3.10+（推荐 3.12）和 Node.js 22.12+（22 系列）。回放和新采集还需要带 libx264 的 FFmpeg。新采集使用 Ubuntu 24.04 系统 Python、ROS 2 Jazzy、Gazebo Harmonic、Transport 13、Msgs 10 和可用的 Ogre2 渲染环境。Windows 通过 WSL2 运行 Gazebo，宿主机运行工作台与结果评价。原生 Ubuntu 可以直接运行两部分。

| 复现层级 | 所需内容 |
|---|---|
| 1 · 安装、单元测试和前端构建 | 本仓库和软件依赖 |
| 2 · Web 配置编辑 | 前端构建和本地服务，可编辑、校验和导出曲线 |
| 3 · 冻结回放 | 对应的真实 U3-G 冻结记录、参考网格和 FFmpeg |
| 4 · 新 Gazebo 采集 | Gazebo、巡检插件、场景网格与纹理、冻结输入锁、有效的 C1 材料登记和独立评价网格 |

第 3、4 层依赖未公开分发的研究资产。有权限的研究人员可使用现有冻结归档，并按[工作台指南](docs/workbench/README.md)核验输入。仓库自带的[开发示例](configs/workbench/development.json)可以直接导入编辑器并验证。实际采集仍需第 4 层资产。

## 快速开始

在仓库根目录执行。Windows 使用已初始化 Conda 的 PowerShell：

```powershell
conda env create -f environment.yml
conda activate uavvisualtracking
python -m pip install -e ".[workbench]"
npm --prefix workbench-ui ci
npm --prefix workbench-ui run build
uav-workbench serve --distro Ubuntu-24.04
```

Ubuntu 24.04 使用系统 Python 创建工作台环境。先准备 Node.js 22.12+，如需编码则准备带 libx264 的 FFmpeg：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[workbench]'
npm --prefix workbench-ui ci
npm --prefix workbench-ui run build
uav-workbench serve --linux-python /usr/bin/python3
```

原生 Ubuntu 如需执行新采集，先在启动服务的同一终端加载 `source /opt/ros/jazzy/setup.bash`，并按[系统依赖和插件构建说明](docs/workbench/README.md#gazebo-环境与插件)准备 Gazebo。Windows 宿主机的 Conda 环境仅用于工作台，WSL 内的 ROS/Gazebo 绑定使用 Ubuntu 系统 Python。

打开 [本机工作台](http://127.0.0.1:8765)，选择场景与策略。双击曲线新增关键点，拖动或用表格精确编辑。导入配置后可预览速度与相位。具备资产时点击运行，完成后在结果区查看覆盖、未完成原因及视频。服务默认仅监听回环地址。

## 命令行与结果

确认已满足第 4 层条件后，运行一条与冻结场景不同的 8 秒开发曲线：

```sh
uav-workbench --version
uav-workbench run --config configs/workbench/development.json
```

默认在 `.local/u4-results/<实验 ID>/` 保存配置、状态、日志、结果与 Demo。新采集写入本次 RGB、遥测及回执，并核对实测运动。冻结回放读取已有真实记录，重新计算评价和视频。任务完成表示处理链结束，巡检是否通过及完成区域数由独立评价单独报告。

第 1 层测试命令：

```sh
python -m unittest discover -s tests -p 'test_public*.py'
python -m unittest discover -s tests -p test_workbench.py
```

## 研究边界与历史基线

当前 C1 材料绑定以可信初相位和预登记的单片材料区域为条件。评价使用已知几何和静态参考网格，现场输入是真实 Gazebo RGB，执行采用运动学模型。结果支持相同条件下的策略对照。跨叶片无锚点巡检、实机飞行动力学与 PX4 尚未验证。

CoTracker3、BootsTAPIR、Stage7C 等历史基线继续保留。推理、评测入口与证据见[历史视觉跟踪基线](docs/baselines.md)和[研究状态](docs/RESEARCH_STATUS.md)。

## Citation 与 License

软件引用可使用 *nightzheng. UAVVisualTracking, version 0.1.0 (2026)*，并附[项目仓库](https://github.com/nightzheng/UAVVisualTracking)和实际使用的提交哈希。

项目原创代码采用 [MIT](LICENSE)。第三方代码、模型和素材适用各自条款，详见[第三方声明](THIRD_PARTY_NOTICES.md)。
