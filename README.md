# sonic_mujoco

面向 SONIC-compatible Unitree G1 的轻量 MuJoCo 遥操作工程。项目包含 G1
仿真、SONIC encoder/decoder、PICO 全身追踪、机器人第一视角视频回传，以及
Sweep 场景。实现保持单进程控制主循环，不引入额外框架。

## 安装

```bash
cd ~/lst/sonic-mujoco
~/.local/bin/uv sync --extra sonic --extra teleop --extra recording
.venv/bin/python scripts/setup_xrobotoolkit.py
```

最后一条命令把 XRoboToolkit SDK 复制到项目的 `.xrobotoolkit/`。该目录不进入
Git。Ubuntu 上仍需安装并保留 `/opt/apps/roboticsservice`；PICO 链路运行时不再
读取旧 GR00T 工程。头显中仍使用 XRRobotKit 客户端。

ONNX 模型不提交到 Git。默认从下面的位置读取：

```text
~/lst/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/
```

也可以通过 `--encoder` 和 `--decoder` 指定其他路径。

## PICO 遥操作

1. 让 PICO 和 Ubuntu 主机处于同一个局域网。
2. 在 PICO 的 XRRobotKit 中填写 Ubuntu 主机地址 `192.168.3.29` 并连接。
3. 在 Ubuntu 上运行：

```bash
cd ~/lst/sonic-mujoco
DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  .venv/bin/python scripts/run_pico_teleop.py --scene sweep
```

默认行为包括：

- 直接启动 XR 服务并读取 PICO 的 24 个身体关节；
- 将真人全身姿态转换为 SONIC 的 SMPL 输入，以 50 Hz 控制 MuJoCo G1；
- 从模型中的 `head_camera` 渲染画面；
- 在 TCP `13579` 端口响应 XRRobotKit，并向头显回传低延迟 H.264 双目画面。

连接后使用与真机一致的核心操作：

- `A+B+X+Y`：启动或停止控制；
- `A+X`：在待机与真人全身 `POSE` 遥操之间切换；
- 按住左菜单键：暂停动作跟随，松开后恢复；
- 左 `grip+A`：开始或结束一段数据录制；
- 左 `grip+B`：放弃当前录制。
- 左 `grip+X`：重置场景、SONIC history 和 controller；若正在录制则放弃该条。

这里不使用 locomotion Planner 或摇杆行走。人的迈步、转身、抬腿和手臂动作会
直接成为 G1 的参考动作。启动后先按 `A+B+X+Y` 进入待机，再按 `A+X` 开始
全身遥操。

录制过程中仍可用 `A+X` 暂停 G1 跟随，再次按下后恢复，当前 episode 不会结束。
暂停期间会继续记录 G1 保持动作与变化中的 PICO 参考，因此不建议在正式数据中
间暂停。头显左上角会显示红色 `REC` 和当前录制时长。

录制结果默认保存到 `records/<采集时间>/`，可通过 `--record-dir` 修改根目录。
目录格式与真机采集一致：`meta/` 保存 schema 和 episode 索引，`data/chunk-000/`
保存 Parquet，`videos/chunk-000/observation.images.ego_view/` 保存第一视角 MP4。
每个控制帧包含完整 MuJoCo `qpos/qvel/ctrl`、PICO SMPL 参考、SONIC token、策略
动作和手柄输入；Sweep 物体状态包含在完整 `qpos` 中。

一次启动可连续采集多条 episode：用 `grip+A` 保存当前条，使用 `grip+X` 重置，
再用 `grip+A` 开始下一条。每次保存后终端会打印帧数和视频时长。

MuJoCo 接触不是模拟 JuQiao 通道，而是在每个物理子步读取 G1 与场景的接触，
再汇总到 50 Hz 控制帧。数据字段 `observation.contact.*` 包含机器人部位、被接触
物体、世界坐标、最大法向/切向力、累计法向冲量和采样次数。每帧按冲量保留最
重要的 16 组部位—物体接触；body ID 对应名称保存在 `meta/info.json`。

每条 episode 结束后还会生成：

- `previews/episode_XXXXXX.html`：视频、曲线和主要接触部位汇总，直接打开即可；
- `previews/episode_XXXXXX_contact.svg`：接触力曲线和接触帧时间轴，可直接打开；
- `previews/episode_XXXXXX_contact.json`：最大力、累计冲量和主要接触部位摘要；
- `videos/.../episode_XXXXXX.mp4`：与 Parquet 同帧数的机器人第一视角录像。

Parquet 保留包括脚—地面在内的全部机器人—场景接触；预览会排除 `world` 地面
支撑力，以免站立重量掩盖手臂、桌面和任务物体的接触曲线。

只调试数值、不需要落盘视频时可加 `--no-record-video`。该参数不影响 PICO 里的
实时画面；`--no-pico-video` 与数据集视频也是两个独立开关。

只在头显里看画面、不显示 Ubuntu viewer：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_pico_teleop.py --scene sweep --headless
```

`--headless` 不会关闭 PICO 画面。仅调试姿态、暂时关闭视频可加
`--no-pico-video`。若端口冲突，可用 `--video-listen 0.0.0.0:其他端口`，并同步
修改 XRRobotKit 中的连接端口。

旧 GR00T PICO manager 仍可作为兼容路径使用：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --endpoint tcp://127.0.0.1:5556 --no-pico-video
```

## Sweep 场景

Sweep 桌面按真机布置由蓝色胶带分成左右两侧。杯子、齿轮、白卡、纸板和小鸭
出生在同一侧；将五个物体全部扫过中线后，终端显示 `Sweep task completed.`。
`env.reset(seed=7)` 可得到可复现的初始位置。

## 代码边界

- `envs/mujoco`：MuJoCo 生命周期、G1 状态和 PD 控制；
- `teleop/pico_direct.py`：XR SDK 加载、SMPL 前向运动学和 50 Hz 姿态采样；
- `teleop/control.py`：启停与真人 `POSE` 模式切换；
- `recording.py`：对齐保存 MuJoCo、PICO 和 SONIC 控制帧；
- `teleop/pico_video.py`：MuJoCo 相机渲染和共享帧；
- `scripts/pico_video_bridge.py`：XRRobotKit 控制协议、H.264 编码和 TCP 回传；
- `controllers/sonic`：与旧 C++ deploy 数值对齐的 encoder/decoder；
- `envs/mujoco/g1/sweep_env.py`：Sweep reset、状态和成功条件。

四元数统一使用 `wxyz`。`RobotState` 和 `RobotCommand` 使用硬件/MuJoCo joint
order，SONIC 内部显式转换为 IsaacLab order。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests
```

测试覆盖模型、PD 控制、SONIC C++ 等价性、PICO 协议、直接姿态转换和视频控制
协议。视频桥还可以在无头显时通过本机回环完成 H.264 编码链路自检。
