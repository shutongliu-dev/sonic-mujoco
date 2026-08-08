# sonic_mujoco

面向 SONIC-compatible Unitree G1 的轻量 MuJoCo 遥操作工程。项目包含 G1
仿真、SONIC encoder/decoder、PICO 全身追踪、机器人第一视角视频回传，以及
Sweep 场景。实现保持单进程控制主循环，不引入额外框架。

## 安装

```bash
cd ~/lst/sonic-mujoco
~/.local/bin/uv sync --extra sonic --extra teleop
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

这里不使用 locomotion Planner 或摇杆行走。人的迈步、转身、抬腿和手臂动作会
直接成为 G1 的参考动作。启动后先按 `A+B+X+Y` 进入待机，再按 `A+X` 开始
全身遥操。

录制结果默认保存到 `records/episode_*.npz`，可通过 `--record-dir` 修改目录。
每个控制帧包含完整 MuJoCo `qpos/qvel/ctrl`、PICO SMPL 参考、根四元数、SONIC
token、策略动作和手柄输入；Sweep 物体状态包含在完整 `qpos` 中。

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

Sweep 包含桌子、三个可移动物体和绿色目标区。将三个物体全部扫入目标区后，
终端显示 `Sweep task completed.`。`env.reset(seed=7)` 可得到可复现的初始位置。

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
