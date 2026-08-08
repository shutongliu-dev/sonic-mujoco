# sonic_mujoco

面向 SONIC-compatible Unitree G1 的轻量 MuJoCo 仿真工程。

当前包含 G1 仿真、统一状态/命令接口、与旧 C++ deploy 数值对齐的 SONIC
encoder/decoder，以及 PICO protocol-v3 姿态流接入。相机和数据采集尚未迁移。

## 安装

```bash
cd ~/lst/sonic-mujoco
~/.local/bin/uv pip install --python .venv/bin/python -e '.[sonic,teleop]'
```

ONNX 模型继续由调用者提供，不复制进本仓库。默认运行命令读取旧工程的
`gear_sonic_deploy/policy/release`。

## 基础仿真

```bash
.venv/bin/python scripts/run_sim.py
.venv/bin/python scripts/run_sim.py --headless --steps 10
```

从 SSH 会话把 viewer 显示到 Ubuntu 桌面：

```bash
DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  .venv/bin/python scripts/run_sim.py
```

## PICO 驱动 SONIC

当前直接复用旧工程的 PICO manager 作为硬件采集进程。先在一个终端启动它：

```bash
cd ~/lst/GR00T-WholeBodyControl
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

再在另一个终端启动 MuJoCo：

```bash
cd ~/lst/sonic-mujoco
DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  .venv/bin/python scripts/run_pico_teleop.py
```

进入 PICO manager 的 `POSE` 模式后，终端会显示
`PICO stream received; SONIC control is running.`，G1 将跟随人体姿态。当前这条路径只接入
SONIC 的 SMPL 姿态模式；planner 导航和手指执行器暂未接入。

## Sweep 场景

```bash
DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  .venv/bin/python scripts/run_pico_teleop.py --scene sweep
```

场景包含一张桌子、三个可移动物体和桌面上的绿色目标区。将三个物体全部扫入目标区后，
终端会显示 `Sweep task completed.`。

```python
state = env.get_scene_state()
print(state.object_position, state.object_quaternion, state.success)
```

`env.reset(seed=7)` 可以得到可复现的物体初始位置。

## 数据边界

- `envs/mujoco`：MuJoCo 生命周期、G1 状态和 PD 控制。
- `teleop/TeleopCommand`：与输入设备无关的参考姿态批次。
- `teleop/PicoTeleop`：只负责读取和校验旧 PICO v3 消息。
- `controllers/sonic/encoder.py`：10 帧滑动窗口和 1762 维 encoder 输入。
- `controllers/sonic/controller.py`：994 维 decoder 输入和动作映射。
- `envs/mujoco/g1/sweep_env.py`：Sweep reset、场景状态和成功条件。

四元数统一使用 `wxyz`。`RobotState` 和 `RobotCommand` 使用旧仿真 `lowstate`
的 hardware/MuJoCo joint order；SONIC 内部显式转换到 IsaacLab order。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests
```

测试会在参考工程可用时编译旧 C++ TensorRT oracle，同时验证 encoder 和 decoder
的输入布局与推理输出。
