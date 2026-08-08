# sonic_mujoco

面向 SONIC-compatible Unitree G1 的最小 MuJoCo 仿真工程。

当前包含 Phase 1–3：加载 G1、空场景、状态读取、PD/力矩控制，以及与旧 C++ deploy 数值对齐的 SONIC decoder 数据路径。PICO、相机与数据采集尚未接入。

## 运行

复用 GR00T 的 MuJoCo 虚拟环境：

```bash
cd ~/lst/sonic-mujoco
~/.local/bin/uv pip install --python ~/lst/GR00T-WholeBodyControl/.venv_sim/bin/python --no-deps -e .
~/lst/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/run_sim.py
```

从 SSH 会话把 viewer 打开到已登录的 Ubuntu 桌面：

```bash
DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  ~/lst/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/run_sim.py
```

无窗口验证：

```bash
~/lst/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/run_sim.py --headless --steps 10
~/lst/GR00T-WholeBodyControl/.venv_sim/bin/python -m unittest discover -s tests
```

## 结构

- `env_base.py`：MuJoCo 通用生命周期。
- `g1_env.py`：G1 模型、SONIC 29-DoF 映射和 PD 控制。
- `interface.py`：`RobotState` 与 `RobotCommand`。
- `empty_env.py`：只选择空场景。
- `assets/mujoco`：机器人定义、必要 meshes 和场景 XML。
- `controllers/sonic`：994 维 observation、ONNX decoder 和 action-to-command 映射。

新增任务时应增加独立的 scene 和薄 task class，不修改 G1 关节语义。

`RobotState` 使用旧仿真 `lowstate` 的硬件/MuJoCo joint order；SONIC observation 内部再显式转换到 IsaacLab order。四元数保持 MuJoCo 的 `wxyz` 顺序。`RobotCommand` 中的五个控制向量长度均为 29。

## SONIC decoder

ONNX Runtime 是可选依赖，模型继续由调用者显式提供，不复制进仓库：

```bash
python -m pip install -e '.[sonic]'
```

```python
from sonic_mujoco.controllers.sonic import SonicController

controller = SonicController.from_onnx("path/to/model_decoder.onnx")
command = controller.act(env.get_robot_state(), token_state)
env.step(command, steps=controller.steps_per_action)
```

`token_state` 为 encoder 或外部 teleop 提供的 64 维 token。当前 Phase 3 只迁移 decoder 闭环；PICO/encoder 输入留到后续阶段。
