# sonic_mujoco

面向 SONIC-compatible Unitree G1 的最小 MuJoCo 仿真工程。

当前只包含 Phase 1：加载 G1、空场景、reset、step 和 viewer。SONIC、DDS、PICO、相机与数据采集尚未接入。

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
- `g1_env.py`：G1 模型与 SONIC 29-DoF 映射校验。
- `empty_env.py`：只选择空场景。
- `assets/mujoco`：机器人定义、必要 meshes 和场景 XML。

新增任务时应增加独立的 scene 和薄 task class，不修改 G1 关节语义。
