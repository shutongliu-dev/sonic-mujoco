# sonic_mujoco

`sonic_mujoco` 是面向 Unitree G1 的轻量 MuJoCo 全身遥操作与数据采集工程。
它在一个进程内完成 PICO 全身姿态接收、SONIC 推理、G1 仿真控制、第一视角
回传、接触触觉反馈和 LeRobot 风格 episode 录制。

项目当前提供空场景、Sweep 扫桌、Chair lean 椅背倚靠和 Bucket carry
抱桶搬运任务。实现强调清晰的数据流和较少的运行时组件，适合仿真遥操作、
示教数据采集及 sim-to-real 实验。

## 功能

- MuJoCo 中的 29-DoF G1、PD 控制与 SONIC encoder/decoder 推理；
- PICO 24 关节全身追踪和 50 Hz 真人到 G1 动作映射；
- G1 头部相机到 PICO 的低延迟 H.264/H.265 画面回传；
- 同一进程内连续录制多条 episode，并显示 `REC` 与录制时长；
- 机器人—场景接触的力、冲量和接触部位记录；
- 按碰撞、按压和滑动状态生成左右手柄触觉反馈；
- 接近真实布置、带物理随机化的 Sweep、Chair lean 和 Bucket carry 场景。

## 环境要求

- Ubuntu 22.04 或 24.04；
- Python 3.10 及以上；
- `uv` Python 包管理器；
- 支持 OpenGL/EGL 的 NVIDIA 驱动；
- PICO 头显与手柄；
- XRRobotKit PC Service 和 Python SDK；
- SONIC encoder/decoder ONNX 权重。

项目不提交 SONIC 权重和 XRRobotKit 二进制。准备以下文件：

```text
models/model_encoder.onnx
models/model_decoder.onnx
<XR_SDK>/xrobotoolkit_sdk*.so
<XR_SDK>/lib/libPXREARobotSDK.so
```

XRRobotKit PC Service 需要安装在 `/opt/apps/roboticsservice`。PICO 客户端的独立
构建说明见 [`pico_client/README.md`](pico_client/README.md)。

## 安装

```bash
git clone https://github.com/shutongliu-dev/sonic-mujoco.git
cd sonic-mujoco

mkdir -p models
# 将 model_encoder.onnx 和 model_decoder.onnx 放入 models/
uv sync --extra sonic --extra teleop --extra recording
.venv/bin/python scripts/setup_xrobotoolkit.py <XR_SDK>
```

视频回传依赖系统 GStreamer：

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly
```

先运行无窗口自检：

```bash
.venv/bin/python scripts/run_sim.py --headless --steps 100
```

也可以通过 `SONIC_POLICY_DIR` 改变默认模型目录，或用 `--encoder`、`--decoder`
分别指定模型文件。

## PICO 遥操作

1. 将 PICO 与 Ubuntu 主机接入同一局域网。
2. 在 PICO 客户端中连接 Ubuntu 主机 IP。
3. 在主机上启动 Sweep 场景：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene sweep \
  --pico-device TestDevice
```

`--pico-device` 使用 XRRobotKit 显示的设备名，并用于发送手柄触觉。也可以通过
环境变量 `SONIC_PICO_DEVICE` 设置。只在 PICO 中看画面时可使用 EGL：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_pico_teleop.py \
  --scene sweep --headless --pico-device TestDevice
```

`--headless` 只关闭主机 viewer，不会关闭 PICO 画面。`--no-pico-video` 用于关闭
实时回传；`--no-record-video` 只关闭数据集视频，两者互不影响。

Chair lean 场景使用相同操作流程：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene chair_lean \
  --pico-device TestDevice
```

Bucket carry 场景也使用相同操作流程：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene bucket_carry \
  --pico-device TestDevice
```

### 手柄操作

| 操作 | 功能 |
| --- | --- |
| `A+B+X+Y` | 启动或停止控制 |
| `A+X` | 在待机和全身 POSE 遥操之间切换 |
| 按住左菜单键 | 暂停动作跟随，松开恢复 |
| 左 `grip+A` | 开始或结束当前 episode |
| 左 `grip+B` | 放弃当前 episode |
| 左 `grip+X` | 重置场景、SONIC history 和控制器 |

`grip` 指手柄中间由中指扣动的握持扳机。一次常用的连续采集流程是：

1. 连接主机后按 `A+B+X+Y` 解锁，再按 `A+X` 进入 POSE 遥操；
2. 调整好人与 G1 的位置，按左 `grip+A` 开始录制；
3. 完成动作后再次按左 `grip+A`，保存当前 episode；
4. 按左 `grip+X` 重置场景，再重复步骤 2–3；
5. 当前动作无效时按左 `grip+B`，直接丢弃该 episode。

项目不使用摇杆规划行走。操作者的迈步、转身、下蹲和手臂动作直接构成 G1 的
参考动作。录制中使用 `A+X` 不会结束 episode，但暂停片段通常不适合作为训练
数据。

## 数据采集

录制默认写入 `records/<session>/`：

```text
records/<session>/
├── meta/                 # schema、episode 索引和 body 名称
├── data/chunk-000/       # Parquet 控制帧
├── videos/chunk-000/     # G1 第一视角 MP4
└── previews/             # episode 与接触数据预览
```

每个控制帧包含 MuJoCo `qpos/qvel/ctrl`、PICO 姿态、SONIC token、策略动作、
手柄输入和最多 16 组重要接触。接触字段包括机器人部位、场景物体、世界坐标、
法向/切向力、法向冲量和物理采样次数。

一次进程可连续采集多条 episode。结束一条后终端会输出文件位置、帧数和时长；
保存期间 PICO 持续收到最后一帧，画面不会因编码和落盘而断流。

## 接触触觉

- 左右手臂接触分别反馈到对应手柄；
- 胸腹、腰和骨盆接触同时反馈到两个手柄；
- 突然碰撞、持续按压和切向滑动使用不同的强度、时长和频率；
- 腿脚接触不触发手柄，避免站立和行走产生持续噪声。

触觉来自 MuJoCo 接触求解结果，不模拟特定实体触觉皮肤。相同的接触数据会写入
episode，可用于训练、回放和后续传感器对齐。

## Sweep 场景

Sweep 场景按真实采集桌面布置：木纹桌面由蓝色胶带纵向分区，杯子、齿轮、
白卡、纸板和小鸭位于同一侧。桌子是带质量、摩擦和顺应性的自由刚体，明显
碰撞会产生移动与晃动；任务物体的质量、摩擦和初始位姿会在 reset 时小范围
随机化。

## Chair lean 场景

Chair lean 按参考椅的 56 cm 宽、54 cm 深、35 cm 座高和 66 cm 总高搭建。椅子
是约 11 kg 的自由刚体，脚垫具有较高摩擦，轻微碰撞会产生真实反作用，强碰撞
仍可能移动或倾倒。椅背采用柔顺接触模型；reset 时小范围随机化椅背宽高、倾角、
软硬度、椅子质量与摩擦，以及 G1 的初始距离、朝向和左右偏移。织物外观使用
Poly Haven 的 CC0 Terlenka 贴图，视觉表面与稳定的简化碰撞体相互独立。

场景不自动判定成功或结束 episode，操作者根据稳定倚靠状态控制录制。

## Bucket carry 场景

Bucket carry 使用两张相同的 90 × 60 × 76 cm 折叠桌，水桶从起始桌搬到带蓝色
区域的目标桌。桌子和水桶都是自由刚体，碰撞会产生反作用、晃动或位移。水桶按
常见 18.9 L / 20 L 饮水机桶的比例搭建，外观与由圆柱、椭球组成的稳定碰撞体分离。

reset 会随机化桶的直径、高度、5–10 kg 验证重量、重心偏移、摩擦和初始姿态，
以及两张桌子的质量与摩擦。待遥操作稳定后，可将重量范围校准到满桶约 19–20 kg。
胸部、上臂和前臂的接触会进入数据集，并按左右接触部位映射为 PICO 手柄震动。
场景不包含自动成功判定。

CGTrader 的免费水桶模型没有随仓库分发：其 Royalty Free 许可允许集成使用，但不允许
以可独立提取的原始资产形式重新发布。默认程序化外观无需额外下载，也能保持碰撞稳定。

## 代码结构

```text
sonic_mujoco/
├── controllers/sonic/        # observation、encoder 和 decoder
├── envs/mujoco/g1/           # G1 环境、PD 控制与任务场景
├── teleop/                    # PICO 姿态、控制状态、视频和触觉
├── contact.py                 # 物理子步接触汇总
└── recording.py               # episode、视频和预览写入
scripts/
├── run_sim.py                 # 最小仿真自检
├── run_pico_teleop.py         # 遥操作与采集入口
├── pico_video_bridge.py       # XRRobotKit 视频协议与编码
└── setup_xrobotoolkit.py      # 安装本地 XR SDK
```

四元数统一使用 `wxyz`。硬件/MuJoCo joint order 与 SONIC 内部 order 的转换均在
边界处显式完成。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests
uvx ruff check .
```

测试覆盖 G1 模型、PD 控制、SONIC observation/action、PICO 协议、姿态转换、
视频回传、连续录制和触觉映射。依赖外部 C++/TensorRT 参考实现的等价性测试会
在未配置相应环境变量时自动跳过。
