# XRRobotKit 触觉补丁

官方 XRRobotKit APK 尚未处理手柄振动命令。构建 APK 前，将同目录补丁应用到
上游 Unity 客户端：

```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-Unity-Client.git
cd XRoboToolkit-Unity-Client
git checkout c932609
git apply /path/to/sonic-mujoco/pico_client/xrobotoolkit-haptics.patch
```

使用 Unity `2022.3.16f1` 构建 Android APK，再执行 `adb install -r <apk>`。
补丁只增加 `HapticImpulse` 命令处理，不改变追踪和视频逻辑。
