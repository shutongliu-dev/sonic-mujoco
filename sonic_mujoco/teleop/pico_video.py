import mmap
import os
from pathlib import Path
import struct
import subprocess
import time

import mujoco
import numpy as np

FRAME_HEADER = struct.Struct("<4sIIII")


class PicoVideo:
    """Render a MuJoCo camera and expose it to the PICO video bridge."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        camera: str = "head_camera",
        listen: str = "0.0.0.0:13579",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._data = data
        self._camera = camera
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._frame_path = Path(f"/dev/shm/sonic_mujoco_pico_{os.getpid()}")
        self._frame_size = 2 * width * height * 3
        self._file = self._frame_path.open("w+b")
        self._file.truncate(FRAME_HEADER.size + self._frame_size)
        self._memory = mmap.mmap(self._file.fileno(), 0)
        self._sequence = 0
        self._period = 1.0 / fps
        self._last_render = 0.0
        self._write_frame(np.zeros((height, width, 3), dtype=np.uint8))

        bridge = Path(__file__).parents[2] / "scripts/pico_video_bridge.py"
        self._process = subprocess.Popen(
            [
                "/usr/bin/python3",
                str(bridge),
                "--frames",
                str(self._frame_path),
                "--listen",
                listen,
            ]
        )

    def render(self) -> None:
        now = time.monotonic()
        if now - self._last_render < self._period:
            return
        self._renderer.update_scene(self._data, camera=self._camera)
        self._write_frame(self._renderer.render())
        self._last_render = now

    def _write_frame(self, frame: np.ndarray) -> None:
        frame = np.concatenate((frame, frame), axis=1)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.nbytes != self._frame_size:
            raise ValueError("video frame dimensions changed")
        self._sequence += 1
        width, height = frame.shape[1], frame.shape[0]
        self._memory[: FRAME_HEADER.size] = FRAME_HEADER.pack(
            b"SMVF", self._sequence, width, height, frame.nbytes
        )
        self._memory[FRAME_HEADER.size :] = frame.tobytes()
        self._sequence += 1
        self._memory[: FRAME_HEADER.size] = FRAME_HEADER.pack(
            b"SMVF", self._sequence, width, height, frame.nbytes
        )

    def close(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._renderer.close()
        self._memory.close()
        self._file.close()
        self._frame_path.unlink(missing_ok=True)
