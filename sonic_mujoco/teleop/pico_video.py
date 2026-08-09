import mmap
import os
import struct
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

FRAME_HEADER = struct.Struct("<4sIIII")

# Minimal 5x7 font keeps the live overlay dependency-free.
_GLYPHS = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 0, 0b00100),
    ":": (0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110),
    "6": (0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110),
    "C": (0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
}


def _recording_overlay(frame: np.ndarray, elapsed_seconds: float) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8).copy()
    elapsed = min(max(float(elapsed_seconds), 0.0), 5999.9)
    minutes = int(elapsed // 60)
    seconds = elapsed - minutes * 60
    label = f"REC {minutes:02d}:{seconds:04.1f}"
    scale, padding = 2, 6
    width = len(label) * 6 * scale - scale + 2 * padding
    height = 7 * scale + 2 * padding
    image[12 : 12 + height, 12 : 12 + width] = (190, 0, 0)
    for index, character in enumerate(label):
        x = 12 + padding + index * 6 * scale
        for row, bits in enumerate(_GLYPHS[character]):
            for column in range(5):
                if bits & (1 << (4 - column)):
                    y0 = 12 + padding + row * scale
                    x0 = x + column * scale
                    image[y0 : y0 + scale, x0 : x0 + scale] = 255
    return image


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

    def render(self, *, recording: bool = False, elapsed_seconds: float = 0.0) -> None:
        now = time.monotonic()
        if now - self._last_render < self._period:
            return
        self._renderer.update_scene(self._data, camera=self._camera)
        frame = self._renderer.render()
        if recording:
            frame = _recording_overlay(frame, elapsed_seconds)
        self._write_frame(frame)
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
