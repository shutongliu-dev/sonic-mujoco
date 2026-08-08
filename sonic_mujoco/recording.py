from datetime import datetime
from pathlib import Path

import numpy as np

from .teleop.base import TeleopCommand
from .teleop.pico_direct import PicoControls


class EpisodeRecorder:
    """Collect aligned MuJoCo, policy, and PICO reference frames."""

    def __init__(self, directory: str | Path, scene: str) -> None:
        self.directory = Path(directory)
        self.scene = scene
        self._frames: list[dict[str, np.ndarray | float]] = []
        self.active = False

    def start(self) -> None:
        self._frames.clear()
        self.active = True

    def append(
        self,
        *,
        time: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        command: TeleopCommand,
        token: np.ndarray,
        action: np.ndarray,
        controls: PicoControls,
    ) -> None:
        if not self.active:
            return
        self._frames.append(
            {
                "time": float(time),
                "qpos": np.asarray(qpos).copy(),
                "qvel": np.asarray(qvel).copy(),
                "ctrl": np.asarray(ctrl).copy(),
                "smpl_joints": command.smpl_joints[-1].copy(),
                "root_quaternion": command.root_quaternion[-1].copy(),
                "reference_joint_position": command.joint_position[-1].copy(),
                "token": np.asarray(token).copy(),
                "action": np.asarray(action).copy(),
                "pico_input": np.array(
                    [
                        controls.left_trigger,
                        controls.right_trigger,
                        controls.left_grip,
                        controls.right_grip,
                        controls.a,
                        controls.b,
                        controls.x,
                        controls.y,
                        controls.menu,
                    ],
                    dtype=np.float64,
                ),
            }
        )

    def finish(self) -> Path | None:
        if not self.active:
            return None
        self.active = False
        if not self._frames:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.directory / f"episode_{timestamp}.npz"
        keys = self._frames[0]
        arrays = {
            key: np.stack([np.asarray(frame[key]) for frame in self._frames])
            for key in keys
        }
        np.savez_compressed(path, scene=np.array(self.scene), **arrays)
        self._frames.clear()
        return path

    def abort(self) -> None:
        self.active = False
        self._frames.clear()
