from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from .g1_env import MujocoG1Env

OBJECT_NAMES = ("sweep_object_0", "sweep_object_1", "sweep_object_2")
SPAWN_POSITIONS = np.array(
    [[0.62, -0.28, 0.79], [0.90, -0.28, 0.79], [0.76, -0.28, 0.79]]
)


@dataclass(frozen=True, slots=True)
class SweepState:
    object_position: NDArray[np.float64]
    object_quaternion: NDArray[np.float64]
    success: bool


class MujocoG1SweepEnv(MujocoG1Env):
    """G1 sweeps three objects across the table's left-right center line."""

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "g1" / "sweep.xml"
        super().__init__(scene, timestep)

        self._object_body_ids = np.array(
            [self._body_id(name) for name in OBJECT_NAMES], dtype=np.int32
        )
        self._object_qpos_addresses = np.array(
            [self._freejoint_qpos_address(f"{name}_joint") for name in OBJECT_NAMES]
        )
        self._target_site_id = self._site_id("sweep_target")

    def reset(self, seed: int | None = None) -> None:
        super().reset()
        rng = np.random.default_rng(seed)
        positions = SPAWN_POSITIONS.copy()
        positions[:, 0] += rng.uniform(-0.025, 0.025, len(OBJECT_NAMES))
        positions[:, 1] += rng.uniform(-0.05, 0.05, len(OBJECT_NAMES))
        for address, position in zip(self._object_qpos_addresses, positions):
            self.data.qpos[address : address + 3] = position
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def get_scene_state(self) -> SweepState:
        position = self.data.xpos[self._object_body_ids].copy()
        quaternion = self.data.xquat[self._object_body_ids].copy()
        return SweepState(position, quaternion, self._positions_in_target(position))

    def is_success(self) -> bool:
        position = self.data.xpos[self._object_body_ids]
        return self._positions_in_target(position)

    def _positions_in_target(self, position: NDArray[np.float64]) -> bool:
        center = self.data.site_xpos[self._target_site_id]
        half_size = self.model.site_size[self._target_site_id]
        inside_xy = np.abs(position[:, :2] - center[:2]) <= half_size[:2]
        on_table = (position[:, 2] >= 0.76) & (position[:, 2] <= 0.85)
        return bool(np.all(inside_xy) and np.all(on_table))

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"missing sweep body: {name}")
        return body_id

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"missing sweep site: {name}")
        return site_id

    def _freejoint_qpos_address(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"missing sweep freejoint: {name}")
        return int(self.model.jnt_qposadr[joint_id])
