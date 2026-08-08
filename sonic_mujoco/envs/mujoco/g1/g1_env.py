from pathlib import Path

import mujoco

from ..env_base import MujocoEnvBase


SONIC_JOINT_NAMES = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class MujocoG1Env(MujocoEnvBase):
    """MuJoCo environment with validated SONIC G1 joint semantics."""

    def __init__(self, xml_path: str | Path, timestep: float = 0.005) -> None:
        super().__init__(xml_path, timestep)
        self.joint_ids = tuple(self._joint_id(name) for name in SONIC_JOINT_NAMES)
        self.actuator_ids = tuple(
            self._actuator_id(name, joint_id)
            for name, joint_id in zip(SONIC_JOINT_NAMES, self.joint_ids)
        )

        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if root_id < 0 or self.model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("G1 model must have a floating_base_joint")

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing G1 joint: {name}")
        return joint_id

    def _actuator_id(self, joint_name: str, joint_id: int) -> int:
        actuator_name = joint_name.removesuffix("_joint")
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if actuator_id < 0:
            raise ValueError(f"missing G1 actuator: {actuator_name}")
        if self.model.actuator_trnid[actuator_id, 0] != joint_id:
            raise ValueError(f"actuator {actuator_name} does not drive {joint_name}")
        return actuator_id
