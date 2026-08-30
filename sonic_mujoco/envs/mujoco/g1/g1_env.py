from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ....contact import ContactRecorder
from ....neck import NECK_DOF, NECK_JOINT_NAMES
from ....tactile import TactileRecorder
from ....tactile_skin import (
    JuQiaoTactileAdapter,
    build_juqiao_skin_layout,
)
from ....teleop.dexhand import DEXHAND_JOINT_NAMES
from ..env_base import MujocoEnvBase
from .interface import RobotCommand, RobotState

SONIC_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
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


@dataclass(frozen=True, slots=True)
class _JointGroup:
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    actuator_indices: np.ndarray
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    joint_range: np.ndarray
    control_range: np.ndarray


class MujocoG1Env(MujocoEnvBase):
    """MuJoCo environment using the G1 hardware joint order."""

    hand_kp = 2.5
    hand_kd = 0.15
    neck_kp = 8.0
    neck_kd = 0.4

    def __init__(
        self,
        xml_path: str | Path,
        timestep: float = 0.005,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        super().__init__(xml_path, timestep, model=model)
        self._body_joints = self._joint_group(SONIC_JOINT_NAMES)
        self._hand_joints = self._joint_group(DEXHAND_JOINT_NAMES)
        self._neck_joints = self._joint_group(NECK_JOINT_NAMES)
        self._publish_joint_group_aliases()
        self._imu_quaternion = self._sensor_slice("imu_quat")
        self._imu_angular_velocity = self._sensor_slice("imu_gyro")
        self._imu_linear_acceleration = self._sensor_slice("imu_acc")
        self.contacts = ContactRecorder(self.model)
        self.tactile_skin_layout = build_juqiao_skin_layout(self.model)
        self.tactile = TactileRecorder(
            self.model,
            layout=self.tactile_skin_layout.taxels,
            mapping_radius=self.tactile_skin_layout.mapping_radius_m,
        )
        self.tactile_adapter = JuQiaoTactileAdapter(self.tactile_skin_layout)
        self.tactile_suit = self.tactile_adapter.last_frame

        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if root_id < 0 or self.model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("G1 model must have a floating_base_joint")

    def get_robot_state(self) -> RobotState:
        return RobotState(
            timestamp=self.time,
            base_position=self.data.qpos[:3].copy(),
            base_quaternion=self.data.qpos[3:7].copy(),
            base_linear_velocity=self.data.qvel[:3].copy(),
            base_angular_velocity=self.data.qvel[3:6].copy(),
            joint_position=self.data.qpos[self._qpos_indices].copy(),
            joint_velocity=self.data.qvel[self._dof_indices].copy(),
            joint_effort=self.data.actuator_force[self._actuator_indices].copy(),
            imu_quaternion=self.data.sensordata[self._imu_quaternion].copy(),
            imu_angular_velocity=self.data.sensordata[
                self._imu_angular_velocity
            ].copy(),
            imu_linear_acceleration=self.data.sensordata[
                self._imu_linear_acceleration
            ].copy(),
        )

    def step(self, command: RobotCommand, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")

        self.contacts.begin()
        self.tactile.begin()
        for _ in range(steps):
            state = self.get_robot_state()
            torque = (
                command.feedforward_torque
                + command.kp * (command.joint_position - state.joint_position)
                + command.kd * (command.joint_velocity - state.joint_velocity)
            )
            torque = np.clip(
                torque, self._control_range[:, 0], self._control_range[:, 1]
            )
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self._actuator_indices] = torque
            if command.hand_joint_position is not None:
                self._apply_position_pd(
                    self._hand_joints,
                    command.hand_joint_position,
                    kp=self.hand_kp,
                    kd=self.hand_kd,
                )
            neck_target = (
                np.zeros(NECK_DOF, dtype=np.float64)
                if command.neck_joint_position is None
                else command.neck_joint_position
            )
            self._apply_position_pd(
                self._neck_joints,
                neck_target,
                kp=self.neck_kp,
                kd=self.neck_kd,
            )
            mujoco.mj_step(self.model, self.data)
            self.contacts.update(self.data)
            self.tactile.update(self.data)
        self.contacts.finish()
        tactile = self.tactile.finish()
        self.tactile_suit = self.tactile_adapter.update(tactile, self.time)

    def reset(self) -> None:
        super().reset()
        self.contacts.reset()
        self.tactile.reset()
        self.tactile_adapter.reset(self.time)
        self.tactile_suit = self.tactile_adapter.last_frame

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

    def _joint_group(self, names: tuple[str, ...]) -> _JointGroup:
        joint_ids = tuple(self._joint_id(name) for name in names)
        actuator_ids = tuple(
            self._actuator_id(name, joint_id)
            for name, joint_id in zip(names, joint_ids, strict=True)
        )
        joint_indices = np.asarray(joint_ids)
        actuator_indices = np.asarray(actuator_ids)
        return _JointGroup(
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            actuator_indices=actuator_indices,
            qpos_indices=self.model.jnt_qposadr[joint_indices],
            dof_indices=self.model.jnt_dofadr[joint_indices],
            joint_range=self.model.jnt_range[joint_indices].copy(),
            control_range=self.model.actuator_ctrlrange[actuator_indices].copy(),
        )

    def _apply_position_pd(
        self,
        group: _JointGroup,
        target: np.ndarray,
        *,
        kp: float,
        kd: float,
    ) -> None:
        target = np.clip(target, group.joint_range[:, 0], group.joint_range[:, 1])
        torque = (
            kp * (target - self.data.qpos[group.qpos_indices])
            - kd * self.data.qvel[group.dof_indices]
        )
        self.data.ctrl[group.actuator_indices] = np.clip(
            torque,
            group.control_range[:, 0],
            group.control_range[:, 1],
        )

    def _publish_joint_group_aliases(self) -> None:
        """Keep the original public arrays while joint groups own their setup."""

        self.joint_ids = self._body_joints.joint_ids
        self.actuator_ids = self._body_joints.actuator_ids
        self._actuator_indices = self._body_joints.actuator_indices
        self._qpos_indices = self._body_joints.qpos_indices
        self._dof_indices = self._body_joints.dof_indices
        self._control_range = self._body_joints.control_range

        self.hand_joint_ids = self._hand_joints.joint_ids
        self.hand_actuator_ids = self._hand_joints.actuator_ids
        self._hand_actuator_indices = self._hand_joints.actuator_indices
        self._hand_qpos_indices = self._hand_joints.qpos_indices
        self._hand_dof_indices = self._hand_joints.dof_indices
        self._hand_joint_range = self._hand_joints.joint_range
        self._hand_control_range = self._hand_joints.control_range

        self.neck_joint_ids = self._neck_joints.joint_ids
        self.neck_actuator_ids = self._neck_joints.actuator_ids
        self._neck_actuator_indices = self._neck_joints.actuator_indices
        self._neck_qpos_indices = self._neck_joints.qpos_indices
        self._neck_dof_indices = self._neck_joints.dof_indices
        self._neck_joint_range = self._neck_joints.joint_range
        self._neck_control_range = self._neck_joints.control_range

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"missing G1 sensor: {name}")
        start = self.model.sensor_adr[sensor_id]
        return slice(start, start + self.model.sensor_dim[sensor_id])
