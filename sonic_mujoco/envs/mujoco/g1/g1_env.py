from pathlib import Path

import mujoco
import numpy as np

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


class MujocoG1Env(MujocoEnvBase):
    """MuJoCo environment using the G1 hardware joint order."""

    def __init__(self, xml_path: str | Path, timestep: float = 0.005) -> None:
        super().__init__(xml_path, timestep)
        self.joint_ids = tuple(self._joint_id(name) for name in SONIC_JOINT_NAMES)
        self.actuator_ids = tuple(
            self._actuator_id(name, joint_id)
            for name, joint_id in zip(SONIC_JOINT_NAMES, self.joint_ids)
        )
        self._actuator_indices = np.asarray(self.actuator_ids)
        self._qpos_indices = self.model.jnt_qposadr[np.asarray(self.joint_ids)]
        self._dof_indices = self.model.jnt_dofadr[np.asarray(self.joint_ids)]
        self._control_range = self.model.actuator_ctrlrange[
            self._actuator_indices
        ].copy()
        self._imu_quaternion = self._sensor_slice("imu_quat")
        self._imu_angular_velocity = self._sensor_slice("imu_gyro")
        self._imu_linear_acceleration = self._sensor_slice("imu_acc")

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
            mujoco.mj_step(self.model, self.data)

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

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"missing G1 sensor: {name}")
        start = self.model.sensor_adr[sensor_id]
        return slice(start, start + self.model.sensor_dim[sensor_id])
