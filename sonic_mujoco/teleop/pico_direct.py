from collections import deque
import ctypes
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from .base import TeleopBase, TeleopCommand

PICO_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
     16, 17, 18, 19, 20, 22],
    dtype=np.int32,
)
SMPL_OUTPUT_JOINTS = np.r_[0:22, 39, 54]


@dataclass(frozen=True, slots=True)
class _PoseFrame:
    pose: np.ndarray
    joints: np.ndarray
    root_quaternion: np.ndarray


def _lerp_quaternion(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    if np.dot(left, right) < 0.0:
        right = -right
    value = (1.0 - alpha) * left + alpha * right
    return value / np.linalg.norm(value)


def _lerp_pose(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    left_q = Rotation.from_rotvec(left).as_quat()
    right_q = Rotation.from_rotvec(right).as_quat()
    signs = np.where(np.sum(left_q * right_q, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    quaternions = (1.0 - alpha) * left_q + alpha * signs * right_q
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return Rotation.from_quat(quaternions).as_rotvec()


class PicoPoseConverter:
    """Convert XRoboToolkit body joints to the representation expected by SONIC."""

    def __init__(self, skeleton: Path | None = None) -> None:
        if skeleton is None:
            skeleton = Path(__file__).parents[1] / "assets/pico/smpl_skeleton.npz"
        with np.load(skeleton) as data:
            self._rest_joints = data["joints"].astype(np.float64)
            self._parents = data["parents"].astype(np.int32)

    def convert(self, body_poses: np.ndarray) -> _PoseFrame:
        body_poses = np.asarray(body_poses, dtype=np.float64)
        if body_poses.shape != (24, 7):
            raise ValueError("PICO body poses must have shape (24, 7)")

        global_rotations = Rotation.from_quat(
            body_poses[:, [6, 3, 4, 5]], scalar_first=True
        )
        global_rotations = global_rotations * Rotation.from_euler("y", np.pi)
        local_rotations = [global_rotations[0]]
        for index in range(1, 24):
            parent = PICO_PARENTS[index]
            local_rotations.append(global_rotations[parent].inv() * global_rotations[index])
        local_pose = np.stack([rotation.as_rotvec() for rotation in local_rotations])

        pose = local_pose[1:22]
        root = Rotation.from_euler("x", np.pi / 2) * Rotation.from_rotvec(local_pose[0])
        joints = self._forward_kinematics(root, pose)
        root = root * Rotation.from_quat([-0.5, -0.5, -0.5, 0.5])
        local_joints = root.inv().apply(joints)
        return _PoseFrame(
            pose=pose,
            joints=local_joints,
            root_quaternion=root.as_quat(scalar_first=True),
        )

    def _forward_kinematics(self, root: Rotation, pose: np.ndarray) -> np.ndarray:
        rotvecs = np.zeros((55, 3), dtype=np.float64)
        rotvecs[0] = root.as_rotvec()
        rotvecs[1:22] = pose
        local_rotations = Rotation.from_rotvec(rotvecs).as_matrix()
        transforms = np.repeat(np.eye(4)[None], 55, axis=0)
        transforms[:, :3, :3] = local_rotations
        transforms[0, :3, 3] = self._rest_joints[0]
        transforms[1:, :3, 3] = (
            self._rest_joints[1:] - self._rest_joints[self._parents[1:]]
        )
        world = np.empty_like(transforms)
        world[0] = transforms[0]
        for index in range(1, 55):
            world[index] = world[self._parents[index]] @ transforms[index]
        return world[SMPL_OUTPUT_JOINTS, :3, 3]


def _wrist_joints(pose: np.ndarray) -> np.ndarray:
    joint_position = np.zeros(29, dtype=np.float64)
    for elbow_index, wrist_index, side in ((17, 19, 1.0), (18, 20, -1.0)):
        elbow = Rotation.from_rotvec(pose[elbow_index])
        quaternion = elbow.as_quat(scalar_first=True)
        twist = np.array([quaternion[0], 0.0, quaternion[2], 0.0])
        norm = np.linalg.norm(twist)
        twist = twist / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
        swing = Rotation.from_quat(twist, scalar_first=True).inv() * elbow
        swing_euler = swing.as_euler("XYZ")
        wrist_euler = Rotation.from_rotvec(pose[wrist_index]).as_euler("XYZ")
        offset = 23 if side > 0 else 24
        joint_position[offset] = side * (swing_euler[0] + wrist_euler[0])
        joint_position[offset + 2] = side * wrist_euler[1]
        joint_position[offset + 4] = swing_euler[2] + wrist_euler[2]
    return joint_position


def load_xrobotoolkit(sdk_dir: Path | None = None):
    """Load the XR SDK copied by scripts/setup_xrobotoolkit.py."""

    if sdk_dir is None:
        configured = os.environ.get("SONIC_XR_SDK_DIR")
        sdk_dir = Path(configured) if configured else Path(__file__).parents[2] / ".xrobotoolkit"
    libraries = list((sdk_dir / "lib").glob("libPXREARobotSDK.so"))
    bindings = list(sdk_dir.glob("xrobotoolkit_sdk*.so"))
    if not libraries or not bindings:
        raise RuntimeError("XR SDK not found; run scripts/setup_xrobotoolkit.py first")
    ctypes.CDLL(str(libraries[0]), mode=ctypes.RTLD_GLOBAL)
    spec = importlib.util.spec_from_file_location("xrobotoolkit_sdk", bindings[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the XR SDK Python binding")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PicoTeleop(TeleopBase):
    """Read PICO body tracking directly from XRoboToolkit at 50 Hz."""

    def __init__(
        self,
        sdk=None,
        *,
        start_service: bool = True,
        target_fps: int = 50,
        buffer_size: int = 5,
    ) -> None:
        if start_service:
            subprocess.Popen(
                ["bash", "/opt/apps/roboticsservice/runService.sh"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        self._sdk = sdk or load_xrobotoolkit()
        self._sdk.init()
        self._converter = PicoPoseConverter()
        self._step_ns = int(1e9 / target_fps)
        self._frames: deque[tuple[int, _PoseFrame]] = deque(maxlen=buffer_size)
        self._previous: tuple[int, _PoseFrame] | None = None
        self._next_target_ns: int | None = None
        self._frame_index = 0

    def read(self) -> TeleopCommand | None:
        if not self._sdk.is_body_data_available():
            return None
        timestamp = int(self._sdk.get_time_stamp_ns())
        if self._previous is not None and timestamp <= self._previous[0]:
            return None
        current = self._converter.convert(np.asarray(self._sdk.get_body_joints_pose()))
        if self._previous is None:
            self._previous = timestamp, current
            self._next_target_ns = timestamp + self._step_ns
            return None

        previous_timestamp, previous = self._previous
        target = self._next_target_ns
        self._previous = timestamp, current
        if target is None or target > timestamp:
            return None
        target = max(target, previous_timestamp)
        alpha = np.clip((target - previous_timestamp) / (timestamp - previous_timestamp), 0.0, 1.0)
        frame = _PoseFrame(
            pose=_lerp_pose(previous.pose, current.pose, alpha),
            joints=(1.0 - alpha) * previous.joints + alpha * current.joints,
            root_quaternion=_lerp_quaternion(
                previous.root_quaternion, current.root_quaternion, alpha
            ),
        )
        self._frames.append((self._frame_index, frame))
        self._frame_index += 1
        self._next_target_ns = target + self._step_ns
        if len(self._frames) < self._frames.maxlen:
            return None

        _, _, right_x, _ = self._controller_axes()
        heading = -1.5 * right_x / 50.0 if abs(right_x) >= 0.15 else 0.0
        return TeleopCommand(
            frame_index=np.array([item[0] for item in self._frames]),
            smpl_joints=np.stack([item[1].joints for item in self._frames]),
            root_quaternion=np.stack([item[1].root_quaternion for item in self._frames]),
            joint_position=np.stack([_wrist_joints(item[1].pose) for item in self._frames]),
            heading_increment=heading,
        )

    def _controller_axes(self) -> tuple[float, float, float, float]:
        try:
            left, right = self._sdk.get_left_axis(), self._sdk.get_right_axis()
            return float(left[0]), float(left[1]), float(right[0]), float(right[1])
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def close(self) -> None:
        self._sdk.close()
