import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
from sonic_mujoco.scan import mujoco_camera_pose


class G1ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1EmptyEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_model_dimensions(self) -> None:
        self.assertEqual((self.env.model.nq, self.env.model.nv), (78, 77))
        self.assertEqual(self.env.model.nu, 71)
        self.assertEqual(len(self.env.joint_ids), 29)
        self.assertEqual(len(self.env.hand_joint_ids), 40)
        self.assertEqual(len(self.env.neck_joint_ids), 2)

    def test_head_cameras_move_with_the_neck(self) -> None:
        head_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "head_link"
        )
        for name in (
            "head_camera",
            "head_camera_scan",
            "head_camera_left",
            "head_camera_right",
        ):
            camera_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            self.assertEqual(self.env.model.cam_bodyid[camera_id], head_id)

    def test_vr_cameras_form_a_parallel_64_mm_stereo_pair(self) -> None:
        self.env.reset()
        view_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "vr_view"
        )
        mocap_id = self.env.model.body_mocapid[view_body_id]
        base_pose = mujoco_camera_pose(self.env.model, self.env.data, "vr_camera_base")
        self.env.data.mocap_pos[mocap_id] = base_pose[:3, 3]
        mujoco.mju_mat2Quat(
            self.env.data.mocap_quat[mocap_id],
            base_pose[:3, :3].reshape(-1),
        )
        mujoco.mj_forward(self.env.model, self.env.data)

        left = mujoco_camera_pose(self.env.model, self.env.data, "vr_camera_left")
        right = mujoco_camera_pose(self.env.model, self.env.data, "vr_camera_right")

        np.testing.assert_allclose((left[:3, 3] + right[:3, 3]) / 2.0, base_pose[:3, 3])
        self.assertAlmostEqual(np.linalg.norm(right[:3, 3] - left[:3, 3]), 0.064)
        np.testing.assert_allclose(left[:3, :3], base_pose[:3, :3])
        np.testing.assert_allclose(right[:3, :3], base_pose[:3, :3])

    def test_reset(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(self.env.time, 0.0)


if __name__ == "__main__":
    unittest.main()
