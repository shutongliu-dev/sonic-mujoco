import unittest

import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv, SONIC_JOINT_NAMES


class G1ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1EmptyEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_model_dimensions(self) -> None:
        self.assertEqual((self.env.model.nq, self.env.model.nv), (50, 49))
        self.assertEqual(self.env.model.nu, 43)
        self.assertEqual(len(self.env.joint_ids), 29)

    def test_sonic_joint_mapping(self) -> None:
        self.assertEqual(len(set(self.env.joint_ids)), len(SONIC_JOINT_NAMES))
        self.assertEqual(len(set(self.env.actuator_ids)), len(SONIC_JOINT_NAMES))
        self.assertEqual(self.env.actuator_ids[:3], (2, 1, 0))
        self.assertEqual(self.env.actuator_ids[22:29], tuple(range(29, 36)))

    def test_reset_and_step(self) -> None:
        self.env.reset()
        initial_qpos = self.env.data.qpos.copy()
        self.env.step()
        self.assertAlmostEqual(self.env.time, 0.005)
        self.assertTrue(np.isfinite(self.env.data.qpos).all())
        self.env.reset()
        np.testing.assert_allclose(self.env.data.qpos, initial_qpos)


if __name__ == "__main__":
    unittest.main()
