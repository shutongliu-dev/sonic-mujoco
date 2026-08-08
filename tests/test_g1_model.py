import unittest

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv


class G1ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1EmptyEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_model_dimensions(self) -> None:
        self.assertEqual((self.env.model.nq, self.env.model.nv), (50, 49))
        self.assertEqual(self.env.model.nu, 43)
        self.assertEqual(len(self.env.joint_ids), 29)

    def test_reset(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(self.env.time, 0.0)


if __name__ == "__main__":
    unittest.main()
