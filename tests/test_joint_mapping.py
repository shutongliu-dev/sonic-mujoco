import unittest

from sonic_mujoco.envs.mujoco.g1 import SONIC_JOINT_NAMES, MujocoG1EmptyEnv


class JointMappingTest(unittest.TestCase):
    def test_hardware_order_maps_by_name(self) -> None:
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)

        self.assertEqual(len(set(env.joint_ids)), len(SONIC_JOINT_NAMES))
        self.assertEqual(len(set(env.actuator_ids)), len(SONIC_JOINT_NAMES))
        self.assertEqual(env.actuator_ids[:3], (0, 1, 2))
        self.assertEqual(env.actuator_ids[22:29], tuple(range(29, 36)))


if __name__ == "__main__":
    unittest.main()
