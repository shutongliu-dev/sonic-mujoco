import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv, RobotCommand


class PdControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1EmptyEnv()
        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_pd_formula_and_actuator_mapping(self) -> None:
        q = np.linspace(-0.2, 0.2, 29)
        dq = np.linspace(-0.1, 0.1, 29)
        self.env.data.qpos[self.env._qpos_indices] = q
        self.env.data.qvel[self.env._dof_indices] = dq
        mujoco.mj_forward(self.env.model, self.env.data)

        command = RobotCommand(
            joint_position=q + 0.1,
            joint_velocity=dq - 0.2,
            feedforward_torque=np.full(29, 0.3),
            kp=np.full(29, 2.0),
            kd=np.full(29, 0.5),
        )
        expected = np.full(29, 0.4)
        self.env.step(command)

        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._actuator_indices], expected, atol=1e-12
        )
        hand_ids = np.setdiff1d(np.arange(self.env.model.nu), self.env._actuator_indices)
        np.testing.assert_array_equal(self.env.data.ctrl[hand_ids], 0.0)

    def test_effort_clipping(self) -> None:
        zeros = np.zeros(29)
        command = RobotCommand(zeros, zeros, np.full(29, 1e6), zeros, zeros)
        self.env.step(command)
        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._actuator_indices], self.env._control_range[:, 1]
        )

    def test_command_rejects_wrong_shape(self) -> None:
        zeros = np.zeros(29)
        with self.assertRaises(ValueError):
            RobotCommand(np.zeros(28), zeros, zeros, zeros, zeros)


if __name__ == "__main__":
    unittest.main()
