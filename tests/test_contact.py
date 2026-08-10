import unittest

import mujoco
import numpy as np

from sonic_mujoco.contact import MAX_CONTACTS, ContactRecorder
from sonic_mujoco.envs.mujoco.g1 import (
    MujocoG1PlushCarryEnv,
    MujocoG1SweepEnv,
    RobotCommand,
)


def zero_command() -> RobotCommand:
    zeros = np.zeros(29)
    return RobotCommand(zeros, zeros, zeros, zeros, zeros)


class ContactRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1SweepEnv()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_contact_frame_has_fixed_empty_shape(self) -> None:
        frame = ContactRecorder(self.env.model).last_frame

        self.assertEqual(frame.robot_body_id.shape, (MAX_CONTACTS,))
        self.assertEqual(frame.position.shape, (MAX_CONTACTS, 3))
        self.assertEqual(frame.count, 0)
        self.assertTrue(np.all(frame.robot_body_id == -1))

    def test_robot_object_contact_is_accumulated(self) -> None:
        hand_body = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        object_joint = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "sweep_object_0_joint"
        )
        address = self.env.model.jnt_qposadr[object_joint]
        self.env.data.qpos[address : address + 3] = self.env.data.xpos[hand_body]
        mujoco.mj_forward(self.env.model, self.env.data)

        self.env.step(zero_command(), steps=2)
        frame = self.env.contacts.last_frame
        active = frame.robot_body_id >= 0

        self.assertGreater(frame.count, 0)
        self.assertTrue(np.any(frame.other_body_id[active] > 0))
        self.assertGreater(frame.normal_impulse[active].sum(), 0.0)

    def test_robot_flex_contact_is_accumulated(self) -> None:
        plush_env = MujocoG1PlushCarryEnv()
        self.addCleanup(plush_env.close)
        plush_env.reset(seed=0)
        hand_body = mujoco.mj_name2id(
            plush_env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_wrist_yaw_link",
        )
        plush_joint = mujoco.mj_name2id(
            plush_env.model, mujoco.mjtObj.mjOBJ_JOINT, "carry_plush_joint"
        )
        address = plush_env.model.jnt_qposadr[plush_joint]
        plush_env.data.qpos[address : address + 3] = plush_env.data.xpos[
            hand_body
        ]
        mujoco.mj_forward(plush_env.model, plush_env.data)

        plush_env.step(zero_command(), steps=2)
        frame = plush_env.contacts.last_frame
        active = frame.robot_body_id >= 0

        self.assertGreater(frame.count, 0)
        self.assertTrue(np.any(frame.other_body_id[active] > 0))
        self.assertGreater(frame.normal_impulse[active].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
