import unittest

import numpy as np

from sonic_mujoco.teleop import PicoPoseConverter, PicoTeleop


def identity_body() -> np.ndarray:
    body = np.zeros((24, 7), dtype=np.float64)
    body[:, 6] = 1.0
    return body


class FakeSdk:
    def __init__(self) -> None:
        self.timestamp = 0
        self.closed = False
        self.buttons = {name: False for name in "ABXY"}
        self.left_grip = 0.0

    def init(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def is_body_data_available(self) -> bool:
        return True

    def get_time_stamp_ns(self) -> int:
        self.timestamp += 20_000_000
        return self.timestamp

    def get_body_joints_pose(self) -> np.ndarray:
        return identity_body()

    def get_left_axis(self) -> tuple[float, float]:
        return 0.0, 0.0

    def get_right_axis(self) -> tuple[float, float]:
        return 0.5, 0.0

    def get_A_button(self) -> bool:
        return self.buttons["A"]

    def get_B_button(self) -> bool:
        return self.buttons["B"]

    def get_X_button(self) -> bool:
        return self.buttons["X"]

    def get_Y_button(self) -> bool:
        return self.buttons["Y"]

    def get_left_grip(self) -> float:
        return self.left_grip


class PicoDirectTest(unittest.TestCase):
    def test_pose_conversion_is_finite_and_has_sonic_shapes(self) -> None:
        frame = PicoPoseConverter().convert(identity_body())

        self.assertEqual(frame.pose.shape, (21, 3))
        self.assertEqual(frame.joints.shape, (24, 3))
        self.assertEqual(frame.root_quaternion.shape, (4,))
        self.assertTrue(np.isfinite(frame.joints).all())
        self.assertAlmostEqual(np.linalg.norm(frame.root_quaternion), 1.0)

    def test_direct_reader_builds_a_five_frame_command(self) -> None:
        sdk = FakeSdk()
        teleop = PicoTeleop(sdk, start_service=False)

        command = None
        for _ in range(6):
            command = teleop.read()

        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command.smpl_joints.shape, (5, 24, 3))
        self.assertEqual(command.joint_position.shape, (5, 29))
        self.assertAlmostEqual(command.heading_increment, -0.015)
        teleop.close()
        self.assertTrue(sdk.closed)

    def test_button_combinations_are_edge_triggered(self) -> None:
        sdk = FakeSdk()
        teleop = PicoTeleop(sdk, start_service=False)

        sdk.buttons.update(A=True, B=True, X=True, Y=True)
        teleop.read()
        events = teleop.pop_events()
        self.assertTrue(events.start_stop)
        self.assertFalse(events.toggle_pose)

        teleop.read()
        self.assertFalse(teleop.pop_events().start_stop)
        sdk.buttons.update(B=False, Y=False)
        teleop.read()
        self.assertFalse(teleop.pop_events().toggle_pose)
        sdk.buttons.update(A=False, X=False)
        teleop.read()
        sdk.buttons.update(A=True, X=True)
        teleop.read()
        self.assertTrue(teleop.pop_events().toggle_pose)

        sdk.buttons.update(A=False, X=False)
        teleop.read()
        sdk.left_grip = 1.0
        sdk.buttons["A"] = True
        teleop.read()
        self.assertTrue(teleop.pop_events().toggle_recording)

        sdk.buttons["A"] = False
        teleop.read()
        sdk.buttons["B"] = True
        teleop.read()
        self.assertTrue(teleop.pop_events().abort_recording)


if __name__ == "__main__":
    unittest.main()
