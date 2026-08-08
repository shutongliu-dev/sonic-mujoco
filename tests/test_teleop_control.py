import tempfile
from pathlib import Path
import unittest

import numpy as np

from sonic_mujoco.recording import EpisodeRecorder
from sonic_mujoco.teleop import (
    PicoControls,
    PicoEvents,
    TeleopCommand,
    TeleopMode,
    next_mode,
)


def command() -> TeleopCommand:
    return TeleopCommand(
        frame_index=np.arange(5),
        smpl_joints=np.zeros((5, 24, 3)),
        root_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
        joint_position=np.zeros((5, 29)),
    )


class TeleopControlTest(unittest.TestCase):
    def test_start_and_pose_combinations_follow_the_real_robot_flow(self) -> None:
        mode = next_mode(TeleopMode.OFF, PicoEvents(start_stop=True))
        self.assertIs(mode, TeleopMode.READY)
        mode = next_mode(mode, PicoEvents(toggle_pose=True))
        self.assertIs(mode, TeleopMode.POSE)
        mode = next_mode(mode, PicoEvents(toggle_pose=True))
        self.assertIs(mode, TeleopMode.READY)
        mode = next_mode(mode, PicoEvents(start_stop=True))
        self.assertIs(mode, TeleopMode.OFF)

    def test_episode_recorder_saves_aligned_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, "sweep")
            recorder.start()
            recorder.append(
                time=0.02,
                qpos=np.arange(40),
                qvel=np.arange(39),
                ctrl=np.arange(29),
                command=command(),
                token=np.arange(64),
                action=np.arange(29),
                controls=PicoControls(left_grip=0.8, a=True),
            )
            path = recorder.finish()

            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(Path(path).is_file())
            with np.load(path) as data:
                self.assertEqual(data["qpos"].shape, (1, 40))
                self.assertEqual(data["smpl_joints"].shape, (1, 24, 3))
                self.assertEqual(data["token"].shape, (1, 64))
                self.assertEqual(data["scene"].item(), "sweep")

    def test_aborted_recording_does_not_write_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, "sweep")
            recorder.start()
            recorder.abort()

            self.assertFalse(recorder.active)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
