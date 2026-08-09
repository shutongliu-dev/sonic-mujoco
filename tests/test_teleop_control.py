import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow.parquet as pq

from sonic_mujoco.contact import ContactFrame, MAX_CONTACTS
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


def contacts() -> ContactFrame:
    robot = np.full(MAX_CONTACTS, -1, dtype=np.int32)
    other = np.full(MAX_CONTACTS, -1, dtype=np.int32)
    robot[0], other[0] = 1, 2
    normal_force = np.zeros(MAX_CONTACTS, dtype=np.float32)
    normal_impulse = np.zeros(MAX_CONTACTS, dtype=np.float32)
    normal_force[0], normal_impulse[0] = 12.0, 0.06
    return ContactFrame(
        robot_body_id=robot,
        other_body_id=other,
        position=np.zeros((MAX_CONTACTS, 3), dtype=np.float32),
        normal_force=normal_force,
        tangent_force=np.zeros(MAX_CONTACTS, dtype=np.float32),
        normal_impulse=normal_impulse,
        sample_count=np.r_[2, np.zeros(MAX_CONTACTS - 1)].astype(np.int32),
        count=1,
    )


def append_frame(recorder: EpisodeRecorder) -> None:
    recorder.append(
        time=0.02,
        qpos=np.arange(40),
        qvel=np.arange(39),
        ctrl=np.arange(29),
        command=command(),
        token=np.arange(64),
        action=np.arange(29),
        controls=PicoControls(left_grip=0.8, a=True),
        contacts=contacts(),
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
            recorder = EpisodeRecorder(
                directory,
                "sweep",
                body_names=("world", "right_wrist", "sweep_object"),
                record_video=False,
            )
            recorder.start()
            append_frame(recorder)
            path = recorder.finish()

            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(Path(path).is_file())
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 1)
            self.assertIn("observation.qpos", table.column_names)
            self.assertIn("observation.contact.normal_force", table.column_names)
            session = path.parents[2]
            self.assertTrue((session / "meta/info.json").is_file())
            self.assertTrue((session / "meta/modality.json").is_file())
            self.assertTrue((session / "meta/episodes.jsonl").is_file())
            self.assertTrue(recorder.last_preview.is_file())

            recorder.start()
            append_frame(recorder)
            second = recorder.finish()
            info = json.loads((session / "meta/info.json").read_text())
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.name, "episode_000001.parquet")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 2)
            self.assertEqual(pq.read_table(second)["index"][0].as_py(), 1)

    def test_aborted_recording_does_not_write_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, "sweep", record_video=False)
            recorder.start()
            append_frame(recorder)
            recorder.abort()

            self.assertFalse(recorder.active)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
