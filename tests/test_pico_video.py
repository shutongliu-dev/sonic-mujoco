import unittest

import numpy as np

from sonic_mujoco.teleop.pico_video import _recording_overlay


class PicoVideoTest(unittest.TestCase):
    def test_recording_overlay_draws_timer_without_mutating_source(self) -> None:
        frame = np.full((80, 240, 3), 17, dtype=np.uint8)

        first = _recording_overlay(frame, 1.8)
        second = _recording_overlay(frame, 61.2)

        self.assertTrue(np.all(frame == 17))
        np.testing.assert_array_equal(first[12, 12], [190, 0, 0])
        self.assertTrue(np.any(np.all(first == 255, axis=2)))
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.all(first[50:] == 17))


if __name__ == "__main__":
    unittest.main()
