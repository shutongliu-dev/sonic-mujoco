import unittest

import numpy as np

from sonic_mujoco.contact import MAX_CONTACTS, ContactFrame
from sonic_mujoco.teleop import ContactHaptics


def contact_frame(*contacts: tuple[int, float]) -> ContactFrame:
    body_id = np.full(MAX_CONTACTS, -1, dtype=np.int32)
    impulse = np.zeros(MAX_CONTACTS, dtype=np.float32)
    for index, (body, value) in enumerate(contacts):
        body_id[index] = body
        impulse[index] = value
    return ContactFrame(
        robot_body_id=body_id,
        other_body_id=np.full(MAX_CONTACTS, -1, dtype=np.int32),
        position=np.zeros((MAX_CONTACTS, 3), dtype=np.float32),
        normal_force=np.zeros(MAX_CONTACTS, dtype=np.float32),
        tangent_force=np.zeros(MAX_CONTACTS, dtype=np.float32),
        normal_impulse=impulse,
        sample_count=np.zeros(MAX_CONTACTS, dtype=np.int32),
        count=len(contacts),
    )


class ContactHapticsTest(unittest.TestCase):
    def test_maps_arm_contacts_to_the_matching_controller(self) -> None:
        sent = []

        def send(*values) -> bool:
            sent.append(values)
            return True

        haptics = ContactHaptics(
            ("world", "left_wrist_yaw_link", "right_hand_index_1_link"),
            send,
        )

        self.assertTrue(haptics.update(contact_frame((1, 0.10), (2, 0.25)), 0.0))

        self.assertEqual(len(sent), 1)
        left, right, duration, frequency = sent[0]
        self.assertGreater(left, 0.0)
        self.assertGreater(right, left)
        self.assertLessEqual(right, 0.75)
        self.assertEqual((duration, frequency), (60, 150))

    def test_ignores_leg_contacts_and_rate_limits_pulses(self) -> None:
        sent = []

        def send(*values) -> bool:
            sent.append(values)
            return True

        haptics = ContactHaptics(
            ("world", "left_ankle_roll_link", "right_wrist_yaw_link"),
            send,
        )

        self.assertTrue(haptics.update(contact_frame((1, 1.0)), 0.0))
        self.assertEqual(sent, [])
        self.assertTrue(haptics.update(contact_frame((2, 0.20)), 0.01))
        self.assertEqual(len(sent), 1)
        self.assertTrue(haptics.update(contact_frame((2, 0.20)), 0.03))
        self.assertEqual(len(sent), 1)
        self.assertTrue(haptics.update(contact_frame((2, 0.20)), 0.07))
        self.assertEqual(len(sent), 2)


if __name__ == "__main__":
    unittest.main()
