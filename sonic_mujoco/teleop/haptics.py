from collections.abc import Callable, Sequence

import numpy as np

from ..contact import ContactFrame

HAPTIC_PERIOD = 0.05
HAPTIC_DURATION_MS = 60
HAPTIC_FREQUENCY_HZ = 150


class ContactHaptics:
    """Convert arm contact impulses into rate-limited controller vibration."""

    def __init__(
        self,
        body_names: Sequence[str],
        send: Callable[[float, float, int, int], bool],
    ) -> None:
        self._side = np.zeros(len(body_names), dtype=np.int8)
        for body_id, name in enumerate(body_names):
            if name.startswith(
                ("left_shoulder", "left_elbow", "left_wrist", "left_hand")
            ):
                self._side[body_id] = 1
            elif name.startswith(
                ("right_shoulder", "right_elbow", "right_wrist", "right_hand")
            ):
                self._side[body_id] = 2
        self._send = send
        self._last_send = -HAPTIC_PERIOD

    def update(self, contacts: ContactFrame, now: float) -> bool:
        if now - self._last_send < HAPTIC_PERIOD:
            return True

        impulse = np.zeros(2)
        for index in range(min(contacts.count, len(contacts.robot_body_id))):
            body_id = contacts.robot_body_id[index]
            if 0 <= body_id < len(self._side) and self._side[body_id]:
                impulse[self._side[body_id] - 1] += contacts.normal_impulse[index]

        amplitude = 0.75 * np.sqrt(np.clip((impulse - 0.01) / 0.30, 0.0, 1.0))
        if not np.any(amplitude):
            return True
        self._last_send = now
        return self._send(
            float(amplitude[0]),
            float(amplitude[1]),
            HAPTIC_DURATION_MS,
            HAPTIC_FREQUENCY_HZ,
        )
