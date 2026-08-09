from pathlib import Path
import runpy
import struct
import unittest

BRIDGE = Path(__file__).parents[1] / "scripts/pico_video_bridge.py"
PROTOCOL = runpy.run_path(str(BRIDGE))


class PicoVideoProtocolTest(unittest.TestCase):
    def test_decodes_xrrobotkit_open_camera_packet(self) -> None:
        camera = b"head_camera"
        ip = b"192.168.3.10"
        data = (
            b"\xca\xfe\x01"
            + struct.pack("<7i", 1280, 720, 30, 8_000_000, 0, 0, 23456)
            + bytes([len(camera)])
            + camera
            + bytes([len(ip)])
            + ip
        )
        command = b"OPEN_CAMERA\0"
        packet = (
            struct.pack("<i", len(command))
            + command
            + struct.pack("<i", len(data))
            + data
        )

        name, payload = PROTOCOL["parse_command"](packet)
        config = PROTOCOL["parse_camera_config"](payload)

        self.assertEqual(name, "OPEN_CAMERA")
        self.assertEqual((config.width, config.height, config.fps), (1280, 720, 30))
        self.assertEqual((config.ip, config.port), ("192.168.3.10", 23456))
        self.assertFalse(config.hevc)

    def test_rejects_truncated_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            PROTOCOL["parse_command"](b"OPEN")


if __name__ == "__main__":
    unittest.main()
