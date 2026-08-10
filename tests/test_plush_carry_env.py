import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1PlushCarryEnv


class PlushCarryEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1PlushCarryEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_scene_contains_a_true_deformable(self) -> None:
        self.assertEqual(self.env.model.nflex, 1)
        self.assertGreater(self.env.model.nv, 170)
        flex_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_FLEX, "plush_flex"
        )
        self.assertGreater(self.env.model.flex_vertnum[flex_id], 100)

    def test_reset_is_seeded(self) -> None:
        self.env.reset(seed=8)
        first = self.env.get_scene_state()
        self.env.reset(seed=8)
        repeated = self.env.get_scene_state()

        np.testing.assert_allclose(first.plush_position, repeated.plush_position)
        np.testing.assert_allclose(first.plush_extent, repeated.plush_extent)
        self.assertEqual(first.plush_mass, repeated.plush_mass)
        self.assertEqual(first.plush_stiffness, repeated.plush_stiffness)

    def test_randomization_stays_in_physical_ranges(self) -> None:
        for seed in range(20):
            self.env.reset(seed=seed)
            state = self.env.get_scene_state()
            self.assertTrue(0.9 <= state.plush_mass <= 2.0)
            self.assertTrue(0.70 <= state.plush_stiffness <= 1.30)
            self.assertTrue(0.60 <= state.plush_friction <= 0.95)
            self.assertGreater(
                state.target_table_position[1] - state.source_table_position[1],
                1.5,
            )

    def test_plush_compresses_under_balanced_force(self) -> None:
        self.env.reset(seed=4)
        initial_width = self.env.get_scene_state().plush_extent[1]
        flex_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_FLEX, "plush_flex"
        )
        address = int(self.env.model.flex_vertadr[flex_id])
        count = int(self.env.model.flex_vertnum[flex_id])
        body_ids = np.unique(
            self.env.model.flex_vertbodyid[address : address + count]
        )
        center_y = self.env.get_scene_state().plush_position[1]
        for body_id in body_ids:
            offset = self.env.data.xpos[body_id, 1] - center_y
            self.env.data.xfrc_applied[body_id, 1] = (
                -np.sign(offset)
            )
        for _ in range(25):
            mujoco.mj_step(self.env.model, self.env.data)

        compressed_width = self.env.get_scene_state().plush_extent[1]
        self.assertLess(compressed_width, initial_width - 0.01)

        self.env.data.xfrc_applied[body_ids] = 0.0
        for _ in range(80):
            mujoco.mj_step(self.env.model, self.env.data)
        self.assertGreater(
            self.env.get_scene_state().plush_extent[1], compressed_width + 0.005
        )


if __name__ == "__main__":
    unittest.main()
