import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
from sonic_mujoco.tactile import TactileRecorder
from sonic_mujoco.tactile_skin import (
    TACTILE_DEVICE_DIM,
    TACTILE_DEVICE_NAMES,
    JuQiaoTactileAdapter,
    SleeveMount,
    build_juqiao_skin_layout,
)


class JuQiaoTactileSkinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = MujocoG1EmptyEnv()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_layout_matches_real_three_device_contract(self) -> None:
        skin = build_juqiao_skin_layout(self.env.model)
        layout = skin.taxels

        self.assertEqual(layout.region_names, TACTILE_DEVICE_NAMES)
        self.assertEqual(len(layout), 624)
        self.assertEqual(np.count_nonzero(layout.region_id == 0), 112)
        self.assertEqual(np.count_nonzero(layout.region_id == 1), 256)
        self.assertEqual(np.count_nonzero(layout.region_id == 2), 256)
        self.assertEqual(len(skin.subregion), 624)
        self.assertTrue(np.all(skin.area_m2 > 0.0))
        self.assertEqual(skin.mapping_radius_m.shape, (624,))
        self.assertTrue(np.all(skin.mapping_radius_m >= 0.03))
        self.assertTrue(np.all(skin.mapping_radius_m <= 0.06))
        self.assertGreater(np.count_nonzero(skin.mapping_radius_m > 0.04), 0)
        pairs = set(zip(layout.region_id.tolist(), layout.channel_id.tolist()))
        self.assertEqual(len(pairs), 624)
        for device_id in (1, 2):
            np.testing.assert_array_equal(
                np.sort(layout.channel_id[layout.region_id == device_id]),
                np.arange(TACTILE_DEVICE_DIM),
            )

    def test_vest_routes_every_collidable_torso_shell(self) -> None:
        skin = build_juqiao_skin_layout(self.env.model)
        recorder = TactileRecorder(self.env.model, layout=skin.taxels)
        torso = mujoco.mj_name2id(
            self.env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso_link",
        )
        collision_geoms = [
            geom_id
            for geom_id in range(self.env.model.ngeom)
            if int(self.env.model.geom_bodyid[geom_id]) == torso
            and (
                int(self.env.model.geom_contype[geom_id]) != 0
                or int(self.env.model.geom_conaffinity[geom_id]) != 0
            )
        ]

        self.assertGreater(len(collision_geoms), 1)
        for geom_id in collision_geoms:
            self.assertGreater(len(recorder.geom_indices(geom_id)), 0)

    def test_exact_layout_maps_a_real_g1_chest_contact(self) -> None:
        scene = (
            Path(__file__).resolve().parents[1]
            / "sonic_mujoco/assets/mujoco/scenes/g1/empty.xml"
        )
        spec = mujoco.MjSpec.from_file(str(scene))
        fixture = spec.add_equality()
        fixture.type = mujoco.mjtEq.mjEQ_WELD
        fixture.objtype = mujoco.mjtObj.mjOBJ_BODY
        fixture.name1 = "pelvis"
        probe = spec.worldbody.add_body(name="tactile_test_probe")
        probe.mocap = True
        probe_geom = probe.add_geom(name="tactile_test_probe_geom")
        probe_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        probe_geom.size = [0.03, 0.0, 0.0]
        probe_geom.density = 100.0

        model = spec.compile()
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        skin = build_juqiao_skin_layout(model)
        recorder = TactileRecorder(
            model,
            layout=skin.taxels,
            mapping_radius=skin.mapping_radius_m,
        )
        front = np.flatnonzero(np.asarray(skin.subregion) == "front_chest")
        taxel = int(front[len(front) // 2])
        body_id = int(skin.taxels.body_id[taxel])
        body_rotation = data.xmat[body_id].reshape(3, 3)
        world_center = (
            data.xpos[body_id] + body_rotation @ skin.taxels.local_center[taxel]
        )
        probe_body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tactile_test_probe",
        )
        data.mocap_pos[int(model.body_mocapid[probe_body])] = world_center

        recorder.begin()
        mujoco.mj_step(model, data)
        recorder.update(data)
        frame = recorder.finish()

        self.assertGreater(frame.mapped_contact_count, 0)
        self.assertGreater(frame.normal_force.sum(), 0.0)
        active_regions = skin.taxels.region_id[frame.contact]
        np.testing.assert_array_equal(active_regions, np.zeros(len(active_regions)))

    def test_vest_unwired_slots_stay_zero(self) -> None:
        skin = build_juqiao_skin_layout(self.env.model)
        recorder = TactileRecorder(self.env.model, layout=skin.taxels)
        adapter = JuQiaoTactileAdapter(
            skin,
            gain_counts_per_newton=1.0,
            gain_variation=0.0,
        )
        force = np.full(len(recorder.layout), 11.0)
        frame = replace(recorder.last_frame, normal_force=force)

        values = adapter.update(frame, time=0.0).device("vest")

        wired = skin.taxels.channel_id[skin.taxels.region_id == 0]
        self.assertEqual(np.count_nonzero(values), 112)
        self.assertTrue(np.all(values[wired] == 11))
        unwired = np.setdiff1d(np.arange(256), wired)
        self.assertTrue(np.all(values[unwired] == 0))

    def test_real_rate_sample_and_hold_uses_independent_device_phases(self) -> None:
        skin = build_juqiao_skin_layout(self.env.model)
        recorder = TactileRecorder(self.env.model, layout=skin.taxels)
        adapter = JuQiaoTactileAdapter(skin, gain_variation=0.0)
        updates = np.zeros(3, dtype=np.int32)
        previous = np.zeros((3, 256), dtype=np.uint8)

        for index in range(50):
            force = np.full(len(recorder.layout), float(index + 1))
            frame = replace(recorder.last_frame, normal_force=force)
            sampled = adapter.update(frame, time=index / 50.0)
            updates += sampled.updated
            for device_id, changed in enumerate(sampled.updated):
                if not changed:
                    np.testing.assert_array_equal(
                        sampled.values[device_id], previous[device_id]
                    )
            previous = sampled.values

        self.assertTrue(np.all((updates >= 13) & (updates <= 15)), updates)
        self.assertFalse(np.array_equal(previous[0], previous[1]))
        self.assertEqual(
            adapter.metadata["output_semantics"],
            "post_baseline_nonnegative_raw_count",
        )
        self.assertEqual(adapter.metadata["policy_deadband_counts"], 2)
        self.assertEqual(adapter.metadata["random_seed"], 0)
        self.assertEqual(len(adapter.metadata["taxel_gain_sha256"]), 64)

    def test_sleeve_mount_axis_is_explicit(self) -> None:
        rows = build_juqiao_skin_layout(
            self.env.model,
            right_mount=SleeveMount(axial_axis="rows"),
        )
        columns = build_juqiao_skin_layout(
            self.env.model,
            right_mount=SleeveMount(axial_axis="cols"),
        )
        mask = rows.taxels.region_id == 2

        self.assertFalse(
            np.array_equal(
                rows.taxels.local_center[mask],
                columns.taxels.local_center[mask],
            )
        )
        np.testing.assert_array_equal(
            rows.taxels.channel_id[mask], columns.taxels.channel_id[mask]
        )

    def test_layout_hash_changes_with_sleeve_calibration(self) -> None:
        base = build_juqiao_skin_layout(self.env.model)
        shifted = build_juqiao_skin_layout(
            self.env.model,
            right_mount=SleeveMount(circular_shift=1),
        )

        self.assertNotEqual(base.sha256, shifted.sha256)


if __name__ == "__main__":
    unittest.main()
