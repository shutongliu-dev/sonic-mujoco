"""JuQiao skin-suit layout and hardware-compatible tactile sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import mujoco
import numpy as np

from .tactile import TactileFrame, TaxelLayout

TACTILE_DEVICE_NAMES = ("vest", "left_arm", "right_arm")
TACTILE_DEVICE_DIM = 256
TACTILE_VALID_COUNT = 624
_MAPPING_RADIUS_MIN_M = 0.03
_MAPPING_RADIUS_MAX_M = 0.06
_MAPPING_RADIUS_MARGIN_M = 0.01
_SLEEVE_RAW_CHANNELS = np.asarray(
    tuple(range(128, TACTILE_DEVICE_DIM)) + tuple(range(128)),
    dtype=np.int32,
).reshape(16, 16)
_SLEEVE_RAW_CHANNELS.setflags(write=False)


def _channels(*rows: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(channel for row in rows for channel in row)


# The vest controller emits a sparse 256-byte packet. Each row below is a
# physical garment row; values are the vendor's one-based electrical slots.
_FRONT_CHEST_CHANNELS = _channels(
    (195, 211, 227, 243, 3, 19, 35, 51),
    (196, 212, 228, 244, 4, 20, 36, 52),
    (197, 213, 229, 245, 5, 21, 37, 53),
    (198, 214, 230, 246, 6, 22, 38, 54),
    (199, 215, 231, 247, 7, 23, 39, 55),
    (200, 216, 232, 248, 8, 24, 40, 56),
)
_BACK_CHANNELS = _channels(
    (58, 42, 26, 10, 250, 234, 218, 202),
    (59, 43, 27, 11, 251, 235, 219, 203),
    (60, 44, 28, 12, 252, 236, 220, 204),
    (61, 45, 29, 13, 253, 237, 221, 205),
    (62, 46, 30, 14, 254, 238, 222, 206),
)
_VEST_REGIONS = (
    ("front_chest", 6, 8, _FRONT_CHEST_CHANNELS),
    ("back", 5, 8, _BACK_CHANNELS),
    ("left_arm", 2, 4, (79, 95, 111, 127, 80, 96, 112, 128)),
    ("left_shoulder", 1, 4, (9, 25, 41, 57)),
    ("right_arm", 2, 4, (177, 162, 146, 130, 178, 161, 145, 129)),
    ("right_shoulder", 1, 4, (249, 233, 217, 201)),
)

_TORSO_PROFILE = (
    (0.020, 0.078, 0.112),
    (0.060, 0.082, 0.120),
    (0.100, 0.086, 0.128),
    (0.140, 0.090, 0.136),
    (0.180, 0.093, 0.143),
    (0.220, 0.095, 0.145),
    (0.260, 0.096, 0.150),
    (0.290, 0.094, 0.155),
    (0.310, 0.091, 0.158),
    (0.328, 0.086, 0.160),
    (0.342, 0.080, 0.160),
)


@dataclass(frozen=True, slots=True)
class SleeveMount:
    """Installation-dependent mapping from a raw 16x16 sleeve to the arm."""

    axial_axis: Literal["rows", "cols"] = "rows"
    flip_axial: bool = False
    flip_circular: bool = False
    circular_shift: int = 0

    def __post_init__(self) -> None:
        if self.axial_axis not in {"rows", "cols"}:
            raise ValueError("sleeve axial_axis must be 'rows' or 'cols'")
        if not isinstance(self.circular_shift, int):
            raise TypeError("sleeve circular_shift must be an integer")

    def physical_indices(self, row: int, column: int) -> tuple[int, int]:
        axial, circular = (row, column) if self.axial_axis == "rows" else (column, row)
        if self.flip_axial:
            axial = 15 - axial
        if self.flip_circular:
            circular = 15 - circular
        return axial, (circular + self.circular_shift) % 16

    def as_dict(self) -> dict[str, object]:
        return {
            "axial_axis": self.axial_axis,
            "flip_axial": self.flip_axial,
            "flip_circular": self.flip_circular,
            "circular_shift": self.circular_shift,
        }


@dataclass(frozen=True, slots=True)
class JuQiaoSkinLayout:
    """Exact 624 wired channels embedded on the MuJoCo G1 upper body."""

    taxels: TaxelLayout
    area_m2: np.ndarray
    mapping_radius_m: np.ndarray
    subregion: tuple[str, ...]
    left_mount: SleeveMount
    right_mount: SleeveMount

    @property
    def sha256(self) -> str:
        payload = self.as_dict(include_taxels=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def matches_taxels(self, layout: TaxelLayout) -> bool:
        """Return whether a physical frame uses this exact ordered layout."""

        return _same_layout(layout, self.taxels)

    def as_dict(self, *, include_taxels: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "hardware": "JuQiao V1.0 triple-device tactile skin",
            "devices": list(TACTILE_DEVICE_NAMES),
            "device_shape": [TACTILE_DEVICE_DIM],
            "device_dtype": "uint8",
            "wired_taxels": len(self.taxels),
            "vest_wired_taxels": 112,
            "left_mount": self.left_mount.as_dict(),
            "right_mount": self.right_mount.as_dict(),
            "coordinate_frame": "owning_mujoco_body_local",
            "contact_routing": "all_collidable_geoms_on_owning_body",
            "mapping_radius_m": {
                "minimum": float(np.min(self.mapping_radius_m)),
                "maximum": float(np.max(self.mapping_radius_m)),
                "rule": "nearest_collision_vertex_plus_10mm",
            },
            "channel_indexing": "zero_based_raw_device_slot",
            "source": (
                "GR00T-WholeBodyControl/JuQiao hardware mapping and "
                "G1 tactile garment geometry"
            ),
        }
        if include_taxels:
            result["taxels"] = [
                {
                    "index": index,
                    "device": self.taxels.region_names[
                        int(self.taxels.region_id[index])
                    ],
                    "channel": int(self.taxels.channel_id[index]),
                    "subregion": self.subregion[index],
                    "body_id": int(self.taxels.body_id[index]),
                    "geom_id": int(self.taxels.geom_id[index]),
                    "center_m": self.taxels.local_center[index].tolist(),
                    "normal": self.taxels.local_normal[index].tolist(),
                    "area_m2": float(self.area_m2[index]),
                    "mapping_radius_m": float(self.mapping_radius_m[index]),
                }
                for index in range(len(self.taxels))
            ]
        return result


@dataclass(frozen=True, slots=True)
class JuQiaoTactileFrame:
    """One 50 Hz-compatible sample of the three real skin-suit packets."""

    values: np.ndarray
    source_time: np.ndarray
    updated: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(TACTILE_DEVICE_NAMES), TACTILE_DEVICE_DIM):
            raise ValueError("JuQiao values must have shape (3, 256)")
        if self.values.dtype != np.uint8:
            raise TypeError("JuQiao values must use uint8")
        if self.source_time.shape != (len(TACTILE_DEVICE_NAMES),):
            raise ValueError("JuQiao source_time must have shape (3,)")
        if self.updated.shape != (len(TACTILE_DEVICE_NAMES),):
            raise ValueError("JuQiao updated must have shape (3,)")

    def device(self, name: str) -> np.ndarray:
        try:
            index = TACTILE_DEVICE_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown tactile device: {name}") from exc
        return self.values[index].copy()


class JuQiaoTactileAdapter:
    """Convert physical taxel force to the real robot's asynchronous packets.

    The real hardware does not provide a force-to-count calibration curve. The
    configurable power-law conversion is therefore deliberately separated from
    the physical force sidecar and labelled provisional in :attr:`metadata`.
    """

    def __init__(
        self,
        layout: JuQiaoSkinLayout,
        *,
        sample_rate_hz: float = 14.0,
        stale_timeout_s: float = 0.1,
        gain_counts_per_newton: float = 4.0,
        gamma: float = 1.0,
        force_offset_n: float = 0.0,
        gain_variation: float = 0.08,
        sample_rate_jitter: float = 0.08,
        seed: int = 0,
    ) -> None:
        values = (
            sample_rate_hz,
            stale_timeout_s,
            gain_counts_per_newton,
            gamma,
            force_offset_n,
            gain_variation,
            sample_rate_jitter,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("tactile adapter parameters must be finite")
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be positive")
        if gain_counts_per_newton <= 0.0 or gamma <= 0.0:
            raise ValueError("tactile gain and gamma must be positive")
        if force_offset_n < 0.0:
            raise ValueError("force_offset_n must be nonnegative")
        if not 0.0 <= gain_variation < 1.0:
            raise ValueError("gain_variation must be in [0, 1)")
        if not 0.0 <= sample_rate_jitter < 1.0:
            raise ValueError("sample_rate_jitter must be in [0, 1)")

        self.layout = layout
        self.sample_rate_hz = float(sample_rate_hz)
        self.stale_timeout_s = float(stale_timeout_s)
        self.gain_counts_per_newton = float(gain_counts_per_newton)
        self.gamma = float(gamma)
        self.force_offset_n = float(force_offset_n)
        self.gain_variation = float(gain_variation)
        self.sample_rate_jitter = float(sample_rate_jitter)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self._device_sample_rate = self.sample_rate_hz * rng.uniform(
            1.0 - sample_rate_jitter,
            1.0 + sample_rate_jitter,
            len(TACTILE_DEVICE_NAMES),
        )
        self._taxel_gain = rng.uniform(
            1.0 - gain_variation,
            1.0 + gain_variation,
            len(layout.taxels),
        )
        self._period = 1.0 / self._device_sample_rate
        self._phase = np.arange(len(TACTILE_DEVICE_NAMES)) / (3.0 * self.sample_rate_hz)
        self._held = np.zeros(
            (len(TACTILE_DEVICE_NAMES), TACTILE_DEVICE_DIM), dtype=np.uint8
        )
        self._source_time = np.full(len(TACTILE_DEVICE_NAMES), -1.0)
        self._next_sample = self._phase.copy()
        self.last_frame = self._frame(np.zeros(3, dtype=bool))

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": "triple",
            "devices": list(TACTILE_DEVICE_NAMES),
            "shape": [TACTILE_DEVICE_DIM],
            "dtype": "uint8",
            "record_rate_hz": 50.0,
            "device_sample_rate_hz": self.sample_rate_hz,
            "realized_device_sample_rate_hz": self._device_sample_rate.tolist(),
            "fixed_device_rate_variation": self.sample_rate_jitter,
            "sampling": "independent_phase_sample_and_hold",
            "stale_timeout_s": self.stale_timeout_s,
            "normal_force_only": True,
            "mapping_radius_m": {
                "minimum": float(np.min(self.layout.mapping_radius_m)),
                "maximum": float(np.max(self.layout.mapping_radius_m)),
                "rule": "nearest_collision_vertex_plus_10mm",
            },
            "output_semantics": "post_baseline_nonnegative_raw_count",
            "real_robot_baseline": "100_packet_per_channel_mean",
            "policy_deadband_counts": 2,
            "gain_counts_per_newton": self.gain_counts_per_newton,
            "gamma": self.gamma,
            "force_offset_n": self.force_offset_n,
            "taxel_gain_variation": self.gain_variation,
            "random_seed": self.seed,
            "taxel_gain_sha256": hashlib.sha256(
                self._taxel_gain.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "calibration_status": "provisional_until_force_fixture_calibration",
            "layout_sha256": self.layout.sha256,
            "source_time": {
                "unit": "s",
                "clock": "mujoco_sim_time",
                "invalid_sentinel": -1.0,
            },
            "updated_semantics": "device_sampled_on_this_record_row",
        }

    def reset(self, time: float = 0.0) -> None:
        if not np.isfinite(time):
            raise ValueError("reset time must be finite")
        self._held.fill(0)
        self._source_time.fill(-1.0)
        self._next_sample = float(time) + self._phase
        self.last_frame = self._frame(np.zeros(3, dtype=bool))

    def update(self, frame: TactileFrame, time: float) -> JuQiaoTactileFrame:
        if not _same_layout(frame.layout, self.layout.taxels):
            raise ValueError("tactile frame uses a different taxel layout")
        if not np.isfinite(time):
            raise ValueError("sample time must be finite")
        raw = self._quantize(frame.normal_force)
        updated = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=bool)
        region_ids = self.layout.taxels.region_id
        channels = self.layout.taxels.channel_id
        for device_id in range(len(TACTILE_DEVICE_NAMES)):
            if time + 1e-12 >= self._next_sample[device_id]:
                packet = np.zeros(TACTILE_DEVICE_DIM, dtype=np.uint8)
                mask = region_ids == device_id
                packet[channels[mask]] = raw[mask]
                self._held[device_id] = packet
                self._source_time[device_id] = time
                updated[device_id] = True
                elapsed = time - self._next_sample[device_id]
                period = self._period[device_id]
                periods = math.floor(max(elapsed, 0.0) / period) + 1
                self._next_sample[device_id] += periods * period
            if time - self._source_time[device_id] > self.stale_timeout_s:
                self._held[device_id].fill(0)
        self.last_frame = self._frame(updated)
        return self.last_frame

    def _quantize(self, normal_force: np.ndarray) -> np.ndarray:
        force = np.asarray(normal_force, dtype=np.float64)
        if force.shape != (len(self.layout.taxels),):
            raise ValueError("normal_force does not match JuQiao layout")
        response = np.maximum(force - self.force_offset_n, 0.0) ** self.gamma
        response *= self.gain_counts_per_newton * self._taxel_gain
        return np.clip(np.rint(response), 0, 255).astype(np.uint8)

    def _frame(self, updated: np.ndarray) -> JuQiaoTactileFrame:
        return JuQiaoTactileFrame(
            values=self._held.copy(),
            source_time=self._source_time.copy(),
            updated=np.asarray(updated, dtype=bool).copy(),
        )


@dataclass(frozen=True, slots=True)
class _SkinTaxel:
    body_name: str
    device_id: int
    channel: int
    center: np.ndarray
    normal: np.ndarray
    area_m2: float
    subregion: str


def build_juqiao_skin_layout(
    model: mujoco.MjModel,
    *,
    left_mount: SleeveMount | None = None,
    right_mount: SleeveMount | None = None,
) -> JuQiaoSkinLayout:
    """Build the current real-robot 3-device skin layout on a MuJoCo G1."""

    left_mount = left_mount or SleeveMount()
    right_mount = right_mount or SleeveMount()
    taxels = [
        *_vest_torso_taxels(),
        *_vest_arm_taxels(),
        *_sleeve_taxels("left", 1, left_mount),
        *_sleeve_taxels("right", 2, right_mount),
    ]
    return _compile_skin_layout(model, taxels, left_mount, right_mount)


def _vest_torso_taxels() -> Iterator[_SkinTaxel]:
    for key, rows, columns, one_based_channels in _VEST_REGIONS[:2]:
        front = key == "front_chest"
        width = 0.208
        height = 0.188 if front else 0.180
        z_center = 0.205
        for row, column in np.ndindex(rows, columns):
            direction = -1.0 if front else 1.0
            y = direction * ((column + 0.5) * width / columns - width / 2.0)
            z = z_center + height / 2.0 - (row + 0.5) * height / rows
            center, normal = _torso_surface(1.0 if front else -1.0, z, y)
            channel = one_based_channels[row * columns + column] - 1
            yield _SkinTaxel(
                body_name="torso_link",
                device_id=0,
                channel=channel,
                center=center,
                normal=normal,
                area_m2=width * height / (rows * columns),
                subregion=key,
            )


def _vest_arm_taxels() -> Iterator[_SkinTaxel]:
    vest_regions = {entry[0]: entry for entry in _VEST_REGIONS[2:]}
    for side in ("left", "right"):
        for band, suffix in enumerate(("shoulder", "arm")):
            key = f"{side}_{suffix}"
            _, rows, columns, one_based_channels = vest_regions[key]
            for row, column in np.ndindex(rows, columns):
                angle = math.radians(-70.0 + column * (140.0 / 3.0))
                z = 0.025 - band * 0.046 - row * 0.020
                radius = 0.051
                center = np.array(
                    (radius * math.cos(angle), radius * math.sin(angle), z)
                )
                normal = np.array((math.cos(angle), math.sin(angle), 0.0))
                channel = one_based_channels[row * columns + column] - 1
                yield _SkinTaxel(
                    body_name=f"{side}_shoulder_yaw_link",
                    device_id=0,
                    channel=channel,
                    center=center,
                    normal=normal,
                    area_m2=0.025 * 0.018,
                    subregion=key,
                )


def _sleeve_taxels(
    side: Literal["left", "right"],
    device_id: int,
    mount: SleeveMount,
) -> Iterator[_SkinTaxel]:
    for row, column in np.ndindex(16, 16):
        axial, circular = mount.physical_indices(row, column)
        angle = 2.0 * math.pi * circular / 16.0
        if axial < 8:
            body_name = f"{side}_shoulder_yaw_link"
            radius = 0.051
            center = np.array(
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0.002 - axial * 0.0165,
                )
            )
            normal = np.array((math.cos(angle), math.sin(angle), 0.0))
            subregion = f"{side}_sleeve_upper"
        else:
            body_name = f"{side}_elbow_link"
            radius = 0.048
            center = np.array(
                (
                    0.002 + (axial - 8) * 0.0165,
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                )
            )
            normal = np.array((0.0, math.cos(angle), math.sin(angle)))
            subregion = f"{side}_sleeve_forearm"
        yield _SkinTaxel(
            body_name=body_name,
            device_id=device_id,
            channel=int(_SLEEVE_RAW_CHANNELS[row, column]),
            center=center,
            normal=normal,
            area_m2=0.0165 * (2.0 * math.pi * radius / 16.0),
            subregion=subregion,
        )


def _compile_skin_layout(
    model: mujoco.MjModel,
    taxels: Iterable[_SkinTaxel],
    left_mount: SleeveMount,
    right_mount: SleeveMount,
) -> JuQiaoSkinLayout:
    body_ids: list[int] = []
    geom_ids: list[int] = []
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    device_ids: list[int] = []
    channel_ids: list[int] = []
    areas: list[float] = []
    subregions: list[str] = []
    mounts: dict[str, tuple[int, int]] = {}

    for taxel in taxels:
        mount = mounts.get(taxel.body_name)
        if mount is None:
            body_id = _body_id(model, taxel.body_name)
            mount = body_id, _primary_collision_geom(model, body_id)
            mounts[taxel.body_name] = mount
        body_id, geom_id = mount
        normal = np.asarray(taxel.normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        body_ids.append(body_id)
        geom_ids.append(geom_id)
        centers.append(np.asarray(taxel.center, dtype=np.float64))
        normals.append(normal)
        device_ids.append(taxel.device_id)
        channel_ids.append(taxel.channel)
        areas.append(taxel.area_m2)
        subregions.append(taxel.subregion)

    layout = TaxelLayout(
        body_id=_readonly(np.asarray(body_ids, dtype=np.int32)),
        geom_id=_readonly(np.asarray(geom_ids, dtype=np.int32)),
        local_center=_readonly(np.asarray(centers, dtype=np.float64)),
        local_normal=_readonly(np.asarray(normals, dtype=np.float64)),
        region_id=_readonly(np.asarray(device_ids, dtype=np.int16)),
        channel_id=_readonly(np.asarray(channel_ids, dtype=np.int32)),
        region_names=TACTILE_DEVICE_NAMES,
    )
    area_m2 = _readonly(np.asarray(areas, dtype=np.float64))
    mapping_radius_m = _mapping_radii(model, layout)
    _validate_layout(layout, area_m2, mapping_radius_m)
    return JuQiaoSkinLayout(
        taxels=layout,
        area_m2=area_m2,
        mapping_radius_m=mapping_radius_m,
        subregion=tuple(subregions),
        left_mount=left_mount,
        right_mount=right_mount,
    )


def _torso_surface(side: float, z: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    radius_x, radius_y = _torso_radii(z)

    def x_at(sample_y: float) -> float:
        normalized = float(np.clip(sample_y / radius_y, -1.0, 1.0))
        magnitude = abs(normalized)
        ellipse = radius_x * math.sqrt(max(1.0 - normalized**2, 0.0))
        flat = radius_x - 0.004 * normalized**2
        blend = float(np.clip((magnitude - 0.70) / 0.27, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        return side * (flat * (1.0 - blend) + ellipse * blend)

    epsilon = 1e-4
    derivative = (x_at(y + epsilon) - x_at(y - epsilon)) / (2.0 * epsilon)
    normal = np.array((side, -side * derivative, 0.0))
    normal /= np.linalg.norm(normal)
    return np.array((x_at(y), y, z)), normal


def _torso_radii(z: float) -> tuple[float, float]:
    lower, upper = _TORSO_PROFILE[0], _TORSO_PROFILE[-1]
    for start, end in pairwise(_TORSO_PROFILE):
        if start[0] <= z <= end[0]:
            lower, upper = start, end
            break
    blend = float(np.clip((z - lower[0]) / max(upper[0] - lower[0], 1e-8), 0.0, 1.0))
    return (
        lower[1] * (1.0 - blend) + upper[1] * blend,
        lower[2] * (1.0 - blend) + upper[2] * blend,
    )


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"missing G1 body for tactile skin: {name}")
    return body_id


def _primary_collision_geom(model: mujoco.MjModel, body_id: int) -> int:
    candidates = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not candidates:
        raise ValueError(f"body {body_id} has no collision geom for tactile skin")
    return max(
        candidates,
        key=lambda geom_id: float(np.prod(2.0 * model.geom_aabb[geom_id, 3:])),
    )


def _mapping_radii(model: mujoco.MjModel, layout: TaxelLayout) -> np.ndarray:
    """Size each contact gate from its garment point to the compiled shell."""

    nearest = np.full(len(layout), np.inf, dtype=np.float64)
    for body_id in np.unique(layout.body_id):
        surface_samples = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != int(body_id):
                continue
            if not (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            ):
                continue
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(model.geom_dataid[geom_id])
                address = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                samples = np.asarray(
                    model.mesh_vert[address : address + count],
                    dtype=np.float64,
                )
            else:
                center = np.asarray(model.geom_aabb[geom_id, :3])
                half = np.asarray(model.geom_aabb[geom_id, 3:])
                corners = np.asarray(
                    [
                        center + half * (x, y, z)
                        for x in (-1.0, 1.0)
                        for y in (-1.0, 1.0)
                        for z in (-1.0, 1.0)
                    ]
                )
                faces = np.concatenate(
                    tuple(
                        center[None, :] + np.eye(3)[axis][None, :] * sign * half[axis]
                        for axis in range(3)
                        for sign in (-1.0, 1.0)
                    )
                )
                samples = np.concatenate((corners, faces))
            rotation = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            surface_samples.append(
                samples @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]
            )
        if not surface_samples:
            raise ValueError(
                f"body {body_id} has no collision surface for tactile skin"
            )
        samples = np.concatenate(surface_samples)
        indices = np.flatnonzero(layout.body_id == body_id)
        for index in indices:
            nearest[index] = np.min(
                np.linalg.norm(samples - layout.local_center[index], axis=1)
            )
    if np.any(~np.isfinite(nearest)):
        raise AssertionError(
            "could not measure every tactile point to a collision shell"
        )
    return _readonly(
        np.clip(
            nearest + _MAPPING_RADIUS_MARGIN_M,
            _MAPPING_RADIUS_MIN_M,
            _MAPPING_RADIUS_MAX_M,
        )
    )


def _validate_layout(
    layout: TaxelLayout,
    area_m2: np.ndarray,
    mapping_radius_m: np.ndarray,
) -> None:
    if len(layout) != TACTILE_VALID_COUNT:
        raise AssertionError(f"JuQiao layout has {len(layout)} taxels, expected 624")
    if area_m2.shape != (len(layout),) or np.any(area_m2 <= 0.0):
        raise AssertionError("JuQiao taxel areas must be positive and fixed-shape")
    if mapping_radius_m.shape != (len(layout),) or np.any(mapping_radius_m <= 0.0):
        raise AssertionError("JuQiao mapping radii must be positive and fixed-shape")
    keys = list(zip(layout.region_id.tolist(), layout.channel_id.tolist()))
    if len(set(keys)) != len(keys):
        raise AssertionError("JuQiao device/channel pairs must be unique")
    if np.any(layout.channel_id < 0) or np.any(layout.channel_id >= TACTILE_DEVICE_DIM):
        raise AssertionError("JuQiao channels must be zero-based uint8 packet slots")
    vest = layout.channel_id[layout.region_id == 0]
    if len(vest) != 112:
        raise AssertionError("JuQiao vest must expose exactly 112 wired channels")
    for device_id in (1, 2):
        channels = np.sort(layout.channel_id[layout.region_id == device_id])
        if not np.array_equal(channels, np.arange(TACTILE_DEVICE_DIM)):
            raise AssertionError("JuQiao sleeve must expose every channel once")


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _same_layout(left: TaxelLayout, right: TaxelLayout) -> bool:
    if left is right:
        return True
    return (
        left.region_names == right.region_names
        and np.array_equal(left.body_id, right.body_id)
        and np.array_equal(left.geom_id, right.geom_id)
        and np.array_equal(left.region_id, right.region_id)
        and np.array_equal(left.channel_id, right.channel_id)
        and np.array_equal(left.local_center, right.local_center)
        and np.array_equal(left.local_normal, right.local_normal)
    )
