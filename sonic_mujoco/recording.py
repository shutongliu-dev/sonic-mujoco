import html
import json
import shutil
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from .contact import MAX_CONTACTS, ContactFrame
from .teleop.base import TeleopCommand
from .teleop.pico_direct import PicoControls

DEFAULT_TASKS = {
    "empty": "move the G1 with full-body teleoperation",
    "sweep": (
        "Use your forearm to sweep all objects across the divider from one side "
        "of the table to the other."
    ),
    "chair_lean": (
        "Walk to the chair from the front, turn around, adjust your position, "
        "and gently lean back against the backrest."
    ),
}


class _VideoWriter:
    def __init__(self, path: Path, model, data, fps: int) -> None:
        try:
            import av
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "video recording requires: uv sync --extra recording"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        self._av = av
        self._data = data
        self._renderer = mujoco.Renderer(model, height=480, width=640)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = 640
        self._stream.height = 480
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "ultrafast", "crf": "23"}
        self._time_base = Fraction(1, fps)
        self._frame_index = 0

    def add_frame(self) -> None:
        self._renderer.update_scene(self._data, camera="head_camera")
        frame = self._av.VideoFrame.from_ndarray(
            self._renderer.render(), format="rgb24"
        )
        frame.pts = self._frame_index
        frame.time_base = self._time_base
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._frame_index += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._renderer.close()

    def cancel(self, path: Path) -> None:
        self._container.close()
        self._renderer.close()
        path.unlink(missing_ok=True)


class EpisodeRecorder:
    """Write aligned MuJoCo episodes in the real-robot LeRobot layout."""

    def __init__(
        self,
        directory: str | Path,
        scene: str,
        *,
        model=None,
        data=None,
        body_names: tuple[str, ...] = (),
        fps: int = 50,
        task: str | None = None,
        record_video: bool = True,
    ) -> None:
        if record_video and (model is None or data is None):
            raise ValueError("model and data are required for video recording")
        self.directory = Path(directory).resolve()
        self.scene = scene
        self.model = model
        self.data = data
        self.body_names = body_names
        self.fps = fps
        self.task = task or DEFAULT_TASKS.get(scene, scene)
        self.record_video = record_video
        self._frames: list[dict[str, np.ndarray | float | int]] = []
        self._session: Path | None = None
        self._video: _VideoWriter | None = None
        self._video_path: Path | None = None
        self._episodes: list[dict] = []
        self._stats: list[dict] = []
        self._features: dict | None = None
        self.active = False
        self.last_preview: Path | None = None

    @property
    def session_directory(self) -> Path | None:
        return self._session

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    def start(self) -> None:
        self._frames.clear()
        self.active = True

    def append(
        self,
        *,
        time: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        command: TeleopCommand,
        token: np.ndarray,
        action: np.ndarray,
        controls: PicoControls,
        contacts: ContactFrame,
    ) -> None:
        if not self.active:
            return
        self._ensure_episode_started()
        self._frames.append(
            {
                "observation.qpos": np.asarray(qpos).copy(),
                "observation.qvel": np.asarray(qvel).copy(),
                "action.ctrl": np.asarray(ctrl).copy(),
                "teleop.smpl_joints": command.smpl_joints[-1].reshape(-1).copy(),
                "teleop.body_quat_w": command.root_quaternion[-1].copy(),
                "teleop.reference_joint_position": (
                    command.joint_position[-1].copy()
                ),
                "action.motion_token": np.asarray(token).copy(),
                "action.policy": np.asarray(action).copy(),
                "teleop.pico_input": self._pico_input(controls),
                "observation.contact.robot_body_id": contacts.robot_body_id.copy(),
                "observation.contact.other_body_id": contacts.other_body_id.copy(),
                "observation.contact.position": contacts.position.reshape(-1).copy(),
                "observation.contact.normal_force": contacts.normal_force.copy(),
                "observation.contact.tangent_force": contacts.tangent_force.copy(),
                "observation.contact.normal_impulse": contacts.normal_impulse.copy(),
                "observation.contact.sample_count": contacts.sample_count.copy(),
                "observation.contact.count": int(contacts.count),
                "sim_time": float(time),
            }
        )
        if self._video is not None:
            self._video.add_frame()

    def finish(self) -> Path | None:
        if not self.active:
            return None
        self.active = False
        if not self._frames:
            return None
        assert self._session is not None
        episode_index = len(self._episodes)
        arrays = self._stack_frames(episode_index)
        data_path = self._episode_path("data/chunk-000", episode_index, "parquet")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_parquet(data_path, arrays)
        if self._video is not None:
            self._video.close()
            self._video = None

        self._episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [self.task],
                "length": len(self._frames),
            }
        )
        self._stats.append(
            {"episode_index": episode_index, "stats": self._episode_stats(arrays)}
        )
        self._features = self._features or self._build_features(arrays)
        self._write_metadata()
        self.last_preview = self._write_contact_preview(episode_index, arrays)
        self._frames.clear()
        self._video_path = None
        return data_path

    def abort(self) -> None:
        self.active = False
        self._frames.clear()
        if self._video is not None and self._video_path is not None:
            self._video.cancel(self._video_path)
            self._video = None
            self._video_path = None
        if self._session is not None and not self._episodes:
            shutil.rmtree(self._session, ignore_errors=True)
            self._session = None

    def _ensure_episode_started(self) -> None:
        if self._session is None:
            self._session = self._new_session_directory()
        if self.record_video and self._video is None:
            index = len(self._episodes)
            self._video_path = self._episode_path(
                "videos/chunk-000/observation.images.ego_view", index, "mp4"
            )
            self._video = _VideoWriter(
                self._video_path, self.model, self.data, self.fps
            )

    def _new_session_directory(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S")
        path = self.directory / name
        suffix = 1
        while path.exists():
            path = self.directory / f"{name}-{suffix:02d}"
            suffix += 1
        return path

    def _episode_path(self, group: str, index: int, suffix: str) -> Path:
        assert self._session is not None
        return self._session / group / f"episode_{index:06d}.{suffix}"

    def _stack_frames(self, episode_index: int) -> dict[str, np.ndarray]:
        arrays = {
            key: np.stack([np.asarray(frame[key]) for frame in self._frames])
            for key in self._frames[0]
        }
        length = len(self._frames)
        arrays.update(
            {
                "timestamp": np.arange(length, dtype=np.float32) / self.fps,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(
                    sum(item["length"] for item in self._episodes),
                    sum(item["length"] for item in self._episodes) + length,
                    dtype=np.int64,
                ),
                "task_index": np.zeros(length, dtype=np.int64),
            }
        )
        return arrays

    @staticmethod
    def _write_parquet(path: Path, arrays: dict[str, np.ndarray]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "parquet recording requires: uv sync --extra recording"
            ) from exc

        columns = {}
        for key, values in arrays.items():
            values = np.asarray(values)
            if values.ndim == 1:
                columns[key] = pa.array(values)
                continue
            width = int(np.prod(values.shape[1:]))
            flat = pa.array(values.reshape(-1))
            columns[key] = pa.FixedSizeListArray.from_arrays(flat, width)
        pq.write_table(pa.table(columns), path, compression="zstd")

    def _build_features(self, arrays: dict[str, np.ndarray]) -> dict:
        features = {}
        if self.record_video:
            features["observation.images.ego_view"] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }
        for key, values in arrays.items():
            shape = list(values.shape[1:]) or [1]
            features[key] = {
                "dtype": str(values.dtype),
                "shape": shape,
                "names": None,
            }
        return features

    def _write_metadata(self) -> None:
        assert self._session is not None and self._features is not None
        meta = self._session / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        total_frames = sum(item["length"] for item in self._episodes)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "g1_29dof_mujoco",
            "total_episodes": len(self._episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(self._episodes) if self.record_video else 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": self._features,
            "script_config": {
                "scene": self.scene,
                "contact_source": "mujoco",
                "max_contacts_per_frame": MAX_CONTACTS,
                "contact_body_names": list(self.body_names),
            },
            "discarded_episode_indices": [],
        }
        (meta / "info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n"
        )
        modality = {
            "state": {
                "qpos": {"original_key": "observation.qpos"},
                "qvel": {"original_key": "observation.qvel"},
            },
            "action": {
                "ctrl": {"original_key": "action.ctrl"},
                "motion_token": {"original_key": "action.motion_token"},
                "policy": {"original_key": "action.policy"},
            },
            "video": (
                {
                    "ego_view": {
                        "original_key": "observation.images.ego_view"
                    }
                }
                if self.record_video
                else {}
            ),
            "annotation": {
                "human.task_description": {"original_key": "task_index"}
            },
            "contact": {
                name: {"original_key": f"observation.contact.{name}"}
                for name in (
                    "robot_body_id",
                    "other_body_id",
                    "position",
                    "normal_force",
                    "tangent_force",
                    "normal_impulse",
                    "sample_count",
                    "count",
                )
            },
        }
        (meta / "modality.json").write_text(
            json.dumps(modality, indent=2, ensure_ascii=False) + "\n"
        )
        self._write_jsonl(
            meta / "tasks.jsonl", [{"task_index": 0, "task": self.task}]
        )
        self._write_jsonl(meta / "episodes.jsonl", self._episodes)
        self._write_jsonl(meta / "episodes_stats.jsonl", self._stats)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path.write_text(text)

    @staticmethod
    def _episode_stats(arrays: dict[str, np.ndarray]) -> dict:
        stats = {}
        for key, values in arrays.items():
            if key in {"episode_index", "index", "task_index"}:
                continue
            values = np.asarray(values)
            stats[key] = {
                "min": np.min(values, axis=0).reshape(-1).tolist(),
                "max": np.max(values, axis=0).reshape(-1).tolist(),
                "mean": np.mean(values, axis=0).reshape(-1).tolist(),
                "std": np.std(values, axis=0).reshape(-1).tolist(),
                "count": [len(values)],
            }
        return stats

    def _write_contact_preview(
        self, episode_index: int, arrays: dict[str, np.ndarray]
    ) -> Path:
        assert self._session is not None
        preview_dir = self._session / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        robot_ids = arrays["observation.contact.robot_body_id"]
        other_ids = arrays["observation.contact.other_body_id"]
        interaction = (robot_ids >= 0) & (other_ids != 0)
        normal = np.where(
            interaction, arrays["observation.contact.normal_force"], 0.0
        ).sum(axis=1)
        impulses = arrays["observation.contact.normal_impulse"]
        impulse = np.where(interaction, impulses, 0.0).sum(axis=1)
        counts = interaction.sum(axis=1)
        pairs = Counter()
        for row in range(len(normal)):
            for slot in range(MAX_CONTACTS):
                if interaction[row, slot]:
                    pair = int(robot_ids[row, slot]), int(other_ids[row, slot])
                    pairs[pair] += float(impulses[row, slot])
        summary = {
            "episode_index": episode_index,
            "frames": len(normal),
            "duration_sec": len(normal) / self.fps,
            "contact_frames": int(np.count_nonzero(counts)),
            "max_normal_force_n": float(np.max(normal)),
            "total_normal_impulse_ns": float(np.sum(impulse)),
            "top_contact_pairs": [
                {
                    "robot_body": self._body_name(pair[0]),
                    "other_body": self._body_name(pair[1]),
                    "normal_impulse_ns": value,
                }
                for pair, value in pairs.most_common(5)
            ],
        }
        stem = f"episode_{episode_index:06d}_contact"
        (preview_dir / f"{stem}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        svg_path = preview_dir / f"{stem}.svg"
        svg_path.write_text(self._contact_svg(normal, counts, summary))
        html_path = preview_dir / f"episode_{episode_index:06d}.html"
        html_path.write_text(self._preview_html(episode_index, stem, summary))
        return html_path

    def _body_name(self, body_id: int) -> str:
        if 0 <= body_id < len(self.body_names):
            return self.body_names[body_id]
        return f"body_{body_id}"

    def _preview_html(self, episode_index: int, stem: str, summary: dict) -> str:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(item['robot_body'])}</td>"
            f"<td>{html.escape(item['other_body'])}</td>"
            f"<td>{item['normal_impulse_ns']:.4f} N·s</td>"
            "</tr>"
            for item in summary["top_contact_pairs"]
        )
        video = ""
        if self.record_video:
            source = (
                "../videos/chunk-000/observation.images.ego_view/"
                f"episode_{episode_index:06d}.mp4"
            )
            video = f'<video controls preload="metadata" src="{source}"></video>'
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MuJoCo contact episode</title>
<style>
body{{font:16px sans-serif;background:#0f172a;color:#e5e7eb;margin:32px}}
main{{max-width:1050px;margin:auto}} video,img{{width:100%;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:8px;border-bottom:1px solid #334155}}
</style></head><body><main>
<h1>Episode {episode_index:06d}</h1>
<p>{summary['frames']} frames · {summary['duration_sec']:.2f} s ·
{summary['contact_frames']} interaction-contact frames</p>
{video}
<img src="{stem}.svg" alt="MuJoCo contact force timeline">
<h2>Top contact pairs</h2>
<table><tr><th>G1 body</th><th>Scene body</th><th>Normal impulse</th></tr>{rows}</table>
</main></body></html>
"""

    @staticmethod
    def _contact_svg(
        normal: np.ndarray, counts: np.ndarray, summary: dict
    ) -> str:
        width, height = 1000, 420
        left, top, plot_width, plot_height = 70, 70, 880, 230
        maximum = max(float(np.max(normal)), 1e-6)
        denominator = max(len(normal) - 1, 1)
        points = " ".join(
            f"{left + plot_width * i / denominator:.1f},"
            f"{top + plot_height * (1.0 - value / maximum):.1f}"
            for i, value in enumerate(normal)
        )
        contact_bars = "".join(
            f'<rect x="{left + plot_width * i / len(counts):.1f}" y="315" '
            f'width="{max(1.0, plot_width / len(counts)):.1f}" height="12" '
            f'fill="#ff9f43" opacity="0.9"/>'
            for i, count in enumerate(counts)
            if count > 0
        )
        pair_text = " · ".join(
            f"{item['robot_body']} → {item['other_body']}"
            for item in summary["top_contact_pairs"][:3]
        ) or "no robot-scene contact"
        title = html.escape(
            f"Episode {summary['episode_index']:06d} MuJoCo contact"
        )
        pair_text = html.escape(pair_text)
        summary_text = (
            f"frames={summary['frames']}  "
            f"contact_frames={summary['contact_frames']}  "
            f"max_force={summary['max_normal_force_n']:.3f} N  "
            f"impulse={summary['total_normal_impulse_ns']:.4f} N·s"
        )
        return "\n".join(
            [
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                    f'height="{height}">'
                ),
                '<rect width="100%" height="100%" fill="#111827"/>',
                (
                    f'<text x="{left}" y="36" fill="white" font-size="22" '
                    f'font-family="sans-serif">{title}</text>'
                ),
                (
                    f'<text x="{left}" y="58" fill="#9ca3af" font-size="13" '
                    f'font-family="sans-serif">{pair_text}</text>'
                ),
                (
                    f'<rect x="{left}" y="{top}" width="{plot_width}" '
                    f'height="{plot_height}" fill="#1f2937"/>'
                ),
                (
                    f'<line x1="{left}" y1="{top + plot_height}" '
                    f'x2="{left + plot_width}" y2="{top + plot_height}" '
                    'stroke="#6b7280"/>'
                ),
                (
                    f'<polyline points="{points}" fill="none" stroke="#34d399" '
                    'stroke-width="2"/>'
                ),
                (
                    f'<text x="12" y="{top + 15}" fill="#34d399" '
                    f'font-size="13" font-family="sans-serif">{maximum:.2f} N</text>'
                ),
                (
                    f'<text x="{left}" y="348" fill="#ff9f43" font-size="13" '
                    'font-family="sans-serif">contact-active control frames</text>'
                ),
                contact_bars,
                (
                    f'<text x="{left}" y="385" fill="#d1d5db" font-size="14" '
                    f'font-family="sans-serif">{summary_text}</text>'
                ),
                "</svg>",
                "",
            ]
        )

    @staticmethod
    def _pico_input(controls: PicoControls) -> np.ndarray:
        return np.array(
            [
                controls.left_trigger,
                controls.right_trigger,
                controls.left_grip,
                controls.right_grip,
                controls.a,
                controls.b,
                controls.x,
                controls.y,
                controls.menu,
            ],
            dtype=np.float64,
        )
