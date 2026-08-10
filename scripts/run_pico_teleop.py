import argparse
import os
import time
from pathlib import Path

from sonic_mujoco.controllers.sonic import SonicController, SonicEncoder
from sonic_mujoco.envs.mujoco.g1 import (
    MujocoG1BucketCarryEnv,
    MujocoG1ChairLeanEnv,
    MujocoG1EmptyEnv,
    MujocoG1PlushCarryEnv,
    MujocoG1SweepEnv,
    RobotCommand,
)
from sonic_mujoco.recording import EpisodeRecorder
from sonic_mujoco.teleop import (
    ContactHaptics,
    PicoControls,
    PicoEvents,
    PicoTeleop,
    PicoVideo,
    PicoZmqTeleop,
    TeleopMode,
    next_mode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = Path(
    os.environ.get("SONIC_POLICY_DIR", str(PROJECT_ROOT / "models"))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the MuJoCo G1 from PICO poses")
    parser.add_argument(
        "--encoder",
        type=Path,
        default=POLICY_DIR / "model_encoder.onnx",
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=POLICY_DIR / "model_decoder.onnx",
    )
    parser.add_argument(
        "--endpoint",
        help="read PICO poses from an external ZMQ endpoint",
    )
    parser.add_argument("--no-pico-video", action="store_true")
    parser.add_argument(
        "--pico-device",
        default=os.environ.get("SONIC_PICO_DEVICE"),
        help="XRRobotKit device name used for controller haptics",
    )
    parser.add_argument("--video-listen", default="0.0.0.0:13579")
    parser.add_argument("--record-dir", type=Path, default=Path("records"))
    parser.add_argument("--no-record-video", action="store_true")
    parser.add_argument("--task")
    parser.add_argument(
        "--scene",
        choices=(
            "empty",
            "sweep",
            "chair_lean",
            "bucket_carry",
            "plush_carry",
        ),
        default="empty",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model in (args.encoder, args.decoder):
        if not model.is_file():
            raise SystemExit(f"SONIC model not found: {model}")
    environments = {
        "empty": MujocoG1EmptyEnv,
        "sweep": MujocoG1SweepEnv,
        "chair_lean": MujocoG1ChairLeanEnv,
        "bucket_carry": MujocoG1BucketCarryEnv,
        "plush_carry": MujocoG1PlushCarryEnv,
    }
    env = environments[args.scene]()
    encoder = SonicEncoder.from_onnx(args.encoder)
    controller = SonicController.from_onnx(args.decoder)
    teleop = PicoZmqTeleop(args.endpoint) if args.endpoint else PicoTeleop()
    direct = isinstance(teleop, PicoTeleop)
    haptics = None
    if direct and args.pico_device:
        haptics = ContactHaptics(
            env.contacts.body_names,
            lambda left, right, duration, frequency: teleop.send_haptics(
                args.pico_device, left, right, duration, frequency
            ),
        )
    video = None
    if not args.endpoint and not args.no_pico_video:
        video = PicoVideo(env.model, env.data, listen=args.video_listen)
    env.reset()
    mode = TeleopMode.OFF if direct else TeleopMode.POSE
    control_fps = round(1.0 / (env.timestep * controller.steps_per_action))
    recorder = EpisodeRecorder(
        args.record_dir,
        args.scene,
        model=env.model,
        data=env.data,
        body_names=env.contacts.body_names,
        fps=control_fps,
        task=args.task,
        record_video=not args.no_record_video,
    )
    latest_command = None
    if args.endpoint:
        print(f"Waiting for PICO pose messages on {args.endpoint} ...")
    else:
        print("Waiting for PICO body tracking from XRRobotKit ...")
        print("Press A+B+X+Y to arm, then A+X to enter full-body POSE teleop.")
        if video is not None:
            print(f"PICO video control is listening on {args.video_listen}.")
        if haptics is not None:
            print(f"PICO contact haptics enabled for {args.pico_device}.")

    completed = 0
    started = False
    task_completed = False
    latest_token = None
    latest_action = None
    latest_robot_command = None
    try:
        if not args.headless:
            env.render()
        while args.steps == 0 or completed < args.steps:
            tick = time.monotonic()
            stepped = False
            command = teleop.read()
            if command is not None:
                latest_command = command
            events = teleop.pop_events() if direct else PicoEvents()
            controls = teleop.controls if direct else PicoControls()

            updated_mode = next_mode(mode, events)
            if updated_mode is not mode:
                if recorder.active and updated_mode is TeleopMode.OFF:
                    _finish_recording(recorder)
                if (
                    mode is TeleopMode.POSE
                    and updated_mode is TeleopMode.READY
                    and latest_robot_command is not None
                ):
                    state = env.get_robot_state()
                    zero = state.joint_position * 0.0
                    latest_robot_command = RobotCommand(
                        joint_position=state.joint_position,
                        joint_velocity=zero,
                        feedforward_torque=zero,
                        kp=latest_robot_command.kp,
                        kd=latest_robot_command.kd,
                    )
                mode = updated_mode
                encoder.reset()
                controller.reset()
                started = False
                print(_mode_message(mode))

            if events.reset_scene:
                if recorder.active:
                    recorder.abort()
                    print("Recording aborted before scene reset.")
                env.reset()
                encoder.reset()
                controller.reset()
                latest_command = None
                latest_token = None
                latest_action = None
                latest_robot_command = None
                started = False
                task_completed = False
                print("Scene, SONIC history, and controller reset.")
                continue

            if events.abort_recording and recorder.active:
                recorder.abort()
                print("Recording aborted; buffered frames were discarded.")
            elif events.toggle_recording:
                if recorder.active:
                    _finish_recording(recorder)
                elif mode is TeleopMode.POSE:
                    recorder.start()
                    print("Recording started.")
                else:
                    print("Enter POSE mode before starting a recording.")

            if mode is TeleopMode.POSE and not controls.menu:
                try:
                    state = env.get_robot_state()
                    token = encoder.encode(state, command)
                except RuntimeError:
                    token = None
                if token is not None:
                    robot_command = controller.act(state, token)
                    env.step(robot_command, steps=controller.steps_per_action)
                    stepped = True
                    latest_token = token
                    latest_action = controller.last_action.copy()
                    latest_robot_command = robot_command
                    completed += 1
                    if not started:
                        print(
                            "PICO stream received; full-body teleoperation is running."
                        )
                        started = True
                    if recorder.active and latest_command is not None:
                        _append_recording(
                            recorder,
                            env,
                            latest_command,
                            latest_token,
                            latest_action,
                            controls,
                        )
            elif (
                recorder.active
                and mode is not TeleopMode.OFF
                and latest_command is not None
                and latest_token is not None
                and latest_action is not None
                and latest_robot_command is not None
            ):
                env.step(latest_robot_command, steps=controller.steps_per_action)
                stepped = True
                completed += 1
                _append_recording(
                    recorder,
                    env,
                    latest_command,
                    latest_token,
                    latest_action,
                    controls,
                )
            if (
                haptics is not None
                and stepped
                and not haptics.update(env.contacts.last_frame, time.monotonic())
            ):
                print("PICO haptics unavailable; disabling contact vibration.")
                haptics = None
            if (
                not task_completed
                and isinstance(env, MujocoG1SweepEnv)
                and env.is_success()
            ):
                print("Sweep task completed.")
                task_completed = True

            if not args.headless:
                if not env.viewer_running:
                    break
                env.render()
            if video is not None:
                video.render(
                    recording=recorder.active,
                    elapsed_seconds=recorder.duration_seconds,
                )
            control_dt = env.timestep * controller.steps_per_action
            time.sleep(max(0.0, control_dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if recorder.active:
            _finish_recording(recorder)
        teleop.close()
        if video is not None:
            video.close()
        env.close()

    if started:
        print(f"Stopped after {completed} SONIC control steps.")


def _mode_message(mode: TeleopMode) -> str:
    if mode is TeleopMode.OFF:
        return "Control stopped. Press A+B+X+Y to arm it again."
    if mode is TeleopMode.READY:
        return "Control ready. Press A+X to enter full-body POSE teleop."
    return "Full-body POSE teleoperation enabled."


def _finish_recording(recorder: EpisodeRecorder) -> None:
    frames = recorder.frame_count
    duration = recorder.duration_seconds
    path = recorder.finish()
    if path is None:
        print("Recording stopped without any control frames.")
    else:
        print(f"Recording saved to {path}.")
        print(f"Episode summary: {frames} frames, {duration:.2f} seconds.")
        print(f"Contact preview saved to {recorder.last_preview}.")


def _append_recording(
    recorder: EpisodeRecorder,
    env,
    command,
    token,
    action,
    controls: PicoControls,
) -> None:
    recorder.append(
        time=env.time,
        qpos=env.data.qpos,
        qvel=env.data.qvel,
        ctrl=env.data.ctrl,
        command=command,
        token=token,
        action=action,
        controls=controls,
        contacts=env.contacts.last_frame,
    )


if __name__ == "__main__":
    main()
