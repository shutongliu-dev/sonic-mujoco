import argparse
import time
from pathlib import Path

from sonic_mujoco.controllers.sonic import SonicController, SonicEncoder
from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv, MujocoG1SweepEnv
from sonic_mujoco.teleop import PicoTeleop

REFERENCE_POLICY = Path(
    "/home/yons/lst/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the MuJoCo G1 from PICO poses")
    parser.add_argument(
        "--encoder",
        type=Path,
        default=REFERENCE_POLICY / "model_encoder.onnx",
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=REFERENCE_POLICY / "model_decoder.onnx",
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--scene", choices=("empty", "sweep"), default="empty")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = MujocoG1SweepEnv() if args.scene == "sweep" else MujocoG1EmptyEnv()
    encoder = SonicEncoder.from_onnx(args.encoder)
    controller = SonicController.from_onnx(args.decoder)
    teleop = PicoTeleop(args.endpoint)
    env.reset()
    print(f"Waiting for PICO pose messages on {args.endpoint} ...")

    completed = 0
    started = False
    task_completed = False
    try:
        if not args.headless:
            env.render()
        while args.steps == 0 or completed < args.steps:
            tick = time.monotonic()
            command = teleop.read()
            try:
                token = encoder.encode(env.get_robot_state(), command)
            except RuntimeError:
                if not args.headless and not env.viewer_running:
                    break
                time.sleep(0.01)
                continue

            if not started:
                print("PICO stream received; SONIC control is running.")
                started = True
            robot_command = controller.act(env.get_robot_state(), token)
            env.step(robot_command, steps=controller.steps_per_action)
            completed += 1
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
            control_dt = env.timestep * controller.steps_per_action
            time.sleep(max(0.0, control_dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        teleop.close()
        env.close()

    if started:
        print(f"Stopped after {completed} SONIC control steps.")


if __name__ == "__main__":
    main()
