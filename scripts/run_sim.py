import argparse
import time

import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv, RobotCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal G1 MuJoCo scene")
    parser.add_argument("--headless", action="store_true", help="do not open the viewer")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = MujocoG1EmptyEnv()
    env.reset()
    state = env.get_robot_state()
    command = RobotCommand(
        joint_position=state.joint_position,
        joint_velocity=np.zeros(29),
        feedforward_torque=np.zeros(29),
        kp=np.full(29, 20.0),
        kd=np.full(29, 1.0),
    )

    print(command)

    try:
        if args.headless:
            env.step(command, args.steps or 1)
            print(f"G1 simulation OK: time={env.time:.3f}s")
            return

        env.render()
        completed = 0
        while env.viewer_running and (args.steps == 0 or completed < args.steps):
            started = time.monotonic()
            env.step(command)
            env.render()
            completed += 1
            time.sleep(max(0.0, env.timestep - (time.monotonic() - started)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
