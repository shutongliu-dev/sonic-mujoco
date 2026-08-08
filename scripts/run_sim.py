import argparse
import time

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal G1 MuJoCo scene")
    parser.add_argument("--headless", action="store_true", help="do not open the viewer")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = MujocoG1EmptyEnv()
    env.reset()

    try:
        if args.headless:
            env.step(args.steps or 1)
            print(f"G1 simulation OK: time={env.time:.3f}s")
            return

        env.render()
        completed = 0
        while env.viewer_running and (args.steps == 0 or completed < args.steps):
            started = time.monotonic()
            env.step()
            env.render()
            completed += 1
            time.sleep(max(0.0, env.timestep - (time.monotonic() - started)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
