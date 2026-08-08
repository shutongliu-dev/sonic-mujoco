import argparse
from pathlib import Path
import shutil


DEFAULT_SOURCE = Path(
    "/home/yons/lst/GR00T-WholeBodyControl/external_dependencies/"
    "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the XR SDK into this project")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    destination = root / ".xrobotoolkit"
    bindings = list(args.source.glob("xrobotoolkit_sdk*.so"))
    library = args.source / "lib/libPXREARobotSDK.so"
    if len(bindings) != 1 or not library.is_file():
        raise SystemExit(f"XR SDK files not found under {args.source}")

    (destination / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bindings[0], destination / bindings[0].name)
    shutil.copy2(library, destination / "lib/libPXREARobotSDK.so")
    print(f"XR SDK copied to {destination}")


if __name__ == "__main__":
    main()
