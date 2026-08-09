# G1 asset source

Copied from `xiaohu-art/GR00T-WholeBodyControl` at commit `ba0f0e83862c19a0e5f86aa0b4f29786da2727cf`:

- `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`
- only the mesh files referenced by that XML

The XML's `meshdir` is adjusted from `meshes` to `../../robots/g1/meshes` because the new root scene lives under `assets/mujoco/scenes/g1`. Robot bodies, joints, actuators, sensors, and meshes are otherwise unchanged.

Source code and software components in the source repository are licensed under Apache License 2.0. See the source repository's `LICENSE` for details.
