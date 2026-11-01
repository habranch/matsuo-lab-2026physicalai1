# Third-party sources

## CRANE+ V2 model

- Source: https://github.com/rt-net/crane_plus
- Revision: `humble`, commit `87486bb`
- Original files: `crane_plus_description/urdf/crane_plus.xacro`, `crane_plus.urdf.xacro` and visual/collision meshes.
- License: Apache License 2.0. The license text is included as `worlds/assets/LICENSE_crane_plus`.

`worlds/assets/` contains the original meshes and reference URDF files. `worlds/crane_plus_reach.xml` is a MuJoCo model using their dimensions, joint definitions and mesh assets. Its header describes the inertia approximation and collision configuration.

## ACT implementation

- Source: https://github.com/huggingface/lerobot
- Revision: `768e36660d1408c71118a2760f831c037fbfa17d`
- License: Apache License 2.0.

LeRobot is installed as a dependency from the pinned revision in `pyproject.toml` and `uv.lock`.

## Training and joint-command references

- https://github.com/matsuolab/pai_training — revision `b4d6b5459066a80433475a83567504988875248a`. Reference for the ACT training loop and the state/action dataset format.
- https://github.com/matsuolab/physicalai_teleop — revision `0ef2a00`. Reference for discrete joint selection and the 0.05 rad command increment.

The local training and data conversion code is in `scripts/train_act.py` and `src/crane_reach/dataset.py`. Program-generated demonstrations use `src/crane_reach/teleop.py`.
