"""Convert synchronous 20 Hz episodes to the LeRobot dataset format.

Each frame pairs the four measured arm-joint angles with the command sent at the
same tick. Joint names determine the state/action order. The gripper is excluded.

Recorded frames are retained without duplicating the final frame. LeRobot's
action_is_pad marks predictions beyond the end of an episode. The environment
state duplicates the joint state to support the state-only ACT configuration.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .sim import ARM_JOINTS

FPS = 20
NAMES = list(ARM_JOINTS)
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (4,), "names": NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (4,), "names": NAMES},
    "action": {"dtype": "float32", "shape": (4,), "names": NAMES},
}
TASK_TEXT = "reach the target over the wall"


def save_raw(path: Path, demo: dict, meta: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, state=demo["state"], action=demo["action"], tip=demo["tip"],
                        err=demo["err"], t=demo["t"], meta=json.dumps(meta))
    return sha256(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(raw_files: list[Path], root: Path, repo_id: str) -> dict:
    """Create a LeRobotDataset at `root` from raw npz episodes (in the given order)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, root=root, robot_type="crane_plus",
                               features=FEATURES, use_videos=False)
    lengths = []
    for f in raw_files:
        z = np.load(f)
        state, action = z["state"], z["action"]
        assert state.shape == action.shape and state.shape[1] == 4
        for i in range(len(state)):
            ds.add_frame({
                "observation.state": state[i],
                "observation.environment_state": state[i],
                "action": action[i],
                "task": TASK_TEXT,
            })
        ds.save_episode()
        lengths.append(len(state))
    return {"n_episodes": len(lengths), "n_frames": int(sum(lengths)),
            "total_sec": sum(lengths) / FPS, "lengths": lengths, "padded_frames": 0}
