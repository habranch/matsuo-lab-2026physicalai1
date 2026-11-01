"""Start states: training start postures (B/C), pilot states, and the 25 evaluation states.

All states are offsets from the base start posture ("ready": joint1 = 0, tip at r=0.15 m, z=0.060 m).
The lists are fixed before any model is evaluated and written to CSV by scripts/make_states.py.
"""

from __future__ import annotations

import numpy as np

from .sim import joint_target

TRAIN_ANGLES = {"B": [0.0], "C": [-10.0, -5.0, 0.0, 5.0, 10.0]}  # joint1 offsets [deg]

# five small variations used inside every evaluation angle group: (d_joint1, d_joint2, d_joint3) [deg]
EVAL_VARIANTS = [(0.0, 0.0, 0.0), (-1.5, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 3.0, -3.0), (0.0, -3.0, 3.0)]
EVAL_GROUPS = [  # (group center joint1 [deg], category)
    (-15.0, "outside"), (-7.5, "inside"), (0.0, "base"), (7.5, "inside"), (15.0, "outside"),
]
# pilot states: used only to choose the number of training updates; disjoint from evaluation states
PILOT_STATES = [(-12.5, 1.0, -1.0), (-2.5, 0.0, 0.0), (2.5, -1.0, 1.0), (12.5, 0.0, 0.0),
                (-4.0, 2.0, 0.0), (4.0, 0.0, -2.0), (-9.0, 0.0, 0.0), (9.0, 0.0, 0.0)]


def ready_pose(task: dict) -> np.ndarray:
    rp = task["ready_pose"]
    return joint_target(0.0, rp["tip_r"], rp["tip_z"], task["tip_pitch"])


def offset_pose(task: dict, dj1: float, dj2: float = 0.0, dj3: float = 0.0) -> np.ndarray:
    q = ready_pose(task).copy()
    q[:3] += np.deg2rad([dj1, dj2, dj3])
    return q


def evaluation_states(task: dict) -> list[dict]:
    rows = []
    for center, cat in EVAL_GROUPS:
        for v, (a, b, c) in enumerate(EVAL_VARIANTS):
            j1 = center + a
            # "known" = identical to a start posture used in that condition's demonstrations
            known_b = j1 in TRAIN_ANGLES["B"] and b == 0.0 and c == 0.0
            known_c = j1 in TRAIN_ANGLES["C"] and b == 0.0 and c == 0.0
            q = offset_pose(task, j1, b, c)
            rows.append({
                "state_id": f"g{center:+05.1f}_v{v}", "group_deg": center, "category": cat,
                "d_joint1_deg": j1, "d_joint2_deg": b, "d_joint3_deg": c,
                "known_B": known_b, "known_C": known_c,
                **{f"q{i + 1}": float(q[i]) for i in range(4)},
            })
    return rows


def pilot_states(task: dict) -> list[dict]:
    rows = []
    for i, (a, b, c) in enumerate(PILOT_STATES):
        q = offset_pose(task, a, b, c)
        rows.append({"state_id": f"pilot{i}", "d_joint1_deg": a, "d_joint2_deg": b, "d_joint3_deg": c,
                     **{f"q{k + 1}": float(q[k]) for k in range(4)}})
    return rows
