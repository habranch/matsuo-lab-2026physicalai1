"""Generate demonstrations with a seeded, discrete joint-command operator.

The command vector starts at the episode's initial posture. One joint is selected
at a time and its target changes by ARM_STEP = 0.05 rad. Commands are sent at 20 Hz.

Each episode varies the command interval, pauses during joint selection, lifting
height, lowering order and occasional corrected overshoots. The planned motion
lifts the arm, rotates over the wall and lowers toward the target. Demonstrations
are program-generated; they are not recordings of human operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import CraneSim, SuccessTracker, joint_target

ARM_STEP = 0.05  # joint-command increment [rad]


@dataclass
class OperatorStyle:
    press_interval: tuple = (0.05, 0.10)   # seconds between W/S presses (key repeat), per press
    switch_pause: tuple = (0.15, 0.45)     # extra pause when switching joints
    overshoot_prob: float = 0.2            # chance to overshoot a joint goal by one press
    hold_after: float = 1.0                # keep recording this long after the last press


def lift_pose(task: dict) -> np.ndarray:
    """Elbow-up posture used to pass over the wall: only joint2 / joint3 targets are used."""
    lp = task["lift_pose"]
    return np.deg2rad([0.0, lp["joint2_deg"], lp["joint3_deg"], 0.0])


def goal_pose(task: dict) -> np.ndarray:
    tg = task["target"]
    return joint_target(np.deg2rad(tg["yaw_deg"]), tg["r"], tg["z"], task["tip_pitch"])


def make_plan(start: np.ndarray, task: dict, rng: np.random.Generator) -> list[tuple[int, float]]:
    """Sequence of (joint index, goal angle) the operator works through, one joint at a time."""
    lift = lift_pose(task)
    goal = goal_pose(task)
    jitter = lambda: ARM_STEP * rng.integers(-1, 2)
    plan = [(2, lift[2] + jitter()), (1, lift[1] + jitter()), (0, goal[0])]
    # joint orders for lowering while keeping the gripper clear of the table
    lower_orders = [(2, 3, 1), (1, 3, 2), (3, 2, 1)]
    plan += [(j, goal[j]) for j in lower_orders[rng.integers(len(lower_orders))]]
    return plan


def run_demo(sim: CraneSim, start: np.ndarray, rng: np.random.Generator,
             style: OperatorStyle = OperatorStyle(), max_time: float = 30.0) -> dict:
    """Run one scripted teleop episode from `start`. Returns recorded 20 Hz arrays and outcome."""
    task = sim.task
    info = sim.reset(start)
    cmd = np.asarray(start, float).copy()  # teleop command starts from the start posture
    plan = make_plan(np.asarray(start), task, rng)
    tracker = SuccessTracker(task)

    obs, act, tips, errs, ts = [], [], [], [], []
    selected = 0
    k_plan = 0
    next_press_t = 0.0
    overshoot_done = False
    last_press_t = None
    contact = table = False
    self_min = np.inf
    outcome = None
    t = 0.0
    while t < max_time:
        # --- operator decides key presses for this tick
        if k_plan < len(plan) and t >= next_press_t - 1e-9:
            j, goal = plan[k_plan]
            if j != selected:
                n_sel = min((j - selected) % 4, (selected - j) % 4)  # Q/E presses
                selected = j
                next_press_t = t + n_sel * 0.05 + rng.uniform(*style.switch_pause)
                overshoot_done = False
            else:
                remaining = goal - cmd[j]
                over = (not overshoot_done) and abs(remaining) < ARM_STEP / 2 and rng.random() < style.overshoot_prob
                if abs(remaining) >= ARM_STEP / 2 or over:
                    if over:
                        overshoot_done = True
                        cmd[j] += ARM_STEP * (1 if rng.random() < 0.5 else -1)
                    else:
                        cmd[j] += ARM_STEP * np.sign(remaining)
                    last_press_t = t
                    next_press_t = t + rng.uniform(*style.press_interval)
                    if over:  # notice the overshoot a bit later
                        next_press_t += rng.uniform(*style.switch_pause)
                else:
                    k_plan += 1
                    overshoot_done = False
        # --- record (state before command, command sent at this tick)
        obs.append(info.qpos)
        act.append(cmd.copy())
        tips.append(info.tip)
        errs.append(info.err)
        ts.append(t)
        info = sim.command(cmd)
        t = info.t
        contact |= info.obstacle_contact
        table |= info.table_contact
        self_min = min(self_min, info.self_min_dist)
        if outcome is None:
            outcome = tracker.update(info)
            if outcome == "collision":
                break
        if k_plan >= len(plan) and last_press_t is not None and t - last_press_t >= style.hold_after - 1e-9:
            break
    return {
        "state": np.array(obs, np.float32),
        "action": np.array(act, np.float32),
        "tip": np.array(tips, np.float32),
        "err": np.array(errs, np.float32),
        "t": np.array(ts, np.float32),
        "outcome": outcome or "timeout",
        "first_inside": tracker.first_inside,
        "final_err": float(info.err),
        "obstacle_contact": bool(contact),
        "table_contact": bool(table),
        "self_min_dist": float(self_min),
        "duration": float(t),
        "plan": [(int(j), float(g)) for j, g in plan],
    }
