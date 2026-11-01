"""MuJoCo simulation of CRANE+ V2 with a 20 Hz joint-position interface.

Each control tick sends a target joint-position vector. The controller reference
moves linearly toward that target over 0.1 s, subject to the joint velocity limit;
position actuators track the reference through the physics simulation.

State/action vectors use ARM_JOINTS name order. The gripper receives a fixed command.
Obstacle contact and non-adjacent link distances are monitored at every physics substep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

EXP_ROOT = Path(__file__).resolve().parents[2]
WORLD_XML = EXP_ROOT / "worlds" / "crane_plus_reach.xml"
TASK_YAML = EXP_ROOT / "configs" / "task.yaml"

ARM_JOINTS = ("crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint3", "crane_plus_joint4")
HAND_JOINT = "crane_plus_joint_hand"
ROBOT_COLLISION_GEOMS = ("link1_c", "link2_c", "link3_c", "link4_c", "hand_c")
# Self-collision is disabled in the physics (contype/conaffinity). Instead, the distance between
# every non-adjacent pair of link collision meshes (convex hulls) is monitored at every physics
# substep, so trajectories that pass links through each other can be flagged and reported.
SELF_PAIRS = (
    ("base_c", "link2_c"), ("base_c", "link3_c"), ("base_c", "link4_c"), ("base_c", "hand_c"),
    ("link1_c", "link3_c"), ("link1_c", "link4_c"), ("link1_c", "hand_c"),
    ("link2_c", "link4_c"), ("link2_c", "hand_c"), ("link3_c", "hand_c"),
)

# planar link lengths (crane_plus.xacro) and tip offset on link4 (site "tip")
SHOULDER_Z = 0.0454 + 0.026
L2, L3, L4_TIP = 0.083, 0.0935, 0.115


def load_task_config(path: Path = TASK_YAML) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- kinematics

def planar_ik(r: float, z: float, pitch: float, elbow_up: bool = True) -> np.ndarray:
    """Return (j2, j3, j4) placing the tip at radius r and height z with link4 at `pitch` from +z.

    Absolute link angles from +z (towards +r): a2 = j2, a3 = j2 - j3, a4 = j2 - j3 - j4
    (joint3/joint4 axes are -y in the URDF).
    """
    wr = r - L4_TIP * np.sin(pitch)
    wz = z - L4_TIP * np.cos(pitch) - SHOULDER_Z
    d2 = wr**2 + wz**2
    c = (d2 - L2**2 - L3**2) / (2 * L2 * L3)
    if abs(c) > 1:
        raise ValueError(f"unreachable r={r} z={z} pitch={pitch}")
    rel = np.arccos(c) * (1 if elbow_up else -1)  # a3 - a2 (elbow up: forearm points further down)
    a2 = np.arctan2(wr, wz) - np.arctan2(L3 * np.sin(rel), L2 + L3 * np.cos(rel))
    a3 = a2 + rel
    return np.array([a2, a2 - a3, a3 - pitch])


def joint_target(yaw: float, r: float, z: float, pitch: float) -> np.ndarray:
    return np.concatenate([[yaw], planar_ik(r, z, pitch)])


# ---------------------------------------------------------------- simulator

@dataclass
class StepInfo:
    t: float
    qpos: np.ndarray
    tip: np.ndarray
    err: float
    obstacle_contact: bool
    table_contact: bool
    contact_geoms: list = field(default_factory=list)
    self_min_dist: float = float("nan")      # min over the tick of the non-adjacent link distance


class CraneSim:
    def __init__(self, xml_path: Path = WORLD_XML, task: dict | None = None):
        self.task = task or load_task_config()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.dadr = np.array([m.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.hand_qadr = m.joint(HAND_JOINT).qposadr[0]
        self.act_ids = np.array([m.actuator(f"a_joint{i}").id for i in (1, 2, 3, 4)])
        self.hand_act = m.actuator("a_hand").id
        self.lo = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 0]
        self.hi = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 1]
        self.robot_geoms = {m.geom(n).id for n in ROBOT_COLLISION_GEOMS}
        self.obstacle_geom = m.geom("obstacle").id
        self.table_geom = m.geom("table").id
        self.tip_site = m.site("tip").id
        self.self_pairs = [(m.geom(a).id, m.geom(b).id) for a, b in SELF_PAIRS]
        self.monitor_self = True
        self.target_pos = m.body("target").pos.copy()

        self.dt_ctrl = 1.0 / self.task["control_hz"]
        self.n_sub = int(round(self.dt_ctrl / m.opt.timestep))
        self.t_traj = self.task["traj_time_from_start"]
        self.vmax = self.task["joint_vel_limit"]
        self.ref = np.zeros(4)
        self._seg_start = np.zeros(4)
        self._seg_goal = np.zeros(4)
        self._seg_t = 0.0

    # -- scene configuration
    def set_target(self, pos):
        self.target_pos = np.asarray(pos, float)
        self.model.body("target").pos[:] = self.target_pos

    # -- episode control
    def reset(self, q0, settle_time: float = 0.5) -> StepInfo:
        """Place the arm at q0 (4,), hold it there, and let it settle (time is not counted)."""
        mujoco.mj_resetData(self.model, self.data)
        q0 = np.clip(np.asarray(q0, float), self.lo, self.hi)
        self.data.qpos[self.qadr] = q0
        self.data.qpos[self.hand_qadr] = self.task["gripper_fixed"]
        self.data.ctrl[self.act_ids] = q0
        self.data.ctrl[self.hand_act] = self.task["gripper_fixed"]
        self.ref = q0.copy()
        self._seg_start = q0.copy()
        self._seg_goal = q0.copy()
        self._seg_t = self.t_traj
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(settle_time / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        return self.info()

    def command(self, target) -> StepInfo:
        """Send one 20 Hz trajectory point (positions, time_from_start=0.1 s) and simulate one tick.

        Contacts are checked at every physics substep; the returned info reports whether any
        obstacle / table contact happened during the tick.
        """
        target = np.clip(np.asarray(target, float), self.lo, self.hi)
        self._seg_start = self.ref.copy()
        self._seg_goal = target
        self._seg_t = 0.0
        obst = table = False
        geoms: set = set()
        self_min = np.inf
        h = self.model.opt.timestep
        for _ in range(self.n_sub):
            self._seg_t = min(self._seg_t + h, self.t_traj)
            desired = self._seg_start + (self._seg_goal - self._seg_start) * (self._seg_t / self.t_traj)
            self.ref = self.ref + np.clip(desired - self.ref, -self.vmax * h, self.vmax * h)
            self.data.ctrl[self.act_ids] = self.ref
            mujoco.mj_step(self.model, self.data)
            o, tb, g = self._contacts()
            obst |= o
            table |= tb
            geoms |= g
            if self.monitor_self:
                self_min = min(self_min, self.self_distance())
        return self.info(obst, table, sorted(geoms), self_min)

    def self_distance(self, distmax: float = 0.05) -> float:
        """Smallest signed distance between non-adjacent link collision meshes (<0: penetration)."""
        return min(
            mujoco.mj_geomDistance(self.model, self.data, a, b, distmax, None) for a, b in self.self_pairs
        )

    def obstacle_distance(self, distmax: float = 0.2) -> float:
        return min(
            mujoco.mj_geomDistance(self.model, self.data, g, self.obstacle_geom, distmax, None)
            for g in self.robot_geoms
        )

    def _contacts(self):
        obst = table = False
        geoms = set()
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0:
                continue
            pair = {c.geom1, c.geom2}
            robot = pair & self.robot_geoms
            if not robot:
                continue
            if self.obstacle_geom in pair:
                obst = True
                geoms |= {self.model.geom(g).name for g in robot}
            if self.table_geom in pair:
                table = True
        return obst, table, geoms

    # -- observations
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site].copy()

    def info(self, obst=False, table=False, geoms=None, self_min=float("nan")) -> StepInfo:
        tip = self.tip()
        return StepInfo(
            t=float(self.data.time),
            qpos=self.qpos(),
            tip=tip,
            err=float(np.linalg.norm(tip - self.target_pos)),
            obstacle_contact=obst,
            table_contact=table,
            contact_geoms=geoms or [],
            self_min_dist=float(self_min),
        )


class SuccessTracker:
    """Success = tip within `radius` of the target continuously for `hold_time` (sim time),
    without obstacle contact, before `time_limit`. Ends on success, contact or timeout."""

    def __init__(self, task: dict):
        s = task["success"]
        self.radius, self.hold, self.limit = s["radius"], s["hold_time"], s["time_limit"]
        self.inside_since: float | None = None
        self.first_inside: float | None = None

    def update(self, info: StepInfo) -> str | None:
        if info.obstacle_contact:
            return "collision"
        if info.err <= self.radius:
            if self.inside_since is None:
                self.inside_since = info.t
                if self.first_inside is None:
                    self.first_inside = info.t
            if info.t - self.inside_since >= self.hold - 1e-9:
                return "success"
        else:
            self.inside_since = None
        if info.t >= self.limit - 1e-9:
            return "timeout"
        return None
