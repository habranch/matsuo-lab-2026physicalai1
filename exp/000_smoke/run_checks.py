"""Validate the robot model, actuation, contact monitoring and data timing.

Checks cover URDF kinematics/meshes/limits, actuator-driven motion, obstacle
contact at physics substeps, self-penetration monitoring, and the 20 Hz dataset's
joint order and action timestamps.

Usage: uv run python exp/000_smoke/run_checks.py
Outputs: exp/000_smoke/results/checks.json and diagnostic figures.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from crane_reach.sim import ARM_JOINTS, EXP_ROOT, CraneSim, load_task_config, joint_target
from crane_reach.teleop import run_demo

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
URDF_DIR = EXP_ROOT / "worlds" / "assets" / "upstream_urdf"
RESULTS: dict = {}


def rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def rpy_mat(r, p, y):
    return rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)


def read_stl(path):
    d = Path(path).read_bytes()
    n = struct.unpack("<I", d[80:84])[0]
    arr = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return arr["v"].reshape(-1, 3).astype(float)


def parse_urdf():
    """Joint origins/axes and visual origins from crane_plus.xacro; limits from crane_plus.urdf.xacro."""
    txt = (URDF_DIR / "crane_plus.xacro").read_text()
    txt = txt.replace("${M_PI}", str(np.pi)).replace("${M_PI/2}", str(np.pi / 2))
    txt = txt.replace("-${M_PI/2}", str(-np.pi / 2))
    for name, val in re.findall(r'<xacro:property name="(MESH_\w+)" value="([^"]+)"/>', txt):
        txt = txt.replace("${%s}" % name, val)
    root = ET.fromstring(txt)
    joints, visuals = [], {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "joint" and el.get("type") == "revolute":
            o = el.find("origin")
            joints.append({
                "name": el.get("name"), "parent": el.find("parent").get("link"),
                "child": el.find("child").get("link"),
                "xyz": [float(x) for x in o.get("xyz").split()],
                "axis": [float(x) for x in el.find("axis").get("xyz").split()],
            })
        if tag == "link":
            v = el.find("visual")
            if v is not None:
                o = v.find("origin")
                mesh = v.find("geometry/mesh").get("filename").split("/")[-1]
                visuals[el.get("name")] = {"rpy": [float(eval(x)) for x in o.get("rpy").split()], "mesh": mesh}
    lim_txt = (URDF_DIR / "crane_plus.urdf.xacro").read_text()
    limits = {}
    for m in re.finditer(r'JOINT_(\w+)_(LOWER|UPPER)_LIMIT" value="\$\{([-0-9.]+) \* TO_RADIAN - SERVO_HOME\}"', lim_txt):
        limits.setdefault(m.group(1), {})[m.group(2)] = np.deg2rad(float(m.group(3)) - 150.0)
    return joints, visuals, limits


def check_model(sim: CraneSim):
    joints, visuals, limits = parse_urdf()
    m, d = sim.model, sim.data
    # macro parameter names -> concrete names used in the MJCF
    name_map = {"${name_joint_1}": "crane_plus_joint1", "${name_joint_2}": "crane_plus_joint2",
                "${name_joint_3}": "crane_plus_joint3", "${name_joint_4}": "crane_plus_joint4",
                "${name_joint_hand}": "crane_plus_joint_hand"}
    link_map = {"${name_link_base}": "crane_plus_base", "${name_link_1}": "crane_plus_link1",
                "${name_link_2}": "crane_plus_link2", "${name_link_3}": "crane_plus_link3",
                "${name_link_4}": "crane_plus_link4", "${name_link_hand}": "crane_plus_link_hand"}
    lim_key = {"crane_plus_joint1": "1", "crane_plus_joint2": "2", "crane_plus_joint3": "3",
               "crane_plus_joint4": "4", "crane_plus_joint_hand": "HAND"}
    chain = [dict(j, name=name_map[j["name"]], child=link_map[j["child"]]) for j in joints]

    res = {"joints": []}
    for j in chain:
        mj = m.joint(j["name"])
        body = m.body(mj.bodyid[0])
        lo, hi = limits[lim_key[j["name"]]]["LOWER"], limits[lim_key[j["name"]]]["UPPER"]
        res["joints"].append({
            "joint": j["name"], "urdf_xyz": j["xyz"], "mjcf_pos": body.pos.round(6).tolist(),
            "urdf_axis": j["axis"], "mjcf_axis": mj.axis.round(6).tolist(),
            "urdf_limit_deg": [round(np.rad2deg(lo), 2), round(np.rad2deg(hi), 2)],
            "mjcf_limit_deg": np.rad2deg(m.jnt_range[mj.id]).round(2).tolist(),
            "match": bool(np.allclose(j["xyz"], body.pos, atol=1e-9) and np.allclose(j["axis"], mj.axis)
                          and np.allclose([lo, hi], m.jnt_range[mj.id], atol=2e-4)),
        })

    # independent forward kinematics from the URDF numbers vs MuJoCo, random configurations
    rng = np.random.default_rng(0)
    jids = [m.joint(j["name"]).id for j in chain]
    max_pos_err = max_rot_err = 0.0
    max_mesh_err = 0.0
    raw = {}
    for _ in range(200):
        q = rng.uniform(m.jnt_range[jids, 0], m.jnt_range[jids, 1])
        d.qpos[:] = 0
        for jid, qi in zip(jids, q):
            d.qpos[m.jnt_qposadr[jid]] = qi
        mujoco.mj_kinematics(m, d)
        R, p = np.eye(3), np.zeros(3)  # base link at world origin
        frames = {"crane_plus_base": (R.copy(), p.copy())}
        for j, qi in zip(chain, q):
            p = p + R @ np.array(j["xyz"])
            R = R @ rot(j["axis"], qi)
            frames[j["child"]] = (R.copy(), p.copy())
            b = m.body(j["child"]).id
            max_pos_err = max(max_pos_err, np.abs(d.xpos[b] - p).max())
            max_rot_err = max(max_rot_err, np.abs(d.xmat[b].reshape(3, 3) - R).max())
        # visual meshes: raw STL vertices placed by the URDF chain vs MuJoCo-placed (recentred) vertices
        for link, vis in visuals.items():
            link = link_map.get(link, link)
            if link not in frames:
                continue
            Rl, pl = frames[link]
            if vis["mesh"] not in raw:
                raw[vis["mesh"]] = read_stl(EXP_ROOT / "worlds" / "assets" / vis["mesh"])
            v_urdf = (Rl @ rpy_mat(*vis["rpy"]) @ raw[vis["mesh"]].T).T + pl
            gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(link).id and m.geom_group[g] == 2][0]
            mid = m.geom_dataid[gid]
            mv = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            v_mj = (d.geom_xmat[gid].reshape(3, 3) @ mv.T).T + d.geom_xpos[gid]
            err = max(np.abs(v_urdf.min(0) - v_mj.min(0)).max(), np.abs(v_urdf.max(0) - v_mj.max(0)).max(),
                      np.abs(v_urdf.mean(0) - v_mj.mean(0)).max() if len(v_urdf) == len(v_mj) else 0)
            max_mesh_err = max(max_mesh_err, err)
    res["fk_random_configs"] = 200
    res["fk_max_body_pos_err_m"] = float(max_pos_err)
    res["fk_max_body_rot_err"] = float(max_rot_err)
    res["visual_mesh_bbox_max_err_m"] = float(max_mesh_err)

    # tip definition: point on link4's z axis, compare with the finger extents in the link4 frame
    d.qpos[:] = 0
    mujoco.mj_kinematics(m, d)
    b4 = m.body("crane_plus_link4").id
    R4, p4 = d.xmat[b4].reshape(3, 3), d.xpos[b4]
    zmax = {}
    for gname, body in (("link4", "crane_plus_link4"), ("hand", "crane_plus_link_hand")):
        gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id and m.geom_group[g] == 2][0]
        mid = m.geom_dataid[gid]
        mv = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        vw = (d.geom_xmat[gid].reshape(3, 3) @ mv.T).T + d.geom_xpos[gid]
        zmax[gname] = float(((vw - p4) @ R4)[:, 2].max())
    res["tip_site_z_in_link4"] = float(m.site("tip").pos[2])
    res["finger_max_z_in_link4"] = zmax
    res["all_joints_match_urdf"] = all(j["match"] for j in res["joints"])
    RESULTS["1_model"] = res
    print("[1] model:", res["all_joints_match_urdf"], "fk err", max_pos_err, max_rot_err, "mesh", max_mesh_err, zmax)


def check_actuation(sim: CraneSim, task: dict):
    src = (EXP_ROOT / "src" / "crane_reach" / "sim.py").read_text()
    writes = [ln.strip() for ln in src.splitlines() if re.search(r"data\.qpos\[.*\]\s*=", ln)]
    res = {"qpos_assignments_in_sim_py": writes}
    ready = joint_target(0, task["ready_pose"]["tip_r"], task["ready_pose"]["tip_z"], task["tip_pitch"])
    # (a) step command held for 1.5 s: response is gradual and limited by the velocity limit
    sim.reset(ready)
    goal = ready.copy()
    goal[0] += 1.0
    goal[2] += 0.6
    log = []
    for _ in range(30):
        sim.command(goal)
        log.append((sim.data.time, *sim.qpos(), *sim.data.actuator_force[sim.act_ids]))
    log = np.array(log)
    vel = np.abs(np.diff(log[:, 1:5], axis=0)).max(0) / sim.dt_ctrl
    res["step_test"] = {
        "command_delta_rad": [1.0, 0.0, 0.6, 0.0],
        "t_reach_90pct_joint1_s": float(log[np.argmax(log[:, 1] - ready[0] >= 0.9), 0]),
        "max_tick_velocity_rad_s": vel.round(3).tolist(),
        "final_tracking_err_rad": (goal - log[-1, 1:5]).round(4).tolist(),
        "max_abs_actuator_force": np.abs(log[:, 5:9]).max(0).round(3).tolist(),
    }
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(log[:, 0], np.rad2deg(log[:, 1:5]))
    ax[0].axhline(np.rad2deg(goal[0]), ls="--", c="C0", lw=0.8)
    ax[0].axhline(np.rad2deg(goal[2]), ls="--", c="C2", lw=0.8)
    ax[0].set(xlabel="sim time [s]", ylabel="joint [deg]", title="step command (held)")
    ax[0].legend(["j1", "j2", "j3", "j4"], fontsize=7)
    ax[1].plot(log[:, 0], log[:, 5:9])
    ax[1].set(xlabel="sim time [s]", ylabel="actuator force [Nm]", title="actuator torque")
    fig.tight_layout()
    fig.savefig(OUT / "actuation_step.png", dpi=120)
    plt.close(fig)
    # (b) gains off: the arm falls under gravity -> it is a dynamic simulation, not kinematic playback
    sim.reset(ready)
    kp = sim.model.actuator_gainprm[sim.act_ids, 0].copy()
    bias = sim.model.actuator_biasprm[sim.act_ids, :3].copy()
    sim.model.actuator_gainprm[sim.act_ids, 0] = 0
    sim.model.actuator_biasprm[sim.act_ids, :3] = 0
    q0 = sim.qpos()
    for _ in range(int(0.5 / sim.model.opt.timestep)):
        mujoco.mj_step(sim.model, sim.data)
    res["gravity_drop_0p5s_deg"] = np.rad2deg(sim.qpos() - q0).round(2).tolist()
    sim.model.actuator_gainprm[sim.act_ids, 0] = kp
    sim.model.actuator_biasprm[sim.act_ids, :3] = bias
    # (c) steady-state error at rest poses
    for name, q in (("ready", ready),):
        sim.reset(q, settle_time=1.0)
        res[f"steady_state_err_deg_{name}"] = np.rad2deg(q - sim.qpos()).round(3).tolist()
    RESULTS["2_actuation"] = res
    print("[2] actuation:", res)


def check_contact(sim: CraneSim, task: dict):
    res = {}
    ready = joint_target(0, task["ready_pose"]["tip_r"], task["ready_pose"]["tip_z"], task["tip_pitch"])
    tg = task["target"]
    goal = joint_target(np.deg2rad(tg["yaw_deg"]), tg["r"], tg["z"], task["tip_pitch"])
    # (a) intentional contact: straight joint interpolation (ignores the wall) over 3 s
    sim.reset(ready)
    first = None
    n_ticks = 0
    for k in range(80):
        info = sim.command(ready + (goal - ready) * min(1, (k + 1) / 60))
        if info.obstacle_contact:
            n_ticks += 1
            if first is None:
                first = {"t": round(info.t, 3), "geoms": info.contact_geoms,
                         "qpos_deg": np.rad2deg(info.qpos).round(1).tolist()}
    res["naive_interpolation"] = {"contact_detected": first is not None, "first": first, "ticks_with_contact": n_ticks}
    # (b) no contact: teleop path over the wall from the base pose and the eval extremes
    res["teleop_paths"] = []
    for dy in (0.0, -16.5, 16.5):
        st = ready.copy()
        st[0] = np.deg2rad(dy)
        demo = run_demo(sim, st, np.random.default_rng(0))
        # minimum clearance along the recorded path (re-simulate with the distance query)
        sim.reset(st)
        clear = min(sim.command(a) and sim.obstacle_distance() for a in demo["action"])
        res["teleop_paths"].append({"start_joint1_deg": dy, "outcome": demo["outcome"],
                                    "contact_detected": demo["obstacle_contact"],
                                    "min_clearance_m": round(float(clear), 4)})
    # (c) consistency with exact geometry: slow yaw sweep at low height into the wall
    sim.reset(ready)
    rows = []
    for k in range(120):
        q = ready.copy()
        q[0] = np.deg2rad(0.5 * (k + 1))
        info = sim.command(q)
        rows.append((info.t, sim.obstacle_distance(), info.obstacle_contact))
        if info.obstacle_contact:
            break
    rows = np.array(rows, dtype=float)
    res["slow_sweep"] = {"first_contact_t": float(rows[-1, 0]) if rows[-1, 2] else None,
                         "geom_distance_at_previous_tick_m": float(rows[-2, 1]),
                         "geom_distance_at_contact_tick_m": float(rows[-1, 1])}
    # (d) per-substep vs tick-end-only detection on fast low swings
    miss = 0
    rng = np.random.default_rng(1)
    trials = 40
    for _ in range(trials):
        sim.reset(ready)
        speed = rng.uniform(0.2, 0.6)  # rad per tick target jump
        any_sub = any_end = False
        for k in range(40):
            q = ready.copy()
            q[0] = min(speed * (k + 1), np.deg2rad(110))
            info = sim.command(q)
            any_sub |= info.obstacle_contact
            end_now = any(
                sim.obstacle_geom in (c.geom1, c.geom2) and ({c.geom1, c.geom2} & sim.robot_geoms) and c.dist <= 0
                for c in sim.data.contact[: sim.data.ncon]
            )
            any_end |= end_now
            if any_sub:
                break
        miss += int(any_sub and not any_end)
    res["fast_swing_tick_end_only_misses"] = {"trials": trials, "missed_by_tick_end_check": miss}
    RESULTS["3_contact"] = res
    print("[3] contact:", json.dumps(res, indent=1))


def check_self(sim: CraneSim, task: dict):
    res = {"pairs": [f"{sim.model.geom(a).name}-{sim.model.geom(b).name}" for a, b in sim.self_pairs]}
    ready = joint_target(0, task["ready_pose"]["tip_r"], task["ready_pose"]["tip_z"], task["tip_pitch"])
    tg = task["target"]
    goal = joint_target(np.deg2rad(tg["yaw_deg"]), tg["r"], tg["z"], task["tip_pitch"])
    poses = {"zero": np.zeros(4), "ready": ready, "goal": goal}
    res["rest_pose_min_dist_m"] = {}
    for k, q in poses.items():
        sim.reset(q)
        res["rest_pose_min_dist_m"][k] = round(sim.self_distance(), 4)
    # deliberately fold the arm onto itself: the monitor must report penetration (<0)
    sim.reset(np.zeros(4))
    folded = np.array([0.0, 0.0, -2.5, -1.8])
    mins = []
    for _ in range(40):
        mins.append(sim.command(folded).self_min_dist)
    res["folded_command_deg"] = np.rad2deg(folded).round(1).tolist()
    res["folded_min_dist_m"] = round(float(min(mins)), 4)
    res["folded_flagged"] = bool(min(mins) < 0)
    RESULTS["4_self"] = res
    print("[4] self:", res)


def check_timing(sim: CraneSim, task: dict):
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from crane_reach.dataset import build_dataset, save_raw

    ready = joint_target(0, task["ready_pose"]["tip_r"], task["ready_pose"]["tip_z"], task["tip_pitch"])
    tmp = Path(tempfile.mkdtemp())
    demos, files = [], []
    for s in range(2):
        st = ready.copy()
        st[0] = np.deg2rad(5 * s)
        demo = run_demo(sim, st, np.random.default_rng(s))
        demos.append(demo)
        files.append(tmp / f"ep{s}.npz")
        save_raw(files[-1], demo, {"seed": s})
    stats = build_dataset(files, tmp / "ds", "smoke")
    chunk = 20
    dts = {"action": [k / 20 for k in range(chunk)]}
    ds = LeRobotDataset("smoke", root=tmp / "ds", delta_timestamps=dts)
    res = {"dataset": stats, "fps": ds.fps, "feature_names": ds.meta.features["action"]["names"],
           "joint_order": list(ARM_JOINTS)}
    max_err = 0.0
    pad_ok = True
    off = 0
    for e, demo in enumerate(demos):
        n = len(demo["action"])
        for i in [0, 1, n // 2, n - 5, n - 1]:
            item = ds[off + i]
            want = demo["action"][i:i + chunk]
            got = item["action"][: len(want)].numpy()
            max_err = max(max_err, np.abs(got - want).max(), np.abs(item["observation.state"].numpy() - demo["state"][i]).max())
            pad = item["action_is_pad"].numpy()
            pad_ok &= bool((pad == (np.arange(chunk) >= n - i)).all())
        off += n
    res["max_abs_diff_loaded_vs_recorded"] = float(max_err)
    res["action_is_pad_matches_episode_end"] = pad_ok
    # state t is measured before command t is sent, i.e. command t moves state t+1
    d0 = demos[0]
    moved = np.where(np.abs(np.diff(d0["action"], axis=0)).max(1) > 0)[0]
    k = int(moved[0]) + 1  # first tick whose command differs from the previous one
    res["first_command_change_tick"] = k
    res["state_change_at_tick_k_rad"] = float(np.abs(d0["state"][k] - d0["state"][k - 1]).max())
    res["state_change_at_tick_k+1_rad"] = float(np.abs(d0["state"][k + 1] - d0["state"][k]).max())
    res["delta_timestamps_action"] = f"[0.00, 0.05, ..., {(chunk - 1) / 20:.2f}] at fps 20, chunk {chunk}"
    RESULTS["5_timing"] = res
    print("[5] timing:", res)


def main():
    task = load_task_config()
    sim = CraneSim(task=task)
    check_model(sim)
    check_actuation(sim, task)
    check_contact(sim, task)
    check_self(sim, task)
    check_timing(sim, task)
    with open(OUT / "checks.json", "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=float)
    print("written", OUT / "checks.json")


if __name__ == "__main__":
    sys.exit(main())
