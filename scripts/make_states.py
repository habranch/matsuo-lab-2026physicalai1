"""Write the fixed state lists and check each start state before any model is evaluated.

For every state: at rest no obstacle / table contact and no self-penetration, and the scripted
teleop operator can reach the target from it (path exists). Model performance is NOT used here.

Usage: uv run python scripts/make_states.py
"""

import numpy as np
import pandas as pd

from crane_reach.sim import EXP_ROOT, CraneSim, load_task_config
from crane_reach.states import evaluation_states, pilot_states
from crane_reach.teleop import run_demo


def check(sim, rows):
    for r in rows:
        q0 = np.array([r[f"q{i}"] for i in range(1, 5)])
        info = sim.reset(q0)
        r["rest_obstacle_dist_mm"] = round(1000 * sim.obstacle_distance(), 1)
        r["rest_self_dist_mm"] = round(1000 * sim.self_distance(), 1)
        r["rest_table_contact"] = any(
            sim.table_geom in (c.geom1, c.geom2) for c in sim.data.contact[: sim.data.ncon]
        )
        r["start_err_deg_max"] = round(float(np.rad2deg(np.abs(info.qpos - q0)).max()), 3)
        demo = run_demo(sim, q0, np.random.default_rng(12345))
        r["teleop_reachable"] = demo["outcome"] == "success" and not demo["table_contact"]
    return rows


def main():
    task = load_task_config()
    sim = CraneSim(task=task)
    ev = pd.DataFrame(check(sim, evaluation_states(task)))
    pi = pd.DataFrame(check(sim, pilot_states(task)))
    ev.to_csv(EXP_ROOT / "exp/003_demo_diversity/evaluation_states.csv", index=False)
    pi.to_csv(EXP_ROOT / "exp/002_obstacle/pilot_states.csv", index=False)
    cols = ["state_id", "d_joint1_deg", "d_joint2_deg", "d_joint3_deg", "rest_obstacle_dist_mm",
            "rest_self_dist_mm", "rest_table_contact", "teleop_reachable"]
    print(ev[cols + ["category", "known_B", "known_C"]].to_string())
    print(pi[cols].to_string())


if __name__ == "__main__":
    main()
