"""Create the 14 candidate start states for exploratory evaluation.

Candidates touching the table or wall at rest, or lacking a successful scripted
trajectory, are excluded before model evaluation. This state set extends the
25-state main evaluation and is analyzed separately.

Usage: uv run python scripts/make_wide_states.py
"""

import numpy as np
import pandas as pd

from crane_reach.sim import EXP_ROOT, CraneSim, load_task_config
from crane_reach.states import offset_pose
from crane_reach.teleop import run_demo

CANDIDATES = [(j1, d2, d3) for j1 in (-45.0, -30.0, -22.5, 20.0, 22.5) for d2, d3 in ((0.0, 0.0), (-8.0, 8.0))]
CANDIDATES += [(0.0, -8.0, 8.0), (0.0, -15.0, 15.0), (0.0, 5.0, -5.0), (0.0, 0.0, -10.0)]


def main():
    task = load_task_config()
    sim = CraneSim(task=task)
    rows = []
    for j1, d2, d3 in CANDIDATES:
        q = offset_pose(task, j1, d2, d3)
        sim.reset(q)
        table = any(sim.table_geom in (c.geom1, c.geom2) for c in sim.data.contact[: sim.data.ncon])
        od, sd = sim.obstacle_distance(), sim.self_distance()
        demo = run_demo(sim, q, np.random.default_rng(12345))
        ok = (not table) and od > 0.005 and sd > 0 and demo["outcome"] == "success" and not demo["table_contact"]
        rows.append({"state_id": f"w{j1:+05.1f}_{d2:+.0f}_{d3:+.0f}", "group_deg": j1,
                     "category": "wide_joint1" if j1 != 0 else "wide_joint23",
                     "d_joint1_deg": j1, "d_joint2_deg": d2, "d_joint3_deg": d3, "known_B": False, "known_C": False,
                     **{f"q{i + 1}": float(q[i]) for i in range(4)},
                     "rest_obstacle_dist_mm": round(1000 * od, 1), "rest_table_contact": table,
                     "teleop_reachable": demo["outcome"] == "success" and not demo["table_contact"], "used": ok})
    df = pd.DataFrame(rows)
    df.to_csv(EXP_ROOT / "exp/004_wide_range/candidate_states.csv", index=False)
    df[df.used].drop(columns="used").to_csv(EXP_ROOT / "exp/004_wide_range/states.csv", index=False)
    print(df[["state_id", "rest_obstacle_dist_mm", "rest_table_contact", "teleop_reachable", "used"]].to_string())


if __name__ == "__main__":
    main()
