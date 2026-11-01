"""Collect 40 program-generated demonstrations for conditions B and C.

An episode is accepted when it reaches the target without obstacle contact,
table contact or self-penetration. Rejected attempts remain in the manifest;
the next seed is used for another attempt. Condition A uses B000-B004.

Usage: uv run python scripts/collect_demos.py
"""

import numpy as np
import pandas as pd

from crane_reach.dataset import build_dataset, save_raw
from crane_reach.sim import EXP_ROOT, CraneSim, load_task_config
from crane_reach.states import TRAIN_ANGLES, offset_pose
from crane_reach.teleop import run_demo

EXP = EXP_ROOT / "exp/003_demo_diversity"
ART = EXP_ROOT / "artifacts/003_demo_diversity"
SEED_BASE = {"B": 10000, "C": 20000}
N_EPISODES = 20


def schedule(cond):
    angles = TRAIN_ANGLES[cond]
    per = N_EPISODES // len(angles)
    return [a for a in angles for _ in range(per)]


def main():
    task = load_task_config()
    sim = CraneSim(task=task)
    rows = []
    for cond in ("B", "C"):
        seed = SEED_BASE[cond]
        for i, ang in enumerate(schedule(cond)):
            q0 = offset_pose(task, ang)
            while True:
                demo = run_demo(sim, q0, np.random.default_rng(seed))
                ok = (demo["outcome"] == "success" and not demo["obstacle_contact"]
                      and not demo["table_contact"] and demo["self_min_dist"] > 0)
                ep_id = f"{cond}{i:03d}" if ok else f"{cond}{i:03d}_rej_s{seed}"
                path = ART / "raw" / cond / f"{ep_id}.npz"
                meta = {"episode_id": ep_id, "condition": cond, "start_joint1_deg": ang, "seed": seed}
                sha = save_raw(path, demo, meta)
                rows.append({
                    "episode_id": ep_id, "condition": cond, "adopted": ok,
                    "reject_reason": "" if ok else f"{demo['outcome']},contact={demo['obstacle_contact']},table={demo['table_contact']}",
                    "start_joint1_deg": ang, **{f"start_q{k + 1}": float(q0[k]) for k in range(4)},
                    **{f"measured_q{k + 1}": float(demo['state'][0][k]) for k in range(4)},
                    "seed": seed, "frames": len(demo["t"]), "duration_s": round(len(demo["t"]) / 20, 2),
                    "first_inside_s": round(demo["first_inside"], 2), "final_err_cm": round(100 * demo["final_err"], 3),
                    "self_min_dist_mm": round(1000 * demo["self_min_dist"], 2),
                    "lower_order": "-".join(str(j + 1) for j, _ in demo["plan"][3:]),
                    "in_A": cond == "B" and ok and i < 5,
                    "raw_path": str(path.relative_to(EXP_ROOT)), "sha256": sha,
                })
                seed += 1
                if ok:
                    break
    df = pd.DataFrame(rows)
    df.to_csv(EXP / "data_manifest.csv", index=False)
    adopted = df[df.adopted]
    summary = []
    for cond, sel in (("A", adopted[adopted.in_A]), ("B", adopted[adopted.condition == "B"]),
                      ("C", adopted[adopted.condition == "C"])):
        files = [EXP_ROOT / p for p in sel.raw_path]
        stats = build_dataset(files, ART / "datasets" / f"crane_reach_{cond}", f"crane_reach_{cond}")
        summary.append({"dataset": cond, "episodes": stats["n_episodes"], "frames": stats["n_frames"],
                        "total_sec": stats["total_sec"], "mean_len_sec": round(stats["total_sec"] / stats["n_episodes"], 2),
                        "min_len_frames": min(stats["lengths"]), "max_len_frames": max(stats["lengths"]),
                        "padded_frames": stats["padded_frames"], "episode_ids": " ".join(sel.episode_id)})
    pd.DataFrame(summary).to_csv(EXP / "dataset_summary.csv", index=False)
    print(df.groupby(["condition", "adopted"]).size())
    print(pd.DataFrame(summary).drop(columns="episode_ids").to_string())


if __name__ == "__main__":
    main()
