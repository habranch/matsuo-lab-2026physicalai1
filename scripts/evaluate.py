"""Evaluate one trained ACT model (or the command-replay reference) on a fixed state list.

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py --run-id 003_B_seed0 \
      --states exp/003_demo_diversity/evaluation_states.csv --eval-id 003_B_seed0_eval
  uv run python scripts/evaluate.py --replay B000 --states ... --eval-id 003_replayB000_eval
Per-trial rows -> exp/<exp>/results/trials_<eval-id>.csv; trajectories -> artifacts/<exp>/evals/<eval-id>/.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from crane_reach.dataset import sha256
from crane_reach.rollout import ACTController, ReplayController, load_policy, run_trial
from crane_reach.sim import EXP_ROOT, CraneSim, load_task_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--model", help="model dir relative to the run (default: final model)")
    ap.add_argument("--replay", help="episode id of a recorded demonstration to replay open-loop")
    ap.add_argument("--states", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--exp", default="003_demo_diversity")
    ap.add_argument("--run-exp", default="003_demo_diversity")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execution", default=None, help="chunk1 / chunk<N> / ensemble (default: configs/train_act.yaml)")
    args = ap.parse_args()

    task = load_task_config()
    sim = CraneSim(task=task)
    states = pd.read_csv(EXP_ROOT / args.states)
    res_dir = EXP_ROOT / "exp" / args.exp / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    art = EXP_ROOT / "artifacts" / args.exp / "evals" / args.eval_id
    art.mkdir(parents=True, exist_ok=True)

    if args.replay:
        man = pd.read_csv(EXP_ROOT / "exp/003_demo_diversity/data_manifest.csv").set_index("episode_id")
        z = np.load(EXP_ROOT / man.loc[args.replay, "raw_path"])
        ctrl = ReplayController(z["action"])
        model_desc = {"replay": args.replay, "raw_sha256": man.loc[args.replay, "sha256"]}
    else:
        run_dir = EXP_ROOT / "artifacts" / args.run_exp / "runs" / args.run_id
        info_path = run_dir / "run_info.json"  # written when training finishes
        info = json.load(open(info_path)) if info_path.exists() else {}
        model_dir = run_dir / (args.model or f"step_{info.get('selected_step', info.get('steps')):06d}")
        execution = args.execution or yaml.safe_load(open(EXP_ROOT / "configs/train_act.yaml"))["execution"]
        ctrl = ACTController(load_policy(model_dir, args.device, execution), args.device)
        model_desc = {"run_id": args.run_id, "model_dir": str(model_dir.relative_to(EXP_ROOT)),
                      "model_sha256": sha256(model_dir / "model.safetensors"), "execution": execution, "dataset": info.get("dataset", ""),
                      "seed": info.get("seed", "")}

    rows = []
    t0 = time.time()
    for _, s in states.iterrows():
        q0 = np.array([s[f"q{i}"] for i in range(1, 5)])
        r = run_trial(sim, ctrl, q0)
        np.savez_compressed(art / f"{s.state_id}.npz", **r["traj"])
        rows.append({
            "eval_id": args.eval_id, "trial_id": f"{args.eval_id}/{s.state_id}", **model_desc,
            **{k: s[k] for k in s.index if not k.startswith("q") and not k.startswith("rest_") and k != "teleop_reachable"},
            "start_err_deg_max": float(np.rad2deg(np.abs(r["start_measured"] - q0)).max()),
            **{k: r[k] for k in ("outcome", "end_time", "final_err_cm", "min_err_cm", "collision", "contact_geoms",
                                 "table_contact", "self_min_dist_mm", "self_penetration", "time_to_success", "first_inside")},
            "success": r["outcome"] == "success",
        })
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / f"trials_{args.eval_id}.csv", index=False)
    json.dump({"states_file": args.states, "states_sha256": sha256(EXP_ROOT / args.states), **model_desc,
               "task": task, "seconds": round(time.time() - t0, 1)},
              open(art / "eval_info.json", "w"), indent=2, default=str)
    print(f"{args.eval_id}: success {df.success.sum()}/{len(df)}, collision {df.collision.sum()}, "
          f"timeout {(df.outcome == 'timeout').sum()}, self-pen {df.self_penetration.sum()}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
