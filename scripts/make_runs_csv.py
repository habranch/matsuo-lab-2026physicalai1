"""Collect run_info.json of all training runs and the evaluation files into exp/003_demo_diversity/runs.csv.

Usage: uv run python scripts/make_runs_csv.py
"""

import json

import pandas as pd

from crane_reach.sim import EXP_ROOT


def main():
    rows = []
    for exp in ("002_obstacle", "003_demo_diversity"):
        for info_path in sorted((EXP_ROOT / "artifacts" / exp / "runs").glob("*/run_info.json")):
            info = json.load(open(info_path))
            run_dir = info_path.parent
            evals = sorted(p.stem.removeprefix("trials_") for p in (EXP_ROOT / "exp").glob(f"*/results/trials_{info['run_id']}_eval*.csv"))
            rows.append({
                "run_id": info["run_id"], "condition": info["dataset"], "seed": info["seed"], "started": info["started"],
                "git_commit": info["git_commit"], "git_dirty": info["git_dirty"],
                "uncommitted_diff": str((run_dir / "uncommitted.diff").relative_to(EXP_ROOT)) if (run_dir / "uncommitted.diff").exists() else "",
                "config": "configs/train_act.yaml", "dataset": info["dataset_root"], "dataset_sha256": info["dataset_sha256"],
                "completed_steps": info["completed_steps"], "selected_step": info.get("selected_step", info["steps"]),
                "evaluated_model": info["final_model"], "train_seconds": info["train_seconds"],
                "device": info["versions"]["device_name"], "artifacts": str(run_dir.relative_to(EXP_ROOT)),
                "evaluations": " ".join(evals), "status": "done",
            })
    df = pd.DataFrame(rows)
    df.to_csv(EXP_ROOT / "exp/003_demo_diversity/runs.csv", index=False)
    print(df[["run_id", "git_commit", "git_dirty", "completed_steps", "evaluations"]].to_string())


if __name__ == "__main__":
    main()
